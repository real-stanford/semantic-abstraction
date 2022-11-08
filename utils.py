from __future__ import annotations
import os
import pickle
import signal
from typing import Optional, Tuple, Type
import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from tqdm import tqdm
from transformers import get_scheduler
from argparse import ArgumentParser
import random
from CLIP.clip import saliency_configs
from net import VirtualGrid
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchtyping import TensorType, patch_typeguard
from arm.optim.lamb import Lamb
from typeguard import typechecked
import logging
from dataset import SceneUnderstandDataset

patch_typeguard()  # use before @typechecked


def config_parser():
    parser = ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--voxel_shape", type=int, default=[128, 128, 128])
    parser.add_argument("--load", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=1024)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--gpus", type=str, nargs="+", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_descs", type=int, default=4)
    parser.add_argument("--saliency_vmin", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--grad_max_norm", type=float, default=2.0)
    parser.add_argument("--xyz_pts_noise", type=float, default=0.0)
    parser.add_argument("--num_input_pts", type=int, default=80000)
    parser.add_argument("--num_output_pts", type=int, default=400000)
    parser.add_argument("--pointing_dim", type=int, default=64)
    parser.add_argument("--unet_f_maps", type=int, default=16)
    parser.add_argument("--unet_num_channels", type=int, default=16)
    parser.add_argument("--unet_num_groups", type=int, default=8)
    parser.add_argument("--unet_num_levels", type=int, default=6)
    parser.add_argument("--num_patches", type=int, default=4)
    parser.add_argument("--patch_mask_cutoff", type=float, default=0.004)
    parser.add_argument("--domain_randomization", action="store_true", default=True)
    parser.add_argument("--use_pts_feat_extractor", action="store_true", default=True)
    parser.add_argument("--pts_feat_extractor_hidden_dim", type=int, default=128)
    parser.add_argument("--subtract_mean_relevancy", action="store_true", default=True)
    parser.add_argument("--offset_patch_mask", action="store_true", default=False)
    parser.add_argument(
        "--balance_positive_negative", action="store_true", default=False
    )
    parser.add_argument(
        "--balance_spatial_relations", action="store_true", default=True
    )
    parser.add_argument(
        "--always_replace_subsample_pts", action="store_true", default=False
    )
    parser.add_argument("--balance_spatial_sampling", action="store_true", default=True)
    parser.add_argument("--decoder_concat_xyz_pts", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dr_pos", type=float, default=0.1)
    parser.add_argument("--dr_orn", type=float, default=0.3)
    parser.add_argument("--dr_scale", type=float, default=0.1)
    parser.add_argument(
        "--scene_bounds", type=list, default=[[-1.0, -1.0, -0.1], [1.0, 1.0, 1.9]]
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pointing_method",
        choices=["cosine_sim", "dot_product", "additive"],
        default="cosine_sim",
    )
    parser.add_argument(
        "--saliency_config", choices=saliency_configs.keys(), default="ours"
    )
    parser.add_argument(
        "--network_inputs",
        nargs="+",
        choices=["patch_masks", "saliency", "rgb", "tsdf"],
        default=["saliency"],
    )
    parser.add_argument(
        "--lr_scheduler_type",
        choices=[
            "constant",
            "linear",
            "cosine",
            "cosine_with_restarts",
            "constant_with_warmup",
        ],
        default="cosine_with_restarts",
    )
    parser.add_argument("--reduce_method", choices=["max", "mean"], default="max")
    return parser


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_experiment(
    args,
    split_file_path: str,
    net_class: Type[torch.nn.Module],
    dataset_class,
    tsdf_shape: Optional[Tuple[int, int, int]] = None,
    return_vis: bool = False,
    **kwargs,
):
    if len(args.gpus) > 1:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        dist.init_process_group(backend="nccl", init_method="env://")
        signal.signal(signal.SIGINT, lambda sig, frame: dist.destroy_process_group())
        if args.device == "cuda":
            torch.cuda.set_device(int(args.gpus[dist.get_rank() % len(args.gpus)]))
    elif args.device == "cuda":
        torch.cuda.set_device(int(args.gpus[0]))
    if not is_main_process():
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.INFO)
    if tsdf_shape is None:
        tsdf_shape = args.voxel_shape
    splits = pickle.load(open(split_file_path, "rb"))
    logging.info("DATASET AT" + args.file_path)
    logging.info(
        " | ".join(
            [
                f"{split_name}: {len(scene_paths)}"
                for split_name, scene_paths in splits.items()
            ]
        )
    )

    loggers = {
        k: SummaryWriter(args.log + f"/{k}") if is_main_process() else None
        for k in splits.keys()
    }
    if is_main_process():
        if os.path.exists(args.log + "/args.pkl"):
            # check if it's very different
            prev_args = pickle.load(open(args.log + "/args.pkl", "rb"))
            logging.warning(
                args.log + "/args.pkl" + " already exists. Differences are;"
            )
            for arg in set(map(str, vars(prev_args).items())) ^ set(
                map(str, vars(args).items())
            ):
                logging.warning(arg)
        else:
            pickle.dump(args, open(args.log + "/args.pkl", "wb"))
    args.scene_bounds = torch.tensor(args.scene_bounds)

    datasets = {
        k: dataset_class(
            scene_paths=splits[k],
            tsdf_shape=tsdf_shape,
            domain_randomized_rgb=(k == "unseen_instances_dr"),
            use_synonyms=(k == "unseen_instances_synonyms"),
            **{
                **vars(args),
                **kwargs,
                **{
                    "domain_randomization": False
                    if k != "train"
                    else args.domain_randomization,
                    "return_vis": k != "train" or return_vis,
                },
            },
        )
        for k in splits.keys()
        if len(splits[k]) > 0
    }

    training_detailed_stats = None
    if os.path.exists(args.log + "/detailed_stats.pkl"):
        training_detailed_stats = pickle.load(
            open(args.log + "/detailed_stats.pkl", "rb")
        )

    net, optimizer, lr_scheduler, start_epoch, scaler = get_net(
        train_dataset=datasets.get("train", None), net_class=net_class, **vars(args)
    )
    return {
        "splits": splits,
        "loggers": loggers,
        "datasets": datasets,
        "net": net,
        "scaler": scaler,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "start_epoch": start_epoch,
        "training_detailed_stats": training_detailed_stats,
    }


def seed_all(seed=0):
    logging.debug(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_net(
    load: str,
    lr: float,
    weight_decay: float,
    lr_scheduler_type: str,
    num_warmup_steps: int,
    epochs: int,
    seed: int,
    net_class: Type[torch.nn.Module],
    use_amp: bool,
    train_dataset: Optional[SceneUnderstandDataset] = None,
    **kwargs,
):
    seed_all(seed)
    device = kwargs["device"]
    batch_size = kwargs["batch_size"]
    kwargs["voxel_shape"] = tuple(kwargs["voxel_shape"])
    net = net_class(**kwargs).to(device)
    if dist.is_initialized():
        net = DistributedDataParallel(
            module=net, device_ids=[device], find_unused_parameters=True
        )
    logging.info(f"NUM PARAMS: {get_n_params(net)}")
    optimizer = Lamb(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        adam=False,
    )
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=epochs * (len(train_dataset) // batch_size)
        if train_dataset is not None
        else 1,
    )
    start_epoch = 0
    if load is not None:
        logging.info(f"loading from {load}")
        ckpt = torch.load(load, map_location=device)
        if dist.is_initialized():
            net.load_state_dict(ckpt["net"])
        else:
            net.load_state_dict(
                {
                    "module.".join(k.split("module.")[1:]): v
                    for k, v in ckpt["net"].items()
                }
            )
        # net.module.steps[...] = 0
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epochs"]
    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.grad_scaler.GradScaler()
    return net, optimizer, lr_scheduler, start_epoch, scaler


def write_to_hdf5(group, key, value, dtype=None, replace=False):
    if value is None:
        return
    if key in group:
        if replace:
            del group[key]
        else:
            raise Exception(f"{key} already present")
    if type(value) == str or type(value) == int or type(value) == float:
        group.attrs[key] = value
    elif type(value) == dict:
        if key in group:
            subgroup = group[key]
        else:
            subgroup = group.create_group(key)
        for subgroup_key, subgroup_value in value.items():
            write_to_hdf5(subgroup, subgroup_key, subgroup_value)
    else:
        group.create_dataset(
            name=key, data=value, dtype=dtype, compression="gzip", compression_opts=9
        )


def compute_grad_norm(net):
    total_norm = 0.0
    for p in net.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


@typechecked
def iou(
    prediction: TensorType[..., "num_points"], label: TensorType[..., "num_points"]
):
    intersection = torch.logical_and(prediction, label).sum(dim=-1).float()
    union = torch.logical_or(prediction, label).sum(dim=-1).float()
    return intersection / union


@typechecked
def prediction_analysis(
    prediction: TensorType["batch", "num_patches", "num_points"],
    label: TensorType["batch", "num_patches", "num_points"],
    ignore: TensorType["batch", "num_patches", "num_points"],
):
    stats = {
        "precision": [],
        "recall": [],
        "false_negative": [],
        "false_positive": [],
        "iou": [],
    }
    for b_i in range(ignore.shape[0]):
        for p_i in range(ignore.shape[1]):
            mask = ~ignore.bool()[b_i, p_i]
            curr_label = label.bool()[b_i, p_i][mask]
            positive_labels = curr_label.bool().float().sum(dim=-1)
            curr_pred = prediction.bool()[b_i, p_i][mask]
            positive_preds = curr_pred.bool().float().sum(dim=-1)
            true_positives = (
                torch.logical_and(curr_label.bool(), curr_pred.bool())
                .float()
                .sum(dim=-1)
            )
            stats["iou"].append(iou(prediction=curr_pred, label=curr_label).item())
            stats["precision"].append(
                true_positives.item() / positive_preds.item()
                if positive_preds.item() != 0
                else np.NAN
            )
            stats["recall"].append(
                true_positives.item() / positive_labels.item()
                if positive_labels.item() != 0
                else np.NAN
            )
            stats["false_negative"].append(
                torch.logical_and(curr_label, ~curr_pred).float().mean(dim=-1).item()
            )
            stats["false_positive"].append(
                torch.logical_and(~curr_label, curr_pred).float().mean(dim=-1).item()
            )
    return stats


def loop(
    net,
    loader,
    pbar,
    get_losses_fn,
    logger: Optional[SummaryWriter] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler=None,
    grad_max_norm: float = 1e5,
    device: torch.device = torch.device("cuda"),
    **kwargs,
):
    epoch_stats = {}
    detailed_stat_df = pd.DataFrame()

    for batch in loader:
        batch = {
            k: (v.to(device) if type(v) == torch.Tensor else v)
            for k, v in batch.items()
        }
        if optimizer:
            stats, detailed_stat = get_losses_fn(net=net, batch=batch, **kwargs)
            optimizer.zero_grad()
            if scaler:
                scaler.scale(stats["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                stats["loss"].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_max_norm)
                optimizer.step()
            lr_scheduler.step()
            if dist.is_initialized():
                net.module.steps += 1
            else:
                net.steps += 1
            stats["gradnorm"] = compute_grad_norm(net)
        else:
            with torch.no_grad():
                stats, detailed_stat = get_losses_fn(net=net, batch=batch, **kwargs)
        # sync stats and detailed_stat_df between different processes
        if dist.is_initialized():
            stats_vector = torch.tensor([stats[k] for k in sorted(stats.keys())]).cuda()
            dist.all_reduce(stats_vector)
            for k, v in zip(sorted(stats.keys()), stats_vector / dist.get_world_size()):
                stats[k] = v.item()

            detailed_stats = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(object_list=detailed_stats, obj=detailed_stat)
            detailed_stat_df = pd.concat([detailed_stat_df] + detailed_stats)
        else:
            detailed_stat_df = pd.concat([detailed_stat_df, detailed_stat])

        for k, v in stats.items():
            v = v.item() if type(v) != float else v
            if k not in epoch_stats:
                epoch_stats[k] = []
            epoch_stats[k].append(v)
            if logger is not None and optimizer is not None:
                logger.add_scalar(
                    k, v, net.module.steps if dist.is_initialized() else net.steps
                )
        if pbar is not None:
            pbar.set_description(
                "|".join(
                    f" {k}: {v*100:.02f} "
                    if any(
                        _k in k
                        for _k in {
                            "iou",
                            "precision",
                            "recall",
                        }
                    )
                    else f" {k}: {v:.04e} "
                    for k, v in stats.items()
                )
            )
            pbar.update()
    epoch_stats = {k: np.nanmean(v) for k, v in epoch_stats.items()}
    if logger is not None and is_main_process():
        for k, v in epoch_stats.items():
            logger.add_scalar(
                f"{k}_mean", v, net.module.steps if dist.is_initialized() else net.steps
            )
    return detailed_stat_df


def train(
    log: str,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    training_detailed_stats: pd.DataFrame,
    start_epoch: int,
    epochs: int,
    datasets: dict,
    loggers: dict,
    splits: dict,
    save_freq: int,
    eval_freq: int,
    num_workers: int,
    batch_size: int,
    get_losses_fn,
    use_amp: bool,
    **kwargs,
):
    for curr_epoch in range(start_epoch, epochs):
        if is_main_process():
            logging.info(f'{"="*10} EPOCH {curr_epoch} {"="*10}')
        for split, dataset in datasets.items():
            if split != "train" and curr_epoch % eval_freq != 0:
                continue
            if split == "train":
                net.train()
            else:
                net.eval()
            if split != "train" and split != "unseen_instances":
                continue
            sampler = None
            if dist.is_initialized():
                sampler = DistributedSampler(
                    dataset=dataset,
                    shuffle=split == "train",
                    drop_last=split == "train",
                )
                sampler.set_epoch(curr_epoch)
            loader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                num_workers=num_workers,
                shuffle=sampler is None and split == "train",
                batch_size=batch_size if split == "train" else 1,
                persistent_workers=num_workers > 0,
            )
            try:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    detailed_stats = loop(
                        net=net,
                        loader=loader,
                        get_losses_fn=get_losses_fn,
                        **{
                            **kwargs,
                            "logger": loggers[split],
                            "optimizer": optimizer if split == "train" else None,
                            "lr_scheduler": lr_scheduler,
                            "pbar": tqdm(
                                total=len(loader),
                                dynamic_ncols=True,
                                unit="batch",
                                smoothing=0.01,
                                postfix=f"| {split.upper()} ",
                            )
                            if is_main_process()
                            else None,
                            "detailed_analysis": False,
                            "cutoffs": [-1.0]
                            if split == "train"
                            else np.arange(-2.7, 0, 0.3),
                        },
                    )
                if is_main_process():
                    ckpt_path = f"{log}/latest.pth"
                    torch.save(
                        {
                            "net": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epochs": curr_epoch + 1,
                        },
                        ckpt_path,
                    )
                detailed_stats["epoch"] = [curr_epoch] * len(detailed_stats)
                detailed_stats["split"] = [split] * len(detailed_stats)
                training_detailed_stats = pd.concat(
                    [training_detailed_stats, detailed_stats]
                )
                training_detailed_stats.to_pickle(log + "/detailed_stats.pkl")
            except Exception as e:
                print(e)
                continue
        epoch_stats = training_detailed_stats[
            training_detailed_stats.epoch == curr_epoch
        ]
        if not is_main_process():
            continue
        for split in splits.keys():
            split_stats = epoch_stats[epoch_stats.split == split]
            if len(split_stats) == 0:
                continue
            logging.info(split.upper())
            for key in filter(
                lambda k: any(
                    metric in k
                    for metric in {
                        "iou",
                        "precision",
                        "recall",
                        "false_negative",
                        "false_positive",
                    }
                ),
                epoch_stats.columns,
            ):
                if len(split_stats) == 0:
                    continue
                best_cutoff = split_stats.groupby("cutoff").mean()[key].idxmax()
                score = split_stats[split_stats.cutoff == best_cutoff][key].mean() * 100
                if pd.isna(score):
                    continue
                logging.info(
                    " " * 4
                    + f"[{key.upper():<30}]:"
                    + f"{score:>6.02f}"
                    + str(best_cutoff).rjust(10)
                )
        logging.info("\n")

        if curr_epoch % save_freq != 0 and curr_epoch != epochs - 1:
            continue
        ckpt_path = f"{log}/ckpt_{curr_epoch}.pth"
        torch.save(
            {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": curr_epoch + 1,
            },
            ckpt_path,
        )
        logging.info(f"Saved checkpoint to {ckpt_path}.\n")


def voxelize_points(
    prediction: TensorType["batch", "num_patches", "num_points"],
    label: TensorType["batch", "num_patches", "num_points"],
    xyz_pts: TensorType["batch", "num_patches", "num_points", 3],
    voxel_shape: Tuple[int, int, int],
    scene_bounds: TensorType[2, 3],
    ignore_pts: TensorType["batch", "num_patches", "num_points"],
    device="cuda",
):
    batch_size, num_patches, num_points = prediction.shape

    prediction = prediction.to(device).float()
    label = (label.to(device).float() - 0.5) * 2
    xyz_pts = xyz_pts.to(device)
    xyz_pts = xyz_pts[:, None, ...].view(batch_size * num_patches, num_points, 3)
    # voxelize
    vg = VirtualGrid(
        scene_bounds=scene_bounds,
        grid_shape=voxel_shape,
        batch_size=batch_size * num_patches,
        device=torch.device(device),
        reduce_method="max",
    )
    voxelized_prediction = vg.scatter_points(
        xyz_pts=xyz_pts, feature_pts=prediction.view(batch_size * num_patches, -1, 1)
    ).view(batch_size, num_patches, *voxel_shape)
    voxelized_label = vg.scatter_points(
        xyz_pts=xyz_pts, feature_pts=label.view(batch_size * num_patches, -1, 1)
    ).view(batch_size, num_patches, *voxel_shape)
    missing_label = voxelized_label == 0.0
    voxelized_label = (voxelized_label > 0).float()
    ignore_vol_mask = (
        vg.scatter_points(
            xyz_pts=xyz_pts,
            feature_pts=ignore_pts.to(device)
            .float()
            .view(batch_size * num_patches, -1, 1),
        )
        .view(batch_size, num_patches, *voxel_shape)
        .bool()
    )
    ignore_vol_mask = torch.logical_or(ignore_vol_mask, missing_label)
    return {
        "prediction": (voxelized_prediction > 0).view(
            batch_size, num_patches, np.prod(voxel_shape)
        ),
        "label": voxelized_label.view(batch_size, num_patches, np.prod(voxel_shape)),
        "ignore": ignore_vol_mask.view(batch_size, num_patches, np.prod(voxel_shape)),
    }


@typechecked
def voxel_score(
    prediction: TensorType["batch", "num_patches", "num_points"],
    label: TensorType["batch", "num_patches", "num_points"],
    xyz_pts: TensorType["batch", "num_patches", "num_points", 3],
    voxel_shape: Tuple[int, int, int],
    scene_bounds: TensorType[2, 3],
    ignore_pts: TensorType["batch", "num_patches", "num_points"],
    out_of_frustum_pts_mask: TensorType["batch", "num_patches", "num_points"],
    score_fn=iou,
    device="cuda",
):
    batch_size, num_patches, num_points = prediction.shape

    prediction = prediction.to(device).float()
    label = (label.to(device).float() - 0.5) * 2
    xyz_pts = xyz_pts.to(device)
    xyz_pts = xyz_pts[:, None, ...].view(batch_size * num_patches, num_points, 3)
    # voxelize
    vg = VirtualGrid(
        scene_bounds=scene_bounds,
        grid_shape=voxel_shape,
        batch_size=batch_size * num_patches,
        device=torch.device(device),
        reduce_method="max",
    )
    voxelized_prediction = vg.scatter_points(
        xyz_pts=xyz_pts, feature_pts=prediction.view(batch_size * num_patches, -1, 1)
    ).view(batch_size, num_patches, *voxel_shape)
    voxelized_label = vg.scatter_points(
        xyz_pts=xyz_pts, feature_pts=label.view(batch_size * num_patches, -1, 1)
    ).view(batch_size, num_patches, *voxel_shape)
    missing_label = voxelized_label == 0.0
    voxelized_label = (voxelized_label > 0).float()
    ignore_vol_mask = (
        vg.scatter_points(
            xyz_pts=xyz_pts,
            feature_pts=torch.logical_or(
                ignore_pts.bool(), out_of_frustum_pts_mask.bool()
            )
            .to(device)
            .float()
            .view(batch_size * num_patches, -1, 1),
        )
        .view(batch_size, num_patches, *voxel_shape)
        .bool()
    )
    ignore_vol_mask = torch.logical_or(ignore_vol_mask, missing_label)
    result = torch.zeros((batch_size, num_patches)).float()
    for b in range(batch_size):
        for p in range(num_patches):
            result[b, p] = score_fn(
                (voxelized_prediction[b, p] > 0)[~ignore_vol_mask[b, p]].bool(),
                (voxelized_label[b, p] > 0)[~ignore_vol_mask[b, p]].bool(),
            )
    return result


@typechecked
def get_bce_weight(
    output_label_pts: TensorType["batch", "num_patches", "num_points"],
    balance_positive_negative: bool,
):
    weight = torch.ones_like(output_label_pts).float()
    if balance_positive_negative:
        weight_total = weight.sum()
        # per instance
        positive_mask = output_label_pts.bool()
        # positive_mask.shape = BATCH x NUM PATCH x NUM PTS
        batch_size, num_patches, num_pts = positive_mask.shape
        percent_positive = positive_mask.float().mean(dim=2).view(-1)
        percent_negative = 1 - percent_positive
        weight = weight.view(-1, num_pts)
        positive_mask = positive_mask.view(-1, num_pts)
        # TODO vectorize this
        assert len(weight) == batch_size * num_patches
        for i in range(len(weight)):
            weight[i, positive_mask[i]] = 1.0 / (percent_positive[i] + 1e-10)
            weight[i, ~positive_mask[i]] = 1.0 / (percent_negative[i] + 1e-10)
        weight = weight.view(output_label_pts.shape)
        weight *= weight_total / weight.sum()
    return weight
