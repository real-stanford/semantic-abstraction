import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
from dataset import ObjectLocalizationDataset, SceneCompletionDataset
from train_vool import get_losses as vool_get_losses, approach as vool_approaches
from train_ovssc import get_losses as ovssc_get_losses, approach as ovssc_approaches
import utils
from torch.utils.data import DataLoader
import pandas as pd
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

if __name__ == "__main__":
    parser = utils.config_parser()
    parser.add_argument("--task", choices=["ovssc", "vool"], required=True)
    args = parser.parse_args()
    with open(os.path.dirname(args.load) + "/args.pkl", "rb") as file:
        exp_args = pickle.load(file)
        for arg in vars(exp_args):
            if any(arg == s for s in ["device", "file_path", "load", "gpus", "task"]):
                continue
            setattr(args, arg, getattr(exp_args, arg))
    args.domain_randomization = False
    args.scene_bounds = torch.tensor(args.scene_bounds)
    args.batch_size = 1
    args.num_workers = 8
    args.balance_spatial_sampling = False
    args.detailed_analysis = True
    ddp = len(args.gpus) > 1
    approaches = ovssc_approaches if args.task == "ovssc" else vool_approaches
    dataset_class = (
        SceneCompletionDataset if args.task == "ovssc" else ObjectLocalizationDataset
    )
    exp_dict = utils.setup_experiment(
        args=args,
        net_class=approaches[args.approach],
        dataset_class=dataset_class,
        split_file_path=args.file_path + "/vool_split.pkl",
        return_vis=True,
        ddp=ddp,
    )
    net = exp_dict["net"]
    net.eval()
    net.requires_grad = False
    epoch = exp_dict["start_epoch"]
    eval_detailed_stats = pd.DataFrame()
    with torch.no_grad():
        for split, dataset in exp_dict["datasets"].items():
            if split == "train":
                continue
            sampler = None
            if ddp:
                sampler = DistributedSampler(
                    dataset=dataset, shuffle=False, drop_last=False
                )
                sampler.set_epoch(0)
            loader = DataLoader(
                dataset=dataset,
                num_workers=args.num_workers,
                batch_size=1,
                sampler=sampler,
            )
            detailed_stats = utils.loop(
                net=net,
                loader=loader,
                get_losses_fn=ovssc_get_losses
                if args.task == "ovssc"
                else vool_get_losses,
                **{
                    **vars(args),
                    "optimizer": None,
                    "lr_scheduler": None,
                    "cutoffs": np.arange(-2.5, -0.0, 0.1),
                    "pbar": tqdm(
                        total=len(loader),
                        dynamic_ncols=True,
                        unit="batch",
                        postfix=f"| {split.upper()} ",
                    ),
                    "detailed_analysis": True,
                },
            )
            detailed_stats["epoch"] = [epoch] * len(detailed_stats)
            detailed_stats["split"] = [split] * len(detailed_stats)
            eval_detailed_stats = pd.concat([eval_detailed_stats, detailed_stats])
            if (ddp and dist.get_rank() == 0) or not ddp:
                stats_path = os.path.splitext(args.load)[0] + f"_eval_stats.pkl"
                eval_detailed_stats.to_pickle(stats_path)
                print("dumped stats to ", stats_path)
