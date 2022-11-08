from typing import Dict, Tuple, Union
import numpy as np
from dataset import ObjectLocalizationDataset
from net import (
    SemAbsVOOL,
    ClipSpatialVOOL,
    SemanticAwareVOOL,
)
import utils
from torch.nn.functional import binary_cross_entropy_with_logits
import torch
import pandas as pd


def get_detailed_stats(
    prediction,
    gt_label,
    xyz_pts,
    scene_ids,
    target_obj_names,
    reference_obj_names,
    spatial_relation_names,
    scene_bounds,
    ignore_pts,
    detailed_analysis=False,
    eval_device="cuda",
    **kwargs,
):
    num_scenes, num_descs = gt_label.shape[:2]

    retvals = {
        "scene_id": np.array([[scene_id] * num_descs for scene_id in scene_ids])
        .reshape(-1)
        .tolist(),
        "target_obj_name": np.array(target_obj_names).T.reshape(-1).tolist(),
        "reference_obj_name": np.array(reference_obj_names).T.reshape(-1).tolist(),
        "spatial_relation_name": np.array(spatial_relation_names)
        .T.reshape(-1)
        .tolist(),
    }
    retvals.update(
        {
            f"point_{k}": v
            for k, v in utils.prediction_analysis(
                prediction=prediction.to(eval_device),
                label=gt_label.to(eval_device),
                ignore=ignore_pts.to(eval_device),
            ).items()
        }
    )
    num_desc_b = 10
    outputs = []
    for i in np.arange(0, num_descs + num_desc_b + 1, num_desc_b):
        if np.prod(prediction[:, i : i + num_desc_b].shape) == 0:
            continue
        outputs.append(
            utils.voxelize_points(
                prediction=prediction[:, i : i + num_desc_b],
                label=gt_label[:, i : i + num_desc_b],
                xyz_pts=xyz_pts[:, i : i + num_desc_b],
                voxel_shape=(32, 32, 32),
                scene_bounds=scene_bounds,
                ignore_pts=ignore_pts[:, i : i + num_desc_b],
                device=eval_device,
            )
        )
    voxelized_pts = {
        k: torch.cat([output[k] for output in outputs], dim=1)
        for k in outputs[0].keys()
    }
    retvals.update(
        {
            "voxel32x32x32_" + k: v
            for k, v in utils.prediction_analysis(
                **{k: v.to(eval_device) for k, v in voxelized_pts.items()}
            ).items()
        }
    )
    if detailed_analysis:
        outputs = []
        for i in np.arange(0, num_descs + num_desc_b + 1, num_desc_b):
            if np.prod(prediction[:, i : i + num_desc_b].shape) == 0:
                continue
            outputs.append(
                utils.voxelize_points(
                    prediction=prediction[:, i : i + num_desc_b],
                    label=gt_label[:, i : i + num_desc_b],
                    xyz_pts=xyz_pts[:, i : i + num_desc_b],
                    voxel_shape=(64, 64, 64),
                    scene_bounds=scene_bounds,
                    ignore_pts=ignore_pts[:, i : i + num_desc_b],
                    device=eval_device,
                )
            )
        voxelized_pts = {
            k: torch.cat([output[k] for output in outputs], dim=1)
            for k in outputs[0].keys()
        }
        retvals.update(
            {
                "voxel64x64x64_" + k: v
                for k, v in utils.prediction_analysis(
                    **{k: v.to(eval_device) for k, v in voxelized_pts.items()}
                ).items()
            }
        )

    for i, spatial_relation in enumerate(
        np.array(spatial_relation_names).T.reshape(-1)
    ):
        if spatial_relation == "[pad]":  # skip padding classes
            for k in retvals.keys():
                if "voxel" in k or "point" in k:
                    retvals[k][i] = np.NAN
    return pd.DataFrame.from_dict(retvals)


def get_losses(
    net, batch: dict, cutoffs=[-2.0], balance_positive_negative: bool = False, **kwargs
) -> Tuple[Dict[str, Union[float, torch.Tensor]], pd.DataFrame]:
    stats = {}
    batch_size, total_num_descs, num_pts = batch["output_label_pts"].shape
    if num_pts <= 500000:
        outputs = net(**batch)
    else:
        num_descs = 1
        # probably CUDA OOM
        outputs = torch.cat(
            [
                net(
                    **{
                        **batch,
                        "input_target_saliency_pts": batch["input_target_saliency_pts"][
                            :, desc_i * num_descs : (desc_i + 1) * num_descs, ...
                        ],
                        "input_reference_saliency_pts": batch[
                            "input_reference_saliency_pts"
                        ][:, desc_i * num_descs : (desc_i + 1) * num_descs, ...],
                        "input_description_saliency_pts": batch[
                            "input_description_saliency_pts"
                        ][:, desc_i * num_descs : (desc_i + 1) * num_descs, ...],
                        "output_xyz_pts": batch["output_xyz_pts"][
                            :, desc_i * num_descs : (desc_i + 1) * num_descs, ...
                        ],
                        "spatial_relation_name": (
                            np.array(batch["spatial_relation_name"])
                            .T[:, desc_i * num_descs : (desc_i + 1) * num_descs]
                            .T
                        ),
                    }
                )
                for desc_i in range(total_num_descs // num_descs + 1)
                if np.prod(
                    batch["output_xyz_pts"][
                        :, desc_i * num_descs : (desc_i + 1) * num_descs, ...
                    ].shape
                )
                > 0
            ],
            dim=1,
        )

    padding_mask = torch.from_numpy(
        np.array(batch["spatial_relation_name"]).T == "[pad]"
    ).bool()
    ignore_pts_mask = torch.zeros_like(outputs).bool()
    # ignore all padding labels
    ignore_pts_mask[padding_mask] = True
    # ignore all points out of bounds
    ignore_pts_mask = torch.logical_or(ignore_pts_mask, batch["out_of_bounds_pts"])
    stats["loss"] = binary_cross_entropy_with_logits(
        outputs,
        batch["output_label_pts"],
        weight=utils.get_bce_weight(
            output_label_pts=batch["output_label_pts"],
            balance_positive_negative=balance_positive_negative,
        ),
    )

    with torch.no_grad():
        accuracy = ((outputs > 0.0).long() == batch["output_label_pts"]).float()[
            ~ignore_pts_mask
        ]
        stats["accuracy"] = accuracy.mean()
        detailed_stats = [
            get_detailed_stats(
                prediction=outputs > cutoff,
                gt_label=batch["output_label_pts"].bool(),
                xyz_pts=batch["output_xyz_pts"],
                ignore_pts=ignore_pts_mask,
                target_obj_names=batch["target_obj_name"],
                reference_obj_names=batch["reference_obj_name"],
                spatial_relation_names=batch["spatial_relation_name"],
                scene_ids=batch["scene_id"],
                eval_device=net.device,
                **kwargs,
            )
            for cutoff in cutoffs
        ]
        for detailed_stat, cutoff in zip(detailed_stats, cutoffs):
            detailed_stat["cutoff"] = [cutoff] * len(detailed_stat)
        detailed_stats = pd.concat(detailed_stats)
        for k in detailed_stats.columns:
            if "iou" in k:
                stats[k] = detailed_stats[k].mean()
    return stats, detailed_stats


approach = {
    "semantic_abstraction": SemAbsVOOL,
    "semantic_aware": SemanticAwareVOOL,
    "clip_spatial": ClipSpatialVOOL,
}

if __name__ == "__main__":
    parser = utils.config_parser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument(
        "--approach", choices=approach.keys(), default="semantic_abstraction"
    )
    args = parser.parse_args()
    if args.approach == "semantic_aware":
        args.network_inputs = ["rgb"]
    utils.train(
        get_losses_fn=get_losses,
        **utils.setup_experiment(
            args=args,
            net_class=approach[args.approach],
            dataset_class=ObjectLocalizationDataset,
            split_file_path=args.file_path + "/vool_split.pkl",
        ),
        **vars(args),
    )
