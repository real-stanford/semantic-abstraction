import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from net import SemAbs3D, SemanticAwareOVSSC
import utils
import pandas as pd
from dataset import SceneCompletionDataset
from typing import Dict, Tuple, Union


def get_detailed_stats(
    prediction,
    gt_label,
    xyz_pts,
    patch_labels,
    scene_ids,
    scene_bounds,
    ignore_pts,
    detailed_analysis=False,
    eval_device="cuda",
    **kwargs,
):
    num_scenes, num_patches = patch_labels.shape
    retvals = {
        "scene_id": np.array([[scene_id] * num_patches for scene_id in scene_ids])
        .reshape(-1)
        .tolist(),
        "label": patch_labels.reshape(-1).tolist(),
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
    voxelized_pts = utils.voxelize_points(
        prediction=prediction,
        label=gt_label,
        xyz_pts=xyz_pts,
        voxel_shape=(32, 32, 32),
        scene_bounds=scene_bounds,
        ignore_pts=ignore_pts,
    )
    retvals.update(
        {
            "voxel32x32x32_" + k: v
            for k, v in utils.prediction_analysis(
                **{k: v.to(eval_device) for k, v in voxelized_pts.items()}
            ).items()
        }
    )
    if detailed_analysis:
        voxelized_pts = utils.voxelize_points(
            prediction=prediction,
            label=gt_label,
            xyz_pts=xyz_pts,
            voxel_shape=(64, 64, 64),
            scene_bounds=scene_bounds,
            ignore_pts=ignore_pts,
        )
        retvals.update(
            {
                "voxel64x64x64_" + k: v
                for k, v in utils.prediction_analysis(
                    **{k: v.to(eval_device) for k, v in voxelized_pts.items()}
                ).items()
            }
        )
    for i, label in enumerate(patch_labels.reshape(-1).tolist()):
        if label == "":  # skip padding classes
            for k in retvals.keys():
                if "voxel" in k or "point" in k:
                    retvals[k][i] = np.NAN
    return pd.DataFrame.from_dict(retvals)


def get_losses(
    net,
    batch: dict,
    cutoffs=[0],
    balance_positive_negative: bool = False,
    **kwargs,
) -> Tuple[Dict[str, Union[float, torch.Tensor]], pd.DataFrame]:
    stats = {}
    num_pts = batch["output_xyz_pts"].shape[2]
    if num_pts <= 500000:
        outputs = net(**batch)
    else:
        num_patches = 1
        # probably CUDA OOM
        outputs = torch.cat(
            [
                net(
                    **{
                        **batch,
                        "input_feature_pts": batch["input_feature_pts"][
                            :, patch_i * num_patches : (patch_i + 1) * num_patches, ...
                        ]
                        if batch["input_feature_pts"].shape[1]
                        == batch["output_xyz_pts"].shape[1]
                        else batch["input_feature_pts"],
                        "output_xyz_pts": batch["output_xyz_pts"][
                            :, patch_i * num_patches : (patch_i + 1) * num_patches, ...
                        ],
                        "semantic_class_features": batch["semantic_class_features"][
                            :, patch_i * num_patches : (patch_i + 1) * num_patches, ...
                        ],
                    }
                )
                for patch_i in range(len(batch["patch_labels"]) // num_patches + 1)
                if np.prod(
                    batch["output_xyz_pts"][
                        :, patch_i * num_patches : (patch_i + 1) * num_patches, ...
                    ].shape
                )
                > 0
            ],
            dim=1,
        )

    batch["patch_labels"] = np.array(batch["patch_labels"]).T
    padding_mask = torch.from_numpy(batch["patch_labels"] == "").bool()
    batch["out_of_bounds_pts"] = batch["out_of_bounds_pts"].view(outputs.shape)
    ignore_pts_mask = torch.zeros_like(outputs).bool()
    # ignore all padding labels
    ignore_pts_mask[padding_mask] = True
    # ignore all points out of bounds
    ignore_pts_mask = torch.logical_or(ignore_pts_mask, batch["out_of_bounds_pts"])
    # don't eval on points outside of frustum
    ignore_pts_mask = torch.logical_or(
        ignore_pts_mask, batch["out_of_frustum_pts_mask"]
    )
    stats["loss"] = binary_cross_entropy_with_logits(
        outputs[~ignore_pts_mask],
        batch["output_label_pts"][~ignore_pts_mask],
        weight=utils.get_bce_weight(
            output_label_pts=batch["output_label_pts"],
            balance_positive_negative=balance_positive_negative,
        )[~ignore_pts_mask],
    )
    with torch.no_grad():
        vision_accuracy_mask = (
            (outputs > 0.0).long() == batch["output_label_pts"]
        ).float()
        stats["accuracy"] = vision_accuracy_mask[~ignore_pts_mask].mean()
        detailed_stats = [
            get_detailed_stats(
                prediction=outputs > cutoff,
                gt_label=batch["output_label_pts"].bool(),
                xyz_pts=batch["output_xyz_pts"],
                ignore_pts=ignore_pts_mask,
                patch_labels=batch["patch_labels"],
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
    "semantic_abstraction": SemAbs3D,
    "semantic_aware": SemanticAwareOVSSC,
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
            ddp=len(args.gpus) > 1,
            net_class=approach[args.approach],
            dataset_class=SceneCompletionDataset,
            split_file_path=args.file_path + "/ssc_split.pkl",
        ),
        **vars(args),
    )
