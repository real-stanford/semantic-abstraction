import numpy as np
import torch
from torch.utils.data import Dataset
from fusion import TSDFVolume
from point_cloud import (
    check_pts_in_frustum,
    filter_pts_bounds,
    get_pointcloud,
)
from typing import List, Optional, Tuple
import h5py
from transforms3d import affines, euler
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


def deref_h5py(dataset, refs):
    return np.array([dataset[ref][0] for ref in refs]).astype(np.float32)


synonyms = {
    "television": "tv",
    "sofa": "couch",
    "house plant": "plant in a pot",
    "bookcase": "bookshelf",
    "baseball bat": "rawlings big stick maple bat",
    "pillow": "cushion",
    "arm chair": "recliner",
    "bread": "loaf of sourdough",
    "cell phone": "mobile phone",
    "desktop": "computer",
    "dresser": "wardrobe",
    "dumbbell": "gym weights",
    "fridge": "refridgerator",
    "garbage can": "trash can",
    "laptop": "computer",
    "outlet": "eletric plug",
    "stairs": "staircase",
}


class SceneUnderstandDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        scene_bounds: TensorType[2, 3],
        network_inputs: List[str],
        domain_randomization: bool,
        num_input_pts: int,
        num_output_pts: int,
        return_vis: bool,
        scene_paths: List[str],
        tsdf_shape: Tuple[int, int, int],
        domain_randomized_rgb: bool,
        offset_patch_mask: bool,
        balance_spatial_relations: bool,
        saliency_config: str,
        use_synonyms: bool,
        subtract_mean_relevancy: bool,
        balance_spatial_sampling: bool,
        saliency_vmin: float,
        dr_pos: float,
        dr_orn: float,
        dr_scale: float,
        xyz_pts_noise: float,
        always_replace_subsample_pts: bool,
        patch_mask_cutoff: float = 0.004,
        load_gt: bool = True,
        **kwargs,
    ):
        # setup
        self.file_path = file_path
        self.keys = list(sorted(scene_paths))
        self.num_input_pts = num_input_pts
        self.num_output_pts = num_output_pts
        self.network_inputs = network_inputs

        # 3D scene
        self.scene_bounds = np.array(scene_bounds)
        self.tsdf_shape = tsdf_shape

        # retval customization
        self.domain_randomized_rgb = domain_randomized_rgb
        self.return_vis = return_vis
        self.domain_randomization = domain_randomization
        self.subtract_mean_relevancy = subtract_mean_relevancy
        self.use_synonyms = use_synonyms
        self.offset_patch_mask = offset_patch_mask
        self.patch_mask_cutoff = patch_mask_cutoff
        self.saliency_config = saliency_config
        self.saliency_vmin = saliency_vmin
        self.xyz_pts_noise = xyz_pts_noise
        self.balance_spatial_relations = balance_spatial_relations
        self.balance_spatial_sampling = balance_spatial_sampling
        self.dr_pos = dr_pos
        self.dr_orn = dr_orn
        self.dr_scale = dr_scale
        self.load_gt = load_gt
        self.always_replace_subsample_pts = always_replace_subsample_pts

    def __len__(self):
        return len(self.keys)

    @staticmethod
    @typechecked
    def transform_filter_subsample(
        xyz_pts,
        scene_bounds,
        num_subsample_pts,
        subsample_probabilities,
        alway_replace_pts: bool,
        transform_matrix=None,
        **kwargs,
    ):
        num_pts = len(xyz_pts)
        retval = {"xyz_pts": xyz_pts}
        retval.update(kwargs)
        if transform_matrix is not None:
            # turn into homogeneous coords
            xyz_pts = torch.cat((xyz_pts, torch.ones(num_pts)[:, None]), dim=-1)
            xyz_pts = torch.matmul(transform_matrix, xyz_pts.permute(1, 0)).permute(
                1, 0
            )[..., :3]
        # filter out of bounds points
        in_bounds_mask = filter_pts_bounds(xyz_pts, scene_bounds).bool()
        retval["xyz_pts"] = xyz_pts[in_bounds_mask, :]
        subsample_probabilities = subsample_probabilities[in_bounds_mask]
        subsample_probabilities /= subsample_probabilities.sum()
        for k, v in kwargs.items():
            if v is None:
                retval[k] = None
            elif v.shape[0] == len(in_bounds_mask):
                retval[k] = v[in_bounds_mask, ...]
            elif v.shape[1] == len(in_bounds_mask):
                retval[k] = v[:, in_bounds_mask, ...]
            else:
                raise Exception(k, v.shape, in_bounds_mask.shape)
        if num_subsample_pts == -1:
            return retval
        try:
            # bias based on description
            indices = np.random.choice(
                a=len(retval["xyz_pts"]),
                size=num_subsample_pts,
                p=subsample_probabilities,
                replace=alway_replace_pts,
            )
        except Exception as e:
            indices = np.random.choice(
                a=len(retval["xyz_pts"]),
                size=num_subsample_pts,
                p=subsample_probabilities,
                replace=True,
            )
        return {
            k: (
                v[indices, ...]
                if len(v) == len(retval["xyz_pts"])
                else v[:, indices, ...]
            )
            if v is not None
            else None
            for k, v in retval.items()
        }


class ObjectLocalizationDataset(SceneUnderstandDataset):
    def __init__(self, num_descs: int, **kwargs):
        super().__init__(**kwargs)
        self.num_descs = num_descs

    @staticmethod
    def get_descriptions(
        scene_group,
        num_subsample_descs: int,
        saliency_config: str,
        rgb_key: str,
        use_synonyms: bool,
        balance_spatial_relations: bool = False,
        only_return_num_descs: bool = False,
    ):
        saliency_prefix = f"saliencies/{rgb_key}|{saliency_config}"
        descriptions = dict()
        desc_group = scene_group["descriptions"]
        num_descs = len(desc_group["spatial_relation_name"])

        descriptions["target_obj_name"] = np.array(
            desc_group["target_obj_name"]
        ).astype(str)
        descriptions["target_obj_id"] = np.array(desc_group["target_obj_id"])
        descriptions["reference_obj_name"] = np.array(
            desc_group["reference_obj_name"]
        ).astype(str)
        descriptions["spatial_relation_name"] = np.array(
            desc_group["spatial_relation_name"]
        ).astype(str)
        description_sentences = ""
        for desc_part in [
            descriptions["target_obj_name"],
            " ",
            descriptions["spatial_relation_name"],
            " a ",
            descriptions["reference_obj_name"],
        ]:
            description_sentences = np.char.add(description_sentences, desc_part)

        if use_synonyms:
            has_synonym = list(
                map(
                    lambda sentence: any(x in sentence for x in synonyms.keys()),
                    description_sentences,
                )
            )
            descriptions["target_obj_name"] = descriptions["target_obj_name"][
                has_synonym
            ]
            descriptions["target_obj_id"] = descriptions["target_obj_id"][has_synonym]
            descriptions["reference_obj_name"] = descriptions["reference_obj_name"][
                has_synonym
            ]
            descriptions["spatial_relation_name"] = descriptions[
                "spatial_relation_name"
            ][has_synonym]
            description_sentences = np.array(description_sentences)[has_synonym]
            num_descs = sum(has_synonym)

        if only_return_num_descs:
            return num_descs

        desc_indices = np.arange(0, num_descs)

        if num_subsample_descs != -1 and num_subsample_descs < num_descs:
            p = np.ones(num_descs).astype(np.float64)
            if balance_spatial_relations:
                spatial_relations = np.array(
                    desc_group["spatial_relation_name"]
                ).tolist()
                unique_relations = list(set(spatial_relations))
                spatial_relations_ids = np.array(
                    list(map(lambda r: unique_relations.index(r), spatial_relations))
                )
                for spatial_relations_id in range(len(unique_relations)):
                    mask = spatial_relations_ids == spatial_relations_id
                    p[mask] = 1 / mask.sum()
            p /= p.sum()
            desc_indices = np.random.choice(
                num_descs, num_subsample_descs, replace=False, p=p
            )
            desc_indices.sort()  # hdf5 indexing must be in order

        descriptions["target_obj_name"] = descriptions["target_obj_name"][desc_indices]
        descriptions["target_obj_id"] = descriptions["target_obj_id"][desc_indices]
        descriptions["reference_obj_name"] = descriptions["reference_obj_name"][
            desc_indices
        ]
        descriptions["spatial_relation_name"] = descriptions["spatial_relation_name"][
            desc_indices
        ]
        description_sentences = description_sentences[desc_indices]

        if use_synonyms:
            descriptions["target_obj_name"] = np.array(
                list(
                    map(
                        lambda x: x if x not in synonyms.keys() else synonyms[x],
                        descriptions["target_obj_name"],
                    )
                )
            )
            descriptions["reference_obj_name"] = np.array(
                list(
                    map(
                        lambda x: x if x not in synonyms.keys() else synonyms[x],
                        descriptions["reference_obj_name"],
                    )
                )
            )

        saliency_text_labels = (
            np.array(scene_group[f"{saliency_prefix}|saliency_text_labels"])
            .astype(str)
            .tolist()
        )
        descriptions["target_obj_saliency_refs"] = [
            scene_group[f"{saliency_prefix}"][idx]
            for idx in map(
                lambda obj_name: saliency_text_labels.index(obj_name),
                descriptions["target_obj_name"],
            )
        ]

        descriptions["reference_obj_saliency_refs"] = [
            scene_group[f"{saliency_prefix}"][idx]
            for idx in map(
                lambda obj_name: saliency_text_labels.index(obj_name),
                descriptions["reference_obj_name"],
            )
        ]

        descriptions["description_saliency_refs"] = [
            scene_group[f"{saliency_prefix}"][idx]
            for idx in map(
                lambda desc: saliency_text_labels.index(desc), description_sentences
            )
        ]

        num_missing_descs = num_subsample_descs - len(
            descriptions["spatial_relation_name"]
        )
        if num_missing_descs > 0 and num_subsample_descs != -1:
            descriptions["target_obj_id"] = np.array(
                descriptions["target_obj_id"].tolist() + [-2] * num_missing_descs
            )
            descriptions["spatial_relation_name"] = np.array(
                descriptions["spatial_relation_name"].tolist()
                + ["[pad]"] * num_missing_descs
            )
            descriptions["target_obj_name"] = np.array(
                descriptions["target_obj_name"].tolist() + ["[pad]"] * num_missing_descs
            )
            descriptions["reference_obj_name"] = np.array(
                descriptions["reference_obj_name"].tolist()
                + ["[pad]"] * num_missing_descs
            )
        descriptions["num_descs"] = len(descriptions["spatial_relation_name"])
        return descriptions

    def __getitem__(self, idx):
        retvals = dict()
        scene_path = self.file_path + "/" + self.keys[idx]
        with h5py.File(scene_path, "r") as f:
            group = f["data"]
            depth = deref_h5py(dataset=f["depth"], refs=group["depth"])[0]
            cam_intr = np.array(group["cam_intr"])
            cam_pose = np.array(group["cam_pose"])
            if self.domain_randomized_rgb:
                retvals["rgb"] = np.array(group["domain_randomized_rgb"]).astype(
                    np.float32
                )[0]
            else:
                retvals["rgb"] = deref_h5py(dataset=f["rgb"], refs=group["rgb"])[0]
            image_shape = retvals["rgb"].shape[:2]
            retvals["rgb"] = torch.from_numpy(retvals["rgb"]) / 255.0

            retvals["input_xyz_pts"] = torch.from_numpy(
                get_pointcloud(depth, None, cam_intr, cam_pose)[0].astype(np.float32)
            )
            retvals["full_objid_pts"] = None
            if "full_objid_pts" in group:
                retvals["output_xyz_pts"] = torch.from_numpy(
                    deref_h5py(dataset=f["full_xyz_pts"], refs=group["full_xyz_pts"])[0]
                )
                retvals["full_objid_pts"] = torch.from_numpy(
                    deref_h5py(
                        dataset=f["full_objid_pts"], refs=group["full_objid_pts"]
                    )[0]
                )

                retvals["out_of_bounds_pts"] = torch.zeros(
                    len(retvals["full_objid_pts"])
                ).float()
            descriptions = self.get_descriptions(
                scene_group=group,
                num_subsample_descs=self.num_descs if not self.return_vis else -1,
                saliency_config=self.saliency_config,
                rgb_key="domain_randomized_rgb"
                if self.domain_randomized_rgb
                else "rgb",
                use_synonyms=self.use_synonyms,
                balance_spatial_relations=self.balance_spatial_relations,
            )

            retvals["spatial_relation_name"] = descriptions[
                "spatial_relation_name"
            ].tolist()

            # gradcam values typically between -0.02 and 0.02
            # so multiply by 50
            retvals["input_target_saliency_pts"] = torch.from_numpy(
                deref_h5py(
                    dataset=f["saliencies"],
                    refs=descriptions["target_obj_saliency_refs"],
                )
            )
            retvals["input_reference_saliency_pts"] = torch.from_numpy(
                deref_h5py(
                    dataset=f["saliencies"],
                    refs=descriptions["reference_obj_saliency_refs"],
                )
            )
            retvals["input_description_saliency_pts"] = torch.from_numpy(
                deref_h5py(
                    dataset=f["saliencies"],
                    refs=descriptions["description_saliency_refs"],
                )
            )
            saliency_prefix = f'data/saliencies/{"domain_randomized_rgb" if self.domain_randomized_rgb else "rgb"}|{self.saliency_config}'
            mean_idx = (
                np.array(f[f"{saliency_prefix}|saliency_text_labels"])
                .astype(str)
                .tolist()
                .index("mean")
            )
            mean_relevancy_map = (
                torch.from_numpy(f["saliencies"][mean_idx]).float().squeeze()
            )
            for k in {
                "input_target_saliency_pts",
                "input_reference_saliency_pts",
                "input_description_saliency_pts",
            }:
                if self.subtract_mean_relevancy:
                    retvals[k] -= mean_relevancy_map
                if self.saliency_vmin is not None:
                    retvals[k] -= self.saliency_vmin
                    retvals[k][retvals[k] < 0] = 0
                retvals[k] = (
                    torch.nn.functional.interpolate(
                        retvals[k][:, None, :, :],
                        size=tuple(image_shape),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .view(len(retvals[k]), -1, 1)
                )
                retvals[k] *= 50
            if "patch_masks" in self.network_inputs:
                assert "saliency" not in self.network_inputs
                retvals["input_target_saliency_pts"] = (
                    retvals["input_target_saliency_pts"] > self.patch_mask_cutoff
                ).float()
                retvals["input_reference_saliency_pts"] = (
                    retvals["input_reference_saliency_pts"] > self.patch_mask_cutoff
                ).float()
                retvals["input_description_saliency_pts"] = (
                    retvals["input_description_saliency_pts"] > self.patch_mask_cutoff
                ).float()
            retvals["input_rgb_pts"] = (
                retvals["rgb"]
                .view(-1, 3)[None, ...]
                .repeat(len(descriptions["spatial_relation_name"]), 1, 1)
            )
            if len(retvals["input_target_saliency_pts"]) < len(
                descriptions["spatial_relation_name"]
            ):
                retvals["input_target_saliency_pts"] = torch.cat(
                    (
                        retvals["input_target_saliency_pts"],
                        torch.zeros(
                            len(descriptions["spatial_relation_name"])
                            - len(retvals["input_target_saliency_pts"]),
                            *list(retvals["input_target_saliency_pts"].shape)[1:],
                        ),
                    ),
                    dim=0,
                )

                retvals["input_reference_saliency_pts"] = torch.cat(
                    (
                        retvals["input_reference_saliency_pts"],
                        torch.zeros(
                            len(descriptions["spatial_relation_name"])
                            - len(retvals["input_reference_saliency_pts"]),
                            *list(retvals["input_reference_saliency_pts"].shape)[1:],
                        ),
                    ),
                    dim=0,
                )
                retvals["input_description_saliency_pts"] = torch.cat(
                    (
                        retvals["input_description_saliency_pts"],
                        torch.zeros(
                            len(descriptions["spatial_relation_name"])
                            - len(retvals["input_description_saliency_pts"]),
                            *list(retvals["input_description_saliency_pts"].shape)[1:],
                        ),
                    ),
                    dim=0,
                )
            retvals["output_label_pts"] = None
            if "full_objid_pts" in retvals and retvals["full_objid_pts"] is not None:
                output_label_pts = torch.zeros(
                    len(descriptions["target_obj_id"]),
                    len(retvals["full_objid_pts"]),
                    dtype=torch.float32,
                )
                for desc_i, target_obj_id in enumerate(descriptions["target_obj_id"]):
                    obj_mask = retvals["full_objid_pts"] == target_obj_id
                    output_label_pts[desc_i, :] = obj_mask
                retvals["output_label_pts"] = output_label_pts

            retvals["scene_id"] = self.keys[idx].split("/")[-1].split(".")[0]
            retvals["target_obj_name"] = descriptions["target_obj_name"].tolist()
            retvals["reference_obj_name"] = descriptions["reference_obj_name"].tolist()
            if self.return_vis:
                retvals["depth"] = depth
                retvals["cam_intr"] = cam_intr
                retvals["cam_pose"] = cam_pose
                retvals["vis_gt_object_labels"] = (
                    np.array(group["objid_to_class"]).astype(str).tolist()
                    if "objid_to_class" in group
                    else []
                )
                if "matterport" in self.file_path or "arkit" in self.file_path:
                    vis_xyz_pts, vis_rgb_pts = get_pointcloud(
                        depth, retvals["rgb"].numpy(), cam_intr, cam_pose
                    )
                    retvals["vis_gt_objid_pts"] = torch.from_numpy(vis_rgb_pts)
                    retvals["vis_gt_xyz_pts"] = torch.from_numpy(vis_xyz_pts)
                else:
                    retvals["vis_gt_objid_pts"] = retvals["full_objid_pts"]
                    retvals["vis_gt_xyz_pts"] = torch.from_numpy(
                        deref_h5py(
                            dataset=f["full_xyz_pts"], refs=group["full_xyz_pts"]
                        )[0]
                    )

        transform_matrix = None
        if self.domain_randomization:
            scene_dims = self.scene_bounds[1, :] - self.scene_bounds[0, :]
            assert (scene_dims >= 0).all()
            translation = torch.randn(3) * scene_dims * self.dr_pos
            rotation = euler.euler2mat(
                (torch.rand(1)[0] - 0.5) * self.dr_orn,
                (torch.rand(1)[0] - 0.5) * self.dr_orn,
                (torch.rand(1)[0] - 0.5) * self.dr_orn
                # full rotation around z axis
            )
            scale = torch.rand(3) * self.dr_scale + 1.0
            transform_matrix = torch.from_numpy(
                affines.compose(T=translation, R=rotation, Z=scale).astype(np.float32)
            )
        # PROCESS INPUTS
        kwargs = {
            "transform_matrix": transform_matrix,
            "scene_bounds": self.scene_bounds,
            "num_subsample_pts": self.num_input_pts,
            "subsample_probabilities": np.ones(len(retvals["input_xyz_pts"])).astype(
                np.float64
            )
            / len(retvals["input_xyz_pts"]),
            "alway_replace_pts": self.always_replace_subsample_pts,
        }
        try:
            processed_pts = SceneUnderstandDataset.transform_filter_subsample(
                xyz_pts=retvals["input_xyz_pts"],
                input_target_saliency_pts=retvals["input_target_saliency_pts"],
                input_reference_saliency_pts=retvals["input_reference_saliency_pts"],
                input_description_saliency_pts=retvals[
                    "input_description_saliency_pts"
                ],
                input_rgb_pts=retvals["input_rgb_pts"],
                **kwargs,
            )
        except Exception as e:
            kwargs["transform_matrix"] = None
            processed_pts = SceneUnderstandDataset.transform_filter_subsample(
                xyz_pts=retvals["input_xyz_pts"],
                input_target_saliency_pts=retvals["input_target_saliency_pts"],
                input_reference_saliency_pts=retvals["input_reference_saliency_pts"],
                input_description_saliency_pts=retvals[
                    "input_description_saliency_pts"
                ],
                input_rgb_pts=retvals["input_rgb_pts"],
                **kwargs,
            )

        retvals["input_xyz_pts"] = processed_pts["xyz_pts"]
        retvals["input_target_saliency_pts"] = processed_pts[
            "input_target_saliency_pts"
        ]
        retvals["input_reference_saliency_pts"] = processed_pts[
            "input_reference_saliency_pts"
        ]
        retvals["input_description_saliency_pts"] = processed_pts[
            "input_description_saliency_pts"
        ]
        retvals["input_rgb_pts"] = processed_pts["input_rgb_pts"]
        if "tsdf" in self.network_inputs:
            voxel_size = (
                (self.scene_bounds[1] - self.scene_bounds[0]) / self.tsdf_shape
            ).min()
            tsdf_vol = TSDFVolume(vol_bnds=self.scene_bounds.T, voxel_size=voxel_size)
            final_transform = cam_pose
            if kwargs["transform_matrix"] is not None:
                final_transform = kwargs["transform_matrix"] @ cam_pose
            tsdf_vol.integrate(
                color_im=retvals["rgb"].numpy(),
                depth_im=depth,
                cam_intr=cam_intr,
                cam_pose=final_transform,
            )
            retvals["tsdf_vol"] = torch.from_numpy(tsdf_vol.get_volume()[0])
        else:
            retvals["tsdf_vol"] = torch.ones(1)

        # PROCESS OUTPUTS
        if "output_label_pts" in retvals and retvals["output_label_pts"] != None:
            kwargs["num_subsample_pts"] = (
                self.num_output_pts if not self.return_vis else -1
            )
            if self.balance_spatial_sampling:
                desc_output_xyz_pts = []
                desc_output_label_pts = []
                desc_ignore_pts = []
                for desc_i in range(len(retvals["output_label_pts"])):
                    subsample_probabilities = np.ones(
                        len(retvals["output_xyz_pts"])
                    ).astype(np.float64)
                    positive_mask = retvals["output_label_pts"][desc_i].bool()
                    if positive_mask.any() and (not positive_mask.all()):
                        subsample_probabilities[positive_mask] = (
                            len(retvals["output_xyz_pts"]) / positive_mask.sum()
                        )
                        subsample_probabilities[~positive_mask] = (
                            len(retvals["output_xyz_pts"]) / (~positive_mask).sum()
                        )
                    subsample_probabilities /= subsample_probabilities.sum()
                    kwargs["subsample_probabilities"] = subsample_probabilities
                    output_pts = SceneUnderstandDataset.transform_filter_subsample(
                        xyz_pts=retvals["output_xyz_pts"],
                        output_label_pts=retvals["output_label_pts"][desc_i][None, :],
                        out_of_bounds_pts=retvals["out_of_bounds_pts"],
                        **kwargs,
                    )
                    desc_output_xyz_pts.append(output_pts["xyz_pts"])
                    desc_output_label_pts.append(output_pts["output_label_pts"])
                    desc_ignore_pts.append(output_pts["out_of_bounds_pts"])
                retvals["output_xyz_pts"] = torch.stack(desc_output_xyz_pts)
                retvals["output_label_pts"] = torch.stack(
                    desc_output_label_pts
                ).squeeze(dim=-2)
                retvals["out_of_bounds_pts"] = torch.stack(desc_ignore_pts)
            else:
                kwargs["subsample_probabilities"] = np.ones(
                    len(retvals["output_xyz_pts"])
                ).astype(np.float64)
                kwargs["subsample_probabilities"] /= kwargs[
                    "subsample_probabilities"
                ].sum()
                processed_pts = SceneUnderstandDataset.transform_filter_subsample(
                    xyz_pts=retvals["output_xyz_pts"],
                    output_label_pts=retvals["output_label_pts"],
                    out_of_bounds_pts=retvals["out_of_bounds_pts"],
                    **kwargs,
                )
                retvals["output_xyz_pts"] = processed_pts["xyz_pts"]
                retvals["out_of_bounds_pts"] = processed_pts["out_of_bounds_pts"]
                retvals["output_xyz_pts"] = retvals["output_xyz_pts"][None].repeat(
                    len(processed_pts["output_label_pts"]), 1, 1
                )
                retvals["output_label_pts"] = processed_pts["output_label_pts"]

        if self.xyz_pts_noise > 0.0:
            retvals["output_xyz_pts"] += (
                torch.randn_like(retvals["output_xyz_pts"]) * self.xyz_pts_noise
            )
            retvals["input_xyz_pts"] += (
                torch.randn_like(retvals["input_xyz_pts"]) * self.xyz_pts_noise
            )
        retvals["out_of_frustum_pts_mask"] = torch.from_numpy(
            np.stack(
                [
                    ~check_pts_in_frustum(
                        xyz_pts=desc_xyz_pts,
                        depth=depth,
                        cam_pose=cam_pose,
                        cam_intr=cam_intr,
                    )
                    for desc_xyz_pts in retvals["output_xyz_pts"]
                ],
                axis=0,
            )
        ).bool()
        return retvals


class SceneCompletionDataset(SceneUnderstandDataset):
    def __init__(self, num_patches: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches

    @staticmethod
    def get_scene_patches(
        file,
        num_subsample_patches: int,
        rgb_key: str,
        saliency_config: str,
        use_synonyms: bool,
        subtract_mean_relevancy: bool,
        full_objid_pts: Optional[torch.Tensor] = None,
        out_of_frustum_pts_mask: Optional[torch.Tensor] = None,
        only_return_num_patches: bool = False,
        use_gt_seg: bool = False,
    ):
        assert only_return_num_patches or saliency_config is not None
        saliency_prefix = f"data/saliencies/{rgb_key}|{saliency_config}"
        has_groundtruth = full_objid_pts is not None

        scene_patches = dict()
        scene_object_labels = np.array(file[f"data/objid_to_class"]).astype(str)
        scene_patches["patch_labels"] = np.array(
            list(map(lambda s: s.split("[")[0], scene_object_labels))
        )

        if has_groundtruth:
            if out_of_frustum_pts_mask is not None:
                scene_obj_ids = set(
                    full_objid_pts[~out_of_frustum_pts_mask].view(-1).long().tolist()
                )
            else:
                scene_obj_ids = set(full_objid_pts.view(-1).long().tolist())
            visible_obj_ids = set(
                np.unique(
                    deref_h5py(dataset=file["seg"], refs=file["data/seg"])[0]
                ).astype(int)
            ) - {-1}
            scene_obj_ids = scene_obj_ids.intersection(visible_obj_ids)
            scene_patches["patch_labels"] = list(
                set(scene_patches["patch_labels"][list(scene_obj_ids)])
                - {"empty", "out of bounds", "unlabelled"}
            )
        if use_synonyms:
            scene_patches["patch_labels"] = list(
                map(
                    lambda x: x if x not in synonyms.keys() else synonyms[x],
                    scene_patches["patch_labels"],
                )
            )

        if use_gt_seg:
            assert has_groundtruth
            assert not subtract_mean_relevancy
            assert not use_synonyms
            patch_objids = dict()
            for patch_label in scene_patches["patch_labels"]:
                patch_objids[patch_label] = set(
                    map(
                        lambda objid: int(objid.split("[")[1].split("]")[0]),
                        filter(
                            lambda objid: objid.split("[")[0] == patch_label,
                            scene_object_labels.tolist(),
                        ),
                    )
                )

            patch_labels = np.array(list(patch_objids.keys()))
            num_patches = len(patch_objids)
            if num_subsample_patches != -1 and num_patches > num_subsample_patches:
                indices = np.random.choice(
                    num_patches, num_subsample_patches, replace=False
                )
                patch_labels = patch_labels[indices]
                patch_objids = {k: patch_objids[k] for k in patch_labels}
            num_patches = len(patch_objids)
            seg = deref_h5py(dataset=file["seg"], refs=file["data/seg"])[0]
            scene_patches["patch_saliencies"] = []
            for patch_label, objids in patch_objids.items():
                # take or of all object segs
                mask = np.zeros_like(seg)
                for objid in objids:
                    mask = np.logical_or(mask, (seg == objid))
                scene_patches["patch_saliencies"].append(mask)
            scene_patches["patch_saliencies"] = (
                torch.from_numpy(np.stack(scene_patches["patch_saliencies"])).float()
                - 0.5
            ) / 50  # because it will be multiplied by 50 later
            scene_patches["patch_labels"] = patch_labels
            scene_patches["patch_objmatches"] = np.array(
                [
                    "|".join(map(str, patch_objids[patch_label]))
                    for patch_label in scene_patches["patch_labels"]
                ]
            )
            scene_patches["num_patches"] = num_patches
            # NOTE HARDCODED, only meant for testing ours, not semaware
            scene_patches["patch_label_features"] = torch.zeros(
                size=(num_patches, 512)
            ).float()
            return scene_patches
        saliency_text_labels = np.array(
            file[f"{saliency_prefix}|saliency_text_labels"]
        ).astype(str)

        # make sure saliencies for scene object labels have been generated
        assert set(scene_patches["patch_labels"]).issubset(saliency_text_labels)
        saliency_indices = np.array(
            list(
                map(
                    lambda l: l[0],
                    # only get index, not actual saliency label
                    filter(
                        lambda l: l[1] in scene_patches["patch_labels"],
                        # make sure saliency text label is in
                        # set of valid patch mask labels
                        enumerate(saliency_text_labels),
                    ),
                )
            )
        )
        num_patches = len(saliency_indices)
        if only_return_num_patches:
            return num_patches
        if num_subsample_patches != -1 and num_patches > num_subsample_patches:
            saliency_indices = np.random.choice(
                saliency_indices, num_subsample_patches, replace=False
            )
            num_patches = num_subsample_patches
            # hdf5 indexing must be in order
            saliency_indices.sort()
        scene_patches["patch_labels"] = np.array(
            file[f"{saliency_prefix}|saliency_text_labels"]
        ).astype(str)[saliency_indices]
        scene_patches["patch_saliencies"] = torch.from_numpy(
            deref_h5py(
                dataset=file[f"saliencies"],
                refs=file[saliency_prefix][saliency_indices],
            )
        ).float()

        if subtract_mean_relevancy:
            mean_idx = (
                np.array(file[f"{saliency_prefix}|saliency_text_labels"])
                .astype(str)
                .tolist()
                .index("mean")
            )
            mean_relevancy = (
                torch.from_numpy(file[f"saliencies"][mean_idx]).float().squeeze()
            )
            scene_patches["patch_saliencies"] -= mean_relevancy

        scene_patches["patch_label_features"] = torch.from_numpy(
            np.array(file[f"{saliency_prefix}|saliency_text_label_features"])
        ).float()[saliency_indices]
        scene_patches["num_patches"] = num_patches

        if has_groundtruth:
            original_patch_labels = scene_patches["patch_labels"]
            if use_synonyms:
                inv_synonyms = {v: k for k, v in synonyms.items()}
                original_patch_labels = map(
                    lambda l: l if l not in synonyms.values() else inv_synonyms[l],
                    original_patch_labels,
                )

            scene_patches["patch_objmatches"] = np.array(
                [
                    "|".join(
                        [
                            str(objid)
                            for objid, obj_label in enumerate(scene_object_labels)
                            if obj_label.split("[")[0] == patch_label
                        ]
                    )
                    for patch_label in original_patch_labels
                ]
            )
        else:
            # matterport
            scene_patches["patch_objmatches"] = np.array([""] * num_patches)
        image_shape = file["rgb"].shape[1:-1]
        scene_patches["patch_saliencies"] = torch.nn.functional.interpolate(
            scene_patches["patch_saliencies"][:, None, :, :],
            size=tuple(image_shape),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return scene_patches

    @classmethod
    def transform_retvals(
        cls,
        retvals: dict,
        num_output_pts: int,
        balance_spatial_sampling: bool,
        scene_bounds: np.ndarray,
        tsdf_shape,
        rgb,
        depth,
        cam_intr,
        cam_pose,
        network_inputs,
        **kwargs,
    ):
        input_pts = SceneUnderstandDataset.transform_filter_subsample(
            xyz_pts=retvals["input_xyz_pts"],
            input_feature_pts=retvals["input_feature_pts"],
            subsample_probabilities=np.ones(len(retvals["input_xyz_pts"])).astype(
                np.float64
            )
            / len(retvals["input_xyz_pts"]),
            scene_bounds=scene_bounds,
            **kwargs,
        )
        kwargs["num_subsample_pts"] = -1
        # PROCESS OUTPUTS
        if "output_label_pts" in retvals:
            kwargs["num_subsample_pts"] = num_output_pts
            if balance_spatial_sampling:
                patch_output_xyz_pts = []
                patch_output_label_pts = []
                patch_ignore_pts = []
                for patch_i in range(len(retvals["output_label_pts"])):
                    subsample_probabilities = np.ones(
                        len(retvals["output_xyz_pts"])
                    ).astype(np.float64)
                    positive_mask = retvals["output_label_pts"][patch_i].bool()
                    if positive_mask.any() and (not positive_mask.all()):
                        subsample_probabilities[positive_mask] = (
                            len(retvals["output_xyz_pts"]) / positive_mask.sum()
                        )
                        subsample_probabilities[~positive_mask] = (
                            len(retvals["output_xyz_pts"]) / (~positive_mask).sum()
                        )
                    subsample_probabilities /= subsample_probabilities.sum()
                    output_pts = SceneUnderstandDataset.transform_filter_subsample(
                        xyz_pts=retvals["output_xyz_pts"],
                        out_of_bounds_pts=retvals["out_of_bounds_pts"],
                        output_label_pts=retvals["output_label_pts"][patch_i][None, :],
                        subsample_probabilities=subsample_probabilities,
                        scene_bounds=scene_bounds,
                        **kwargs,
                    )
                    patch_output_xyz_pts.append(output_pts["xyz_pts"])
                    patch_output_label_pts.append(output_pts["output_label_pts"])
                    patch_ignore_pts.append(output_pts["out_of_bounds_pts"])
                retvals["output_xyz_pts"] = torch.stack(patch_output_xyz_pts)
                retvals["out_of_bounds_pts"] = torch.stack(patch_ignore_pts)
                retvals["output_label_pts"] = torch.stack(
                    patch_output_label_pts
                ).squeeze(dim=-2)
            else:
                output_pts = SceneUnderstandDataset.transform_filter_subsample(
                    xyz_pts=retvals["output_xyz_pts"],
                    output_label_pts=retvals["output_label_pts"],
                    out_of_bounds_pts=retvals["out_of_bounds_pts"],
                    subsample_probabilities=np.ones(
                        len(retvals["output_xyz_pts"])
                    ).astype(np.float64)
                    / len(retvals["output_xyz_pts"]),
                    scene_bounds=scene_bounds,
                    **kwargs,
                )
                retvals["output_xyz_pts"] = output_pts["xyz_pts"][None, ...].repeat(
                    len(output_pts["output_label_pts"]), 1, 1
                )
                retvals["out_of_bounds_pts"] = output_pts["out_of_bounds_pts"][
                    None, ...
                ].repeat(len(output_pts["output_label_pts"]), 1, 1)
                retvals["output_label_pts"] = output_pts["output_label_pts"]
        retvals["input_xyz_pts"] = input_pts["xyz_pts"]
        retvals["input_feature_pts"] = input_pts["input_feature_pts"]

        # construct the tsdf vol
        if "tsdf" in network_inputs:
            voxel_size = ((scene_bounds[1] - scene_bounds[0]) / tsdf_shape).min()
            tsdf_vol = TSDFVolume(vol_bnds=scene_bounds.T, voxel_size=voxel_size)
            final_transform = cam_pose
            if kwargs["transform_matrix"] is not None:
                final_transform = kwargs["transform_matrix"] @ cam_pose
            tsdf_vol.integrate(
                color_im=rgb.numpy(),
                depth_im=depth,
                cam_intr=cam_intr,
                cam_pose=final_transform,
            )
            retvals["tsdf_vol"] = torch.from_numpy(tsdf_vol.get_volume()[0])
        else:
            retvals["tsdf_vol"] = torch.ones(1)

    def __getitem__(self, idx):
        retvals = dict()
        scene_path = self.file_path + "/" + self.keys[idx]
        with h5py.File(scene_path, "r") as f:
            group = f["data"]
            depth = deref_h5py(dataset=f["depth"], refs=group["depth"])[0]
            cam_intr = np.array(group["cam_intr"])
            cam_pose = np.array(group["cam_pose"])

            if self.domain_randomized_rgb:
                retvals["rgb"] = np.array(group["domain_randomized_rgb"][0])
            else:
                retvals["rgb"] = np.array(f["rgb"][group["rgb"][0]][0])
            retvals["rgb"] = torch.from_numpy(retvals["rgb"]).float()
            retvals["input_xyz_pts"] = torch.from_numpy(
                get_pointcloud(depth, None, cam_intr, cam_pose)[0]
            ).float()
            retvals["full_objid_pts"] = None
            if "full_objid_pts" in group:
                retvals["output_xyz_pts"] = torch.from_numpy(
                    deref_h5py(dataset=f["full_xyz_pts"], refs=group["full_xyz_pts"])[0]
                ).float()
                retvals["full_objid_pts"] = torch.from_numpy(
                    deref_h5py(
                        dataset=f["full_objid_pts"], refs=group["full_objid_pts"]
                    )[0]
                ).long()
                retvals["out_of_frustum_pts_mask"] = ~check_pts_in_frustum(
                    xyz_pts=retvals["output_xyz_pts"],
                    depth=depth,
                    cam_pose=cam_pose,
                    cam_intr=cam_intr,
                )
            scene_patches = self.get_scene_patches(
                file=f,
                num_subsample_patches=self.num_patches if not self.return_vis else -1,
                full_objid_pts=retvals["full_objid_pts"],
                out_of_frustum_pts_mask=retvals["out_of_frustum_pts_mask"],
                saliency_config=self.saliency_config,
                subtract_mean_relevancy=self.subtract_mean_relevancy,
                use_synonyms=self.use_synonyms,
                rgb_key="domain_randomized_rgb"
                if self.domain_randomized_rgb
                else "rgb",
            )

            feature_pts = []
            feature_dim = 0
            if "rgb" in self.network_inputs:
                # if rgb is in network inputs, then approach must be semantic aware
                # therefore, no other inputs
                feature_pts.append(retvals["rgb"][None, ...] / 255.0)
                feature_dim += 3
            else:
                if "patch_masks" in self.network_inputs:
                    if self.offset_patch_mask:
                        feature_pts.append(
                            (
                                scene_patches["patch_saliencies"][..., None]
                                > self.patch_mask_cutoff
                            )
                            * 2
                            - 1
                        )
                    else:
                        feature_pts.append(
                            (
                                scene_patches["patch_saliencies"][..., None]
                                > self.patch_mask_cutoff
                            )
                        )
                    feature_dim += 1
                if "saliency" in self.network_inputs:
                    patch_saliencies = scene_patches["patch_saliencies"][..., None]
                    if self.saliency_vmin is not None:
                        patch_saliencies -= self.saliency_vmin
                        patch_saliencies[patch_saliencies < 0] = 0
                    feature_pts.append(patch_saliencies * 50)
                    # gradcam values typically between -0.02 and 0.02
                    feature_dim += 1

            retvals["input_feature_pts"] = torch.cat(feature_pts, dim=-1)
            retvals["input_feature_pts"] = retvals["input_feature_pts"].view(
                len(retvals["input_feature_pts"]), -1, feature_dim
            )
            if (
                self.num_patches > len(retvals["input_feature_pts"])
                and not self.return_vis
                and "rgb" not in self.network_inputs
            ):
                retvals["input_feature_pts"] = torch.cat(
                    (
                        retvals["input_feature_pts"],
                        torch.zeros(
                            self.num_patches - len(retvals["input_feature_pts"]),
                            *list(retvals["input_feature_pts"].shape[1:]),
                        ),
                    ),
                    dim=0,
                )
            retvals["semantic_class_features"] = scene_patches["patch_label_features"]
            if (
                self.num_patches > len(scene_patches["patch_label_features"])
                and not self.return_vis
            ):
                retvals["semantic_class_features"] = torch.cat(
                    (
                        retvals["semantic_class_features"],
                        torch.randn(
                            [self.num_patches - len(retvals["semantic_class_features"])]
                            + list(retvals["semantic_class_features"].shape[1:]),
                        ),
                    ),
                    dim=0,
                )

            if (
                self.load_gt
                and "full_objid_pts" in retvals
                and retvals["full_objid_pts"] is not None
            ):
                gt_seg = deref_h5py(dataset=f["seg"], refs=group["seg"])[0]
                retvals["seg"] = gt_seg
                output_label_pts = torch.zeros(
                    len(retvals["semantic_class_features"]),
                    len(retvals["full_objid_pts"]),
                    dtype=float,
                )
                for patch_i, patch_matches in enumerate(
                    scene_patches["patch_objmatches"]
                ):
                    for objid in patch_matches.split("|"):
                        if objid == "":
                            continue
                        output_label_pts[
                            patch_i, retvals["full_objid_pts"] == int(objid)
                        ] = 1.0
                retvals["output_label_pts"] = output_label_pts
                retvals["out_of_bounds_pts"] = torch.zeros(
                    len(retvals["full_objid_pts"])
                ).float()
                object_labels = np.array(group["objid_to_class"]).astype(str).tolist()
                if "out of bounds" in object_labels:
                    oob_idx = object_labels.index("out of bounds")
                    retvals["out_of_bounds_pts"] = (
                        retvals["full_objid_pts"] == oob_idx
                    ).float()
            retvals["patch_labels"] = scene_patches["patch_labels"].tolist()
            assert all(map(lambda l: l != "", retvals["patch_labels"]))
            retvals["patch_labels"] += (
                [""]
                * max(self.num_patches - len(retvals["patch_labels"]), 0)
                * int(not self.return_vis)
            )
            retvals["scene_id"] = self.keys[idx].split("/")[-1].split(".")[0]
            if self.return_vis:
                retvals["depth"] = depth
                retvals["cam_intr"] = cam_intr
                retvals["cam_pose"] = cam_pose
                retvals["patch_objmatches"] = scene_patches["patch_objmatches"].tolist()
                retvals["vis_gt_object_labels"] = (
                    np.array(group["objid_to_class"]).astype(str).tolist()
                    if "objid_to_class" in group
                    else []
                )
                if "matterport" in self.file_path or "arkit" in self.file_path:
                    vis_xyz_pts, vis_rgb_pts = get_pointcloud(
                        depth, retvals["rgb"].numpy(), cam_intr, cam_pose
                    )
                    retvals["vis_gt_objid_pts"] = torch.from_numpy(vis_rgb_pts).float()
                    retvals["vis_gt_xyz_pts"] = torch.from_numpy(vis_xyz_pts).float()
                else:
                    retvals["vis_gt_objid_pts"] = retvals["full_objid_pts"]
                    retvals["vis_gt_xyz_pts"] = torch.from_numpy(
                        deref_h5py(
                            dataset=f["full_xyz_pts"], refs=group["full_xyz_pts"]
                        )[0]
                    ).float()
                    empty_mask = (
                        retvals["vis_gt_objid_pts"] == retvals["vis_gt_objid_pts"].max()
                    )
                    retvals["vis_gt_objid_pts"] = retvals["vis_gt_objid_pts"][
                        ~empty_mask
                    ]
                    retvals["vis_gt_xyz_pts"] = retvals["vis_gt_xyz_pts"][~empty_mask]
                retvals["patch_saliencies"] = scene_patches["patch_saliencies"]

        transform_matrix = None
        if self.domain_randomization:
            scene_dims = self.scene_bounds[1, :] - self.scene_bounds[0, :]
            assert (scene_dims >= 0).all()
            translation = torch.randn(3) * scene_dims * 0.05
            rotation = euler.euler2mat(
                (torch.rand(1)[0] - 0.5) * 0.3,
                (torch.rand(1)[0] - 0.5) * 0.3,
                (torch.rand(1)[0] - 0.5) * 0.3
                # full rotation around z axis
            )
            scale = torch.rand(3) * 0.1 + 1.0
            transform_matrix = torch.tensor(
                affines.compose(T=translation, R=rotation, Z=scale)
            ).float()
        # filter out points with invalid depth
        if (depth == 0.0).any():
            invalid_depth_mask = (depth == 0.0).reshape(-1)
            for k in retvals.keys():
                if "input" in k:
                    if retvals[k].shape[0] == len(invalid_depth_mask):
                        retvals[k] = retvals[k][~invalid_depth_mask]
                    elif retvals[k].shape[1] == len(invalid_depth_mask):
                        retvals[k] = retvals[k][:, ~invalid_depth_mask]
                    else:
                        raise Exception()

        # PROCESS INPUTS
        kwargs = {
            "transform_matrix": transform_matrix,
            "scene_bounds": self.scene_bounds,
            "num_subsample_pts": self.num_input_pts,
            "alway_replace_pts": self.always_replace_subsample_pts,
            "depth": depth,
            "cam_intr": cam_intr,
            "cam_pose": cam_pose,
            "balance_spatial_sampling": self.balance_spatial_sampling,
            "tsdf_shape": self.tsdf_shape,
            "retvals": retvals,
            "num_output_pts": self.num_output_pts if not self.return_vis else -1,
            "rgb": retvals["rgb"],
            "network_inputs": self.network_inputs,
        }
        try:
            self.transform_retvals(**kwargs)
        except Exception as e:
            kwargs["transform_matrix"] = None
            self.transform_retvals(**kwargs)

        retvals["out_of_frustum_pts_mask"] = ~torch.from_numpy(
            np.stack(
                [
                    check_pts_in_frustum(
                        xyz_pts=xyz_pts,
                        depth=depth,
                        cam_pose=cam_pose,
                        cam_intr=cam_intr,
                    )
                    for xyz_pts in retvals["output_xyz_pts"].cpu().numpy()
                ]
            )
        )

        if self.xyz_pts_noise > 0.0:
            retvals["output_xyz_pts"] += (
                torch.randn_like(retvals["output_xyz_pts"]) * self.xyz_pts_noise
            )
            retvals["input_xyz_pts"] += (
                torch.randn_like(retvals["input_xyz_pts"]) * self.xyz_pts_noise
            )
        return {
            k: v.float() if type(v) == torch.Tensor else v
            for k, v in retvals.items()
            if v is not None
        }
