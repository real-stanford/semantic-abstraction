import logging
import re
from copy import deepcopy
import shutil
from argparse import ArgumentParser
from typing import List
import ray
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from matplotlib import pyplot as plt
import numpy as np
import torch

from transforms3d import affines, euler
from fusion import TSDFVolume, rigid_transform
from generate_relevancy import get_datastructure, init_dataset, resize_and_add_data

from net import VirtualGrid
import pickle
import os
from tqdm import tqdm
from point_cloud import filter_pts_bounds, get_pointcloud
from utils import write_to_hdf5
import h5py
from numba import njit, prange

fov_w = 80.0
width = 224 * 4
height = 224 * 4
num_output_pts = 1000000
scene_bounds = np.array([[-1, -1, -0.1], [1, 1, 1.9]])
focal_length = (width / 2) / np.tan((np.pi * fov_w / 180) / 2)
cam_intr = np.array(
    [[focal_length, 0, height / 2], [0, focal_length, width / 2], [0, 0, 1]]
)

kitchens = [f"FloorPlan{i}_physics" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}_physics" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}_physics" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}_physics" for i in range(1, 31)]

test_scenes = kitchens[-5:] + living_rooms[-5:] + bedrooms[-5:] + bathrooms[-5:]


def parse_gt(scene_name: str, path_to_exported_scenes: str):
    pickle_path = f"{path_to_exported_scenes}/{scene_name}.pkl"
    scene_gt = None
    if os.path.exists(pickle_path):
        try:
            scene_gt = pickle.load(open(pickle_path, "rb"))
        except Exception as e:
            logging.error(e)
            logging.error(pickle_path)

    # cache this pre-processing
    if scene_gt is None:
        labels = []
        semantic = []
        full_xyz_pts = np.array(
            list(
                map(
                    lambda l: list(map(float, l.rstrip().split("|"))),
                    open(
                        f"{path_to_exported_scenes}/{scene_name}/full_xyz_pts.txt"
                    ).readlines(),
                )
            )
        )
        full_objid_pts = list(
            map(
                lambda l: l.rstrip(),
                open(
                    f"{path_to_exported_scenes}/{scene_name}/full_objid_pts.txt"
                ).readlines(),
            )
        )
        receptacle_infos = list(
            map(
                process_receptacle_line,
                open(
                    f"{path_to_exported_scenes}/{scene_name}_receptacles.txt"
                ).readlines(),
            )
        )
        receptacle_masks = {
            receptacle_info["receptacle_name"]: check_inside_receptacle(
                xyz_pts=full_xyz_pts, receptacle_info=receptacle_info
            )
            for receptacle_info in receptacle_infos
        }

        unique_obj_ids = list(set(full_objid_pts))
        unique_labels = list(set(map(class_reduction_rule, unique_obj_ids)))
        for objid in full_objid_pts:
            label = class_reduction_rule(objid)
            labels.append(label)
            semantic.append(unique_labels.index(label))
        semantic = np.array(semantic).astype(int)
        scene_gt = {
            "full_xyz_pts": full_xyz_pts,
            "full_objid_pts": full_objid_pts,
            "semantic": semantic,
            "labels": labels,
            "unique_labels": unique_labels,
            "receptacle_masks": receptacle_masks,
        }
        pickle.dump(scene_gt, open(pickle_path, "wb"))
    return scene_gt


def check_inside_receptacle(xyz_pts, receptacle_info):
    local_pts = (
        np.linalg.inv(receptacle_info["transform_matrix"])
        @ np.concatenate((xyz_pts, np.ones(len(xyz_pts))[:, None]), axis=1).T
    ).T[:, :3]
    # in and out
    bbox = np.array(
        [
            -receptacle_info["bbox_size"] / 2,
            receptacle_info["bbox_size"] / 2,
        ]
    )
    mask_pts = np.logical_and(
        (local_pts >= bbox[0]).all(axis=-1), (local_pts <= bbox[1]).all(axis=-1)
    )
    return mask_pts


def process_receptacle_line(line):
    receptacle_name, transform_matrix, bbox_size, bbox_center = (
        line.rstrip().lstrip().split("|")
    )
    transform_matrix = np.array(
        transform_matrix.replace(")(", ",").replace(")", "").replace("(", "").split(",")
    ).astype(float)
    bbox_size = np.array(bbox_size[1 : len(bbox_size) - 1].split(",")).astype(float)
    bbox_center = np.array(bbox_center[1 : len(bbox_center) - 1].split(",")).astype(
        float
    )
    return {
        "receptacle_name": receptacle_name,
        "transform_matrix": transform_matrix.reshape(4, 4),
        "bbox_size": bbox_size,
        "bbox_center": bbox_center,
    }


@njit(parallel=True)
def cam2pix(cam_pts, intr):
    # from https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py#L181-L193
    """Convert camera coordinates to pixel coordinates."""
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
        pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
        pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix


def xyz_pts_to_cam_pix(xyz_pts, cam_pose, cam_intr):
    cam_pts = rigid_transform(xyz_pts, np.linalg.inv(cam_pose))
    pix_z = cam_pts[:, 2]
    pix = cam2pix(cam_pts, cam_intr)
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    return pix_x, pix_y, pix_z


def get_all_relations(
    scene_data,
    receptacle_masks,
    objects_info,
    remapped_visible_obj_ids,
    all_remapped_obj_ids,
    visibility_pts_mask,
    container_obj_classes={
        "cabinet",
        "fridge",
        "drawer",
        "bathtub basin",
        "bowl",
        "box",
        "cup",
        "desk",
        "garbage can",
        "laundry hamper",
        "microwave",
        "mug",
        "pot",
        "safe",
        "sink basin",
        "toaster",
    },
    no_localization_obj_classes={
        "wall",
        "ceiling",
        "floor",
        "empty",
        "countertop",
        "drawer",
        "counter",
        "banana",
    },
    direction_dot_threshold=0.6,
):
    objects_in_scene = set(np.unique(scene_data["full_objid_pts"]))
    descriptions = set()
    unfiltered_descriptions = list()

    def should_add_relation(target_obj_name, spatial_relation, reference_obj_name):
        if target_obj_name == reference_obj_name:
            # unhelpful
            return False
        if (
            "ceiling" in reference_obj_name
            or reference_obj_name
            in {"floor", "rug", "baseboard", "light fixture", "decal"}
            or target_obj_name
            in {"floor", "rug", "baseboard", "light fixture", "decal"}
        ):
            # people don't localize objects in reference to these objects
            return False
        if (
            f"{target_obj_name} {spatial_relation} a {reference_obj_name}"
            in descriptions
        ):
            # duplicate
            return False
        if spatial_relation not in {"in", "on"} and (
            (f"{target_obj_name} in a {reference_obj_name}" in descriptions)
            or (f"{target_obj_name} on a {reference_obj_name}" in descriptions)
            or (f"{reference_obj_name} on a {target_obj_name}" in descriptions)
            or (f"{reference_obj_name} in a {target_obj_name}" in descriptions)
        ):
            # if target obj is on or in reference obj, then it shouldn't also be
            # left of, right of, behind, or in front of
            return False
        return True

    retval = {
        "target_obj_name": [],
        "target_obj_material": [],
        "target_obj_id": [],
        "reference_obj_name": [],
        "reference_obj_material": [],
        "spatial_relation_name": [],
    }
    # map from object id to obj class name
    for target_obj_id, obj_info in objects_info.items():
        target_obj_name = " ".join(
            map(lambda c: c.lower(), camel_case_split(obj_info["objectType"]))
        )
        if obj_info["parentReceptacles"] is not None:
            for reference_obj_id in obj_info["parentReceptacles"]:
                if reference_obj_id not in remapped_visible_obj_ids.keys():
                    # parent obj not visible
                    continue
                if target_obj_id not in all_remapped_obj_ids:
                    logging.warning(
                        target_obj_id + " not in mapped objids " + reference_obj_id
                    )
                    continue
                if (
                    all_remapped_obj_ids[target_obj_id] not in objects_in_scene
                    or all_remapped_obj_ids[reference_obj_id] not in objects_in_scene
                ):
                    # target or reference object doesn't even appear in scene bounds
                    continue
                parent_obj_info = objects_info[reference_obj_id]
                if parent_obj_info["objectType"] == "Floor":
                    continue

                reference_obj_name = " ".join(
                    map(
                        lambda c: c.lower(),
                        camel_case_split(parent_obj_info["objectType"]),
                    )
                )
                spatial_relation_name = (
                    "in" if reference_obj_name in container_obj_classes else "on"
                )
                unfiltered_descriptions.append(
                    f"{target_obj_name} {spatial_relation_name} a {reference_obj_name}"
                )
                if should_add_relation(
                    target_obj_name=target_obj_name,
                    spatial_relation=spatial_relation_name,
                    reference_obj_name=reference_obj_name,
                ):
                    descriptions.add(
                        f"{target_obj_name} {spatial_relation_name} a {reference_obj_name}"
                    )
                    retval["target_obj_name"].append(target_obj_name)
                    retval["target_obj_id"].append(all_remapped_obj_ids[target_obj_id])
                    retval["target_obj_material"].append(
                        "|".join(obj_info["salientMaterials"])
                        if obj_info["salientMaterials"] is not None
                        else ""
                    )

                    retval["reference_obj_name"].append(reference_obj_name)
                    retval["reference_obj_material"].append(
                        "|".join(parent_obj_info["salientMaterials"])
                        if parent_obj_info["salientMaterials"] is not None
                        else ""
                    )
                    retval["spatial_relation_name"].append(spatial_relation_name)

                    target_obj_is_visible = (
                        target_obj_id in remapped_visible_obj_ids.keys()
                    )
                    if not target_obj_is_visible:
                        # if target obj not visible then should
                        # supervise entire region
                        matching_receptacle_masks = {
                            rk: rv
                            for rk, rv in receptacle_masks.items()
                            if " ".join(
                                map(
                                    lambda c: c.lower(),
                                    camel_case_split(rk.split("_")[0]),
                                )
                            )
                            == retval["reference_obj_name"][-1]
                        }
                        if len(matching_receptacle_masks) == 0:
                            continue
                        receptacle_mask = np.logical_or.reduce(
                            tuple(
                                receptacle_mask["mask"]
                                for receptacle_mask in matching_receptacle_masks.values()
                            )
                        )
                        scene_data["full_objid_pts"][
                            :, np.logical_and(receptacle_mask, ~visibility_pts_mask)
                        ] = all_remapped_obj_ids[target_obj_id]

        # augment with inside relation
        if target_obj_name in container_obj_classes:
            container_name = target_obj_name
            container_obj_id = target_obj_id
            if container_obj_id not in remapped_visible_obj_ids.keys():
                continue
            matching_receptacle_masks = {
                rk: rv
                for rk, rv in receptacle_masks.items()
                if " ".join(
                    map(lambda c: c.lower(), camel_case_split(rk.split("_")[0]))
                )
                == container_name
            }
            if len(matching_receptacle_masks) == 0:
                continue
            description = f"banana in a {container_name}"
            unfiltered_descriptions.append(description)
            if should_add_relation(
                target_obj_name="banana",
                spatial_relation="in",
                reference_obj_name=container_name,
            ):
                descriptions.add(description)
                receptacle_mask = np.logical_or.reduce(
                    tuple(
                        receptacle_mask["mask"]
                        for receptacle_mask in matching_receptacle_masks.values()
                    )
                )
                hidden_obj_id = len(scene_data["objid_to_class"])
                retval["reference_obj_name"].append(container_name)
                retval["reference_obj_material"].append(
                    "|".join(obj_info["salientMaterials"])
                    if obj_info["salientMaterials"] is not None
                    else ""
                )
                hidden_obj_name = "banana"
                retval["target_obj_name"].append(hidden_obj_name)
                retval["target_obj_id"].append(hidden_obj_id)
                retval["target_obj_material"].append("")
                retval["spatial_relation_name"].append("in")
                scene_data["objid_to_class"] = np.array(
                    scene_data["objid_to_class"].astype(str).tolist()
                    + [f"banana[{hidden_obj_id}]"]
                ).astype("S")
                scene_data["full_objid_pts"][
                    :, np.logical_and(receptacle_mask, ~visibility_pts_mask)
                ] = hidden_obj_id

    # FIND ALL SPATIAL RELATIONS IN SCENE
    for reference_obj_key, reference_obj_id in remapped_visible_obj_ids.items():
        for target_obj_id in set(scene_data["full_objid_pts"][0]):
            target_obj_name = (
                scene_data["objid_to_class"][target_obj_id]
                .decode("utf-8")
                .split("[")[0]
            )
            reference_obj_name = (
                scene_data["objid_to_class"][reference_obj_id]
                .decode("utf-8")
                .split("[")[0]
            )
            if reference_obj_id == target_obj_id:
                continue
            if (
                target_obj_name in no_localization_obj_classes
                or reference_obj_name in no_localization_obj_classes
            ):
                continue
            target_obj_mask = scene_data["full_objid_pts"][0] == target_obj_id
            target_obj_xyz_pts = scene_data["full_xyz_pts"][0][target_obj_mask, :]
            reference_obj_mask = scene_data["full_objid_pts"][0] == reference_obj_id
            if not reference_obj_mask.any() or not target_obj_mask.any():
                continue
            reference_obj_xyz_pts = scene_data["full_xyz_pts"][0][reference_obj_mask, :]
            displacement = reference_obj_xyz_pts.mean(axis=0) - target_obj_xyz_pts.mean(
                axis=0
            )
            distance = np.linalg.norm(displacement)
            direction = displacement / distance
            reference_obj_bounds = reference_obj_xyz_pts.max(
                axis=0
            ) - reference_obj_xyz_pts.min(axis=0)
            distance_threshold = min(
                max(max(reference_obj_bounds[0], reference_obj_bounds[1]) * 2.0, 0.1),
                1.0,
            )
            if distance > distance_threshold:
                # too far away, probably not an actual spatial relation
                continue
            reference_material = (
                "|".join(objects_info[reference_obj_key]["salientMaterials"])
                if reference_obj_key in objects_info
                and objects_info[reference_obj_key]["salientMaterials"] is not None
                else ""
            )
            target_obj_is_visible = target_obj_id in scene_data["seg"]
            unfiltered_descriptions.append(
                f"{target_obj_name} behind a {reference_obj_name}"
            )
            if np.dot(
                direction, [-1, 0, 0]
            ) > direction_dot_threshold and should_add_relation(
                target_obj_name=target_obj_name,
                spatial_relation="behind",
                reference_obj_name=reference_obj_name,
            ):
                descriptions.add(f"{target_obj_name} behind a {reference_obj_name}")
                retval["target_obj_name"].append(target_obj_name)
                retval["target_obj_material"].append("")
                retval["target_obj_id"].append(target_obj_id)
                retval["reference_obj_name"].append(reference_obj_name)
                retval["reference_obj_material"].append(reference_material)
                retval["spatial_relation_name"].append("behind")
                if not target_obj_is_visible:
                    empty_id = list(
                        map(
                            lambda c: c.split("[")[0],
                            scene_data["objid_to_class"].astype(str),
                        )
                    ).index("empty")
                    empty_mask = scene_data["full_objid_pts"][0] == empty_id
                    reference_class_mask_pts = np.logical_or.reduce(
                        tuple(
                            scene_data["full_objid_pts"][0] == objid
                            for objid, objclass in enumerate(
                                scene_data["objid_to_class"].astype(str)
                            )
                            if objclass.split("[")[0] == reference_obj_name
                        )
                    )
                    im_h, im_w = scene_data["depth"][0].shape
                    resize_scale = 10
                    pix_x, pix_y, pix_z = xyz_pts_to_cam_pix(
                        xyz_pts=scene_data["full_xyz_pts"][0],
                        cam_pose=scene_data["cam_pose"],
                        cam_intr=scene_data["cam_intr"],
                    )
                    # effectively resize

                    ref_pix_x, ref_pix_y, ref_pix_z = xyz_pts_to_cam_pix(
                        xyz_pts=scene_data["full_xyz_pts"][0][
                            reference_class_mask_pts, :
                        ],
                        cam_pose=scene_data["cam_pose"],
                        cam_intr=scene_data["cam_intr"],
                    )

                    full_pix_xy = np.stack((pix_x, pix_y), axis=1)
                    corner = full_pix_xy.min(axis=0)
                    full_pix_xy -= corner

                    ref_pix_xy = np.stack((ref_pix_x, ref_pix_y), axis=1)
                    ref_pix_xy -= corner

                    full_pix_xy[:, 0] = np.digitize(
                        full_pix_xy[:, 0], bins=np.arange(0, im_w, resize_scale)
                    )
                    full_pix_xy[:, 1] = np.digitize(
                        full_pix_xy[:, 1], bins=np.arange(0, im_h, resize_scale)
                    )
                    ref_pix_xy[:, 0] = np.digitize(
                        ref_pix_xy[:, 0], bins=np.arange(0, im_w, resize_scale)
                    )
                    ref_pix_xy[:, 1] = np.digitize(
                        ref_pix_xy[:, 1], bins=np.arange(0, im_h, resize_scale)
                    )

                    ref_backsize = -np.ones(
                        (full_pix_xy[:, 0].max() + 1, full_pix_xy[:, 1].max() + 1)
                    ).astype(float)
                    # get back side of object in each pixel
                    for pix_xy in np.unique(ref_pix_xy, axis=0):
                        mask = (ref_pix_xy == pix_xy).all(axis=1)
                        ref_backsize[pix_xy[0], pix_xy[1]] = ref_pix_z[mask].max()
                    accessed_depth = ref_backsize[full_pix_xy[:, 0], full_pix_xy[:, 1]]
                    behind_mask = np.logical_and(
                        accessed_depth < pix_z, accessed_depth != -1
                    )
                    target_obj_mask = np.logical_and.reduce(
                        (behind_mask, ~visibility_pts_mask, empty_mask)
                    )
                    scene_data["full_objid_pts"][:, target_obj_mask] = target_obj_id
            # some objects shouldn't allow behind
            if reference_obj_name in {"cabinet"}:
                continue
            # if in front of, left of, or right of, then target object
            # should be visible
            if target_obj_id not in remapped_visible_obj_ids.values():
                continue

            if np.dot(direction, [0, 1, 0]) > direction_dot_threshold:
                unfiltered_descriptions.append(
                    f"{target_obj_name} on the right of a {reference_obj_name}"
                )
            elif np.dot(direction, [0, -1, 0]) > direction_dot_threshold:
                unfiltered_descriptions.append(
                    f"{target_obj_name} on the left of a {reference_obj_name}"
                )
            elif np.dot(direction, [1, 0, 0]) > direction_dot_threshold:
                unfiltered_descriptions.append(
                    f"{target_obj_name} in front of a {reference_obj_name}"
                )

            if np.dot(
                direction, [0, 1, 0]
            ) > direction_dot_threshold and should_add_relation(
                target_obj_name=target_obj_name,
                spatial_relation="on the right of",
                reference_obj_name=reference_obj_name,
            ):
                descriptions.add(
                    f"{target_obj_name} on the right of a {reference_obj_name}"
                )
                retval["target_obj_name"].append(target_obj_name)
                retval["target_obj_material"].append("")
                retval["target_obj_id"].append(target_obj_id)
                retval["reference_obj_name"].append(reference_obj_name)
                retval["reference_obj_material"].append(reference_material)
                retval["spatial_relation_name"].append("on the right of")
            elif np.dot(
                direction, [0, -1, 0]
            ) > direction_dot_threshold and should_add_relation(
                target_obj_name=target_obj_name,
                spatial_relation="on the left of",
                reference_obj_name=reference_obj_name,
            ):
                descriptions.add(
                    f"{target_obj_name} on the left of a {reference_obj_name}"
                )
                retval["target_obj_name"].append(target_obj_name)
                retval["target_obj_material"].append("")
                retval["target_obj_id"].append(target_obj_id)
                retval["reference_obj_name"].append(reference_obj_name)
                retval["reference_obj_material"].append(reference_material)
                retval["spatial_relation_name"].append("on the left of")
            elif np.dot(
                direction, [1, 0, 0]
            ) > direction_dot_threshold and should_add_relation(
                target_obj_name=target_obj_name,
                spatial_relation="in front of",
                reference_obj_name=reference_obj_name,
            ):
                descriptions.add(
                    f"{target_obj_name} in front of a {reference_obj_name}"
                )
                retval["target_obj_name"].append(target_obj_name)
                retval["target_obj_material"].append("")
                retval["target_obj_id"].append(target_obj_id)
                retval["reference_obj_name"].append(reference_obj_name)
                retval["reference_obj_material"].append(reference_material)
                retval["spatial_relation_name"].append("in front of")
    return retval


def camel_case_split(str):
    return re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", str)


def class_reduction_rule(raw_class_name):
    if "FP326:PS_326_" in raw_class_name:
        raw_class_name = raw_class_name.split("FP326:PS_326_")[1]
    class_name = (
        raw_class_name.split("_")[0]
        .split("Height")[0]
        .split("Standard")[-1]
        .split("|")[0]
        .split("Size")[0]
        .split("Done")[0]
    )
    if class_name.upper() == class_name:
        return class_name
    if len(camel_case_split(class_name)):
        class_name = " ".join(c.lower() for c in camel_case_split(class_name))
    class_name = "".join(class_name.split("mesh")).rstrip().lstrip()

    if "f " == class_name[:2]:
        class_name = class_name[2:]

    if "ladel" in class_name or "ladle" in class_name:
        return "ladle"
    if class_name == "towl":
        return "towel"
    if class_name == "plate stack":
        return "plate"

    if (
        "deco" in class_name
        and "decor" not in class_name
        and "decorative" not in class_name
        and "decoration" not in class_name
    ):
        class_name = class_name.replace("deco", "decoration")
    elif (
        "decor" in class_name
        and "decorative" not in class_name
        and "decoration" not in class_name
    ):
        class_name = class_name.replace("decor", "decoration")
    class_name = class_name.replace("counter top", "countertop")
    class_name = class_name.replace("fire place", "fireplace")
    class_name = class_name.replace("base board", "baseboard")
    class_name = class_name.replace("dish washer", "dishwasher")
    class_name = class_name.replace("dish washer", "dishwasher")
    class_name = class_name.replace("dish washer", "dishwasher")
    class_name = class_name.replace("bath tub", "bathtub")
    class_name = class_name.replace("base board", "baseboard")
    if "book" == class_name or "book stack" == class_name:
        return "book"
    if "rug" == class_name[-3:]:
        return "rug"
    if (
        class_name[-len("bottles") :] == "bottles"
        or class_name[-len("wires") :] == "wires"
        or class_name[-len("windows") :] == "windows"
        or class_name[-len("pans") :] == "pans"
        or class_name[-len("decals") :] == "decals"
        or class_name[-len("cups") :] == "cups"
        or class_name[-len("walls") :] == "walls"
        or class_name[-len("rods") :] == "rods"
        or class_name[-len("cans") :] == "cans"
        or class_name[-len("lights") :] == "lights"
    ):
        return class_name[:-1]
    if class_name[-len("glasses") :] == "glasses":
        return class_name[:-2]
    if "cloth" in class_name:
        return "cloth"
    if "island" in class_name:
        return "kitchen island"
    if "ceiling" in class_name:
        return class_name
    if "cabinet" in class_name:
        return "cabinet"
    if "fridge" in class_name:
        return "fridge"
    if "shelf" in class_name or "shelving" in class_name or "shelves" in class_name:
        return "shelf"
    if "knife" in class_name:
        return "knife"
    if "stove" in class_name:
        return "stove"
    if "wall" in class_name:
        return "wall"
    if "window" in class_name:
        return "window"
    if "door" in class_name:
        return "door"
    return class_name


def process_class_name(c):
    return c.split("|")[0].split(" ")[0]


def run_simulator(
    scene_id: str,
    domain_randomization: bool,
    np_rand: np.random.RandomState,
    num_attempts: int = 10,
    dist: float = 3.0,
    debug: bool = False,
):
    controller = None
    try:
        controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene=scene_id,
            # step sizes
            gridSize=0.05,
            snapToGrid=False,
            rotateStepDegrees=5,
            # image modalities
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            # camera properties
            width=width,
            height=height,
            fieldOfView=fov_w,
            # render headless
            platform=CloudRendering,
        )
    except Exception as e:
        logging.error(e)
        if controller is not None:
            controller.stop()
        return

    datapoint = None
    reachable_positions = controller.step(action="GetReachablePositions").metadata[
        "actionReturn"
    ]
    for _ in range(num_attempts):
        sampled_position = np_rand.choice(reachable_positions)
        sampled_rotation = dict(x=0, y=np_rand.uniform(0, 360), z=0)
        try:
            event = controller.step(
                action="Teleport",
                position=sampled_position,
                rotation=sampled_rotation,
                horizon=0,
                standing=True,
            )
        except Exception as e:
            logging.error(e)
            controller.stop()
            return
        classes = list(set(map(process_class_name, event.color_to_object_id.values())))
        semantic_img = np.zeros(event.instance_segmentation_frame.shape[:2]).astype(int)
        for color, objname in event.color_to_object_id.items():
            objname = process_class_name(objname)
            obj_mask = (event.instance_segmentation_frame == color).all(axis=-1)
            semantic_img[obj_mask] = classes.index(objname)

        # reflective surfaces in Unity shows depth of reflection probe
        reflective_surface_mask = event.depth_frame > 10.0
        depth = deepcopy(event.depth_frame)
        depth[reflective_surface_mask] = np.interp(
            np.flatnonzero(reflective_surface_mask),
            np.flatnonzero(~reflective_surface_mask),
            depth[~reflective_surface_mask],
        )
        if "Wall" in classes and (semantic_img == classes.index("Wall")).mean() > 0.8:
            continue
        # ideally most objects are between 1.5 and 3.5 meters away
        pixel_in_good_range = np.logical_and(
            depth < dist + 1.0,
            depth > dist - 1.0,
        )
        if len(np.unique(semantic_img)) < 4:
            if debug:
                plt.imshow(semantic_img)
                plt.show()
                logging.debug("not enough interesting objects")
            continue
        if pixel_in_good_range.mean() < 0.2:
            if debug:
                logging.debug("not enough pixels in good range")
                fig, axes = plt.subplots(1, 3)
                axes[0].axis("off")
                axes[1].axis("off")
                axes[2].axis("off")
                axes[0].imshow(depth)
                axes[1].imshow(pixel_in_good_range.astype(int))
                axes[2].imshow(event.frame)
                plt.show()
            continue
        domain_randomized_rgb = np.zeros(1)
        if domain_randomization:
            controller.step(action="RandomizeMaterials")
            domain_randomized_rgb = controller.step(action="RandomizeMaterials").frame
        controller.stop()
        datapoint = {
            "scene_id": scene_id,
            "rgb": deepcopy(event.frame),
            "depth": depth,
            "instance": deepcopy(event.instance_segmentation_frame),
            "color_to_object_id": deepcopy(event.color_to_object_id),
            "semantic": semantic_img,
            "classes": classes,
            "position": list(event.metadata["agent"]["position"].values()),
            "camera_horizon": event.metadata["agent"]["cameraHorizon"],
            "rotation": list(event.metadata["agent"]["rotation"].values()),
            "objects_info": event.metadata["objects"],
            "sampled_position": sampled_position,
            "sampled_rotation": sampled_rotation,
            "domain_randomized_rgb": domain_randomized_rgb,
        }
        break
    if datapoint is None:
        controller.stop()
        logging.debug("attempts ran out")
        return
    return datapoint


def scene_data_from_thor_datapoint(
    np_rand,
    datapoint: dict,
    dist: float,
    path_to_exported_scenes: str,
    debug: bool = False,
):
    cam_pose = affines.compose(
        T=datapoint["position"],
        R=euler.euler2mat(
            datapoint["rotation"][2] * np.pi / 180,
            datapoint["rotation"][1] * np.pi / 180,
            datapoint["rotation"][0] * np.pi / 180,
        ),
        Z=np.ones(3),
    )
    xyz_pts, rgb_pts = get_pointcloud(
        depth_img=datapoint["depth"],
        color_img=datapoint["rgb"],
        cam_intr=cam_intr,
        cam_pose=cam_pose,
    )
    # compute transform to align ground truth with view
    transform = (
        affines.compose(T=[0, 0, 2], R=euler.euler2mat(0, 0, 0), Z=np.array([1, 1, 1]))
        @ affines.compose(
            T=[0, 0, 0], R=euler.euler2mat(0, 0, 0), Z=np.array([1, 1, -1])
        )
        @ affines.compose(
            T=[0, 0, 0], R=euler.euler2mat(np.pi / 2, 0, 0), Z=np.ones(3) * 0.6
        )
        @ affines.compose(T=[0, 0, 0], R=euler.euler2mat(0, np.pi, 0), Z=np.ones(3))
        @ affines.compose(
            T=[dist - 0.5, 2.0, 0], R=euler.euler2mat(0, np.pi / 2, 0), Z=np.ones(3)
        )
        @ affines.compose(
            T=[0, 0, 0], R=euler.euler2mat(0, -np.pi, -np.pi), Z=np.ones(3)
        )
        @ np.linalg.inv(cam_pose)
    )
    scene_gt = parse_gt(
        scene_name=datapoint["scene_id"],
        path_to_exported_scenes=path_to_exported_scenes,
    )
    full_xyz_pts = scene_gt["full_xyz_pts"]
    remapped_full_objid_pts = scene_gt["full_objid_pts"]
    full_objid_unique = scene_gt["objids"]
    objid_to_class = scene_gt["objid_to_class"]
    receptacle_masks = scene_gt["receptacle_masks"]

    original_xyz_pts = full_xyz_pts.copy()

    full_xyz_pts = (
        transform
        @ np.concatenate(
            (original_xyz_pts, np.ones(len(original_xyz_pts))[:, None]), axis=1
        ).T
    ).T[:, :3]
    if debug:
        from plot_utils import plot_pointcloud

        mask = filter_pts_bounds(xyz=full_xyz_pts, bounds=scene_bounds)
        fig, ax = plt.subplots(1)
        ax.imshow(datapoint["rgb"])
        plot_pointcloud(
            xyz=full_xyz_pts[mask],
            features=remapped_full_objid_pts[mask],
            object_labels=np.array(objid_to_class),
            show_plot=False,
            delete_fig=False,
        )
        xyz_pts, rgb_pts = get_pointcloud(
            depth_img=datapoint["depth"],
            color_img=datapoint["rgb"],
            cam_intr=cam_intr,
            cam_pose=transform @ cam_pose,
        )
        plot_pointcloud(
            xyz=xyz_pts,
            features=rgb_pts,
            show_plot=True,
        )

        plt.show()

    # process instance
    remapped_seg = -np.ones(datapoint["instance"].shape[:2]).astype(int)
    objects_in_view = {
        color: instance_key
        for color, instance_key in datapoint["color_to_object_id"].items()
        if (datapoint["instance"] == color).all(axis=-1).any()
    }

    remapped_visible_obj_ids = dict()
    for obj_color, instance_key in objects_in_view.items():
        obj_mask = (datapoint["instance"] == obj_color).all(axis=-1)
        if instance_key in full_objid_unique:
            remapped_objid = full_objid_unique.index(instance_key)
        else:
            # project out to 3D, then find class in gt which is spatially closest
            # to projected mask
            xyz_pts, _ = get_pointcloud(
                depth_img=datapoint["depth"],
                color_img=None,
                cam_intr=cam_intr,
                cam_pose=transform @ cam_pose,
            )
            partial_obj_xyz_pts = xyz_pts[obj_mask.reshape(-1), :]
            partial_to_full_distances = dict()
            for int_obj_id, gt_obj_id in enumerate(full_objid_unique):
                if gt_obj_id == "empty":
                    continue
                gt_obj_mask = remapped_full_objid_pts == int_obj_id
                full_obj_xyz_pts = full_xyz_pts[gt_obj_mask, :]
                if len(full_obj_xyz_pts) == 0:
                    continue
                elif len(full_obj_xyz_pts) > 100:
                    full_obj_xyz_pts = full_obj_xyz_pts[
                        np_rand.choice(len(full_obj_xyz_pts), 100, replace=False), :
                    ]
                distances = (
                    (full_obj_xyz_pts[None, ...] - partial_obj_xyz_pts[:, None, ...])
                    ** 2
                ).sum(axis=2)
                all_distances = distances.min(axis=1).sum(axis=0)
                partial_to_full_distances[gt_obj_id] = all_distances
            gt_obj_id = min(partial_to_full_distances.items(), key=lambda v: v[1])[0]
            remapped_objid = full_objid_unique.index(gt_obj_id)
        remapped_visible_obj_ids[instance_key] = remapped_objid
        remapped_seg[obj_mask] = remapped_objid
    mask = filter_pts_bounds(xyz=full_xyz_pts, bounds=scene_bounds)
    full_xyz_pts = full_xyz_pts[mask, :]
    remapped_full_objid_pts = remapped_full_objid_pts[mask]
    logging.debug(f"NUM PTS: { len(full_xyz_pts)}")
    try:
        indices = np_rand.choice(len(full_xyz_pts), size=num_output_pts, replace=False)
    except Exception as e:
        logging.error("Not enough points")
        logging.error(e)
        return
    remapped_obj_ids = deepcopy(remapped_visible_obj_ids)
    for remapped_id, objid in enumerate(full_objid_unique):
        if objid not in remapped_obj_ids:
            remapped_obj_ids[objid] = remapped_id
    vox_size = 64
    tsdf_vol = TSDFVolume(vol_bnds=scene_bounds.T, voxel_size=2.0 / vox_size)
    tsdf_vol.integrate(
        color_im=datapoint["rgb"],
        depth_im=datapoint["depth"],
        cam_intr=cam_intr,
        cam_pose=transform @ cam_pose,
    )

    tsdf_xyz_pts = tsdf_vol.vox2world(
        tsdf_vol._vol_origin, tsdf_vol.vox_coords, tsdf_vol._voxel_size
    )
    tsdf_value_pts = tsdf_vol.get_volume()[0].reshape(-1)
    for objid in range(len(objid_to_class)):
        objid_to_class[objid] = objid_to_class[objid] + f"[{objid}]"

    scene_data = {
        "rgb": datapoint["rgb"][None, ...],
        "domain_randomized_rgb": datapoint["domain_randomized_rgb"][None, ...],
        "depth": datapoint["depth"][None, ...],
        "seg": remapped_seg[None, ...],
        "cam_intr": cam_intr,
        "cam_pose": transform @ cam_pose,
        "scene_bounds": scene_bounds,
        "tsdf_value_pts": tsdf_value_pts[None, ...],
        "tsdf_xyz_pts": tsdf_xyz_pts[None, ...],
        "full_xyz_pts": full_xyz_pts[indices, :][None, ...],
        "full_objid_pts": remapped_full_objid_pts[indices][None, ...],
        "objid_to_class": np.array(objid_to_class).astype("S"),
    }
    vg = VirtualGrid(
        scene_bounds=scene_bounds, grid_shape=tuple([vox_size] * 3), batch_size=1
    )
    query_points = torch.from_numpy(scene_data["full_xyz_pts"])
    grid_indices = (
        vg.get_points_grid_idxs(query_points, cast_to_int=True)[0].cpu().numpy()
    )
    tsdf_vol = tsdf_vol.get_volume()[0]
    visibility_pts_mask = (
        tsdf_vol[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]] > 0.0
    )
    scene_data["descriptions"] = get_all_relations(
        scene_data=scene_data,
        receptacle_masks={
            receptacle_name: {
                "mask": receptacle_mask[mask][indices],
                "xyz_pts": original_xyz_pts[receptacle_mask],
            }
            for receptacle_name, receptacle_mask in receptacle_masks.items()
        },
        objects_info={
            obj_info["objectId"]: obj_info for obj_info in datapoint["objects_info"]
        },
        remapped_visible_obj_ids=remapped_visible_obj_ids,
        all_remapped_obj_ids=remapped_obj_ids,
        visibility_pts_mask=visibility_pts_mask,
    )
    return scene_data


@ray.remote(num_cpus=1, num_gpus=0.05)
def generate_datapoint(
    scene_ids,
    dataset_dir_path: str,
    seed: int,
    path_to_exported_scenes: str,
    dist: float = 3.0,
    **kwargs,
):
    np_rand = np.random.RandomState(seed=seed)
    scene_id = np_rand.choice(scene_ids)
    output_path = f"{dataset_dir_path}/{seed:05d}|{scene_id}.hdf5"
    if os.path.exists(output_path):
        return
    domain_randomization = scene_id in test_scenes
    datapoint = run_simulator(
        scene_id=scene_id,
        dist=dist,
        np_rand=np_rand,
        domain_randomization=domain_randomization,
        **kwargs,
    )
    if datapoint is None:
        return
    scene_data = scene_data_from_thor_datapoint(
        datapoint=datapoint,
        dist=dist,
        np_rand=np_rand,
        path_to_exported_scenes=path_to_exported_scenes,
    )
    if scene_data is None:
        return
    init_dataset(output_path, data_structure=data_structure)
    with h5py.File(output_path, "a") as file:
        group = file.create_group(f"data")
        for key, value in scene_data.items():
            if key in data_structure.keys():
                region_references = resize_and_add_data(dataset=file[key], data=value)
                write_to_hdf5(group, key, region_references, dtype=h5py.regionref_dtype)
            else:
                write_to_hdf5(group, key, value)


def generate_gt_scenes(
    scene_ids: List[str], path_to_exported_scenes: str, path_to_custom_unity: str
):
    np.random.shuffle(scene_ids)
    for scene_id in scene_ids:
        if os.path.exists(
            f"{path_to_exported_scenes}/{scene_id}/full_xyz_pts.txt"
        ) and os.path.exists(
            f"{path_to_exported_scenes}/{scene_id}/full_objid_pts.txt"
        ):
            continue
        controller = None
        try:
            controller = Controller(
                local_executable_path=path_to_custom_unity,
                agentMode="default",
                visibilityDistance=1.5,
                scene=scene_id,
                # step sizes
                gridSize=0.25,
                snapToGrid=True,
                rotateStepDegrees=90,
                # image modalities
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                # camera properties
                width=width,
                height=height,
                fieldOfView=fov_w,
                # render headless
                platform=CloudRendering,
            )
        except Exception as e:
            logging.error(e)
        finally:
            if controller is not None:
                controller.stop()
    exit()


def generate_datapoints(
    dataset_dir_path: str,
    path_to_custom_unity: str,
    path_to_exported_scenes: str,
    num_processes: int,
    num_pts: int,
    start_seed: int,
    local: bool,
):
    ray.init(
        log_to_driver=True,
        local_mode=local,
    )
    tasks = []
    scene_ids = sorted(kitchens + living_rooms + bathrooms + bedrooms)
    not_gt_scene_ids = list(
        filter(
            lambda scene_id: not (
                os.path.exists(f"{path_to_exported_scenes}/{scene_id}/full_xyz_pts.txt")
                and os.path.exists(
                    f"{path_to_exported_scenes}/{scene_id}/full_objid_pts.txt"
                )
            ),
            scene_ids,
        )
    )
    logging.info("scenes without gts: " + ", ".join(not_gt_scene_ids))
    if (
        len(not_gt_scene_ids) > 0
        and input(f"There are {len(not_gt_scene_ids)} scenes without gt. Generate?")
        == "y"
    ):
        generate_gt_scenes(
            not_gt_scene_ids, path_to_exported_scenes, path_to_custom_unity
        )
    scene_ids = list(
        filter(
            lambda scene_id: (
                os.path.exists(f"{path_to_exported_scenes}/{scene_id}/full_xyz_pts.txt")
                and os.path.exists(
                    f"{path_to_exported_scenes}/{scene_id}/full_objid_pts.txt"
                )
            ),
            scene_ids,
        )
    )
    seed = start_seed
    tasks = [
        generate_datapoint.remote(
            scene_ids=scene_ids,
            dataset_dir_path=dataset_dir_path,
            path_to_exported_scenes=path_to_exported_scenes,
            seed=seed + i,
        )
        for i in range(num_processes)
    ]
    seed += num_processes

    pbar = tqdm(total=num_pts, smoothing=0.001)
    offset = 0

    while seed < start_seed + num_pts:
        readies, tasks = ray.wait(tasks, num_returns=1)
        pbar.update((seed - start_seed) - pbar.n)
        offset += len(readies)
        tasks.extend(
            [
                generate_datapoint.remote(
                    scene_ids=scene_ids,
                    dataset_dir_path=dataset_dir_path,
                    path_to_exported_scenes=path_to_exported_scenes,
                    seed=seed + i,
                )
                for i in range(len(readies))
            ]
        )
        seed += len(readies)
        pbar.set_description(f"CURR SEED: {seed:06d}")
        try:
            ray.get(readies)
        except Exception as e:
            logging.error(e)
            pass


data_structure = get_datastructure(
    image_shape=(width, height),
    relevancy_shape=(128, 128),
    clip_hidden_dim=512,
    tsdf_dim=(64, 64, 64),
    num_output_pts=num_output_pts,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir_path", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_pts", type=int, default=50000)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--path_to_custom_unity", type=str)
    parser.add_argument("--path_to_exported_scenes", type=str)
    args = parser.parse_args()
    if os.path.exists(args.dataset_dir_path) and (
        input(f"{args.dataset_dir_path} exists. replace?") == "y"
    ):
        shutil.rmtree(args.dataset_dir_path)
        os.mkdir(args.dataset_dir_path)
    elif not os.path.exists(args.dataset_dir_path):
        os.mkdir(args.dataset_dir_path)
    data = generate_datapoints(**vars(args))
