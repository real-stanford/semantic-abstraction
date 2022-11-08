from typing import List
from pathlib import Path
import h5py
import torch
from tqdm import tqdm
import ray
from utils import write_to_hdf5
from filelock import FileLock
import numpy as np
from CLIP.clip import ClipWrapper, saliency_configs, imagenet_templates
from dataset import synonyms, deref_h5py
import typer
import imageio
from matplotlib import pyplot as plt
import cv2
from time import time

app = typer.Typer()


def resize_and_add_data(dataset, data):
    data_shape = np.array(data.shape)
    dataset_shape = np.array(dataset.shape)
    assert (dataset_shape[1:] == data_shape[1:]).all()
    dataset.resize(dataset_shape[0] + data_shape[0], axis=0)
    dataset[-data_shape[0] :, ...] = data
    return [
        dataset.regionref[dataset_shape[0] + i, ...]
        for i in np.arange(0, data_shape[0])
    ]


def get_datastructure(image_shape, relevancy_shape, tsdf_dim, num_output_pts, **kwargs):
    image_shape = list(image_shape)
    relevancy_shape = list(relevancy_shape)
    return {
        "rgb": {"dtype": "uint8", "item_shape": image_shape + [3]},
        "depth": {"dtype": "f", "item_shape": image_shape},
        "seg": {"dtype": "i", "item_shape": image_shape},
        "saliencies": {"dtype": "f", "item_shape": relevancy_shape},
        "tsdf_value_pts": {"dtype": "f", "item_shape": [np.prod(tsdf_dim)]},
        "tsdf_xyz_pts": {"dtype": "f", "item_shape": [np.prod(tsdf_dim), 3]},
        "full_xyz_pts": {"dtype": "f", "item_shape": [num_output_pts, 3]},
        "full_objid_pts": {"dtype": "i", "item_shape": [num_output_pts]},
    }


def init_dataset(file_path, data_structure):
    with h5py.File(file_path, mode="w") as file:
        # setup
        for key, data_info in data_structure.items():
            file.create_dataset(
                name=key,
                shape=tuple([0] + data_info["item_shape"]),
                dtype=data_info["dtype"],
                chunks=tuple([1] + data_info["item_shape"]),
                compression="gzip",
                compression_opts=9,
                maxshape=tuple([None] + data_info["item_shape"]),
            )


@ray.remote
def generate_saliency_helper(
    clip_wrapper, rgb_inputs, prompts, text_labels, scene_path, replace
):
    saliencies = {
        rgb_name: {
            saliency_config_name: ray.get(
                clip_wrapper.get_clip_saliency.remote(
                    img=rgb,
                    text_labels=text_labels,
                    prompts=prompts
                    if "imagenet_prompt_ensemble"
                    not in saliency_config(img_dim=min(rgb.shape[:2]))
                    or not saliency_config(img_dim=min(rgb.shape[:2]))[
                        "imagenet_prompt_ensemble"
                    ]
                    else imagenet_templates,
                    **saliency_config(img_dim=min(rgb.shape[:2])),
                )
            )
            for saliency_config_name, saliency_config in saliency_configs.items()
        }
        for rgb_name, rgb in rgb_inputs.items()
    }
    with FileLock(scene_path + ".lock"):
        with h5py.File(scene_path, mode="a") as f:
            saliency_group = f["data"].create_group("saliencies")
            for rgb_name, rgb_saliencies in saliencies.items():
                for (
                    saliency_config_name,
                    (config_saliency, text_label_features),
                ) in rgb_saliencies.items():
                    storage_dims = np.array(f["saliencies"].shape)[1:]
                    config_saliency = torch.nn.functional.interpolate(
                        config_saliency[:, None, :, :],
                        size=tuple(storage_dims),
                        mode="nearest-exact"
                        # mode='bilinear',
                        # align_corners=False
                    )[:, 0]

                    config_saliency = torch.cat(
                        [config_saliency, config_saliency.mean(dim=0, keepdim=True)],
                        dim=0,
                    )
                    text_label_features = torch.cat(
                        [
                            text_label_features,
                            text_label_features.mean(dim=0, keepdim=True),
                        ],
                        dim=0,
                    )
                    text_label_features /= text_label_features.norm(
                        dim=-1, keepdim=True
                    )
                    write_to_hdf5(
                        saliency_group,
                        key=rgb_name
                        + "|"
                        + saliency_config_name
                        + "|saliency_text_labels",
                        value=np.array(text_labels + ["mean"]).astype("S"),
                        replace=replace,
                    )
                    write_to_hdf5(
                        saliency_group,
                        key=rgb_name
                        + "|"
                        + saliency_config_name
                        + "|saliency_text_label_features",
                        value=text_label_features,
                        replace=replace,
                    )
                    region_references = resize_and_add_data(
                        dataset=f["saliencies"], data=config_saliency
                    )
                    write_to_hdf5(
                        saliency_group,
                        key=rgb_name + "|" + saliency_config_name,
                        dtype=h5py.regionref_dtype,
                        value=region_references,
                        replace=replace,
                    )
    return clip_wrapper


@app.command()
def dataset(
    file_path: str,
    num_processes: int,
    local: bool,
    prompts: List[str] = ["a render of a {} in a game engine."],
    replace=False,
):
    if "matterport" in file_path or "nyu" in file_path:
        prompts = ["a photograph of a {} in a home."]
    print(prompts)
    tasks = []
    ray.init(log_to_driver=True, local_mode=local)
    num_cuda_devices = torch.cuda.device_count()
    assert num_cuda_devices > 0
    print(f"[INFO] FOUND {num_cuda_devices} CUDA DEVICE")
    wrapper_actor_cls = ray.remote(ClipWrapper)
    available_clip_wrappers = [
        wrapper_actor_cls.options(num_gpus=num_cuda_devices / num_processes).remote(
            clip_model_type="ViT-B/32", device="cuda"
        )
        for _ in range(num_processes)
    ]

    scene_paths = list(reversed(sorted(map(str, Path(file_path).rglob("*.hdf5")))))
    if replace:
        if input("Replace = True. Delete existing relevancies? [y/n]") != "y":
            exit()
        for scene_path in tqdm(
            scene_paths, dynamic_ncols=True, desc="deleting existing relevancies"
        ):
            try:
                with h5py.File(scene_path, mode="a") as f:
                    for k in f["data"]:
                        if "salienc" in k:
                            del f[f"data/{k}"]
                    if "saliencies" in f:
                        data_shape = list(f["saliencies"].shape[1:])
                        del f["saliencies"]
                        f.create_dataset(
                            name="saliencies",
                            shape=tuple([0] + data_shape),
                            dtype="f",
                            chunks=tuple([1] + data_shape),
                            compression="gzip",
                            compression_opts=9,
                            maxshape=tuple([None] + data_shape),
                        )
            except Exception as e:
                print(e, scene_path)
        exit()
    for scene_path in tqdm(
        scene_paths, dynamic_ncols=True, desc="generating relevancies", smoothing=0.001
    ):
        assert len(available_clip_wrappers) > 0
        try:
            with h5py.File(scene_path, mode="a") as f:
                scene_already_done = "saliencies" in f["data"]
                if not scene_already_done or replace:
                    if scene_already_done:
                        for k in f["data"]:
                            if "salienc" in k:
                                del f[f"data/{k}"]
                        data_shape = f["saliencies"].shape[1:]
                        if "saliencies" in f:
                            del f["saliencies"]
                            f.create_dataset(
                                name="saliencies",
                                shape=tuple([0] + data_shape),
                                dtype="f",
                                chunks=tuple([1] + data_shape),
                                compression="gzip",
                                compression_opts=9,
                                maxshape=tuple([None] + data_shape),
                            )

                    if "data/visible_scene_obj_labels" in f:
                        del f["data/visible_scene_obj_labels"]
                    objid_to_class = np.array(f[f"data/objid_to_class"]).astype(str)
                    text_labels = objid_to_class.copy()
                    scene_has_groundtruth = (
                        "seg" in f["data"] and "full_objid_pts" in f["data"]
                    )
                    visible_scene_obj_labels = text_labels.copy()
                    if scene_has_groundtruth:
                        objids_in_scene = list(
                            set(
                                deref_h5py(
                                    dataset=f["full_objid_pts"],
                                    refs=f["data/full_objid_pts"],
                                )
                                .astype(int)
                                .reshape(-1)
                            )
                            - {-1}
                        )  # remove empty
                        scene_object_labels = text_labels.copy()[objids_in_scene]

                        # remove objects which are not in view
                        gt_seg = deref_h5py(dataset=f["seg"], refs=f["data"]["seg"])[0]
                        visible_obj_ids = list(map(int, set(np.unique(gt_seg)) - {-1}))
                        visible_obj_labels = text_labels[visible_obj_ids]
                        visible_scene_obj_labels = list(
                            set(visible_obj_labels).intersection(
                                set(scene_object_labels)
                            )
                        )
                        visible_scene_obj_labels = list(
                            sorted(
                                set(
                                    map(
                                        lambda c: c.split("[")[0].lstrip().rstrip(),
                                        visible_scene_obj_labels,
                                    )
                                )
                            )
                        )
                        # visible_scene_obj_labels used to filter
                        # objects both visible and in scene
                        text_labels = visible_obj_labels.copy()
                    text_labels = set(text_labels)

                    # create saliency maps necessary for descriptions
                    if (
                        "descriptions" in f["data"]
                        and len(np.array(f["data/descriptions/spatial_relation_name"]))
                        > 0
                    ):
                        target_obj_names = np.array(
                            f["data/descriptions/target_obj_name"]
                        ).astype(str)
                        reference_obj_names = np.array(
                            f["data/descriptions/reference_obj_name"]
                        ).astype(str)
                        spatial_relation_names = np.array(
                            f["data/descriptions/spatial_relation_name"]
                        ).astype(str)
                        text_labels = text_labels.union(
                            target_obj_names.tolist() + reference_obj_names.tolist()
                        )

                        # gradcam for clip spatial
                        descriptions = ""
                        for desc_part in [
                            target_obj_names,
                            " ",
                            spatial_relation_names,
                            " a ",
                            reference_obj_names,
                        ]:
                            descriptions = np.char.add(descriptions, desc_part)
                        text_labels = text_labels.union(descriptions)
                        # descriptions with synonyms
                        descriptions = ""
                        for desc_part in [
                            np.array(
                                list(
                                    map(
                                        lambda x: x
                                        if x not in synonyms.keys()
                                        else synonyms[x],
                                        target_obj_names,
                                    )
                                )
                            ),
                            " ",
                            spatial_relation_names,
                            " a ",
                            np.array(
                                list(
                                    map(
                                        lambda x: x
                                        if x not in synonyms.keys()
                                        else synonyms[x],
                                        reference_obj_names,
                                    )
                                )
                            ),
                        ]:
                            descriptions = np.char.add(descriptions, desc_part)
                        text_labels = text_labels.union(descriptions)
                    text_labels = set(
                        map(lambda c: c.split("[")[0].lstrip().rstrip(), text_labels)
                    )

                    # do synonyms
                    text_labels = text_labels.union(
                        map(
                            lambda text_label: synonyms[text_label],
                            filter(
                                lambda text_label: text_label in synonyms, text_labels
                            ),
                        )
                    )
                    for remove_label in {"unlabelled", "empty", "out of bounds"}:
                        if remove_label in text_labels:
                            text_labels.remove(remove_label)
                    text_labels = list(sorted(text_labels))

                    rgb_inputs = {"rgb": np.array(f["rgb"][f["data"]["rgb"][0]][0])}
                    if (
                        "domain_randomized_rgb" in f["data"]
                        and len(np.array(f["data/domain_randomized_rgb"])[0].shape) > 1
                    ):
                        rgb_inputs["domain_randomized_rgb"] = np.array(
                            f["data/domain_randomized_rgb"]
                        )[0]
                    write_to_hdf5(
                        f["data"],
                        key="visible_scene_obj_labels",
                        value=np.array(visible_scene_obj_labels).astype("S"),
                        replace=replace,
                    )
                    clip_wrapper = available_clip_wrappers.pop()
                    tasks.append(
                        generate_saliency_helper.remote(
                            clip_wrapper=clip_wrapper,
                            scene_path=scene_path,
                            rgb_inputs=rgb_inputs,
                            text_labels=text_labels,
                            prompts=prompts,
                            replace=replace,
                        )
                    )
        except Exception as e:
            print(e)
            print(scene_path, "invalid hdf5 file")
        if len(available_clip_wrappers) == 0:
            readies, tasks = ray.wait(tasks, num_returns=1)
            num_readies = len(readies)
            try:
                available_clip_wrappers.extend(ray.get(readies))
            except Exception as e:
                print(e)
                available_clip_wrappers.extend(
                    [
                        wrapper_actor_cls.options(
                            num_gpus=num_cuda_devices / num_processes
                        ).remote(clip_model_type="ViT-B/32", device="cuda")
                        for _ in range(num_readies)
                    ]
                )
    ray.get(tasks)


@app.command()
def image(
    file_path: str = typer.Argument(
        default="matterport.png", help="path of image file"
    ),
    labels: List[str] = typer.Option(
        default=[
            "basketball jersey",
            "nintendo switch",
            "television",
            "ping pong table",
            "vase",
            "fireplace",
            "abstract painting of a vespa",
            "carpet",
            "wall",
        ],
        help='list of object categories (e.g.: "nintendo switch")',
    ),
    prompts: List[str] = typer.Option(
        default=["a photograph of a {} in a home."],
        help="prompt template to use with CLIP.",
    ),
):
    """
    Generates a multi-scale relevancy for image at `file_path`.
    """
    img = np.array(imageio.imread(file_path))
    assert img.dtype == np.uint8
    h, w, c = img.shape
    start = time()
    grads = ClipWrapper.get_clip_saliency(
        img=img,
        text_labels=np.array(labels),
        prompts=prompts,
        **saliency_configs["ours"](h),
    )[0]
    print(f"get gradcam took {float(time() - start)} seconds", grads.shape)
    grads -= grads.mean(axis=0)
    grads = grads.cpu().numpy()
    fig, axes = plt.subplots(3, 3)
    axes = axes.flatten()
    vmin = 0.002
    cmap = plt.get_cmap("jet")
    vmax = 0.008
    for ax, label_grad, label in zip(axes, grads, labels):
        ax.axis("off")
        ax.imshow(img)
        ax.set_title(label, fontsize=12)
        grad = np.clip((label_grad - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)
        colored_grad = cmap(grad)
        grad = 1 - grad
        colored_grad[..., -1] = grad * 0.7
        ax.imshow(colored_grad)
    plt.tight_layout(pad=0)
    plt.savefig("grads.png")
    print("dumped relevancy to grads.png")
    plt.show()


if __name__ == "__main__":
    app()
