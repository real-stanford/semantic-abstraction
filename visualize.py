import io
import logging
from pathlib import Path
import textwrap
from typing import Any, Dict, List, Tuple
from skimage.measure import marching_cubes
import numpy as np
import torch
import os
import pickle
from net import SemAbs3D, SemAbsVOOL
from point_cloud import (
    check_pts_in_frustum,
    filter_pts_bounds,
    get_pointcloud,
    meshwrite,
)
import utils
import os
from utils import config_parser
from CLIP.clip import ClipWrapper, saliency_configs
from fusion import TSDFVolume
import typer
from matplotlib import pyplot as plt
from rich.progress import Progress
import open3d as o3d
from transforms3d import affines, euler
import imageio
from PIL import Image
import cv2

app = typer.Typer(pretty_exceptions_enable=False)
Point3D = Tuple[float, float, float]


def visualize_relevancies(
    rgb: np.ndarray,
    relevancies: np.ndarray,
    obj_classes: List[str],
    dump_path: str,
):
    fig, axes = plt.subplots(4, int(np.ceil(len(obj_classes) / 4)), figsize=(15, 15))
    axes = axes.flatten()
    vmin = 0.000
    cmap = plt.get_cmap("jet")
    vmax = 0.01
    [ax.axis("off") for ax in axes]
    for ax, label_grad, label in zip(axes, relevancies, obj_classes):
        ax.imshow(rgb)
        ax.set_title(label, fontsize=12)
        grad = np.clip((label_grad - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0)
        colored_grad = cmap(grad)
        grad = 1 - grad
        colored_grad[..., -1] = grad * 0.7
        ax.imshow(colored_grad)
    plt.tight_layout(pad=0)
    plt.savefig(dump_path)
    plt.close(fig)


def prep_data(
    data_pickle_path: str,
    scene_bounds: Tuple[Point3D, Point3D],
    subtract_mean: bool,
    dump_path: str,
):
    scene_id = data_pickle_path.split("/")[-1].split(".pkl")[0]
    data = pickle.load(open(data_pickle_path, "rb"))
    rgb = data["rgb"]
    assert rgb.dtype == np.uint8
    depth = data["depth"]
    assert depth.dtype == np.float32
    cam_intr = data["cam_intr"]
    assert depth.dtype == np.float32
    cam_extr = data["cam_extr"]
    assert depth.dtype == np.float32
    scene_dump_path = f"{dump_path}/{scene_id}"
    if not os.path.exists(scene_dump_path):
        Path(scene_dump_path).mkdir(parents=True, exist_ok=True)
    if "img_shape" in data:
        rgb = cv2.resize(rgb, data["img_shape"])
        depth = cv2.resize(depth, data["img_shape"])
    descriptions = data["descriptions"]
    target_obj_classes = [d[0] for d in descriptions]
    spatial_relation_names = [d[1] for d in descriptions]
    reference_obj_classes = [d[2] for d in descriptions]
    ovssc_obj_classes = data["ovssc_obj_classes"]
    relevancy_keys = list(
        set(ovssc_obj_classes).union(target_obj_classes).union(reference_obj_classes)
    )

    h, w, c = rgb.shape
    relevancies = (
        ClipWrapper.get_clip_saliency(
            img=rgb,
            text_labels=np.array(relevancy_keys),
            prompts=["a photograph of a {} in a home."],
            **saliency_configs["ours"](h),
        )[0]
        * 50
    )
    assert len(relevancy_keys) == len(relevancies)
    input_xyz_pts = torch.from_numpy(
        get_pointcloud(depth, None, cam_intr, cam_extr)[0].astype(np.float32)
    )
    in_bounds_mask = filter_pts_bounds(input_xyz_pts, np.array(scene_bounds)).bool()
    input_xyz_pts = input_xyz_pts[in_bounds_mask]
    input_rgb_pts = rgb.reshape(-1, 3)[in_bounds_mask.cpu().numpy()]
    if subtract_mean:
        relevancies -= relevancies.mean(dim=0, keepdim=True)
    visualize_relevancies(
        rgb=rgb,
        relevancies=relevancies.cpu().numpy() / 50,
        obj_classes=relevancy_keys,
        dump_path=scene_dump_path + "/relevancies.png",
    )
    ovssc_input_feature_pts = torch.stack(
        [
            relevancies[relevancy_keys.index(obj_class)].view(-1)[in_bounds_mask]
            for obj_class in ovssc_obj_classes
        ]
    )

    input_target_saliency_pts = torch.stack(
        [
            relevancies[relevancy_keys.index(obj_class)].view(-1)[in_bounds_mask]
            for obj_class in target_obj_classes
        ]
    )
    input_reference_saliency_pts = torch.stack(
        [
            relevancies[relevancy_keys.index(obj_class)].view(-1)[in_bounds_mask]
            for obj_class in reference_obj_classes
        ]
    )

    batch = {
        "input_xyz_pts": input_xyz_pts,
        "input_rgb_pts": input_rgb_pts,
        "relevancies": relevancies,
        "input_feature_pts": ovssc_input_feature_pts,
        "ovssc_obj_classes": ovssc_obj_classes,
        "rgb": rgb,
        "depth": depth,
        "cam_intr": cam_intr,
        "cam_extr": cam_extr,
        "scene_id": scene_id,
        "input_target_saliency_pts": input_target_saliency_pts,
        "input_reference_saliency_pts": input_reference_saliency_pts,
        "spatial_relation_name": spatial_relation_names,
        "tsdf_vol": None,
        "descriptions": [f"the {d[0]} {d[1]} the {d[2]}" for d in data["descriptions"]],
    }
    return batch


def process_batch_ovssc(
    net: SemAbs3D,
    batch: Dict[str, Any],
    scene_bounds: Tuple[Point3D, Point3D],
    device: str,
    num_input_pts: int,
    sampling_shape: Tuple[int, int, int] = (240, 240, 240),
    num_pts_per_pass: int = int(2**20),
    cutoff: float = -3.0,
) -> Dict[str, torch.Tensor]:

    grid_points = get_sample_points(
        sampling_shape=sampling_shape, scene_bounds=scene_bounds, device=device
    )
    assert filter_pts_bounds(
        grid_points.cpu().numpy(), bounds=np.array(scene_bounds)
    ).all()

    label_outputs = {}
    with Progress() as progress:
        inference_task = progress.add_task(
            "Running completion", total=len(batch["ovssc_obj_classes"])
        )
        for class_idx, obj_class in enumerate(batch["ovssc_obj_classes"]):
            label_outputs[obj_class] = []
            for j in np.arange(
                0,
                ((len(grid_points) // num_pts_per_pass) + 1) * num_pts_per_pass,
                num_pts_per_pass,
            ):
                if len(grid_points[j : j + num_pts_per_pass, :]) == 0:
                    break
                output_xyz_pts = grid_points[j : j + num_pts_per_pass, :][
                    None, None, ...
                ]
                input_xyz_pts = batch["input_xyz_pts"]
                indices = np.random.choice(input_xyz_pts.shape[-2], size=num_input_pts)
                label_outputs[obj_class].append(
                    net(
                        **{
                            **batch,
                            **{
                                "output_xyz_pts": output_xyz_pts.float().to(device),
                                "input_feature_pts": batch["input_feature_pts"][
                                    None, None, [class_idx], indices, None
                                ].to(device),
                                "input_xyz_pts": input_xyz_pts[..., indices, :]
                                .float()
                                .to(device),
                            },
                        }
                    )
                    .detach()
                    .cpu()
                )
            progress.update(inference_task, advance=1)
    label_outputs = {
        class_idx: torch.cat(patch_output, dim=-1).squeeze().view(*sampling_shape)
        for class_idx, patch_output in label_outputs.items()
    }
    tsdf_vol = TSDFVolume(
        vol_bnds=np.array(scene_bounds).T,
        voxel_size=(scene_bounds[1][0] - scene_bounds[0][0]) / sampling_shape[0],
    )
    tsdf_vol.integrate(
        color_im=batch["rgb"],
        depth_im=batch["depth"],
        cam_intr=batch["cam_intr"],
        cam_pose=batch["cam_extr"],
    )
    tsdf_vol = tsdf_vol.get_volume()[0]
    logprobs = torch.stack(
        [label_outputs[label] for label in batch["ovssc_obj_classes"]], dim=-1
    )
    prediction = logprobs.argmax(dim=-1)
    empty_mask = (logprobs < cutoff).all(dim=-1)
    empty_mask = empty_mask.view(*sampling_shape)
    in_frustum_mask = check_pts_in_frustum(
        xyz_pts=grid_points.cpu().numpy(),
        depth=batch["depth"],
        cam_pose=batch["cam_extr"],
        cam_intr=batch["cam_intr"],
    )
    in_frustum_mask = torch.from_numpy(in_frustum_mask).view(*sampling_shape)
    prediction_volumes = {}
    for class_idx, class_label in enumerate(batch["ovssc_obj_classes"]):
        patch_prediction = (prediction == class_idx).float().view(*sampling_shape)
        patch_prediction[empty_mask] = 0.0
        patch_prediction[~in_frustum_mask] = 0.0
        patch_prediction[tsdf_vol > 0.0] = 0.0
        prediction_volumes[class_label] = patch_prediction.cpu().numpy()
    return prediction_volumes


def export_obj(vol, filename, level=0.5):
    vol[:, :, -1] = -np.inf
    vol[:, :, 0] = -np.inf
    vol[:, -1, :] = -np.inf
    vol[:, 0, :] = -np.inf
    vol[-1, :, :] = -np.inf
    vol[0, :, :] = -np.inf
    if (vol < level).all():
        return
    verts, faces, norms, _ = marching_cubes(vol, level=level)

    vol_shape = np.array(vol.shape)
    verts -= vol_shape / 2
    verts = verts / vol_shape
    # Write header
    obj_file = open(filename, "w")

    # Write vertex list
    for i in range(verts.shape[0]):
        obj_file.write("v %f %f %f\n" % (verts[i, 0], verts[i, 1], verts[i, 2]))

    for i in range(norms.shape[0]):
        obj_file.write("vn %f %f %f\n" % (norms[i, 0], norms[i, 1], norms[i, 2]))

    faces = faces.copy()
    faces += 1

    for i in range(faces.shape[0]):
        obj_file.write("f %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))
    obj_file.close()


def get_sample_points(
    sampling_shape: Tuple[int, int, int],
    scene_bounds: Tuple[Point3D, Point3D],
    device: str,
):
    axis_coords = [torch.arange(0, x, device=device) for x in sampling_shape]
    coords_per_axis = torch.meshgrid(*axis_coords, indexing="ij")
    grid_idxs = torch.stack(coords_per_axis, dim=-1).to(device)
    lc = torch.tensor(scene_bounds[0], device=device, dtype=torch.float32)
    uc = torch.tensor(scene_bounds[1], device=device, dtype=torch.float32)
    idx_scale = torch.tensor(sampling_shape, device=device, dtype=torch.float32) - 1
    scales = (uc - lc) / idx_scale
    offsets = lc
    grid_idxs_f = grid_idxs.to(torch.float32)
    grid_points = grid_idxs_f * scales + offsets
    return grid_points.view(-1, 3)


@app.command()
def ovssc_inference(
    data_pickle_path: str,
    model_ckpt_path: str,
    dump_path: str = "visualization/",
):
    args = config_parser().parse_args(
        args=["--load", model_ckpt_path, "--file_path", data_pickle_path]
    )
    with open(os.path.dirname(args.load) + "/args.pkl", "rb") as file:
        exp_args = pickle.load(file)
        for arg in vars(exp_args):
            if any(arg == s for s in ["device", "file_path", "load"]):
                continue
            setattr(args, arg, getattr(exp_args, arg))
    args.domain_randomization = False
    scene_bounds = tuple(args.scene_bounds)
    logging.info("Preparing batch")
    batch = prep_data(
        data_pickle_path=data_pickle_path,
        scene_bounds=scene_bounds,
        subtract_mean=args.subtract_mean_relevancy,
        dump_path=dump_path,
    )
    logging.info(
        f"Fetched {len(batch['ovssc_obj_classes'])} classes: "
        + ", ".join(batch["ovssc_obj_classes"])
    )
    pickle.dump(batch, open("new-input.pkl", "wb"))
    batch = pickle.load(open("new-input.pkl", "rb"))
    if not os.path.exists(f"{dump_path}/{batch['scene_id']}"):
        Path(f"{dump_path}/{batch['scene_id']}").mkdir(parents=True, exist_ok=True)
    net = utils.get_net(net_class=SemAbs3D, **vars(args))[0]
    net.eval()
    prediction_volumes = process_batch_ovssc(
        net=net,
        batch=batch,
        scene_bounds=scene_bounds,
        device=args.device,
        num_input_pts=args.num_input_pts,
    )
    logging.info(f"Dumping meshes to {dump_path}/{batch['scene_id']}")
    for obj_class, vol in prediction_volumes.items():
        try:
            export_obj(
                vol=vol,
                filename=f"{dump_path}/{batch['scene_id']}/{obj_class}.obj",
                level=0.5,
            )
        except RuntimeError as e:
            print(f"{obj_class} probably empty: {e}")


def process_batch_vool(
    net: SemAbs3D,
    batch: Dict[str, Any],
    scene_bounds: Tuple[Point3D, Point3D],
    device: str,
    num_input_pts: int,
    sampling_shape: Tuple[int, int, int] = (240, 240, 240),
    num_pts_per_pass: int = int(2**20),
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

    grid_points = get_sample_points(
        sampling_shape=sampling_shape, scene_bounds=scene_bounds, device=device
    )
    assert filter_pts_bounds(
        grid_points.cpu().numpy(), bounds=np.array(scene_bounds)
    ).all()

    desc_predictions = {}
    with Progress() as progress:
        inference_task = progress.add_task(
            "Running localization", total=len(batch["descriptions"])
        )
        for desc_idx, desc in enumerate(batch["descriptions"]):
            desc_predictions[desc] = []
            for j in np.arange(
                0,
                ((len(grid_points) // num_pts_per_pass) + 1) * num_pts_per_pass,
                num_pts_per_pass,
            ):
                if len(grid_points[j : j + num_pts_per_pass, :]) == 0:
                    break
                output_xyz_pts = grid_points[j : j + num_pts_per_pass, :][
                    None, None, ...
                ]
                input_xyz_pts = batch["input_xyz_pts"]
                indices = np.random.choice(input_xyz_pts.shape[-2], size=num_input_pts)
                desc_predictions[desc].append(
                    net(
                        **{
                            **batch,
                            **{
                                "output_xyz_pts": output_xyz_pts.float().to(device),
                                "input_target_saliency_pts": batch[
                                    "input_target_saliency_pts"
                                ][None, None, [desc_idx], indices, None].to(device),
                                "input_reference_saliency_pts": batch[
                                    "input_reference_saliency_pts"
                                ][None, None, [desc_idx], indices, None].to(device),
                                "spatial_relation_name": [
                                    [batch["spatial_relation_name"][desc_idx]]
                                ],
                                "input_xyz_pts": input_xyz_pts[..., indices, :]
                                .float()
                                .to(device),
                            },
                        }
                    )
                    .detach()
                    .cpu()
                )
            progress.update(inference_task, advance=1)
    desc_predictions = {
        desc: torch.cat(patch_output, dim=-1).squeeze().view(*sampling_shape)
        for desc, patch_output in desc_predictions.items()
    }
    return desc_predictions, grid_points


@app.command()
def vool_inference(
    data_pickle_path: str,
    model_ckpt_path: str,
    dump_path: str = "visualization/",
):
    args = config_parser().parse_args(
        args=["--load", model_ckpt_path, "--file_path", data_pickle_path]
    )
    with open(os.path.dirname(args.load) + "/args.pkl", "rb") as file:
        exp_args = pickle.load(file)
        for arg in vars(exp_args):
            if any(arg == s for s in ["device", "file_path", "load"]):
                continue
            setattr(args, arg, getattr(exp_args, arg))
    args.domain_randomization = False
    scene_bounds = tuple(args.scene_bounds)
    logging.info("Preparing batch")
    batch = prep_data(
        data_pickle_path=data_pickle_path,
        scene_bounds=scene_bounds,
        subtract_mean=args.subtract_mean_relevancy,
        dump_path=dump_path,
    )
    logging.info(
        f"Fetched {len(batch['descriptions'])} descriptions: "
        + ", ".join(batch["descriptions"])
    )
    pickle.dump(batch, open("new-input.pkl", "wb"))
    batch = pickle.load(open("new-input.pkl", "rb"))
    net = utils.get_net(net_class=SemAbsVOOL, **vars(args))[0]
    net.eval()
    desc_predictions, grid_points = process_batch_vool(
        net=net,
        batch=batch,
        scene_bounds=scene_bounds,
        device=args.device,
        num_input_pts=args.num_input_pts,
    )
    logging.info(f"Dumping pointclouds to {dump_path}/{batch['scene_id']}")
    cmap = plt.get_cmap("jet")
    for desc, prediction in desc_predictions.items():
        prediction = prediction.squeeze().view(-1)
        keep_mask = prediction > prediction.max() - 0.15
        desc_points = grid_points[keep_mask]
        logprobs = prediction[keep_mask]
        logprobs = logprobs.exp().numpy()
        vmin = logprobs.min()
        vmax = logprobs.max()
        logprobs = (logprobs - vmin) / (vmax - vmin)
        colors = cmap(logprobs)[..., :3]
        meshwrite(
            filename=f"{dump_path}/{batch['scene_id']}/{desc}.ply",
            verts=desc_points.cpu().numpy(),
            colors=(colors * 255).astype(np.uint8),
        )
    indices = np.arange(len(batch["input_xyz_pts"]))
    if len(batch["input_xyz_pts"]) > 100000:
        indices = np.random.choice(
            len(batch["input_xyz_pts"]), size=100000, replace=False
        )
    meshwrite(
        filename=f"{dump_path}/{batch['scene_id']}/scene_rgb.ply",
        verts=batch["input_xyz_pts"].cpu().numpy()[indices],
        colors=batch["input_rgb_pts"][indices],
    )

# color palette from https://sashamaps.net/docs/resources/20-colors/
twenty_color_palette = (
    np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 212],
            [0, 128, 128],
            [220, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
            [0, 0, 128],
            [128, 128, 128],
            [255, 255, 255],
            [0, 0, 0],
        ]
    )
    / 255
)


def render_animation(geometries, n_frames=220, point_size=6, **kwargs):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=1024)
    vis.get_render_option().point_size = point_size
    for geom in geometries:
        vis.add_geometry(geom)
    images = []
    with Progress() as progress:
        render_task = progress.add_task("Rendering", total=n_frames)
        for _ in range(n_frames):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            vis.update_renderer()
            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            images.append((img * 255).astype(np.uint8))
            progress.update(render_task, advance=1)
    vis.destroy_window()
    return images


def generate_legend(legend):
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", c) for c in legend.values()]
    legend = plt.legend(
        handles, list(legend.keys()), loc=3, framealpha=0, frameon=False
    )
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches=bbox)
    buf.seek(0)
    img = np.array(Image.open(buf)).astype(np.uint8)
    return img


@app.command()
def ovssc_visualize(output_path: str):
    geometries = []
    rotate = affines.compose(
        T=[0, 0, 0], R=euler.euler2mat(-np.pi / 2, 0, 0), Z=[1, 1, 1]
    )
    legend = {}
    for idx, path in enumerate(Path(output_path).rglob("*.obj")):
        path = str(path)
        mesh = o3d.io.read_triangle_mesh(path)
        mesh = mesh.transform(rotate)
        class_name = "\n".join(textwrap.wrap(path.split("/")[-1].split(".obj")[0], 30))
        # color mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        pcd.paint_uniform_color(twenty_color_palette[idx % 20])
        legend[class_name] = twenty_color_palette[idx % 20]
        geometries.append(pcd)
    output_path = f"{output_path}/completion.mp4"
    legend_img = generate_legend(legend)[:, :, :3]
    h, w, _ = legend_img.shape
    mask = (legend_img != 255).any(axis=2)
    with imageio.get_writer(output_path, fps=24) as writer:
        for img in render_animation(geometries=geometries, point_size=4):
            img[:h, :w, :][mask] = legend_img[mask]
            writer.append_data(img)
    print(output_path)


@app.command()
def vool_visualize(output_path: str):
    pointclouds = {
        str(path).split("/")[-1].split(".ply")[0]: o3d.io.read_point_cloud(str(path))
        for path in Path(output_path).rglob("*.ply")
    }
    rotate = affines.compose(
        T=[0, 0, 0], R=euler.euler2mat(-np.pi / 2, 0, 0), Z=[1, 1, 1]
    )
    scene = pointclouds["scene_rgb"].voxel_down_sample(voxel_size=0.03)
    scene = scene.transform(rotate)

    for desc, localization in pointclouds.items():
        if desc == "scene_rgb":
            continue
        localization = localization.transform(rotate)
        with imageio.get_writer(f"{output_path}/{desc}.mp4", fps=24) as writer:
            for image in render_animation(geometries=[scene, localization]):
                writer.append_data(image)
        print(f"{output_path}/{desc}.mp4")


if __name__ == "__main__":
    app()


"""
### scene_4_living-room-1.pkl (NO, VOOL messed up for some reason..., should look into this)
python visualize.py ovssc-inference matterport/scene_4_living-room-1.pkl models/ours/ovssc/ovssc.pth
python visualize.py ovssc-visualize visualization/scene_4_living-room-1
python visualize.py vool-inference matterport/scene_4_living-room-1.pkl models/ours/vool/vool.pth
python visualize.py vool-visualize visualization/scene_4_living-room-1

### scene_1_kitchen-5.pkl (YES)
python visualize.py ovssc-inference matterport/scene_1_kitchen-5.pkl models/ours/ovssc/ovssc.pth
python visualize.py ovssc-visualize visualization/scene_1_kitchen-5
python visualize.py vool-inference matterport/scene_1_kitchen-5.pkl models/ours/vool/vool.pth
python visualize.py vool-visualize visualization/scene_1_kitchen-5

### 00754-EqZacbtdApE_living-room-1 (YES)
python visualize.py ovssc-inference matterport/00754-EqZacbtdApE_living-room-1.pkl models/ours/ovssc/ovssc.pth
python visualize.py ovssc-visualize visualization/00754-EqZacbtdApE_living-room-1
python visualize.py vool-inference matterport/00754-EqZacbtdApE_living-room-1.pkl models/ours/vool/vool.pth
python visualize.py vool-visualize visualization/00754-EqZacbtdApE_living-room-1

scene_2_hallway-2 (YES)

310_kitchen-6 (BAD OVSSC)

scene_2_bedroom-8 (COMPLETION AND LOCALIZATION MESSED UP)

vn_poster (Good completion)
"""
