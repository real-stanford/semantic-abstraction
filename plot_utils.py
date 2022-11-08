import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import io
from PIL import Image
import open3d as o3d
from skimage.measure import block_reduce
import matplotlib.cm as cm
import matplotlib as mpl


def plot_to_png(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = np.array(Image.open(buf)).astype(np.uint8)
    return img


def set_view_and_save_img(fig, ax, views):
    for elev, azim in views:
        ax.view_init(elev=elev, azim=azim)
        yield plot_to_png(fig)


def plot_pointcloud(
    xyz,
    features,
    object_labels=None,
    background_color=(0.1, 0.1, 0.1, 0.99),
    num_points=50000,
    views=[(45, 135)],
    pts_size=3,
    alpha=0.5,
    plot_empty=False,
    visualize_ghost_points=False,
    object_colors=None,
    delete_fig=True,
    show_plot=False,
    bounds=[[-1, -1, -0.1], [1, 1, 1.9]],
):
    is_semantic = len(features.shape) == 1
    if type(alpha) is float:
        alpha = np.ones(xyz.shape[0]).astype(np.float32) * alpha
    if not plot_empty and is_semantic and object_labels is not None:
        mask = np.ones_like(alpha).astype(bool)
        for remove_label in ["empty", "unlabelled", "out of bounds"]:
            if remove_label in object_labels.tolist():
                remove_idx = object_labels.tolist().index(remove_label)
                mask = np.logical_and(mask, features != remove_idx)
        xyz = xyz[mask, :]
        features = features[mask, ...]
        alpha = alpha[mask]
        if type(pts_size) != int and type(pts_size) != float:
            pts_size = pts_size[mask]
    # subsample
    if xyz.shape[0] > num_points:
        indices = np.random.choice(xyz.shape[0], size=num_points, replace=False)
        xyz = xyz[indices, :]
        features = features[indices, ...]
        alpha = alpha[indices]
        if type(pts_size) != int and type(pts_size) != float:
            pts_size = pts_size[indices]

    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ax.set_facecolor(background_color)
    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)
    # ax._axis3don = False

    if is_semantic and object_labels is not None:
        object_ids = list(np.unique(features))
        object_labels = object_labels[object_ids].tolist()
        if object_colors is not None:
            object_colors = object_colors[object_ids]
        features = features.astype(np.int)
        # repack object ids
        repacked_obj_ids = np.zeros(features.shape).astype(np.uint32)
        for i, j in enumerate(object_ids):
            repacked_obj_ids[features == j] = i
        features = repacked_obj_ids

        object_ids = list(np.unique(features))
        colors = np.zeros((len(features), 4)).astype(np.uint8)
        if object_colors is None:
            cmap = plt.get_cmap("tab20")
            object_colors = (255 * cmap(np.array(object_ids) % 20)).astype(np.uint8)
        for obj_id in np.unique(features):
            colors[features == obj_id, :] = object_colors[obj_id]
        colors = colors.astype(float) / 255.0
        object_colors = object_colors.astype(float) / 255
        handles = [
            Patch(facecolor=c, edgecolor="grey", label=label)
            for label, c in zip(object_labels, object_colors)
        ]

        l = ax.legend(
            handles=handles,
            labels=object_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=4,
            facecolor=(0, 0, 0, 0.1),
            fontsize=8,
            framealpha=0,
        )
        plt.setp(l.get_texts(), color=(0.8, 0.8, 0.8))
    else:
        colors = features.astype(float)
        if colors.max() > 1.0:
            colors /= 255.0
            assert colors.max() <= 1.0
    # ensure alpha has same dims as colors
    if colors.shape[-1] == 4:
        colors[:, -1] = alpha
    ax.scatter(x, y, z, c=colors, s=pts_size)
    if visualize_ghost_points:
        x, y, z = np.array(np.unique(xyz, axis=0)).T
        ax.scatter(x, y, z, color=[1.0, 1.0, 1.0, 0.02], s=pts_size)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axes.set_xlim3d(left=bounds[0][0], right=bounds[1][0])
    ax.axes.set_ylim3d(bottom=bounds[0][1], top=bounds[1][1])
    ax.axes.set_zlim3d(bottom=bounds[0][2], top=bounds[1][2])
    plt.tight_layout(pad=0)
    imgs = list(set_view_and_save_img(fig, ax, views))
    if show_plot:
        plt.show()
    if delete_fig:
        plt.close(fig)
    return imgs


# meshes = []
# for class_id in np.unique(features):
#     mask = features == class_id
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz[mask, :])
#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     radii = [0.005, 0.01, 0.02, 0.04]
#     rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#         pcd, o3d.utility.DoubleVector(radii))
#     rec_mesh.paint_uniform_color(object_colors[class_id][:3])
#     meshes.append(rec_mesh)
# o3d.visualization.draw_geometries(meshes)


def view_tsdf(tsdf, simplify=True):
    main_color = "#00000055"
    mpl.rcParams["text.color"] = main_color
    mpl.rcParams["axes.labelcolor"] = main_color
    mpl.rcParams["xtick.color"] = main_color
    mpl.rcParams["ytick.color"] = main_color
    mpl.rc("axes", edgecolor=main_color)
    mpl.rcParams["grid.color"] = "#00000033"

    if simplify:
        tsdf = block_reduce(tsdf, block_size=(8, 8, 8), func=np.mean)
        print("block reduced", tsdf.shape)

    x = np.arange(tsdf.shape[0])[:, None, None]
    y = np.arange(tsdf.shape[1])[None, :, None]
    z = np.arange(tsdf.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    c = cm.plasma((tsdf.ravel() + 1))
    alphas = (tsdf.ravel() < 0).astype(float)
    c[..., -1] = alphas

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=c, s=1)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # Hide axes ticks
    ax.tick_params(axis="x", colors=(0.0, 0.0, 0.0, 0.0))
    ax.tick_params(axis="y", colors=(0.0, 0.0, 0.0, 0.0))
    ax.tick_params(axis="z", colors=(0.0, 0.0, 0.0, 0.0))
    ax.view_init(20, -110)

    plt.show()
