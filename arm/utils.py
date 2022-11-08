# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/utils.py

import torch
import numpy as np
from scipy.spatial.transform import Rotation

import pyrender
import trimesh
from pyrender.trackball import Trackball


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler("xyz", degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


def point_to_voxel_index(
    point: np.ndarray, voxel_size: np.ndarray, coord_bounds: np.ndarray
):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one
    )
    return voxel_indicy


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    scale = scene.scale
    # if scale == 0.0:
    #     scale = DEFAULT_SCENE_SCALE
    scale = 4.0
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp


def _from_trimesh_scene(trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {
        name: pyrender.Mesh.from_trimesh(geom, smooth=False)
        for name, geom in trimesh_scene.geometry.items()
    }
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def create_voxel_scene(
    voxel_grid: np.ndarray,
    q_attention: np.ndarray = None,
    highlight_coordinate: np.ndarray = None,
    highlight_gt_coordinate: np.ndarray = None,
    highlight_alpha: float = 1.0,
    voxel_size: float = 0.1,
    show_bb: bool = False,
    alpha: float = 0.5,
):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1) / 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = q > 0.75
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate(
            [q, np.zeros_like(q), np.zeros_like(q), np.clip(q, 0, 1)], axis=-1
        )
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]

    if highlight_gt_coordinate is not None:
        x, y, z = highlight_gt_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 0.0, 1.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0)
    )
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform
    )
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    if show_bb:
        assert d == h == w
        _create_bounding_box(scene, voxel_size, d)
    return scene


def visualise_voxel(
    voxel_grid: np.ndarray,
    q_attention: np.ndarray = None,
    highlight_coordinate: np.ndarray = None,
    highlight_gt_coordinate: np.ndarray = None,
    highlight_alpha: float = 1.0,
    rotation_amount: float = 0.0,
    show: bool = False,
    voxel_size: float = 0.1,
    offscreen_renderer: pyrender.OffscreenRenderer = None,
    show_bb: bool = False,
    alpha: float = 0.5,
    render_gripper=False,
    gripper_pose=None,
    gripper_mesh_scale=1.0,
):
    scene = create_voxel_scene(
        voxel_grid,
        q_attention,
        highlight_coordinate,
        highlight_gt_coordinate,
        highlight_alpha,
        voxel_size,
        show_bb,
        alpha,
    )
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=1920, viewport_height=1080, point_size=1.0
        )
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8], bg_color=[1.0, 1.0, 1.0]
        )
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height
        )
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)

        if render_gripper:
            gripper_trimesh = trimesh.load("peract_colab/meshes/hand.dae", force="mesh")
            gripper_trimesh.vertices *= gripper_mesh_scale
            radii = np.linalg.norm(
                gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1
            )
            gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(
                radii * gripper_mesh_scale, color_map="winter"
            )
            gripper_mesh = pyrender.Mesh.from_trimesh(
                gripper_trimesh, poses=np.array([gripper_pose]), smooth=False
            )
            s.add(gripper_mesh)
        color, depth = r.render(s)
        return color.copy()


def get_gripper_render_pose(
    voxel_scale, scene_bound_origin, continuous_trans, continuous_quat
):
    # finger tip to gripper offset
    offset = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1 * voxel_scale], [0, 0, 0, 1]]
    )

    # scale and translate by origin
    translation = (continuous_trans - (np.array(scene_bound_origin[:3]))) * voxel_scale
    mat = np.eye(4, 4)
    mat[:3, :3] = Rotation.from_quat(
        [continuous_quat[0], continuous_quat[1], continuous_quat[2], continuous_quat[3]]
    ).as_matrix()
    offset_mat = np.matmul(mat, offset)
    mat[:3, 3] = translation - offset_mat[:3, 3]
    return mat
