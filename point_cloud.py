import pybullet_data
import numpy as np
from numba import njit, prange
import pybullet as p
import matplotlib.pyplot as plt


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)
    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3, :3], xyz_pts.T)  # apply rotation
    # apply translation
    xyz_pts = xyz_pts + np.tile(
        rigid_transform[:3, 3].reshape(3, 1), (1, xyz_pts.shape[1])
    )
    return xyz_pts.T


def filter_pts_bounds(xyz, bounds):
    mask = xyz[:, 0] >= bounds[0, 0]
    mask = np.logical_and(mask, xyz[:, 0] <= bounds[1, 0])
    mask = np.logical_and(mask, xyz[:, 1] >= bounds[0, 1])
    mask = np.logical_and(mask, xyz[:, 1] <= bounds[1, 1])
    mask = np.logical_and(mask, xyz[:, 2] >= bounds[0, 2])
    mask = np.logical_and(mask, xyz[:, 2] <= bounds[1, 2])
    return mask


def get_pointcloud(depth_img, color_img, cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.

    Args:
        depth_img: HxW float array of depth values in meters aligned with color_img
        color_img: HxWx3 uint8 array of color image
        cam_intr: 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix

    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
        color_pts: Nx3 uint8 array of color points
    """

    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x, pixel_y = np.meshgrid(
        np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h)
    )
    cam_pts_x = np.multiply(pixel_x - cam_intr[0, 2], depth_img / cam_intr[0, 0])
    cam_pts_y = np.multiply(pixel_y - cam_intr[1, 2], depth_img / cam_intr[1, 1])
    cam_pts_z = depth_img
    cam_pts = (
        np.array([cam_pts_x, cam_pts_y, cam_pts_z]).transpose(1, 2, 0).reshape(-1, 3)
    )

    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)
    color_pts = None if color_img is None else color_img.reshape(-1, 3)
    # TODO check memory leak here
    return cam_pts, color_pts


def project_pts_to_2d(pts, camera_view_matrix, camera_intrisic):
    """Project points to 2D.
    Args:
        pts: Nx3 float array of 3D points in world coordinates.
        camera_view_matrix: 4x4 float array. A wrd2cam transformation defining camera's totation and translation.
        camera_intrisic: 3x3 float array. [ [f,0,0],[0,f,0],[0,0,1] ]. f is focal length.
    Returns:
        coord_2d: Nx3 float array of 2D pixel. (w, h, d) the last one is depth
    """
    pts_c = transform_pointcloud(pts, camera_view_matrix[0:3, :])
    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    pts_c = transform_pointcloud(pts_c, rot_algix)  # Nx3
    coord_2d = np.dot(camera_intrisic, pts_c.T)  # 3xN
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[:, 2]
    coord_2d = np.array([coord_2d[1], coord_2d[0], coord_2d[2]])
    return coord_2d.T


def check_pts_in_frustum(xyz_pts, depth, cam_pose, cam_intr):
    # xyz_pts (N,3)
    cam_pts = transform_pointcloud(
        xyz_pts=xyz_pts, rigid_transform=np.linalg.inv(cam_pose)
    )
    cam_pts_x = cam_pts[..., 0]
    cam_pts_y = cam_pts[..., 1]
    pix_z = cam_pts[..., 2]

    pix_x = (cam_intr[0, 0] / pix_z) * cam_pts_x + cam_intr[0, 2]
    pix_y = (cam_intr[1, 1] / pix_z) * cam_pts_y + cam_intr[1, 2]

    # camera to pixel space
    h, w = depth.shape

    valid_pix = np.logical_and(
        pix_x >= 0,
        np.logical_and(
            pix_x < w, np.logical_and(pix_y >= 0, np.logical_and(pix_y < h, pix_z > 0))
        ),
    )
    in_frustum_mask = valid_pix.reshape(-1)
    return in_frustum_mask


def meshwrite(filename, verts, colors, faces=None):
    """Save 3D mesh to a polygon .ply file.
    Args:
        filename: string; path to mesh file. (suffix should be .ply)
        verts: [N, 3]. Coordinates of each vertex
        colors: [N, 3]. RGB or each vertex. (type: uint8)
        faces: (optional) [M, 4]
    """
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    if faces is not None:
        ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write(
                "4 %d %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2], faces[i, 3])
            )

    ply_file.close()


@njit(parallel=True)
def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates."""
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
        pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
        pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix


def compute_empty_mask(
    scene_bounds, depth_img, intrinsic_matrix, extrinsic_matrix, voxel_resolution=20
):
    # parts taken from
    # https://github.com/andyzeng/tsdf-fusion-python/blob/3f22a940d90f684145b1f29b1feaa92e09eb1db6/fusion.py#L170

    # start off all empty
    grid_shape = [voxel_resolution] * 3
    mask = np.ones(grid_shape).astype(int)
    # get volume points
    lc = scene_bounds[0]
    uc = scene_bounds[1]

    # get voxel indices
    grid_idxs = np.stack(
        np.meshgrid(*[np.arange(0, dim) for dim in grid_shape]), axis=-1
    )

    # voxel indices to world pts
    idx_scale = np.array(grid_shape) - 1
    scales = (uc - lc) / idx_scale
    offsets = lc
    grid_points = grid_idxs.astype(float) * scales + offsets

    flattened_grid_points = grid_points.reshape(-1, 3)
    print(flattened_grid_points.min(axis=0), flattened_grid_points.max(axis=0))

    # world pts to camera centric frame pts
    xyz_h = np.hstack(
        [
            flattened_grid_points,
            np.ones((len(flattened_grid_points), 1), dtype=np.float32),
        ]
    )
    xyz_t_h = np.dot(np.linalg.inv(extrinsic_matrix), xyz_h.T).T
    cam_pts = xyz_t_h[:, :3]
    pix_z = cam_pts[:, 2]
    pix = cam2pix(cam_pts, intrinsic_matrix)
    pix_x, pix_y = pix[:, 0], pix[:, 1]
    im_w, im_h = depth_img.shape

    valid_pix = np.logical_and(
        pix_x >= 0,
        np.logical_and(
            pix_x < im_w,
            np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
        ),
    )
    inframe_indices = grid_idxs.reshape(-1, 3)[valid_pix, :]

    # depth_val = np.zeros(pix_x.shape)
    # depth_val[valid_pix] = depth_img[pix_y[valid_pix], pix_x[valid_pix]]
    observed_indices = inframe_indices[
        (depth_img[pix_y[valid_pix], pix_x[valid_pix]] > pix_z[valid_pix])
    ]

    print("before:", mask.mean(), mask.shape, observed_indices.shape)
    for idx in observed_indices:
        mask[tuple(idx)] = 0
    print(mask.mean())
    print(observed_indices.shape, mask.shape)
    # mask[observed_indices] = 0
    print("after:", mask.mean())

    ax = plt.figure().add_subplot(projection="3d")
    ax.voxels(mask)
    # pts = grid_points[mask, :]
    # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    plt.show()
    return mask.astype(bool)


def subsample(seg_pts, num_pts, random_state, balanced=True):
    probabilities = np.ones(seg_pts.shape).astype(np.float64)
    if balanced:
        unique_semantic_ids = np.unique(seg_pts)
        num_semantic_ids = len(unique_semantic_ids)
        for semantic_id in unique_semantic_ids:
            mask = seg_pts == semantic_id
            probabilities[mask] = 1.0 / (int((mask).sum().item()) * num_semantic_ids)
    else:
        probabilities /= probabilities.sum()
    indices = random_state.choice(
        seg_pts.shape[0], size=num_pts, replace=False, p=probabilities
    )
    return indices


if __name__ == "__main__":
    # TODO change this to filter input sampled points out based on
    # view point
    from datagen.simulation.asset import make_object, occluder_objects, partnet_objs
    from datagen.simulation import Camera

    object_keys = [k for k in occluder_objects]
    object_def = occluder_objects[object_keys[10]]
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=4.0,
        cameraYaw=270,
        cameraPitch=-20,
        cameraTargetPosition=(0, 0, 0.4),
    )
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(False)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    planeUid = p.loadURDF(fileName="plane.urdf", useFixedBase=True)
    occluder_obj = make_object(**object_def)
    camera = Camera(position=[-1, 1, 1], lookat=[0, 0, 0.5])
    view = camera.get_image(return_pose=True, segmentation_mask=True)
    mask = compute_empty_mask(
        scene_bounds=np.array([[-1.0, -1.0, -0.1], [1.0, 1.0, 1.9]]),
        depth_img=view[1],
        intrinsic_matrix=view[-2],
        extrinsic_matrix=view[-1],
    )
