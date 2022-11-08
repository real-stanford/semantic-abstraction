# Copyright (c) 2018 Andy Zeng
# Source: https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py
# BSD 2-Clause License

# Copyright (c) 2019, Princeton University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numba import njit, prange
from skimage import measure


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, vol_bnds, voxel_size):
        """Constructor.
        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        assert (vol_bnds[:, 0] < vol_bnds[:, 1]).all()

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        # Assume all unobserved regions are occupied
        self._tsdf_vol_cpu = -np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = (
            np.concatenate(
                [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
            )
            .astype(int)
            .T
        )

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
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

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume."""
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.0):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
        """
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2] * self._color_const
            + color_im[..., 1] * 256
            + color_im[..., 0]
        )

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
        cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = self.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        # Eliminate pixels outside view frustum
        valid_pix = np.logical_and(
            pix_x >= 0,
            np.logical_and(
                pix_x < im_w,
                np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
            ),
        )
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # Integrate TSDF
        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
        dist = np.maximum(-1, np.minimum(1, depth_diff / self._trunc_margin))
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        valid_dist = dist[valid_pts]
        tsdf_vol_new, w_new = self.integrate_tsdf(
            tsdf_vals, valid_dist, w_old, obs_weight
        )
        self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
        self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

        # Integrate color
        old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        old_b = np.floor(old_color / self._color_const)
        old_g = np.floor((old_color - old_b * self._color_const) / 256)
        old_r = old_color - old_b * self._color_const - old_g * 256
        new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
        new_b = np.floor(new_color / self._color_const)
        new_g = np.floor((new_color - new_b * self._color_const) / 256)
        new_r = new_color - new_b * self._color_const - new_g * 256
        new_b = np.minimum(
            255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new)
        )
        new_g = np.minimum(
            255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new)
        )
        new_r = np.minimum(
            255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new)
        )
        self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = (
            new_b * self._color_const + new_g * 256 + new_r
        )

    def get_volume(self):
        # Fold RGB color image into a single channel image
        color_vol = np.zeros([3] + list(self._color_vol_cpu.shape)).astype(np.uint8)
        color_vol[2, ...] = np.floor(self._color_vol_cpu / self._color_const)
        color_vol[1, ...] = np.floor(
            (self._color_vol_cpu - color_vol[2, ...] * self._color_const) / 256
        )
        color_vol[0, ...] = (
            self._color_vol_cpu
            - color_vol[2, ...] * self._color_const
            - color_vol[1, ...] * 256
        )
        return self._tsdf_vol_cpu, color_vol

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        # voxel grid coordinates to world coordinates
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image"""
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )
        )
