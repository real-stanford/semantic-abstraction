from typing import List, Tuple
import torch
from torch.nn import (
    Sequential,
    LeakyReLU,
    Linear,
    Module,
    Dropout,
    ParameterDict,
)
from torch.nn.parameter import Parameter
from torch.nn.functional import grid_sample
from torch_scatter import scatter
import numpy as np
from unet3d import ResidualUNet3D
from CLIP.clip import ClipWrapper
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@typechecked
class VirtualGrid:
    def __init__(
        self,
        scene_bounds,
        grid_shape: Tuple[int, int, int] = (32, 32, 32),
        batch_size: int = 8,
        device: torch.device = torch.device("cpu"),
        int_dtype: torch.dtype = torch.int64,
        float_dtype: torch.dtype = torch.float32,
        reduce_method: str = "mean",
    ):
        self.lower_corner = tuple(scene_bounds[0])
        self.upper_corner = tuple(scene_bounds[1])
        self.grid_shape = tuple(grid_shape)
        self.batch_size = int(batch_size)
        self.device = device
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self.reduce_method = reduce_method

    @property
    def num_grids(self):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        return int(np.prod((batch_size,) + grid_shape))

    def get_grid_idxs(self, include_batch=True):
        batch_size = self.batch_size
        grid_shape = self.grid_shape
        device = self.device
        int_dtype = self.int_dtype
        dims = grid_shape
        if include_batch:
            dims = (batch_size,) + grid_shape
        axis_coords = [torch.arange(0, x, device=device, dtype=int_dtype) for x in dims]
        coords_per_axis = torch.meshgrid(*axis_coords, indexing="ij")
        grid_idxs = torch.stack(coords_per_axis, dim=-1)
        return grid_idxs

    def get_grid_points(self, include_batch=True):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        device = self.device
        grid_idxs = self.get_grid_idxs(include_batch=include_batch)

        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, dtype=float_dtype, device=device) - 1
        scales = (uc - lc) / idx_scale
        offsets = lc

        grid_idxs_no_batch = grid_idxs
        if include_batch:
            grid_idxs_no_batch = grid_idxs[:, :, :, :, 1:]
        grid_idxs_f = grid_idxs_no_batch.to(float_dtype)
        grid_points = grid_idxs_f * scales + offsets
        return grid_points

    def get_points_grid_idxs(self, points, cast_to_int=True, batch_idx=None):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        int_dtype = self.int_dtype
        float_dtype = self.float_dtype
        device = self.device
        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, dtype=float_dtype, device=device) - 1
        offsets = -lc
        scales = idx_scale / (uc - lc)
        points_idxs_i = (points + offsets) * scales
        if cast_to_int:
            points_idxs_i = points_idxs_i.to(dtype=int_dtype)
        points_idxs = torch.empty_like(points_idxs_i)
        for i in range(3):
            points_idxs[..., i] = torch.clamp(
                points_idxs_i[..., i], min=0, max=grid_shape[i] - 1
            )
        final_points_idxs = points_idxs
        if batch_idx is not None:
            final_points_idxs = torch.cat(
                [
                    batch_idx.view(*points.shape[:-1], 1).to(dtype=points_idxs.dtype),
                    points_idxs,
                ],
                dim=-1,
            )
        return final_points_idxs

    def flatten_idxs(self, idxs, keepdim=False):
        grid_shape = self.grid_shape
        batch_size = self.batch_size

        coord_size = idxs.shape[-1]
        target_shape = None
        if coord_size == 4:
            # with batch
            target_shape = (batch_size,) + grid_shape
        elif coord_size == 3:
            # without batch
            target_shape = grid_shape
        else:
            raise RuntimeError("Invalid shape {}".format(str(idxs.shape)))
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)
        flat_idxs = (
            idxs * torch.tensor(target_stride, dtype=idxs.dtype, device=idxs.device)
        ).sum(dim=-1, keepdim=keepdim, dtype=idxs.dtype)
        return flat_idxs

    def unflatten_idxs(self, flat_idxs, include_batch=True):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        target_shape = grid_shape
        if include_batch:
            target_shape = (batch_size,) + grid_shape
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)

        source_shape = tuple(flat_idxs.shape)
        if source_shape[-1] == 1:
            source_shape = source_shape[:-1]
            flat_idxs = flat_idxs[..., 0]
        source_shape += (4,) if include_batch else (3,)

        idxs = torch.empty(
            size=source_shape, dtype=flat_idxs.dtype, device=flat_idxs.device
        )
        mod = flat_idxs
        for i in range(source_shape[-1]):
            idxs[..., i] = mod / target_stride[i]
            mod = mod % target_stride[i]
        return idxs

    def idxs_to_points(self, idxs):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        int_dtype = idxs.dtype
        device = idxs.device

        source_shape = idxs.shape
        point_idxs = None
        if source_shape[-1] == 4:
            # has batch idx
            point_idxs = idxs[..., 1:]
        elif source_shape[-1] == 3:
            point_idxs = idxs
        else:
            raise RuntimeError("Invalid shape {}".format(tuple(source_shape)))

        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, dtype=float_dtype, device=device) - 1
        offsets = lc
        scales = (uc - lc) / idx_scale

        idxs_points = point_idxs * scales + offsets
        return idxs_points

    def scatter_points(self, xyz_pts, feature_pts, reduce_method=None, **kwargs):
        if reduce_method is None:
            reduce_method = self.reduce_method
        batch_size = feature_pts.shape[0]
        idxs = self.get_points_grid_idxs(xyz_pts)
        # idxs.shape = [B, num_pts, 3]
        flat_idxs = self.flatten_idxs(idxs, keepdim=False)
        # flat_idxs.shape = [B, num_pts]
        vol_features = scatter(
            src=feature_pts,
            index=flat_idxs,
            dim=-2,
            dim_size=np.prod(self.grid_shape),
            reduce=self.reduce_method,
            **kwargs
        ).view(batch_size, *self.grid_shape, -1)
        return vol_features.permute(0, 4, 1, 2, 3).contiguous()


class ImplicitVolumetricDecoder(Module):
    def __init__(self, hidden_size: int, output_dim: int, concat_xyz_pts: bool = False):
        super().__init__()
        self.concat_xyz_pts = concat_xyz_pts
        self.mlp = Sequential(
            Linear(hidden_size + int(self.concat_xyz_pts) * 3, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        features_vol: TensorType["batch", "channel", "width", "height", "length"],
        virtual_grid: VirtualGrid,
        query_points: TensorType["batch", "num_points", 3],
    ) -> TensorType["batch", "num_points", "channel"]:
        query_points = virtual_grid.get_points_grid_idxs(
            query_points, cast_to_int=False
        ).float()
        for i in range(len(virtual_grid.grid_shape)):
            query_points[..., i] = query_points[..., i] / virtual_grid.grid_shape[i]
        # query_points now between 0 and 1
        # normalize query points to (-1, 1), which is
        # required by grid_sample
        query_points_normalized = 2.0 * query_points - 1.0
        query_points = query_points_normalized.view(
            *(query_points_normalized.shape[:2] + (1, 1, 3))
        )
        sampled_features = grid_sample(
            input=features_vol,
            grid=query_points,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled_features = (
            sampled_features.view(sampled_features.shape[:3])
            .permute(0, 2, 1)
            .contiguous()
        )
        B, L, C = sampled_features.shape
        # return sampled_features
        sampled_features = sampled_features.view(B * L, C).contiguous()
        if self.concat_xyz_pts:
            sampled_features = torch.cat(
                (sampled_features, query_points.view(B * L, 3)), dim=-1
            )

        out_features = (
            self.mlp(sampled_features).view(B, L, self.output_dim).contiguous()
        )
        return out_features


class PointingAttention(Module):
    def __init__(self, pointing_dim, method="dot_product", pointing_temperature=0.07):
        super().__init__()
        self.method = method
        self.pointing_dim = pointing_dim
        if method == "dot_product":
            self.forward = self.dot_product
        elif method == "cosine_sim":
            self.cosine_sim_temp = pointing_temperature
            self.forward = self.cosine_sim
        elif method == "additive":
            self.pointer_v = Linear(pointing_dim, 1, bias=False)
            self.forward = self.additive
        else:
            raise Exception()

    @staticmethod
    def prep_input(key, query):
        """
        key.shape = BxKx[ABC]xD
        query.shape = BxQx[XYZ]xD
        output attention should be: Bx[ABC]x[XYZ]xD
        """
        if key.shape == query.shape:
            return key, query
        for _ in range(len(key.shape) - 3):
            query = query.unsqueeze(2)
        # Now, query.shape = BxQx[1,1,1]x[XYZ]xD
        for _ in range(len(query.shape) - len(key.shape)):
            key = key.unsqueeze(-2)
        # Now, key.shape = BxKx[ABC]x[1,1,1]xD
        key = key.unsqueeze(dim=2)
        query = query.unsqueeze(dim=1)
        return key, query

    def dot_product(self, key, query):
        key, query = self.prep_input(key, query)
        dotprod = (query * key).sum(dim=-1)
        pointing_attn = dotprod / np.sqrt(self.pointing_dim)
        return pointing_attn

    def cosine_sim(self, key, query):
        """
        key.shape = BxDxKx...
        query.shape = BxDxQx...
        """
        key, query = self.prep_input(key, query)
        pointing_attn = (
            torch.cosine_similarity(key, query, dim=-1) / self.cosine_sim_temp
        )
        return pointing_attn

    def additive(self, key, query):
        key, query = self.prep_input(key, query)
        additive_kq = query + key
        additive_kq = torch.tanh(additive_kq)
        pointing_attn = self.pointer_v(additive_kq).squeeze(dim=-1)
        return pointing_attn


class SemAbs3D(Module):
    def __init__(
        self,
        voxel_shape: Tuple[int, int, int],
        scene_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        unet_num_channels: int,
        unet_f_maps: int,
        unet_num_groups: int,
        unet_num_levels: int,
        network_inputs: List[str],
        use_pts_feat_extractor: bool,
        pts_feat_extractor_hidden_dim: int,
        reduce_method: str,
        output_dim=1,
        device: str = "cuda",
        decoder_concat_xyz_pts: bool = False,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.vg = VirtualGrid(
            scene_bounds=np.array(scene_bounds),
            batch_size=kwargs["batch_size"],
            grid_shape=voxel_shape,
            device=torch.device(device),
        )
        self.register_buffer("steps", torch.zeros(1))
        self.network_inputs = network_inputs
        self.use_pts_feat_extractor = use_pts_feat_extractor
        self.reduce_method = reduce_method
        self.pts_feature_dim = (
            ("saliency" in self.network_inputs)
            + ("rgb" in self.network_inputs) * 3
            + ("patch_masks" in self.network_inputs)
        )
        vol_feature_extractor_input_channels = self.pts_feature_dim + (
            "tsdf" in self.network_inputs
        )
        if self.use_pts_feat_extractor:
            self.pts_feat_extractor = Sequential(
                Linear(self.pts_feature_dim + 3, pts_feat_extractor_hidden_dim),
                LeakyReLU(),
                Linear(pts_feat_extractor_hidden_dim, pts_feat_extractor_hidden_dim),
                LeakyReLU(),
                Linear(
                    pts_feat_extractor_hidden_dim,
                    unet_num_channels - int("tsdf" in self.network_inputs),
                ),
            )
            vol_feature_extractor_input_channels = unet_num_channels
            assert self.reduce_method == "max"
        self.vol_feature_extractor = ResidualUNet3D(
            in_channels=vol_feature_extractor_input_channels,
            out_channels=unet_num_channels,
            f_maps=unet_f_maps,
            num_groups=unet_num_groups,
            num_levels=unet_num_levels,
        )
        self.visual_sampler = ImplicitVolumetricDecoder(
            hidden_size=unet_num_channels,
            output_dim=output_dim,
            concat_xyz_pts=decoder_concat_xyz_pts,
        )

    def forward(
        self, input_xyz_pts, input_feature_pts, tsdf_vol, output_xyz_pts, **kwargs
    ):
        batch_size, num_patches, input_num_pts = input_feature_pts.shape[:3]
        input_xyz_pts = (
            input_xyz_pts.unsqueeze(dim=1)
            .repeat(1, num_patches, 1, 1)
            .view(batch_size * num_patches, input_num_pts, 3)
        )
        input_feature_pts = input_feature_pts.view(
            batch_size * num_patches, input_num_pts, self.pts_feature_dim
        )
        if self.use_pts_feat_extractor:
            input_feature_pts = self.pts_feat_extractor(
                torch.cat(
                    (
                        input_xyz_pts,
                        input_feature_pts,
                    ),
                    dim=-1,
                )
            )
        visual_volumetric_features = self.vg.scatter_points(
            xyz_pts=input_xyz_pts,
            feature_pts=input_feature_pts,
            reduce_method=self.reduce_method,
        )
        batch_size, num_patches, num_output_pts = output_xyz_pts.shape[:3]
        if visual_volumetric_features.shape[0] < batch_size * num_patches:
            visual_volumetric_features = (
                visual_volumetric_features[:, None, ...]
                .repeat(1, num_patches, 1, 1, 1, 1)
                .view(batch_size * num_patches, *visual_volumetric_features.shape[1:])
            )
        if "tsdf" in self.network_inputs:
            visual_volumetric_features = torch.cat(
                (
                    tsdf_vol.unsqueeze(dim=1).repeat(num_patches, 1, 1, 1, 1),
                    visual_volumetric_features,
                ),
                dim=1,
            )
        self.visual_volumetric_features = self.vol_feature_extractor(
            visual_volumetric_features
        )
        output_xyz_pts = output_xyz_pts.view(
            batch_size * num_patches, num_output_pts, 3
        )
        return (
            self.visual_sampler(
                features_vol=self.visual_volumetric_features,
                virtual_grid=self.vg,
                query_points=output_xyz_pts,
            )
            .view(batch_size, num_patches, num_output_pts, -1)
            .squeeze(dim=-1)
        )


class SemanticAwareOVSSC(SemAbs3D):
    def __init__(self, pointing_method: str, clip_hidden_dim: int = 512, **kwargs):
        super().__init__(output_dim=clip_hidden_dim, **kwargs)
        self.semantic_class_pointer = PointingAttention(
            pointing_dim=clip_hidden_dim, method=pointing_method
        )

    def forward(self, semantic_class_features, **kwargs):
        sampled_features = super().forward(**kwargs)
        assert sampled_features.shape[1] == semantic_class_features.shape[1]
        num_patches = semantic_class_features.shape[1]
        return (
            torch.stack(
                [
                    self.semantic_class_pointer(
                        key=semantic_class_features[:, patch_i, ...][:, None, ...],
                        query=sampled_features[:, patch_i, ...][:, None, ...],
                    )
                    for patch_i in range(num_patches)
                ],
                dim=1,
            )
            .squeeze(dim=2)
            .squeeze(dim=2)
        )


class SemAbsVOOL(Module):
    def __init__(
        self,
        pointing_method: str,
        pointing_dim: int,
        device: str,
        decoder_concat_xyz_pts: bool,
        **kwargs
    ):
        super().__init__()
        self.register_buffer("steps", torch.zeros(1))
        self.device = device
        self.completion_net = SemAbs3D(device=device, **kwargs).to(device)
        self.spatial_sampler = ImplicitVolumetricDecoder(
            hidden_size=2 * kwargs["unet_num_channels"],
            output_dim=pointing_dim,
            concat_xyz_pts=decoder_concat_xyz_pts,
        )
        self.pointer = PointingAttention(
            method=pointing_method, pointing_dim=pointing_dim
        )
        self.relation_embeddings = ParameterDict(
            {
                k: Parameter(torch.randn(pointing_dim))
                for k in [
                    "in",
                    "behind",
                    "in front of",
                    "on the left of",
                    "on the right of",
                    "on",
                    "[pad]",
                ]
            }
        )

    def get_region_pointing_features(self, spatial_relation_name, **kwargs):
        # spatial_relation_name.shape NUMDESCxBATCHxWORD
        region_pointing_features = (
            torch.stack(
                [
                    torch.stack(
                        [
                            self.relation_embeddings[
                                spatial_relation_name[desc_i][batch_i]
                            ]
                            for batch_i in range(len(spatial_relation_name[desc_i]))
                        ],
                        dim=0,
                    )
                    for desc_i in range(len(spatial_relation_name))
                ],
                dim=0,
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        return region_pointing_features

    def get_feature_vol(
        self,
        input_xyz_pts,
        input_target_saliency_pts,
        input_reference_saliency_pts,
        tsdf_vol,
        num_descs,
        **kwargs
    ):
        place_holder_output_xyz_pts = torch.zeros_like(input_xyz_pts)[
            ..., None, 0:1, :
        ].repeat(1, num_descs, 1, 1)
        self.completion_net(
            input_xyz_pts=input_xyz_pts,
            input_feature_pts=input_target_saliency_pts,
            tsdf_vol=tsdf_vol,
            # placeholder
            output_xyz_pts=place_holder_output_xyz_pts,
        )
        target_feature_vol = self.completion_net.visual_volumetric_features
        self.completion_net(
            input_xyz_pts=input_xyz_pts,
            input_feature_pts=input_reference_saliency_pts,
            tsdf_vol=tsdf_vol,
            # placeholder
            output_xyz_pts=place_holder_output_xyz_pts,
        )
        reference_feature_vol = self.completion_net.visual_volumetric_features
        feature_vol = torch.cat((target_feature_vol, reference_feature_vol), dim=1)
        return feature_vol

    def forward(self, output_xyz_pts, spatial_relation_name, **kwargs):
        batch_size, num_descs = np.array(spatial_relation_name).T.shape
        feature_vol = self.get_feature_vol(num_descs=num_descs, **kwargs)
        num_output_pts = output_xyz_pts.shape[-2]
        sampled_locator_feature_pts = self.spatial_sampler(
            features_vol=feature_vol,
            virtual_grid=self.completion_net.vg,
            query_points=output_xyz_pts.view(batch_size * num_descs, num_output_pts, 3),
        )

        # region_pointing_features.shape BATCH x NUMDESC x WORD
        region_pointing_features = self.get_region_pointing_features(
            spatial_relation_name=spatial_relation_name
        )

        return self.pointer(
            key=sampled_locator_feature_pts,
            query=region_pointing_features.contiguous().view(
                batch_size * num_descs, 1, -1
            ),
        ).view(batch_size, num_descs, num_output_pts)


class SemanticAwareVOOL(SemAbsVOOL):
    def __init__(self, pointing_dim: int, clip_hidden_dim=512, **kwargs):
        super().__init__(output_dim=pointing_dim, pointing_dim=pointing_dim, **kwargs)
        self.mlp = Linear(clip_hidden_dim * 2 + pointing_dim, pointing_dim)

    def get_region_pointing_features(
        self, target_obj_name, reference_obj_name, **kwargs
    ):
        with torch.no_grad():
            target_obj_name = np.array(target_obj_name).T
            reference_obj_name = np.array(reference_obj_name).T
            batch_size, num_descs = target_obj_name.shape
            target_obj_feature_names = torch.from_numpy(
                ClipWrapper.get_clip_text_feature(target_obj_name.reshape(-1))
            ).to(self.device)
            target_obj_feature_names = target_obj_feature_names.view(
                batch_size, num_descs, -1
            )
            reference_obj_feature_names = torch.from_numpy(
                ClipWrapper.get_clip_text_feature(reference_obj_name.reshape(-1))
            ).to(self.device)
            reference_obj_feature_names = reference_obj_feature_names.view(
                batch_size, num_descs, -1
            )

        region_pointing_features = super().get_region_pointing_features(**kwargs)
        return self.mlp(
            torch.cat(
                (
                    target_obj_feature_names,
                    reference_obj_feature_names,
                    region_pointing_features,
                ),
                dim=-1,
            )
        )

    def forward(self, input_rgb_pts, spatial_relation_name, **kwargs):
        # prepare inputs
        batch_size, num_desc, _, _ = input_rgb_pts.shape
        num_output_pts = kwargs["output_xyz_pts"].shape[-2]
        sampled_locator_feature_pts = self.completion_net(
            input_feature_pts=input_rgb_pts, **kwargs
        )
        region_pointing_features = self.get_region_pointing_features(
            spatial_relation_name=spatial_relation_name, **kwargs
        )
        return self.pointer(
            key=sampled_locator_feature_pts.view(
                batch_size * num_desc, num_output_pts, -1
            ),
            query=region_pointing_features.contiguous().view(
                batch_size * num_desc, 1, -1
            ),
        ).view(batch_size, num_desc, -1)


class ClipSpatialVOOL(Module):
    def __init__(self, device: str, decoder_concat_xyz_pts: bool, **kwargs):
        super().__init__()
        self.register_buffer("steps", torch.zeros(1))
        self.device = device
        self.completion_net = SemAbs3D(device=device, **kwargs).to(device)
        self.spatial_sampler = ImplicitVolumetricDecoder(
            hidden_size=kwargs["unet_num_channels"],
            output_dim=1,
            concat_xyz_pts=decoder_concat_xyz_pts,
        )

    def get_feature_vol(
        self,
        input_xyz_pts,
        input_description_saliency_pts,
        tsdf_vol,
        num_descs,
        **kwargs
    ):
        self.completion_net(
            input_xyz_pts=input_xyz_pts,
            input_feature_pts=input_description_saliency_pts,
            tsdf_vol=tsdf_vol,
            # placeholder
            output_xyz_pts=torch.zeros_like(input_xyz_pts)[..., None, 0:1, :].repeat(
                1, num_descs, 1, 1
            ),
        )
        return self.completion_net.visual_volumetric_features

    def forward(self, output_xyz_pts, spatial_relation_name, **kwargs):
        batch_size, num_descs = np.array(spatial_relation_name).T.shape
        feature_vol = self.get_feature_vol(num_descs=num_descs, **kwargs)
        num_output_pts = output_xyz_pts.shape[-2]
        return self.spatial_sampler(
            features_vol=feature_vol,
            virtual_grid=self.completion_net.vg,
            query_points=output_xyz_pts.view(batch_size * num_descs, num_output_pts, 3),
        ).view(batch_size, num_descs, num_output_pts)
