from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from functools import partial
from . import pointnet2_utils
from ..pointnet2_stack import voxel_query_utils
from ....utils import voxel_aggregation_utils, common_utils
from ....utils.spconv_utils import replace_feature, spconv
# from ....models.backbones_3d.spconv_backbone import post_act_block as block

from spconv.core_cc.csrc.sparse.all.ops3d import Point2Voxel as Point2VoxelGPU3d
from spconv.core_cc.csrc.sparse.all.ops_cpu3d import Point2VoxelCPU as Point2VoxelCPU3d


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.farthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class _PointnetSAModuleFSBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.npoint_list = []
        self.sample_range_list = [[0, -1]]
        self.sample_method_list = ['d-fps']
        self.radii = []

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0
        self.skip_connection = False

        self.aggregation_mlp = None
        self.confidence_mlp = None

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None,
                scores=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            for i in range(len(self.sample_method_list)):
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous()
                if self.sample_method_list[i] == 'd-fps':
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                elif self.sample_method_list[i] == 's-fps':
                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )
                else:
                    raise NotImplementedError

                sample_idx_list.append(sample_idx + self.sample_range_list[i][0])

            sample_idx = torch.cat(sample_idx_list, dim=-1)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)

            if self.skip_connection:
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)

        for i in range(len(self.groupers)):
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
            idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
            new_features = new_features * idx_cnt_mask

            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)

        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)

        new_features = torch.cat(new_features_list, dim=1)
        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features)

        if self.confidence_mlp is not None:
            new_scores = self.confidence_mlp(new_features)
            new_scores = new_scores.squeeze(1)  # (B, npoint)
            return new_xyz, new_features, new_scores

        return new_xyz, new_features, None


class PointnetSAModuleFSMSG(_PointnetSAModuleFSBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None):
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.radii = radii
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        former_radius = 0.0
        in_channels, out_channels = 0, 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if dilated_radius_group:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                )
            former_radius = radius
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlp = []
            for k in range(len(mlp_spec) - 1):
                shared_mlp.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlp))
            in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.skip_connection = skip_connection
        self.weight_gamma = weight_gamma

        if skip_connection:
            out_channels += in_channels

        if aggregation_mlp is not None:
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_mlp = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_mlp = None

        if confidence_mlp is not None:
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, 1, kernel_size=1, bias=True),
            )
            self.confidence_mlp = nn.Sequential(*shared_mlp)
        else:
            self.confidence_mlp = None


class PointnetSAModuleFS(PointnetSAModuleFSMSG):
    """Pointnet set abstraction layer with fusion sampling"""

    def __init__(self, *,
                 mlp: List[int],
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps, f-fps or c-fps
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__(
            mlps=[mlp], npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, radii=[radius], nsamples=[nsample],
            bn=bn, use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            skip_connection=skip_connection, weight_gamma=weight_gamma,
            aggregation_mlp=aggregation_mlp, confidence_mlp=confidence_mlp
        )


class _VoxelPointnetSAModuleFSBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.spconv_mlps = None
        self.npoint_list = []
        self.sample_range_list = [[0, -1]]
        self.sample_method_list = ['d-fps']
        self.sp_stride = None
        self.radii = []
        self.point_mlps = None

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0
        self.skip_connection = False

        self.aggregation_mlp = None
        self.confidence_mlp = None

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None,
                scores=None,
                part_scores=None,
                sp_tensor=None,
                unique_idxs=None,
                switch=False,
                centroids=None,
                centroid_voxel_idxs=None,):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        batch_size = len(xyz)
        ori_scores = None

        xyz_flipped = xyz.transpose(1, 2).contiguous()

        if scores is not None:
            # scores[:, 1] = scores[:, 1] * 5.0
            ori_scores = torch.max(scores.sigmoid(), dim=1, keepdim=True)[0]

            scores, idx_cls_pred = torch.max(scores, dim=1, keepdim=True)
            # mask_pc = idx_cls_pred > 0
            # scores[mask_pc] = 100
            # ori_scores = scores.sigmoid()
            if unique_idxs is not None:
                scores = scores[unique_idxs]
                scores = scores.view(batch_size, -1)
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            for i in range(len(self.sample_method_list)):
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous()
                if self.sample_method_list[i] == 'd-fps':

                    # sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                    if self.sa_layer_idx == 0:
                        sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                    else:
                        sample_idx = torch.range(0, self.npoint_list[i]-1, device=xyz_slice.device, dtype=torch.int32)
                        sample_idx = sample_idx.unsqueeze(0).repeat(len(xyz_slice), 1)

                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                elif self.sample_method_list[i] == 's-fps':
                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )
                elif self.sample_method_list[i] == 's-topk':
                    assert scores is not None
                    scores, sample_idx = torch.topk(scores, k=self.npoint_list[i], dim=-1)
                    sample_idx = sample_idx.int()
                elif self.sample_method_list[i] == 'd-fps-faraware':
                    pts_depth = torch.norm(xyz_slice, p=2, dim=-1)
                    sorted_depth, sorted_idx = torch.sort(pts_depth, dim=-1)
                    pts_near_idx = sorted_idx[:, :-256]
                    pts_far_idx = sorted_idx[:, -256:]
                    pts_near = []
                    for index_near in range(batch_size):
                        pts_near.append(xyz_slice[index_near][pts_near_idx[index_near]].unsqueeze(0))
                    pts_near = torch.cat(pts_near, dim=0)
                    sample_idx_base_near = pointnet2_utils.furthest_point_sample(pts_near, self.npoint_list[i]-256)
                    sample_near_idx = []
                    for index_near_idx in range(batch_size):
                        sample_near_idx.append(pts_near_idx[index_near_idx][sample_idx_base_near[index_near_idx].long()].unsqueeze(0))
                    sample_near_idx = torch.cat(sample_near_idx, dim=0)
                    sample_idx = torch.cat([sample_near_idx, pts_far_idx], dim=-1).int()

                else:
                    raise NotImplementedError

                sample_idx_list.append(sample_idx + self.sample_range_list[i][0])

            sample_idx = torch.cat(sample_idx_list, dim=-1)

            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)

            if self.skip_connection:
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)

        if unique_idxs is not None:
            batch_size = sample_idx.shape[0]
            for i in range(batch_size):
                sample_idx[i, :] = sample_idx[i, :]+i*xyz.shape[1]
            sample_idx = sample_idx.view(-1).long()
            unique_idxs = unique_idxs[sample_idx]

        point_cloud_range = self.voxel_config.POINT_CLOUD_RANGE
        voxel_size = self.voxel_config.VOXEL_SIZE
        if sp_tensor is not None:
            v2p_ind_tensor = common_utils.generate_voxel2pinds(sp_tensor)
            batch_size, num_points, _ = new_xyz.shape
            point_grid_coords = new_xyz.clone().view(-1, 3)
            point_grid_coords[:, 0] = (point_grid_coords[:, 0] - point_cloud_range[0]) / voxel_size[0]
            point_grid_coords[:, 1] = (point_grid_coords[:, 1] - point_cloud_range[1]) / voxel_size[1]
            point_grid_coords[:, 2] = (point_grid_coords[:, 2] - point_cloud_range[2]) / voxel_size[2]
            point_grid_cnt = new_xyz.new_zeros(batch_size).int()
            point_grid_cnt = point_grid_cnt + num_points
            overlapping_indices_nonempty, overlapping_nonempty_mask = \
                voxel_aggregation_utils.get_nonempty_voxel_feature_indices(centroid_voxel_idxs, sp_tensor)
            sp_coords = sp_tensor.indices
            voxel_xyz = common_utils.get_voxel_centers(
                sp_coords[:, 1:4],
                downsample_times=self.sp_stride,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range
            )
            voxel_xyz_batch_cnt = sp_coords.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                voxel_xyz_batch_cnt[bs_idx] = (sp_coords[:, 0] == bs_idx).sum()

            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            point_grid_coords = point_grid_coords / self.sp_stride
            point_batch_idx = new_xyz.new_zeros(size=(batch_size, num_points))
            for cur_batch_idx in range(batch_size):
                point_batch_idx[cur_batch_idx] = point_batch_idx[cur_batch_idx] + cur_batch_idx
            point_batch_idx = point_batch_idx.view(-1, 1).long()
            point_grid_coords = torch.cat([point_batch_idx, point_grid_coords], dim=-1)
            point_grid_coords = point_grid_coords.int()
            point_grid_coords = point_grid_coords[:, [0, 3, 2, 1]].contiguous()
            # voxel neighbor aggregation
            voxel_xyz[overlapping_indices_nonempty] = centroids[:, 1:4][overlapping_nonempty_mask]
            features_in = sp_tensor.features.contiguous()

        for i in range(len(self.groupers)):
            if sp_tensor is None:
                idx_cnt, grouped_features, grouped_xyz = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                batch_size, npoint, _ = new_xyz.shape
                idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
                idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
                grouped_features = grouped_features * idx_cnt_mask
                new_features = self.point_mlps[i](grouped_features)
            else:
                grouped_features, grouped_xyz, empty_ball_mask, density_score = self.groupers[i](
                    new_coords=point_grid_coords,
                    xyz=voxel_xyz.contiguous(),
                    xyz_batch_cnt=voxel_xyz_batch_cnt,
                    new_xyz=new_xyz.view(-1, 3),
                    new_xyz_batch_cnt=point_grid_cnt,
                    features=features_in,
                    voxel2point_indices=v2p_ind_tensor
                )

                # (B, C, npoint, nsample)
                batch_size, npoint, _ = new_xyz.shape
                nchannel, nsample = grouped_features.shape[1:]
                grouped_new_xyz = new_xyz.view(-1, 3).unsqueeze(-1)
                grouped_features[empty_ball_mask] = 0
                grouped_xyz = grouped_xyz - grouped_new_xyz
                grouped_xyz[empty_ball_mask] = 0
                grouped_features = grouped_features.view(batch_size, npoint, nchannel, nsample)
                grouped_features = grouped_features.permute(0, 2, 1, 3)
                grouped_features = self.point_mlps[i](grouped_features)
                grouped_xyz = grouped_xyz.view(batch_size, npoint, 3, nsample)
                grouped_xyz = grouped_xyz.permute(0, 2, 1, 3)
                grouped_xyz = self.pos_mlps[i](grouped_xyz)
                new_features = self.relu(grouped_features+grouped_xyz)

            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'weight_pool':
                pos_weights = self.pos_mlps[i](new_features)
                new_features = new_features * pos_weights
                pooled_features = torch.sum(new_features, dim=-1)
            else:
                raise NotImplementedError

            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)

        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)

        new_features = torch.cat(new_features_list, dim=1)

        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features)

        if sp_tensor is None:
            batch_size, channel, num_points = new_features.shape
            voxel_idxs = voxel_aggregation_utils.get_voxel_indices(
                new_xyz.clone().view(-1, 3),
                downsample_times=self.sp_stride,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range)  # get index of all points  within the range

            # Add batch_idx
            batch_idx = new_xyz.new_zeros(size=(batch_size, num_points))
            for i in range(batch_size):
                batch_idx[i] = batch_idx[i] + i
            batch_idx = batch_idx.view(-1, 1).long()
            # voxel_idxs_valid_mask = (voxel_idxs != -1).all(-1)
            voxel_idxs = torch.cat((batch_idx, voxel_idxs), dim=-1)
            # Filter out points that are outside the valid point cloud range (invalid indices have -1)
            # voxel_idxs_valid = voxel_idxs[voxel_idxs_valid_mask]
            # Convert voxel_indices from (bxyz) to (bzyx) format for properly indexing voxelization layer
            voxel_idxs = voxel_idxs[:, [0, 3, 2, 1]]
            xyz_for_voxel = new_xyz.view(-1, 3)
            xyz_for_voxel = torch.cat([batch_idx, xyz_for_voxel], dim=-1)
            features_for_voxel = new_features.permute(0, 2, 1).contiguous().view(-1, channel)
            point_for_voxel = torch.cat([xyz_for_voxel, features_for_voxel], dim=-1) # bxyz+features
            # points_valid = point_for_voxel[voxel_idxs_valid_mask]

            centroids_coords_features, centroid_voxel_idxs, num_points_in_voxel, unique_idxs = voxel_aggregation_utils.get_centroid_per_voxel(
                point_for_voxel, voxel_idxs)
            grid_size = (np.array(point_cloud_range[3:6]) - np.array(point_cloud_range[0:3])) / (np.array(voxel_size) * self.sp_stride)
            sparse_shape = grid_size[::-1].astype(np.int64)
            centroids = centroids_coords_features[:, 0:4].contiguous()
            centroids_features = centroids_coords_features[:, 4:]
            sp_tensor = spconv.SparseConvTensor(
                features=centroids_features.contiguous(),
                indices=centroid_voxel_idxs.int(),
                spatial_shape=sparse_shape,
                batch_size=batch_size
            )
        else:
            if new_features is not None and (self.sa_layer_idx > 0) and (self.sa_layer_idx < 3):

                batch_size, last_channel, num_points = new_features.shape
                num_points_for_update = num_points
                new_features_for_update = new_features
                new_xyz_for_update = new_xyz
                new_point_idxs = voxel_aggregation_utils.get_voxel_indices(
                    new_xyz_for_update.view(-1, 3),
                    downsample_times=self.sp_stride,
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range)
                # Add batch_idx
                new_batch_idx = new_xyz_for_update.new_zeros(size=(batch_size, num_points_for_update))
                for i in range(batch_size):
                    new_batch_idx[i] = new_batch_idx[i] + i
                new_batch_idx = new_batch_idx.view(-1, 1).long()
                # new_voxel_idxs_valid_mask = (new_point_idxs != -1).all(-1)
                new_voxel_idxs = torch.cat((new_batch_idx, new_point_idxs), dim=-1)

                # Filter out points that are outside the valid point cloud range (invalid indices have -1)
                # new_voxel_idxs_valid = new_voxel_idxs[new_voxel_idxs_valid_mask]
                new_voxel_idxs = new_voxel_idxs[:, [0, 3, 2, 1]]
                new_xyz_for_voxel = new_xyz_for_update.view(-1, 3)
                new_xyz_for_voxel = torch.cat([new_batch_idx, new_xyz_for_voxel], dim=-1)
                new_features_for_voxel = new_features_for_update.permute(0, 2, 1).contiguous().view(-1, last_channel)
                point_for_voxel = torch.cat([new_xyz_for_voxel, new_features_for_voxel], dim=-1)
                # points_valid = point_for_voxel[new_voxel_idxs_valid_mask]
                new_centroids, new_centroid_voxel_idxs, new_num_points_in_voxel, _ = \
                    voxel_aggregation_utils.get_centroid_per_voxel(point_for_voxel, new_voxel_idxs)
                update_indices_nonempty, update_nonempty_mask = \
                    voxel_aggregation_utils.get_nonempty_voxel_feature_indices(new_centroid_voxel_idxs, sp_tensor)
                source_tesor_features = new_centroids.new_zeros([sp_tensor.features.shape[0], new_centroids.shape[1]-4])
                source_tesor_features[update_indices_nonempty] = new_centroids[:, 4:][update_nonempty_mask]
                source_tesor = spconv.SparseConvTensor(
                    features=source_tesor_features.contiguous(),
                    indices=centroid_voxel_idxs.int(),
                    spatial_shape=sp_tensor.spatial_shape,
                    batch_size=batch_size
                )
                # sp_tensor = source_tesor
                sp4x_tensor = self.spconv4x_mlps(source_tesor)
                sp8x_tensor = self.spconv8x_mlps(sp4x_tensor)
                sp16x_tensor = self.spconv16x_mlps(sp8x_tensor)
                spinv16x_tensor = self.spconvinv16x_mlps(sp16x_tensor)
                spinv16x_tensor = replace_feature(spinv16x_tensor, spinv16x_tensor.features + sp16x_tensor.features)
                spinv8x_tensor = self.spconvinv8x_mlps(spinv16x_tensor)
                spinv8x_tensor = replace_feature(spinv8x_tensor, spinv8x_tensor.features+sp8x_tensor.features)
                spinv4x_tensor = self.spconvinv4x_mlps(spinv8x_tensor)
                spinv4x_tensor = replace_feature(spinv4x_tensor, spinv4x_tensor.features + sp4x_tensor.features)
                dest_tensor = self.spconv_out_mlps(spinv4x_tensor)
                sp_tensor = self.spconv_mlps(sp_tensor)
                sp_tensor = replace_feature(sp_tensor, self.update_relu(sp_tensor.features + ori_scores * dest_tensor.features))
                # sp_tensor = replace_feature(sp_tensor,
                #                             self.update_relu(ori_scores * dest_tensor.features))

        if self.confidence_mlp is not None:
            features_for_confidence = sp_tensor.features
            features_for_confidence = features_for_confidence.unsqueeze(-1)
            new_scores = self.confidence_mlp(features_for_confidence)
            new_scores = new_scores.squeeze(2)  # (B, npoint)

            return new_xyz.contiguous(), new_features.contiguous(), new_scores.contiguous(), \
                   sp_tensor, centroids, centroid_voxel_idxs.contiguous(), unique_idxs, None

        return new_xyz.contiguous(), new_features.contiguous(), None, \
               sp_tensor, centroids, centroid_voxel_idxs.contiguous(), unique_idxs, None


class VoxelPointnetSAModuleFSMSG(_VoxelPointnetSAModuleFSBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 query_range: List[List[int]] = None,
                 sp_stride: int = None,
                 stride: List[List[int]] = None,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 spconv_mlps: List[int] = None,
                 spconv_mlps_post: List[int] = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None,
                 voxel_config=None,
                 sa_layer_idx=1):
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.query_range = query_range
        self.sp_stride = sp_stride
        self.stride = stride
        self.radii = radii
        self.spconv_mlps = spconv_mlps
        self.spconv_mlps_post = spconv_mlps_post
        self.groupers = nn.ModuleList()
        self.sa_layer_idx = sa_layer_idx
        self.pool_method = pool_method


        if mlps[0]:
            self.point_mlps = nn.ModuleList()
            if self.sa_layer_idx > 0:
                self.pos_mlps = nn.ModuleList()
        if self.pool_method == "weight_pool":
            self.pos_mlps = nn.ModuleList()
        self.voxel_config = voxel_config

        former_radius = 0.0
        in_channels, out_channels = 0, 0
        for i in range(len(radii)):
            radius = radii[i]
            query_ranges = query_range[i]
            strides = stride[i]
            nsample = nsamples[i]
            if dilated_radius_group:
                if sa_layer_idx == 0:
                    self.groupers.append(
                        pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz)
                    )
                else:
                    self.groupers.append(
                        voxel_query_utils.VoxelQueryAndGroupingDilated(
                            query_ranges, strides, former_radius, radius, nsample)
                    )
            else:
                if sa_layer_idx == 0:
                    self.groupers.append(
                        pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                    )
                else:
                    self.groupers.append(
                        voxel_query_utils.VoxelQueryAndGrouping(query_ranges, radius, nsample)
                    )

            former_radius = radius

            if mlps[0] and self.sa_layer_idx==0:
                mlp_spec = mlps[i]
                if use_xyz:
                    mlp_spec[0] += 3
                ori_mlp_spec_in = mlp_spec[0]
                shared_point_mlp = []
                for k in range(len(mlp_spec) - 1):
                    shared_point_mlp.extend([
                        nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(mlp_spec[k + 1]),
                        nn.ReLU()
                    ])
                self.point_mlps.append(nn.Sequential(*shared_point_mlp))
                mlp_spec[0] = ori_mlp_spec_in
                in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
                out_channels += mlp_spec[-1]
            else:
                mlp_spec = mlps[i]
                ori_mlp_spec_in = mlp_spec[0]
                shared_point_mlp = []
                for k in range(len(mlp_spec) - 2):
                    shared_point_mlp.extend([
                        nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(mlp_spec[k + 1]),
                        nn.ReLU()
                    ])
                shared_point_mlp.extend([
                    nn.Conv2d(mlp_spec[-2], mlp_spec[-1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[-1]),
                ])
                self.point_mlps.append(nn.Sequential(*shared_point_mlp))
                self.pos_mlps.append(nn.Sequential(
                    nn.Conv2d(3, mlp_spec[-1]//2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[-1]//2),
                    nn.ReLU(),
                    nn.Conv2d(mlp_spec[-1]//2, mlp_spec[-1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[-1]),
                ))
                self.relu = nn.ReLU()
                mlp_spec[0] = ori_mlp_spec_in
                in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
                out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.skip_connection = skip_connection
        self.weight_gamma = weight_gamma

        if skip_connection:
            out_channels += in_channels

        if aggregation_mlp is not None:
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_mlp = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_mlp = None

        if (self.sa_layer_idx <= 2) and (self.sa_layer_idx > 0):
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            sp_in_channels = spconv_mlps[0]
            tagspconv8x = "spconv8x%d" % sa_layer_idx
            tagspconv16x = "spconv16x%d" % sa_layer_idx
            n_EnDe = int(out_channels//2)
            n_EnDe2x = n_EnDe
            n_EnDe4x = n_EnDe * 2

            self.spconv4x_mlps = spconv.SparseSequential(
                spconv.SubMConv3d(out_channels, n_EnDe, 1, padding=0, bias=False, indice_key="subm4x"),
                norm_fn(n_EnDe),
                nn.ReLU(),
            )
            self.spconv8x_mlps = spconv.SparseSequential(
                spconv.SparseConv3d(n_EnDe, n_EnDe2x, 3, stride=2, padding=1, bias=False, indice_key=tagspconv8x),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
            )
            self.spconv16x_mlps = spconv.SparseSequential(
                spconv.SparseConv3d(n_EnDe2x, n_EnDe4x, 3, stride=2, padding=1, bias=False, indice_key=tagspconv16x),
                norm_fn(n_EnDe4x),
                nn.ReLU(),
            )
            self.spconvinv16x_mlps = spconv.SparseSequential(
                spconv.SubMConv3d(n_EnDe4x, n_EnDe4x, 3, padding=1, bias=False, indice_key="subm16x"),
                norm_fn(n_EnDe4x),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe4x, n_EnDe4x, 3, padding=1, bias=False, indice_key="subm16x"),
                norm_fn(n_EnDe4x),
                nn.ReLU(),
            )

            self.spconvinv8x_mlps = spconv.SparseSequential(
                spconv.SparseInverseConv3d(n_EnDe4x, n_EnDe2x, 3, indice_key=tagspconv16x, bias=False),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe2x, n_EnDe2x, 3, padding=1, bias=False, indice_key="subm8x"),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe2x, n_EnDe2x, 3, padding=1, bias=False, indice_key="subm8x"),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
            )
            self.spconvinv4x_mlps = spconv.SparseSequential(
                spconv.SparseInverseConv3d(n_EnDe2x, n_EnDe, 3, indice_key=tagspconv8x, bias=False),
                norm_fn(n_EnDe),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe, n_EnDe, 3, padding=1, bias=False, indice_key="subm4x"),
                norm_fn(n_EnDe),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe, n_EnDe, 3, padding=1, bias=False, indice_key="subm4x"),
                norm_fn(n_EnDe),
                nn.ReLU(),
            )
            self.spconv_out_mlps = spconv.SparseSequential(
                spconv.SubMConv3d(n_EnDe, out_channels, 1, padding=0, bias=False, indice_key="submencoder"),
                norm_fn(out_channels),
                # nn.ReLU(),
            )
            shared_update_mlp = []
            # for k in range(len(spconv_mlps)-2):
            #     shared_update_mlp.extend([
            #         spconv.SubMConv3d(spconv_mlps[k], spconv_mlps[k+1], 3, padding=0, bias=False, indice_key="subm"),
            #         norm_fn(spconv_mlps[k+1]),
            #         nn.ReLU(),
            #     ])
            shared_update_mlp.extend([
                spconv.SubMConv3d(spconv_mlps[-2], spconv_mlps[-1], 1, padding=0, bias=False, indice_key="subm"),
                norm_fn(spconv_mlps[-1]),
            ])
            self.spconv_mlps = spconv.SparseSequential(*shared_update_mlp)
            self.update_relu = nn.ReLU()
            out_channels = spconv_mlps[-1]

        if confidence_mlp is not None:
            out_part_channels = out_channels
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, 3, kernel_size=1, bias=True),
            )
            self.confidence_mlp = nn.Sequential(*shared_mlp)
            pi = 0.01
            nn.init.constant_(self.confidence_mlp[3].bias, -np.log((1 - pi) / pi))
        else:
            self.confidence_mlp = None


class VoxelPointnetSAModuleFS(VoxelPointnetSAModuleFSMSG):
    """Pointnet set abstraction layer with fusion sampling"""

    def __init__(self, *,
                 mlp: List[int],
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 query_range: List[List[int]] = None,
                 sp_stride: int = None,
                 stride: List[List[int]] = None,
                 spconv_mlps: List[int] = None,
                 spconv_mlps_post: List[int] = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None,
                 voxel_config=None,
                 sa_layer_idx=None):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps, f-fps or c-fps
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__(
            mlps=[mlp], npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, query_range=query_range, sp_stride=sp_stride, stride=stride,
            radii=[radius], nsamples=[nsample], spconv_mlps=spconv_mlps, spconv_mlps_post=spconv_mlps_post,
            bn=bn, use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            skip_connection=skip_connection, weight_gamma=weight_gamma,
            aggregation_mlp=aggregation_mlp, confidence_mlp=confidence_mlp, voxel_config=voxel_config,
            sa_layer_idx=sa_layer_idx
        )


class _VoxelPointnetSAModuleFSDistillationBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.spconv_mlps = None
        self.npoint_list = []
        self.sample_range_list = [[0, -1]]
        self.sample_method_list = ['d-fps']
        self.sp_stride = None
        self.radii = []
        self.point_mlps = None

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0
        self.skip_connection = False

        self.aggregation_mlp = None
        self.confidence_mlp = None
        self.voxel_size = [],
        self.grid_size = [],
        self.point_cloud_range = []
        self.voxel_size_tensor = [],
        self.grid_size_tensor = [],
        self.point_cloud_range_tensor = []

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None,
                scores=None,
                part_scores=None,
                sp_tensor=None,
                unique_idxs=None,
                switch=False,
                centroids=None,
                centroid_voxel_idxs=None,):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        batch_size = len(xyz)
        ori_scores = None
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if scores is not None:
            ori_scores = torch.max(scores.sigmoid(), dim=1, keepdim=True)[0]
            scores, idx_cls_pred = torch.max(scores, dim=1, keepdim=True)
            # mask_pc = idx_cls_pred > 0
            # scores[mask_pc] = 100
            # ori_scores = scores.sigmoid()
            if unique_idxs is not None:
                scores = scores[unique_idxs]
                scores = scores.view(batch_size, -1)
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            for i in range(len(self.sample_method_list)):
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous()
                if self.sample_method_list[i] == 'd-fps':

                    # sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                    if self.sa_layer_idx == 0:
                        sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                    else:
                        sample_idx = torch.range(0, self.npoint_list[i]-1, device=xyz_slice.device, dtype=torch.int32)
                        sample_idx = sample_idx.unsqueeze(0).repeat(len(xyz_slice), 1)

                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                elif self.sample_method_list[i] == 's-fps':

                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )

                elif self.sample_method_list[i] == 's-topk':
                    assert scores is not None
                    scores, sample_idx = torch.topk(scores, k=self.npoint_list[i], dim=-1)
                    sample_idx = sample_idx.int()
                elif self.sample_method_list[i] == 'd-fps-faraware':
                    pts_depth = torch.norm(xyz_slice, p=2, dim=-1)
                    sorted_depth, sorted_idx = torch.sort(pts_depth, dim=-1)
                    pts_near_idx = sorted_idx[:, :-256]
                    pts_far_idx = sorted_idx[:, -256:]
                    pts_near = []
                    for index_near in range(batch_size):
                        pts_near.append(xyz_slice[index_near][pts_near_idx[index_near]].unsqueeze(0))
                    pts_near = torch.cat(pts_near, dim=0)
                    sample_idx_base_near = pointnet2_utils.furthest_point_sample(pts_near, self.npoint_list[i]-256)
                    sample_near_idx = []
                    for index_near_idx in range(batch_size):
                        sample_near_idx.append(pts_near_idx[index_near_idx][sample_idx_base_near[index_near_idx].long()].unsqueeze(0))
                    sample_near_idx = torch.cat(sample_near_idx, dim=0)
                    sample_idx = torch.cat([sample_near_idx, pts_far_idx], dim=-1).int()

                else:
                    raise NotImplementedError

                sample_idx_list.append(sample_idx + self.sample_range_list[i][0])

            sample_idx = torch.cat(sample_idx_list, dim=-1)

            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)

            if self.skip_connection:
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)

        if unique_idxs is not None:
            batch_size = sample_idx.shape[0]
            for i in range(batch_size):
                sample_idx[i, :] = sample_idx[i, :]+i*xyz.shape[1]
            sample_idx = sample_idx.view(-1).long()
            unique_idxs = unique_idxs[sample_idx]

        if sp_tensor is not None:

            v2p_ind_tensor = common_utils.generate_voxel2pinds(sp_tensor)
            batch_size, num_points, _ = new_xyz.shape
            point_grid_coords = new_xyz.clone().view(-1, 3)

            point_grid_coords_x = (point_grid_coords[:, 0:1] - self.point_cloud_range_tensor[0]) / self.voxel_size_tensor[0]
            point_grid_coords_y = (point_grid_coords[:, 1:2] - self.point_cloud_range_tensor[1]) / self.voxel_size_tensor[1]
            point_grid_coords_z = (point_grid_coords[:, 2:] - self.point_cloud_range_tensor[2]) / self.voxel_size_tensor[2]
            point_grid_cnt = new_xyz.new_zeros(batch_size).int()
            point_grid_cnt = point_grid_cnt + num_points

            sp_coords = sp_tensor.indices
            voxel_xyz_batch_cnt = sp_coords.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                voxel_xyz_batch_cnt[bs_idx] = (sp_coords[:, 0] == bs_idx).sum()

            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            point_batch_idx = new_xyz.new_zeros(size=(batch_size, num_points))
            for cur_batch_idx in range(batch_size):
                point_batch_idx[cur_batch_idx] = point_batch_idx[cur_batch_idx] + cur_batch_idx
            point_batch_idx = point_batch_idx.view(-1, 1).long()

            point_grid_coords = torch.cat([point_batch_idx, point_grid_coords_z, point_grid_coords_y, point_grid_coords_x], dim=-1).contiguous()
            point_grid_coords = point_grid_coords.int()

            voxel_xyz = centroids[:, 1:4]
            features_in = sp_tensor.features.contiguous()

        for i in range(len(self.groupers)):

            if sp_tensor is None:

                idx_cnt, grouped_features, grouped_xyz = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                batch_size, npoint, _ = new_xyz.shape
                idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
                idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
                grouped_features = grouped_features * idx_cnt_mask
                new_features = self.point_mlps[i](grouped_features)

            else:

                grouped_features, grouped_xyz, empty_ball_mask, density_score = self.groupers[i](
                    new_coords=point_grid_coords,
                    xyz=voxel_xyz.contiguous(),
                    xyz_batch_cnt=voxel_xyz_batch_cnt,
                    new_xyz=new_xyz.view(-1, 3),
                    new_xyz_batch_cnt=point_grid_cnt,
                    features=features_in,
                    voxel2point_indices=v2p_ind_tensor
                )

                # (B, C, npoint, nsample)
                batch_size, npoint, _ = new_xyz.shape
                nchannel, nsample = grouped_features.shape[1:]
                grouped_new_xyz = new_xyz.view(-1, 3).unsqueeze(-1)
                grouped_features[empty_ball_mask] = 0
                grouped_xyz = grouped_xyz - grouped_new_xyz
                grouped_xyz[empty_ball_mask] = 0
                grouped_features = grouped_features.view(batch_size, npoint, nchannel, nsample)
                grouped_features = grouped_features.permute(0, 2, 1, 3)
                grouped_features = self.point_mlps[i](grouped_features)
                grouped_xyz = grouped_xyz.view(batch_size, npoint, 3, nsample)
                grouped_xyz = grouped_xyz.permute(0, 2, 1, 3)
                grouped_xyz = self.pos_mlps[i](grouped_xyz)
                new_features = self.relu(grouped_features+grouped_xyz)

            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'weight_pool':
                pos_weights = self.pos_mlps[i](new_features)
                new_features = new_features * pos_weights
                pooled_features = torch.sum(new_features, dim=-1)
            else:
                raise NotImplementedError

            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)


        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)

        new_features = torch.cat(new_features_list, dim=1)

        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features)

        if sp_tensor is None:

            batch_size, channel, num_points = new_features.shape
            voxel_idxs = voxel_aggregation_utils.get_voxel_indices(
                new_xyz.clone().view(-1, 3).contiguous(),
                voxel_size=self.voxel_size_tensor,
                point_cloud_range=self.point_cloud_range_tensor)  # get index of all points  within the range
            # Add batch_idx
            batch_idx = new_xyz.new_zeros(size=(batch_size, num_points))
            for i in range(batch_size):
                batch_idx[i] = batch_idx[i] + i

            batch_idx = batch_idx.view(-1, 1).long()
            voxel_idxs = torch.flip(voxel_idxs, dims=[1])
            voxel_idxs = torch.cat((batch_idx, voxel_idxs), dim=-1)
            # Filter out points that are outside the valid point cloud range (invalid indices have -1)
            # voxel_idxs_valid = voxel_idxs[voxel_idxs_valid_mask]
            # Convert voxel_indices from (bxyz) to (bzyx) format for properly indexing voxelization layer


            # voxel_idxs = voxel_idxs[:, [0, 3, 2, 1]]
            xyz_for_voxel = new_xyz.view(-1, 3)

            xyz_for_voxel = torch.cat([batch_idx, xyz_for_voxel], dim=-1)
            features_for_voxel = new_features.permute(0, 2, 1).contiguous().view(-1, channel)
            point_for_voxel = torch.cat([xyz_for_voxel, features_for_voxel], dim=-1) # bxyz+features
            # start_time = time.time()
            centroids_coords_features, centroid_voxel_idxs, num_points_in_voxel, unique_idxs = voxel_aggregation_utils.get_centroid_per_voxel(
                point_for_voxel, voxel_idxs)
            # delta = time.time() - start_time
            # delta_print = 'Generate label finished(sec_per_forward_example: %.4f second).' % delta
            # print(delta_print)
            # centroids_coords_features = point_for_voxel
            # centroid_voxel_idxs = voxel_idxs
            # unique_idxs = torch.range(0, len(centroid_voxel_idxs)-1, device="cuda:0").long()
            sparse_shape = self.grid_size[::-1].astype(np.int64)
            centroids = centroids_coords_features[:, 0:4].contiguous()
            centroids_features = centroids_coords_features[:, 4:]


            sp_tensor = spconv.SparseConvTensor(
                features=centroids_features.contiguous(),
                indices=centroid_voxel_idxs.int(),
                spatial_shape=sparse_shape,
                batch_size=batch_size
            )


        else:
            if new_features is not None and (self.sa_layer_idx > 0) and (self.sa_layer_idx < 3):
                # start_time = time.time()
                batch_size, last_channel, num_points = new_features.shape
                num_points_for_update = num_points
                new_features_for_update = new_features
                new_xyz_for_update = new_xyz
                new_point_idxs = voxel_aggregation_utils.get_voxel_indices(
                    new_xyz_for_update.view(-1, 3),
                    voxel_size=self.voxel_size_tensor,
                    point_cloud_range=self.point_cloud_range_tensor)
                # Add batch_idx
                new_batch_idx = new_xyz_for_update.new_zeros(size=(batch_size, num_points_for_update))
                for i in range(batch_size):
                    new_batch_idx[i] = new_batch_idx[i] + i
                new_batch_idx = new_batch_idx.view(-1, 1).long()
                # new_voxel_idxs_valid_mask = (new_point_idxs != -1).all(-1)
                new_voxel_idxs = torch.cat((new_batch_idx, new_point_idxs), dim=-1)

                # Filter out points that are outside the valid point cloud range (invalid indices have -1)
                # new_voxel_idxs_valid = new_voxel_idxs[new_voxel_idxs_valid_mask]
                new_voxel_idxs = new_voxel_idxs[:, [0, 3, 2, 1]]
                new_xyz_for_voxel = new_xyz_for_update.view(-1, 3)
                new_xyz_for_voxel = torch.cat([new_batch_idx, new_xyz_for_voxel], dim=-1)
                new_features_for_voxel = new_features_for_update.permute(0, 2, 1).contiguous().view(-1, last_channel)
                point_for_voxel = torch.cat([new_xyz_for_voxel, new_features_for_voxel], dim=-1)
                new_centroids, new_centroid_voxel_idxs, new_num_points_in_voxel, _ = \
                    voxel_aggregation_utils.get_centroid_per_voxel(point_for_voxel, new_voxel_idxs)
                update_indices_nonempty, update_nonempty_mask = \
                    voxel_aggregation_utils.get_nonempty_voxel_feature_indices(new_centroid_voxel_idxs, sp_tensor)
                source_tesor_features = new_centroids.new_zeros([sp_tensor.features.shape[0], new_centroids.shape[1]-4])
                source_tesor_features[update_indices_nonempty] = new_centroids[:, 4:][update_nonempty_mask]
                source_tesor = spconv.SparseConvTensor(
                    features=source_tesor_features.contiguous(),
                    indices=centroid_voxel_idxs.int(),
                    spatial_shape=sp_tensor.spatial_shape,
                    batch_size=batch_size
                )

                sp4x_tensor = self.spconv4x_mlps(source_tesor)
                sp8x_tensor = self.spconv8x_mlps(sp4x_tensor)
                sp16x_tensor = self.spconv16x_mlps(sp8x_tensor)
                spinv16x_tensor = self.spconvinv16x_mlps(sp16x_tensor)
                spinv16x_tensor = replace_feature(spinv16x_tensor, spinv16x_tensor.features + sp16x_tensor.features)
                spinv8x_tensor = self.spconvinv8x_mlps(spinv16x_tensor)
                spinv8x_tensor = replace_feature(spinv8x_tensor, spinv8x_tensor.features+sp8x_tensor.features)
                spinv4x_tensor = self.spconvinv4x_mlps(spinv8x_tensor)
                spinv4x_tensor = replace_feature(spinv4x_tensor, spinv4x_tensor.features + sp4x_tensor.features)
                dest_tensor = self.spconv_out_mlps(spinv4x_tensor)
                sp_tensor = self.spconv_mlps(sp_tensor)
                sp_tensor = replace_feature(sp_tensor, self.update_relu(sp_tensor.features + ori_scores * dest_tensor.features))
                # delta = time.time() - start_time
                # delta_print = 'Generate label finished(sec_per_forward_example: %.4f second).' % delta
                # print(delta_print)

        if self.confidence_mlp is not None:
            features_for_confidence = sp_tensor.features
            features_for_confidence = features_for_confidence.unsqueeze(-1)
            new_scores = self.confidence_mlp(features_for_confidence)
            new_scores = new_scores.squeeze(2)  # (B, npoint)

            return new_xyz.contiguous(), new_features.contiguous(), new_scores.contiguous(), \
                   sp_tensor, centroids, centroid_voxel_idxs.contiguous(), unique_idxs, None

        return new_xyz.contiguous(), new_features.contiguous(), None, \
               sp_tensor, centroids, centroid_voxel_idxs.contiguous(), unique_idxs, None


class VoxelPointnetSAModuleFSMSGDistillation(_VoxelPointnetSAModuleFSDistillationBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 query_range: List[List[int]] = None,
                 sp_stride: int = None,
                 stride: List[List[int]] = None,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 spconv_mlps: List[int] = None,
                 spconv_mlps_post: List[int] = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None,
                 sa_layer_idx=1,
                 voxel_size=None,
                 grid_size=None,
                 point_cloud_range=None
                 ):
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.query_range = query_range
        self.sp_stride = sp_stride
        self.stride = stride
        self.radii = radii
        self.spconv_mlps = spconv_mlps
        self.spconv_mlps_post = spconv_mlps_post
        self.groupers = nn.ModuleList()
        self.sa_layer_idx = sa_layer_idx
        self.pool_method = pool_method
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range

        self.voxel_size_tensor = torch.tensor(voxel_size, device='cuda:0').float()
        self.point_cloud_range_tensor = torch.tensor(point_cloud_range, device='cuda:0').float()

        if mlps[0]:
            self.point_mlps = nn.ModuleList()
            if self.sa_layer_idx > 0:
                self.pos_mlps = nn.ModuleList()
        if self.pool_method == "weight_pool":
            self.pos_mlps = nn.ModuleList()

        former_radius = 0.0
        in_channels, out_channels = 0, 0
        for i in range(len(radii)):
            radius = radii[i]
            query_ranges = query_range[i]
            strides = stride[i]
            nsample = nsamples[i]
            if dilated_radius_group:
                if sa_layer_idx == 0:
                    self.groupers.append(
                        pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz)
                    )
                else:
                    self.groupers.append(
                        voxel_query_utils.VoxelQueryAndGroupingDilated(
                            query_ranges, strides, former_radius, radius, nsample)
                    )
            else:
                if sa_layer_idx == 0:
                    self.groupers.append(
                        pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                    )
                else:
                    self.groupers.append(
                        voxel_query_utils.VoxelQueryAndGrouping(query_ranges, radius, nsample)
                    )

            former_radius = radius

            if mlps[0] and self.sa_layer_idx==0:
                mlp_spec = mlps[i]
                if use_xyz:
                    mlp_spec[0] += 3
                ori_mlp_spec_in = mlp_spec[0]
                shared_point_mlp = []
                for k in range(len(mlp_spec) - 1):
                    shared_point_mlp.extend([
                        nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(mlp_spec[k + 1]),
                        nn.ReLU()
                    ])
                self.point_mlps.append(nn.Sequential(*shared_point_mlp))
                mlp_spec[0] = ori_mlp_spec_in
                in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
                out_channels += mlp_spec[-1]
            else:
                mlp_spec = mlps[i]
                ori_mlp_spec_in = mlp_spec[0]
                shared_point_mlp = []
                for k in range(len(mlp_spec) - 2):
                    shared_point_mlp.extend([
                        nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm2d(mlp_spec[k + 1]),
                        nn.ReLU()
                    ])
                shared_point_mlp.extend([
                    nn.Conv2d(mlp_spec[-2], mlp_spec[-1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[-1]),
                ])
                self.point_mlps.append(nn.Sequential(*shared_point_mlp))
                self.pos_mlps.append(nn.Sequential(
                    nn.Conv2d(3, mlp_spec[-1]//2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[-1]//2),
                    nn.ReLU(),
                    nn.Conv2d(mlp_spec[-1]//2, mlp_spec[-1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[-1]),
                ))
                self.relu = nn.ReLU()
                mlp_spec[0] = ori_mlp_spec_in
                in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
                out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.skip_connection = skip_connection
        self.weight_gamma = weight_gamma

        if skip_connection:
            out_channels += in_channels

        if aggregation_mlp is not None:
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_mlp = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_mlp = None

        if (self.sa_layer_idx <= 2) and (self.sa_layer_idx > 0):
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            sp_in_channels = spconv_mlps[0]
            tagspconv8x = "spconv8x%d" % sa_layer_idx
            tagspconv16x = "spconv16x%d" % sa_layer_idx
            n_EnDe = int(out_channels//2)
            n_EnDe2x = n_EnDe
            n_EnDe4x = n_EnDe * 2

            self.spconv4x_mlps = spconv.SparseSequential(
                spconv.SubMConv3d(out_channels, n_EnDe, 1, padding=0, bias=False, indice_key="subm4x"),
                norm_fn(n_EnDe),
                nn.ReLU(),
            )
            self.spconv8x_mlps = spconv.SparseSequential(
                spconv.SparseConv3d(n_EnDe, n_EnDe2x, 3, stride=2, padding=1, bias=False, indice_key=tagspconv8x),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
            )
            self.spconv16x_mlps = spconv.SparseSequential(
                spconv.SparseConv3d(n_EnDe2x, n_EnDe4x, 3, stride=2, padding=1, bias=False, indice_key=tagspconv16x),
                norm_fn(n_EnDe4x),
                nn.ReLU(),
            )
            self.spconvinv16x_mlps = spconv.SparseSequential(
                spconv.SubMConv3d(n_EnDe4x, n_EnDe4x, 3, padding=1, bias=False, indice_key="subm16x"),
                norm_fn(n_EnDe4x),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe4x, n_EnDe4x, 3, padding=1, bias=False, indice_key="subm16x"),
                norm_fn(n_EnDe4x),
                nn.ReLU(),
            )

            self.spconvinv8x_mlps = spconv.SparseSequential(
                spconv.SparseInverseConv3d(n_EnDe4x, n_EnDe2x, 3, indice_key=tagspconv16x, bias=False),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe2x, n_EnDe2x, 3, padding=1, bias=False, indice_key="subm8x"),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe2x, n_EnDe2x, 3, padding=1, bias=False, indice_key="subm8x"),
                norm_fn(n_EnDe2x),
                nn.ReLU(),
            )
            self.spconvinv4x_mlps = spconv.SparseSequential(
                spconv.SparseInverseConv3d(n_EnDe2x, n_EnDe, 3, indice_key=tagspconv8x, bias=False),
                norm_fn(n_EnDe),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe, n_EnDe, 3, padding=1, bias=False, indice_key="subm4x"),
                norm_fn(n_EnDe),
                nn.ReLU(),
                spconv.SubMConv3d(n_EnDe, n_EnDe, 3, padding=1, bias=False, indice_key="subm4x"),
                norm_fn(n_EnDe),
                nn.ReLU(),
            )
            self.spconv_out_mlps = spconv.SparseSequential(
                spconv.SubMConv3d(n_EnDe, out_channels, 1, padding=0, bias=False, indice_key="submencoder"),
                norm_fn(out_channels),
                # nn.ReLU(),
            )
            shared_update_mlp = []
            # for k in range(len(spconv_mlps)-2):
            #     shared_update_mlp.extend([
            #         spconv.SubMConv3d(spconv_mlps[k], spconv_mlps[k+1], 3, padding=0, bias=False, indice_key="subm"),
            #         norm_fn(spconv_mlps[k+1]),
            #         nn.ReLU(),
            #     ])
            shared_update_mlp.extend([
                spconv.SubMConv3d(spconv_mlps[-2], spconv_mlps[-1], 1, padding=0, bias=False, indice_key="subm"),
                norm_fn(spconv_mlps[-1]),
            ])
            self.spconv_mlps = spconv.SparseSequential(*shared_update_mlp)
            self.update_relu = nn.ReLU()
            out_channels = spconv_mlps[-1]

        if confidence_mlp is not None:
            out_part_channels = out_channels
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, 3, kernel_size=1, bias=True),
            )
            self.confidence_mlp = nn.Sequential(*shared_mlp)
            pi = 0.01
            nn.init.constant_(self.confidence_mlp[3].bias, -np.log((1 - pi) / pi))
        else:
            self.confidence_mlp = None


class VoxelPointnetSAModuleFSDistillation(VoxelPointnetSAModuleFSMSGDistillation):
    """Pointnet set abstraction layer with fusion sampling"""

    def __init__(self, *,
                 mlp: List[int],
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 query_range: List[List[int]] = None,
                 sp_stride: int = None,
                 stride: List[List[int]] = None,
                 spconv_mlps: List[int] = None,
                 spconv_mlps_post: List[int] = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None,
                 sa_layer_idx=None,
                 voxel_size=None,
                 grid_size=None,
                 point_cloud_range=None
                 ):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps, f-fps or c-fps
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__(
            mlps=[mlp], npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, query_range=query_range, sp_stride=sp_stride, stride=stride,
            radii=[radius], nsamples=[nsample], spconv_mlps=spconv_mlps, spconv_mlps_post=spconv_mlps_post,
            bn=bn, use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            skip_connection=skip_connection, weight_gamma=weight_gamma,
            aggregation_mlp=aggregation_mlp, confidence_mlp=confidence_mlp,
            sa_layer_idx=sa_layer_idx, voxel_size=voxel_size, grid_size=grid_size, point_cloud_range=point_cloud_range
        )

if __name__ == "__main__":
    pass
