import time

import torch
import torch.nn as nn
import numpy as np

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """
    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)

        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1], known_feats=l_features[i]
            )

        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict


class PointNet2FSMSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        use_xyz = self.model_cfg.SA_CONFIG.get('USE_XYZ', True)
        dilated_group = self.model_cfg.SA_CONFIG.get('DILATED_RADIUS_GROUP', False)
        skip_connection = self.model_cfg.SA_CONFIG.get('SKIP_CONNECTION', False)
        weight_gamma = self.model_cfg.SA_CONFIG.get('WEIGHT_GAMMA', 1.0)

        self.aggregation_mlps = self.model_cfg.SA_CONFIG.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = self.model_cfg.SA_CONFIG.get('CONFIDENCE_MLPS', None)

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINT_LIST.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            if skip_connection:
                channel_out += channel_in

            if self.aggregation_mlps and self.aggregation_mlps[k]:
                aggregation_mlp = self.aggregation_mlps[k].copy()
                if aggregation_mlp.__len__() == 0:
                    aggregation_mlp = None
                else:
                    channel_out = aggregation_mlp[-1]
            else:
                aggregation_mlp = None

            if self.confidence_mlps and self.confidence_mlps[k]:
                confidence_mlp = self.confidence_mlps[k].copy()
                if confidence_mlp.__len__() == 0:
                    confidence_mlp = None
            else:
                confidence_mlp = None

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleFSMSG(
                    npoint_list=self.model_cfg.SA_CONFIG.NPOINT_LIST[k],
                    sample_range_list=self.model_cfg.SA_CONFIG.SAMPLE_RANGE_LIST[k],
                    sample_method_list=self.model_cfg.SA_CONFIG.SAMPLE_METHOD_LIST[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    dilated_radius_group=dilated_group,
                    skip_connection=skip_connection,
                    weight_gamma=weight_gamma,
                    aggregation_mlp=aggregation_mlp,
                    confidence_mlp=confidence_mlp
                )
            )
            self.num_points_each_layer.append(
                sum(self.model_cfg.SA_CONFIG.NPOINT_LIST[k]))
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.num_point_features = channel_out

        fp_mlps = self.model_cfg.get('FP_MLPS', None)
        if fp_mlps is not None:
            self.FP_modules = nn.ModuleList()
            l_skipped = self.model_cfg.SA_CONFIG.NPOINT_LIST.__len__() - self.model_cfg.FP_MLPS.__len__()
            for k in range(fp_mlps.__len__()):
                pre_channel = fp_mlps[k + 1][-1] if k + 1 < len(fp_mlps) else channel_out
                self.FP_modules.append(
                    pointnet2_modules.PointnetFPModule(
                        mlp=[pre_channel + skip_channel_list[k + l_skipped]] + fp_mlps[k]
                    )
                )
            self.num_point_features = fp_mlps[0][-1]
        else:
            self.FP_modules = None

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                point_coords: (N, 3)
                point_features: (N, C)
                point_confidence_scores: (N, 1)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3).contiguous()
        features = features.view(batch_size, -1, features.shape[-1]) if features is not None else None
        features = features.permute(0, 2, 1).contiguous() if features is not None else None

        batch_idx = batch_idx.view(batch_size, -1).float()

        l_xyz, l_features, l_scores = [xyz], [features], [None]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_scores = self.SA_modules[i](
                l_xyz[i], l_features[i], scores=l_scores[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_scores.append(li_scores)

        # prepare for confidence loss
        l_xyz_flatten, l_scores_flatten = [], []
        for i in range(1, len(l_xyz)):
            l_xyz_flatten.append(torch.cat([
                batch_idx[:, :l_xyz[i].size(1)].reshape(-1, 1),
                l_xyz[i].reshape(-1, 3)
            ], dim=1))  # (N, 4)
        for i in range(1, len(l_scores)):
            if l_scores[i] is None:
                l_scores_flatten.append(None)
            else:
                l_scores_flatten.append(l_scores[i].reshape(-1, 1))  # (N, 1)
        batch_dict['point_coords_list'] = l_xyz_flatten
        batch_dict['point_scores_list'] = l_scores_flatten

        if self.FP_modules is not None:
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )  # (B, C, N)
        else:  # take l_xyz[i - 1] and l_features[i - 1]
            i = 0

        point_features = l_features[i - 1].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((
            batch_idx[:, :l_xyz[i - 1].size(1)].reshape(-1, 1).float(),
            l_xyz[i - 1].view(-1, 3)), dim=1)
        batch_dict['point_scores'] = l_scores[-1]  # (B, N)
        return batch_dict


class VoxelPointNet2FSMSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        use_xyz = self.model_cfg.SA_CONFIG.get('USE_XYZ', True)
        dilated_group = self.model_cfg.SA_CONFIG.get('DILATED_RADIUS_GROUP', False)
        skip_connection = self.model_cfg.SA_CONFIG.get('SKIP_CONNECTION', False)
        weight_gamma = self.model_cfg.SA_CONFIG.get('WEIGHT_GAMMA', 1.0)

        self.aggregation_mlps = self.model_cfg.SA_CONFIG.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = self.model_cfg.SA_CONFIG.get('CONFIDENCE_MLPS', None)

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]

        last_spconv_mlps = []

        for k in range(self.model_cfg.SA_CONFIG.NPOINT_LIST.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            spconv_mlps = self.model_cfg.SA_CONFIG.SPCONV_MLPS_PRE[k].copy()

            channel_out = 0
            if k<=2:
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]
            else:
                channel_out = last_spconv_mlps[-1]
            # if k<2:
            #     for idx in range(mlps.__len__()):
            #         mlps[idx] = [channel_in] + mlps[idx]
            #         channel_out += mlps[idx][-1]
            # else:
            #     for idx in range(mlps.__len__()):
            #         mlps[idx] = [last_spconv_mlps[-1]] + mlps[idx]
            #         channel_out += mlps[idx][-1]
            # for idx in range(mlps.__len__()):
            #     mlps[idx] = [channel_in] + mlps[idx]
            #     channel_out += mlps[idx][-1]
            if skip_connection:
                channel_out += channel_in

            if self.aggregation_mlps and self.aggregation_mlps[k]:
                aggregation_mlp = self.aggregation_mlps[k].copy()
                if aggregation_mlp.__len__() == 0:
                    aggregation_mlp = None
                else:
                    channel_out = aggregation_mlp[-1]
            else:
                aggregation_mlp = None

            if k==0:
                spconv_mlps = [channel_out] + spconv_mlps
            else:
                spconv_mlps = [last_spconv_mlps[-1]] + spconv_mlps



            if self.confidence_mlps and self.confidence_mlps[k]:
                confidence_mlp = self.confidence_mlps[k].copy()
                if confidence_mlp.__len__() == 0:
                    confidence_mlp = None
            else:
                confidence_mlp = None

            self.SA_modules.append(
                pointnet2_modules.VoxelPointnetSAModuleFSMSG(
                    npoint_list=self.model_cfg.SA_CONFIG.NPOINT_LIST[k],
                    sample_range_list=self.model_cfg.SA_CONFIG.SAMPLE_RANGE_LIST[k],
                    sample_method_list=self.model_cfg.SA_CONFIG.SAMPLE_METHOD_LIST[k],
                    sp_stride=self.model_cfg.SA_CONFIG.SPARSE_TENSOR_STRIDE[k],
                    query_range=self.model_cfg.SA_CONFIG.QUERY_RANGE[k],
                    stride=self.model_cfg.SA_CONFIG.STRIDE[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    spconv_mlps=spconv_mlps,
                    pool_method=self.model_cfg.SA_CONFIG.POOL_METHOD[k],
                    # spconv_mlps_post=spconv_mlps_post,
                    use_xyz=use_xyz,
                    dilated_radius_group=dilated_group,
                    skip_connection=skip_connection,
                    weight_gamma=weight_gamma,
                    aggregation_mlp=aggregation_mlp,
                    confidence_mlp=confidence_mlp,
                    voxel_config=self.model_cfg.VOXEL_CONFIG,
                    sa_layer_idx=k
                )
            )
            self.num_points_each_layer.append(
                sum(self.model_cfg.SA_CONFIG.NPOINT_LIST[k]))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
            last_spconv_mlps = spconv_mlps

        self.num_point_features = channel_out

        fp_mlps = self.model_cfg.get('FP_MLPS', None)
        if fp_mlps is not None:
            self.FP_modules = nn.ModuleList()
            l_skipped = self.model_cfg.SA_CONFIG.NPOINT_LIST.__len__() - self.model_cfg.FP_MLPS.__len__()
            for k in range(fp_mlps.__len__()):
                pre_channel = fp_mlps[k + 1][-1] if k + 1 < len(fp_mlps) else channel_out
                self.FP_modules.append(
                    pointnet2_modules.PointnetFPModule(
                        mlp=[pre_channel + skip_channel_list[k + l_skipped]] + fp_mlps[k]
                    )
                )
            self.num_point_features = fp_mlps[0][-1]
        else:
            self.FP_modules = None

        self.switch = True

        self.num_class = 3
        self.raw_score = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, self.num_class, kernel_size=1, bias=True),
        )
        pi = 0.01
        nn.init.constant_(self.raw_score[3].bias, -np.log((1 - pi) / pi))

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                point_coords: (N, 3)
                point_features: (N, C)
                point_confidence_scores: (N, 1)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3).contiguous()
        features = features.view(batch_size, -1, features.shape[-1]) if features is not None else None
        # dist = torch.norm(xyz, p=2, dim=-1, keepdim=True)
        raw_features = torch.cat([xyz, features], dim=-1).permute(0, 2, 1).contiguous()
        features = features.permute(0, 2, 1).contiguous() if features is not None else None

        raw_score = self.raw_score(raw_features).permute(0, 2, 1).contiguous().view(-1, self.num_class)

        batch_idx = batch_idx.view(batch_size, -1).float()

        l_xyz, l_features, l_scores = [xyz], [features], [raw_score]
        # for i in range(len(self.SA_modules)):
        #     li_xyz, li_features, li_scores, li_sp_tensor, li_centroids, li_centroid_voxel_idxs = self.SA_modules[i](
        #         l_xyz[i], l_features[i], scores=l_scores[i],)
        #     l_xyz.append(li_xyz)
        #     l_features.append(li_features)
        #     l_scores.append(li_scores)

        l_sp_tensor, l_centroids, l_centroid_voxel_idxs, l_unique_idxs = [None], [None], [None], [None]
        l_part_scores = [None]

        if self.training:
            self.switch = not self.switch
            # accumulated_iter = batch_dict['accumulated_iter']
            # if accumulated_iter % 100 ==0:
            #     self.switch = not self.switch

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_scores, li_sp_tensor, li_centroids, \
            li_centroid_voxel_idxs, li_unique_idxs, li_part_scores = self.SA_modules[i](
                l_xyz[i], l_features[i], scores=l_scores[i], part_scores=l_part_scores[i],
                sp_tensor=l_sp_tensor[i], centroids=l_centroids[i],
                centroid_voxel_idxs=l_centroid_voxel_idxs[i], unique_idxs=l_unique_idxs[i])

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_scores.append(li_scores)
            l_sp_tensor.append(li_sp_tensor)
            l_centroids.append(li_centroids)
            l_centroid_voxel_idxs.append(li_centroid_voxel_idxs)
            l_unique_idxs.append(li_unique_idxs)
            l_part_scores.append(li_part_scores)

        batch_dict['last_sp_tensor'] = l_sp_tensor[-1]
        batch_dict['last_centroids'] = l_centroids[-1]
        batch_dict['last_features'] = l_features[-1]
        batch_dict['last_centroid_voxel_idxs'] = l_centroid_voxel_idxs[-1]
        batch_dict['last_scores'] = l_scores[-1]
        batch_dict['last_unique_idxs'] = l_unique_idxs[-1]

        # prepare for confidence loss
        l_xyz_flatten, l_scores_flatten, l_part_scores_flatten  = [], [], []

        l_xyz_flatten.append(torch.cat([
            batch_idx.reshape(-1, 1),
            xyz.reshape(-1, 3)
        ], dim=1))  # (N, 4)
        l_scores_flatten.append(raw_score)
        for i in range(1, len(l_xyz)):
            # l_xyz_flatten.append(torch.cat([
            #     batch_idx[:, :l_xyz[i].size(1)].reshape(-1, 1),
            #     l_xyz[i].reshape(-1, 3)
            # ], dim=1))  # (N, 4)
            li_centroids_temp = l_centroids[i]
            # if i == 1:
            #     li_centroids_temp[:, 1:] = li_centroids_temp[:, 1:] + l_part_scores[i]
            l_xyz_flatten.append(li_centroids_temp)  # (N, 4)
        for i in range(1, len(l_scores)):
            if l_scores[i] is None:
                l_scores_flatten.append(None)
                l_part_scores_flatten.append(None)
            else:
                l_scores_flatten.append(l_scores[i].reshape(-1, self.num_class))  # (N, 1)
                # l_part_scores_flatten.append(l_part_scores[i].reshape(-1, 3))
        batch_dict['point_coords_list'] = l_xyz_flatten
        batch_dict['point_scores_list'] = l_scores_flatten
        batch_dict['point_part_scores_list'] = l_part_scores_flatten

        if self.FP_modules is not None:
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
                )  # (B, C, N)
        else:  # take l_xyz[i - 1] and l_features[i - 1]
            i = 0

        point_features = l_features[i - 1].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((
           batch_idx[:, :l_xyz[i - 1].size(1)].reshape(-1, 1).float(),
            l_xyz[i - 1].view(-1, 3)), dim=1)
        batch_dict['point_scores'] = l_scores[-1]  # (B, N)

        batch_dict['statistic_feature'] = l_sp_tensor[-1].features
        return batch_dict


class VoxelPointNet2FSMSGDistillation(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        use_xyz = self.model_cfg.SA_CONFIG.get('USE_XYZ', True)
        dilated_group = self.model_cfg.SA_CONFIG.get('DILATED_RADIUS_GROUP', False)
        skip_connection = self.model_cfg.SA_CONFIG.get('SKIP_CONNECTION', False)
        weight_gamma = self.model_cfg.SA_CONFIG.get('WEIGHT_GAMMA', 1.0)

        self.aggregation_mlps = self.model_cfg.SA_CONFIG.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = self.model_cfg.SA_CONFIG.get('CONFIDENCE_MLPS', None)

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]

        last_spconv_mlps = []

        for k in range(self.model_cfg.SA_CONFIG.NPOINT_LIST.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            spconv_mlps = self.model_cfg.SA_CONFIG.SPCONV_MLPS_PRE[k].copy()

            channel_out = 0
            if k<=2:
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]
            else:
                channel_out = last_spconv_mlps[-1]

            if skip_connection:
                channel_out += channel_in

            if self.aggregation_mlps and self.aggregation_mlps[k]:
                aggregation_mlp = self.aggregation_mlps[k].copy()
                if aggregation_mlp.__len__() == 0:
                    aggregation_mlp = None
                else:
                    channel_out = aggregation_mlp[-1]
            else:
                aggregation_mlp = None

            if k==0:
                spconv_mlps = [channel_out] + spconv_mlps
            else:
                spconv_mlps = [last_spconv_mlps[-1]] + spconv_mlps

            if self.confidence_mlps and self.confidence_mlps[k]:
                confidence_mlp = self.confidence_mlps[k].copy()
                if confidence_mlp.__len__() == 0:
                    confidence_mlp = None
            else:
                confidence_mlp = None

            self.SA_modules.append(
                pointnet2_modules.VoxelPointnetSAModuleFSMSGDistillation(
                    npoint_list=self.model_cfg.SA_CONFIG.NPOINT_LIST[k],
                    sample_range_list=self.model_cfg.SA_CONFIG.SAMPLE_RANGE_LIST[k],
                    sample_method_list=self.model_cfg.SA_CONFIG.SAMPLE_METHOD_LIST[k],
                    sp_stride=self.model_cfg.SA_CONFIG.SPARSE_TENSOR_STRIDE[k],
                    query_range=self.model_cfg.SA_CONFIG.QUERY_RANGE[k],
                    stride=self.model_cfg.SA_CONFIG.STRIDE[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    spconv_mlps=spconv_mlps,
                    pool_method=self.model_cfg.SA_CONFIG.POOL_METHOD[k],
                    use_xyz=use_xyz,
                    dilated_radius_group=dilated_group,
                    skip_connection=skip_connection,
                    weight_gamma=weight_gamma,
                    aggregation_mlp=aggregation_mlp,
                    confidence_mlp=confidence_mlp,
                    sa_layer_idx=k,
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                    point_cloud_range=point_cloud_range
                )
            )
            self.num_points_each_layer.append(
                sum(self.model_cfg.SA_CONFIG.NPOINT_LIST[k]))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
            last_spconv_mlps = spconv_mlps

        self.num_point_features = channel_out

        self.switch = True
        self.num_class = 3

        # student ***************************************************************************
        self.S_SA_modules = nn.ModuleList()
        channel_in = last_spconv_mlps[0]
        self.s_aggregation_mlps = self.model_cfg.S_SA_CONFIG.get('AGGREGATION_MLPS', None)
        self.s_confidence_mlps = self.model_cfg.S_SA_CONFIG.get('CONFIDENCE_MLPS', None)
        self.s_num_points_each_layer = []
        skip_channel_list = [input_channels - 3]

        s_last_spconv_mlps = [last_spconv_mlps[0]]

        for k in range(1, self.model_cfg.S_SA_CONFIG.NPOINT_LIST.__len__()):
            mlps = self.model_cfg.S_SA_CONFIG.MLPS[k].copy()
            spconv_mlps = self.model_cfg.S_SA_CONFIG.SPCONV_MLPS_PRE[k].copy()

            channel_out = 0
            if k <= 2:
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]
            else:
                channel_out = last_spconv_mlps[-1]

            if skip_connection:
                channel_out += channel_in

            if self.s_aggregation_mlps and self.s_aggregation_mlps[k]:
                aggregation_mlp = self.s_aggregation_mlps[k].copy()
                if aggregation_mlp.__len__() == 0:
                    aggregation_mlp = None
                else:
                    channel_out = aggregation_mlp[-1]
            else:
                aggregation_mlp = None

            if k == 0:
                spconv_mlps = [channel_out] + spconv_mlps
            else:
                spconv_mlps = [s_last_spconv_mlps[-1]] + spconv_mlps

            if self.s_confidence_mlps and self.s_confidence_mlps[k]:
                confidence_mlp = self.s_confidence_mlps[k].copy()
                if confidence_mlp.__len__() == 0:
                    confidence_mlp = None
            else:
                confidence_mlp = None

            self.S_SA_modules.append(
                pointnet2_modules.VoxelPointnetSAModuleFSMSGDistillation(
                    npoint_list=self.model_cfg.S_SA_CONFIG.NPOINT_LIST[k],
                    sample_range_list=self.model_cfg.S_SA_CONFIG.SAMPLE_RANGE_LIST[k],
                    sample_method_list=self.model_cfg.S_SA_CONFIG.SAMPLE_METHOD_LIST[k],
                    sp_stride=self.model_cfg.S_SA_CONFIG.SPARSE_TENSOR_STRIDE[k],
                    query_range=self.model_cfg.S_SA_CONFIG.QUERY_RANGE[k],
                    stride=self.model_cfg.S_SA_CONFIG.STRIDE[k],
                    radii=self.model_cfg.S_SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.S_SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    spconv_mlps=spconv_mlps,
                    pool_method=self.model_cfg.S_SA_CONFIG.POOL_METHOD[k],
                    use_xyz=use_xyz,
                    dilated_radius_group=dilated_group,
                    skip_connection=skip_connection,
                    weight_gamma=weight_gamma,
                    aggregation_mlp=aggregation_mlp,
                    confidence_mlp=confidence_mlp,
                    sa_layer_idx=k,
                    voxel_size=voxel_size,
                    grid_size=grid_size,
                    point_cloud_range=point_cloud_range
                )
            )
            self.s_num_points_each_layer.append(
                sum(self.model_cfg.S_SA_CONFIG.NPOINT_LIST[k]))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
            s_last_spconv_mlps = spconv_mlps

        self.FP_modules = None

        self.s_num_point_features = channel_out

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                point_coords: (N, 3)
                point_features: (N, C)
                point_confidence_scores: (N, 1)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3).contiguous()
        features = features.view(batch_size, -1, features.shape[-1]) if features is not None else None
        features = features.permute(0, 2, 1).contiguous() if features is not None else None

        batch_idx = batch_idx.view(batch_size, -1).float()
        l_xyz, l_features, l_scores = [xyz], [features], [None]
        l_sp_tensor, l_centroids, l_centroid_voxel_idxs, l_unique_idxs = [None], [None], [None], [None]
        l_part_scores = [None]

        with torch.no_grad():

            if self.training:
                aggregation_num = len(self.SA_modules)
            else:
                aggregation_num = len(self.SA_modules)-1
            for i in range(aggregation_num):
                # start_time = time.time()
                li_xyz, li_features, li_scores, li_sp_tensor, li_centroids, \
                li_centroid_voxel_idxs, li_unique_idxs, li_part_scores = self.SA_modules[i](
                    l_xyz[i], l_features[i], scores=l_scores[i], part_scores=l_part_scores[i],
                    sp_tensor=l_sp_tensor[i], centroids=l_centroids[i],
                    centroid_voxel_idxs=l_centroid_voxel_idxs[i], unique_idxs=l_unique_idxs[i])
                # delta = time.time() - start_time
                # delta_print = 'Generate label finished(sec_per_forward_example: %.4f second).' % delta
                # print(delta_print)
                l_xyz.append(li_xyz)
                l_features.append(li_features)
                l_scores.append(li_scores)
                l_sp_tensor.append(li_sp_tensor)
                l_centroids.append(li_centroids)
                l_centroid_voxel_idxs.append(li_centroid_voxel_idxs)
                l_unique_idxs.append(li_unique_idxs)
                l_part_scores.append(li_part_scores)



# student ###############################################################################################


        li_xyz, li_features, li_scores, li_sp_tensor, li_centroids, \
        li_centroid_voxel_idxs, li_unique_idxs, li_part_scores = self.S_SA_modules[0](
            l_xyz[1], l_features[1], scores=l_scores[1], part_scores=l_part_scores[1],
            sp_tensor=l_sp_tensor[1], centroids=l_centroids[1],
            centroid_voxel_idxs=l_centroid_voxel_idxs[1], unique_idxs=l_unique_idxs[1])



        l_xyz.append(li_xyz)
        l_features.append(li_features)
        l_scores.append(li_scores)
        l_sp_tensor.append(li_sp_tensor)
        l_centroids.append(li_centroids)
        l_centroid_voxel_idxs.append(li_centroid_voxel_idxs)
        l_unique_idxs.append(li_unique_idxs)
        l_part_scores.append(li_part_scores)

# student ------------------------------------------------------------------------------------------------
        if self.training:
            batch_dict['last_sp_tensor'] = l_sp_tensor[-2]
            batch_dict['last_centroids'] = l_centroids[-2]
            batch_dict['last_features'] = l_features[-2]
            batch_dict['last_centroid_voxel_idxs'] = l_centroid_voxel_idxs[-2]
            batch_dict['last_scores'] = l_scores[-2]
            batch_dict['last_unique_idxs'] = l_unique_idxs[-2]
# student======================================================================
        batch_dict['s_last_sp_tensor'] = l_sp_tensor[-1]
        batch_dict['s_last_centroids'] = l_centroids[-1]
        batch_dict['s_last_features'] = l_features[-1]
        batch_dict['s_last_centroid_voxel_idxs'] = l_centroid_voxel_idxs[-1]
        batch_dict['s_last_scores'] = l_scores[-1]
        batch_dict['s_last_unique_idxs'] = l_unique_idxs[-1]
# student---------------------------------------------------------------------

        # prepare for confidence loss
        l_xyz_flatten, l_scores_flatten, l_part_scores_flatten = [], [], []
        for i in range(1, len(l_xyz)):
            li_centroids_temp = l_centroids[i]
            l_xyz_flatten.append(li_centroids_temp)  # (N, 4)
        for i in range(1, len(l_scores)):
            if l_scores[i] is None:
                l_scores_flatten.append(None)
                l_part_scores_flatten.append(None)
            else:
                l_scores_flatten.append(l_scores[i].reshape(-1, self.num_class))  # (N, 1)
        batch_dict['point_coords_list'] = l_xyz_flatten
        batch_dict['point_scores_list'] = l_scores_flatten
        batch_dict['point_part_scores_list'] = l_part_scores_flatten

        if self.training:
            point_features = l_features[-2].permute(0, 2, 1).contiguous()  # (B, N, C)
            batch_dict['point_features'] = point_features.view(-1, point_features.shape[-2])
            batch_dict['point_coords'] = torch.cat((
                batch_idx[:, :l_xyz[-2].size(1)].reshape(-1, 1).float(), l_xyz[-2].view(-1, 3)), dim=1)
            batch_dict['point_scores'] = l_scores[-2]  # (B, N)
            batch_dict['statistic_feature'] = l_sp_tensor[-2].features

# student==========================================================================================================
        s_point_features = l_features[-1].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['s_point_features'] = s_point_features.view(-1, s_point_features.shape[-1])
        batch_dict['s_point_coords'] = torch.cat((
            batch_idx[:, :l_xyz[-1].size(1)].reshape(-1, 1).float(), l_xyz[-1].view(-1, 3)), dim=1)
        batch_dict['s_point_scores'] = l_scores[-1]  # (B, N)
        batch_dict['s_statistic_feature'] = l_sp_tensor[-1].features
        return batch_dict
# student---------------------------------------------------------------------------------------------------------

