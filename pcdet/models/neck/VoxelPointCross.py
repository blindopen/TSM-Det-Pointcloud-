# -*- coding: utf-8 -*- 
# @Time : 2021/11/29 下午3:40 
# @Author : Peng Hao 
# @File : VoxelPointCross.py
import torch
from torch import nn
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
# from ..backbones_3d.spconv_backbone import post_act_block
# from functools import partial
# import spconv

class VoxelPointCross(nn.Module):
    def __init__(self, model_cfg,  voxel_size, point_cloud_range, backbone_channels):
        super(VoxelPointCross, self).__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.pool_cfg = model_cfg.POINT_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])

        # block = post_act_block
        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.v_input = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v2p_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v2p_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v2p_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v2p_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v2p_5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.p2v_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.p2v_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.p2v_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.p2v_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
        )
        self.v3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
        )
        self.v4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
        )

        # self.r_input = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )
        # self.r2p_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        # )
        # self.r2p_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # self.r2p_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # self.p2r_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # self.p2r_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )

        # self.pos_encoder1 = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=128, bias=False),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU()
        # )
        # self.pos_encoder2 = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=128, bias=False),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU()
        # )
        # self.pos_encoder3 = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=128, bias=False),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU()
        # )
        self.point_features = nn.Sequential(
            nn.Linear(in_features=32, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        # self.point_features = spconv.SparseSequential(
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )
        # self.p1_out = spconv.SparseSequential(
        #     block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', active=False, norm=True),
        # )
        # self.p2_out = spconv.SparseSequential(
        #     block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', active=False, norm=True),
        # )
        # self.p3_out = spconv.SparseSequential(
        #     block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', active=False, norm=True),
        # )
        # self.p4_out = spconv.SparseSequential(
        #     block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', active=False, norm=True),
        # )
        # self.p5_out = spconv.SparseSequential(
        #     block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', active=False, norm=True),
        # )

        self.p1_out = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.p2_out = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

        )
        self.p3_out = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.p4_out = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.p5_out = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        # self.relu = nn.ReLU()
        self.num_voxel_neck_features = 128
        self.num_point_neck_features = 128

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        batch_idxs = keypoints[:, 0]
        n_points = len(keypoints)
        features_dim = (bev_features.shape)[1]

        point_bev_features = bev_features.new_zeros((n_points, features_dim))
        for k in range(batch_size):
            batch_mask = batch_idxs == k
            cur_x_idxs = x_idxs[batch_mask]
            cur_y_idxs = y_idxs[batch_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            cur_point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features[batch_mask] = cur_point_bev_features
            # point_bev_features_list.append(point_bev_features)

        # point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def point_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        point_coords = batch_dict['point_coords']

        point_grid_coords = point_coords.clone()
        point_grid_coords[:, 1] = (point_grid_coords[:, 1] - self.point_cloud_range[0]) // self.voxel_size[0]
        point_grid_coords[:, 2] = (point_grid_coords[:, 2] - self.point_cloud_range[1]) // self.voxel_size[1]
        point_grid_coords[:, 3] = (point_grid_coords[:, 3] - self.point_cloud_range[2]) // self.voxel_size[2]
        point_grid_coords = point_grid_coords.int()
        point_grid_cnt = point_coords.new_zeros(batch_size).int()
        for i in range(batch_size):
            point_grid_cnt[i] = (point_grid_coords[:, 0] == i).sum()
        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = point_grid_coords[:,1:] // cur_stride
            cur_roi_grid_coords = torch.cat([point_grid_coords[:, 0:1], cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=point_coords[:, 1:].contiguous().view(-1, 3),
                new_xyz_batch_cnt=point_grid_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features_list.append(pooled_features)

        pooled_features = torch.cat(pooled_features_list, dim=-1)

        return pooled_features


    def forward(self, batch_dict):
        encoded_bev_features_init = batch_dict['spatial_features']
        # point_features = batch_dict['point_features']
        point_coords = batch_dict['points'][:, 0:4]
        encoded_spconv_tensor_stride = batch_dict['encoded_spconv_tensor_stride']
        # encoded_spconv_tensor_stride = 2
        batch_size = batch_dict['batch_size']
        batch_dict['point_coords'] = point_coords
        # xyz_features1 = batch_dict['x_conv_transform1']
        # xyz_features2 = batch_dict['x_conv_transform2']
        # xyz_features3 = batch_dict['x_conv_transform3']
        # xyz_features4 = batch_dict['x_conv_transform4']
        # xyz_features5 = batch_dict['x_conv_transform5']
        # point_sp = batch_dict['point_sp']
        # range_features = batch_dict['range_features']

        encoded_bev_features = self.v_input(encoded_bev_features_init)

        point_multi_features = self.point_grid_pool(batch_dict)
        # point_sp.features = torch.cat([point_features, point_multi_features], dim=-1)
        # point_init_features = batch_dict['points'][:, 1:]
        point_features = self.point_features(point_multi_features)
        # point_features = self.point_features(point_multi_features)
        # range_features = self.r_input(range_features)

        v2p1_features = self.v2p_1(encoded_bev_features)
        p2v1_features = self.p2v_1(v2p1_features)
        v1_features = self.v1(torch.cat((encoded_bev_features, p2v1_features), dim=1))

        v2p2_features = self.v2p_2(v1_features)
        p2v2_features = self.p2v_2(v2p2_features)
        v2_features = self.v2(torch.cat((v1_features, p2v2_features), dim=1))

        v2p3_features = self.v2p_3(v2_features)
        p2v3_features = self.p2v_3(v2p3_features)
        v3_features = self.v3(torch.cat((v2_features, p2v3_features), dim=1))

        v2p4_features = self.v2p_4(v3_features)
        p2v4_features = self.p2v_4(v2p4_features)
        v4_features = self.v4(torch.cat((v3_features, p2v4_features), dim=1))

        v2p5_features = self.v2p_5(v4_features)

        # r2p1_features = self.r2p_1(range_features)
        # p2r1_features = self.p2r_1(r2p1_features)
        # r2p2_features = self.r2p_2(p2r1_features)
        # p2r2_features = self.p2r_2(r2p2_features)
        # r2p3_features = self.r2p_3(p2r2_features)

        v2p_features = torch.cat([v2p1_features, v2p2_features, v2p3_features, v2p4_features, v2p5_features], dim=1)
        # r2p_features = torch.cat([r2p1_features, r2p2_features, r2p3_features], dim=1)

        encoded_features_to_pointwise = self.interpolate_from_bev_features(
            keypoints=point_coords,
            bev_features=v2p_features,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
        )
        p1 = encoded_features_to_pointwise[:, :128]
        p2 = encoded_features_to_pointwise[:, 128:256]
        p3 = encoded_features_to_pointwise[:, 256:384]
        p4 = encoded_features_to_pointwise[:, 384:512]
        p5 = encoded_features_to_pointwise[:, 512:640]
        # point_range_features_list = []
        # for i in range(batch_size):
        #     cur_range_image_features = range_features[i]
        #     cur_points_features = cur_range_image_features[:, yx_img_list[i][0], yx_img_list[i][1]]
        #     point_range_features_list.append(cur_points_features)
        # point_range_features = torch.cat(point_range_features_list, dim=-1).permute(1, 0)
        # r1 = point_range_features[:, :64]
        # r2 = point_range_features[:, 64:128]
        # r3 = point_range_features[:, 128:192]

        # point_features = torch.cat([point_features, p1], dim=-1)
        # p1_out = self.relu(self.p1_out(point_features))
        # point_features = torch.cat([p1_out, p2], dim=-1)
        # p2_out = self.relu(self.p2_out(point_features))
        # point_features = torch.cat([p2_out, p3], dim=-1)
        # p3_out = self.relu(self.p3_out(point_features))
        # point_features = torch.cat([p3_out, p4], dim=-1)
        # p4_out = self.relu(self.p4_out(point_features))
        # point_features = torch.cat([p4_out, p5], dim=-1)
        # p5_out = self.relu(self.p5_out(point_features))
        p1_out = self.p1_out(torch.cat([point_features, p1], dim=-1))
        p2_out = self.p2_out(torch.cat([p1_out, p2], dim=-1))
        p3_out = self.p3_out(torch.cat([p2_out, p3], dim=-1))
        p4_out = self.p4_out(torch.cat([p3_out, p4], dim=-1))
        p5_out = self.p5_out(torch.cat([p4_out, p5], dim=-1))
        # point_features.features = torch.cat([p3_out, p4], dim=-1)
        # p4_out = self.relu(self.p4_out(point_features).features + xyz_features1)
        # point_features.features = torch.cat([p4_out, p5], dim=-1)
        # p5_out = self.relu(self.p5_out(point_features).features + xyz_features1)
        # point_features.features = p5_out

        batch_dict['spatial_features_2d'] = v4_features
        batch_dict['encoded_point_features'] = p5_out
        # batch_dict['encoded_point_sp'] = point_features
        # bev_point_features = p2v2_features.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 128)
        # bev_point_coords = batch_dict['bev_point_coords']
        # point_coords_for_head = []
        # point_features_for_head = []
        # for i in range(batch_size):
        #     batch_point_mask = point_coords[:, 0]==i
        #     batch_bev_point_mask = bev_point_coords[:, 0]==i
        #     cur_point_coords = point_coords[batch_point_mask]
        #     cur_bev_point_coords = bev_point_coords[batch_bev_point_mask]
        #     cur_point_features = p3_out[batch_point_mask]
        #     cur_bev_point_features = bev_point_features[i]
        #     point_coords_for_head.append(cur_point_coords)
        #     point_coords_for_head.append(cur_bev_point_coords)
        #     point_features_for_head.append(cur_point_features)
        #     point_features_for_head.append(cur_bev_point_features)

        # point_features_for_head = torch.cat(point_features_for_head, dim=0)
        # point_coords_for_head = torch.cat(point_coords_for_head, dim=0)
        # batch_dict['point_coords'] = point_coords_for_head
        # batch_dict['encoded_point_features'] = point_features_for_head

        # bev_sum = torch.sum(p2v2_features, dim=1,keepdim=True)
        # for i in range(batch_size):
        #     cur_bev = bev_sum[i]
        #     nonzero_index = torch.nonzero(cur_bev)

        return batch_dict


