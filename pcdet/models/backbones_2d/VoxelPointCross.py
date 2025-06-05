# -*- coding: utf-8 -*- 
# @Time : 2021/11/29 下午3:40 
# @Author : Peng Hao 
# @File : VoxelPointCross.py
import torch
import numpy as np
from torch import nn
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules, pointnet2_utils

# from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
# from ...utils import common_utils
# from ..backbones_3d.spconv_backbone import post_act_block
# from functools import partial
# from ...utils.spconv_utils import spconv


class VoxelPointCross(nn.Module):
    def __init__(self, model_cfg,  input_channels, voxel_size, point_cloud_range, backbone_channels):
        super(VoxelPointCross, self).__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.n_block_scale1 = model_cfg.N_BLOCK[0]
        self.n_block_scale2 = model_cfg.N_BLOCK[1]
        self.n_block = self.n_block_scale1+self.n_block_scale2
        fg_candidate_points = self.model_cfg.FG_CORNER_POINTS
        self.fg_points = fg_candidate_points[0]
        self.fg_points_all = sum(self.fg_points)
        self.candidate_points = fg_candidate_points[1]
        self.candidate_points_all = sum(self.candidate_points)
        self.sample_fps = self.model_cfg.SAMPLE_FPS

        # self.pool_cfg = model_cfg.POINT_GRID_POOL
        # layer_cfg = self.pool_cfg.POOL_LAYERS
        # c_out = 0
        # self.roi_grid_pool_layers = nn.ModuleList()
        # for src_name in self.pool_cfg.FEATURES_SOURCE:
        #     mlps = layer_cfg[src_name].MLPS
        #     for k in range(len(mlps)):
        #         mlps[k] = [backbone_channels[src_name]] + mlps[k]
        #     pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
        #         query_ranges=layer_cfg[src_name].QUERY_RANGES,
        #         nsamples=layer_cfg[src_name].NSAMPLE,
        #         radii=layer_cfg[src_name].POOL_RADIUS,
        #         mlps=mlps,
        #         pool_method=layer_cfg[src_name].POOL_METHOD,
        #     )
        #     self.roi_grid_pool_layers.append(pool_layer)
        #     c_out += sum([x[-1] for x in mlps])
        self.SA_modules = nn.ModuleList()
        channel_in = 256
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

        self.v_input_scale1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=5),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
        self.v_input_scale2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False, groups=5),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
        self.point_features = nn.Sequential(
            nn.Linear(in_features=96, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.p_input_scale1 = nn.Sequential(
            nn.Linear(in_features=131, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.p_input_scale2 = nn.Sequential(
            nn.Linear(in_features=131, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.fg_pred_layer = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3, bias=True),
        )
        self.neighborhood_offset = nn.Sequential(
            nn.Linear(in_features=32, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.offset = nn.Sequential(
            nn.Linear(in_features=3, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.corner_preds = nn.Sequential(
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=24, bias=True),
        )
        self.candidate_preds = nn.Sequential(
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )
        self.candidate_features = nn.Sequential(
            nn.Linear(in_features=385, out_features=256, bias=False),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=False),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.p_block = nn.ModuleList()
        self.pos_block = nn.ModuleList()
        self.neighborhood_offset_block = nn.ModuleList()
        self.offset_block = nn.ModuleList()
        self.channel_wise_block = nn.ModuleList()
        self.channel_aggregation_block = nn.ModuleList()
        self.local_block = nn.ModuleList()
        self.local_se_block = nn.ModuleList()
        self.global_block = nn.ModuleList()
        self.global_se_block = nn.ModuleList()
        self.local_global_attention_block = nn.ModuleList()

        for i in range(self.n_block):
            if i < self.n_block_scale1:
                group = 5
                neighborhood_input = 64
            else:
                group = 5
                neighborhood_input = 64
            self.local_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=group),
                    nn.BatchNorm2d(160),
                    nn.ReLU(),
                    # nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                    #           groups=group),
                    # nn.BatchNorm2d(160),
                    # nn.ReLU(),
                )
            )
            self.global_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=group),
                    nn.BatchNorm2d(160),
                    nn.ReLU(),
                    # nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                    #           groups=group),
                    # nn.BatchNorm2d(160),
                    # nn.ReLU(),
                )
            )
            self.channel_wise_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=group),
                    nn.BatchNorm2d(160),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=group),
                    nn.BatchNorm2d(160),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=group),
                    nn.BatchNorm2d(160),
                    nn.ReLU(),
                    # nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                    #           groups=group),
                    # nn.BatchNorm2d(160),
                    # nn.ReLU(),

                )
            )
            self.channel_aggregation_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=1),
                    nn.BatchNorm2d(160),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=160, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=1),
                    nn.BatchNorm2d(320),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=320, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=1),
                    nn.BatchNorm2d(160),
                    nn.ReLU(),
                    # nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False,
                    #           groups=1),
                    # nn.BatchNorm2d(160),
                    # nn.ReLU(),
                )
            )
            self.p_block.append(
                nn.Sequential(
                    nn.Linear(in_features=256, out_features=128, bias=False),
                    nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                    # nn.Linear(in_features=128, out_features=128, bias=False),
                    # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    # nn.ReLU(),
                )
            )
            # self.pos_block.append(
            #     nn.Sequential(
            #         nn.Linear(in_features=3, out_features=128, bias=False),
            #         nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #         nn.ReLU(),
            #     )
            # )
            self.local_se_block.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(in_channels=160, out_channels=16, kernel_size=1, groups=1),
                    # nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=160, kernel_size=1, groups=1),
                    # nn.BatchNorm2d(160),
                    nn.Sigmoid()
                )
            )
            self.global_se_block.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(in_channels=160, out_channels=16, kernel_size=1, groups=1),
                    # nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=160, kernel_size=1, groups=1),
                    # nn.BatchNorm2d(160),
                    nn.Sigmoid()
                )
            )
            # if i !=self.n_block:
            #     self.v_backbone.append(
            #         nn.Sequential(
            #             nn.Conv2d(in_channels=160, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            #             nn.BatchNorm2d(128),
            #             nn.ReLU(),
            #             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            #             nn.BatchNorm2d(64),
            #             nn.ReLU(),
            #             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            #             nn.BatchNorm2d(32),
            #             nn.ReLU(),
            #             # nn.Conv2d(in_channels=320, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False, groups=5),
            #             # nn.BatchNorm2d(160),
            #             # nn.ReLU(),
            #             # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            #             # nn.BatchNorm2d(64),
            #             # nn.ReLU(),
            #             # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            #             # nn.BatchNorm2d(64),
            #             # nn.ReLU(),
            #         )
            #     )
            #     self.p2v_block.append(
            #         nn.Sequential(
            #             nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False, groups=5),
            #             nn.BatchNorm2d(160),
            #             nn.ReLU(),
            #         )
            #     )
            #     self.v_block.append(
            #         nn.Sequential(
            #             nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False, groups=5),
            #             nn.BatchNorm2d(160),
            #             nn.ReLU(),
            #         )
            #     )
            self.neighborhood_offset_block.append(
                nn.Sequential(
                    nn.Linear(in_features=neighborhood_input, out_features=128, bias=False),
                    nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                )
            )
            self.offset_block.append(
                nn.Sequential(
                    nn.Linear(in_features=3, out_features=128, bias=False),
                    nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                )
            )
            self.local_global_attention_block.append(
                nn.Sequential(
                    nn.Linear(in_features=64, out_features=32, bias=False),
                    nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(),
                    nn.Linear(in_features=32, out_features=2, bias=False),
                    nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Sigmoid(),
                )
            )

        self.num_voxel_neck_features = 256
        self.num_point_features = 256

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.fg_pred_layer[3].bias, -np.log((1 - pi) / pi))
        nn.init.constant_(self.candidate_preds[3].bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.corner_preds[3].weight, mean=0, std=0.001)
        nn.init.constant_(self.corner_preds[3].bias, 0)

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

    def bev_to_points(self, keypoints, bev_features, batch_size, bev_stride, index=None):
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        batch_idxs = keypoints[:, 0]
        n_points = len(keypoints)
        # ==================================
        z_idxs = (keypoints[:, 3] - self.point_cloud_range[2]) / self.voxel_size[2]
        z_idxs = z_idxs / 8

        if index is not None:
            point_bev_features = bev_features.new_zeros((n_points, 128))
        else:
            point_bev_features = bev_features.new_zeros((n_points, 64))
        for k in range(batch_size):
            batch_mask = batch_idxs == k
            cur_x_idxs = x_idxs[batch_mask]
            cur_y_idxs = y_idxs[batch_mask]
            cur_z_idxs = z_idxs[batch_mask]
            cur_bev_features = bev_features[k].permute(2, 3, 1, 0)  # (H, W, C)
            im, x, y, z = cur_bev_features, cur_x_idxs, cur_y_idxs, cur_z_idxs

            x0 = torch.floor(x).long()
            # x1 = x0 + 1

            y0 = torch.floor(y).long()
            # y1 = y0 + 1

            z0 = torch.floor(z).long()
            # z1 = z0 + 1

            x0 = torch.clamp(x0, 0, im.shape[1] - 1)
            # x1 = torch.clamp(x1, 0, im.shape[1] - 1)
            y0 = torch.clamp(y0, 0, im.shape[0] - 1)
            # y1 = torch.clamp(y1, 0, im.shape[0] - 1)
            z0 = torch.clamp(z0, 0, im.shape[2] - 1)
            # z1 = torch.clamp(z1, 0, im.shape[2] - 1)

            Ia = im[y0, x0, z0]
            # Ib = im[y1, x0, z0]
            # Ic = im[y0, x1, z0]
            # Id = im[y1, x1, z0]
            # Ie = im[y0, x0, z1]
            # If = im[y1, x0, z1]
            # Ig = im[y0, x1, z1]
            # Ih = im[y1, x1, z1]

            offset_x0 = (x-x0.type_as(x)-0.5).view(-1, 1)
            offset_y0 = (y-y0.type_as(x)-0.5).view(-1, 1)
            offset_z0 = (z-z0.type_as(x)-0.5).view(-1, 1)

            # offset_x0 = (x0.type_as(x) - x).view(-1, 1)
            # # offset_x1 = (x1.type_as(x) - x).view(-1, 1)
            # offset_y0 = (y0.type_as(x) - y).view(-1, 1)
            # # offset_y1 = (y1.type_as(x) - y).view(-1, 1)
            # offset_z0 = (z0.type_as(x) - z ).view(-1, 1)

            offset_Ia = torch.cat([offset_x0, offset_y0, offset_z0], dim=-1)
            # offset_Ib = torch.cat([offset_x0, offset_y1, offset_z0], dim=-1)
            # offset_Ic = torch.cat([offset_x1, offset_y0, offset_z0], dim=-1)
            # offset_Id = torch.cat([offset_x1, offset_y1, offset_z0], dim=-1)
            # offset_Ie = torch.cat([offset_x0, offset_y0, offset_z1], dim=-1)
            # offset_If = torch.cat([offset_x0, offset_y1, offset_z1], dim=-1)
            # offset_Ig = torch.cat([offset_x1, offset_y0, offset_z1], dim=-1)
            # offset_Ih = torch.cat([offset_x1, offset_y1, offset_z1], dim=-1)

            # offset = torch.cat([offset_Ia, offset_Ib, offset_Ic, offset_Id, offset_Ie, offset_If, offset_Ig,
            #                     offset_Ih], dim=-1)
            # neighborhood_features = torch.cat([Ia, Ib, Ic, Id, Ie, If, Ig, Ih], dim=-1)
            if index is not None:
                offset = self.offset_block[index](offset_Ia)
                local_global_attention = self.local_global_attention_block[index](Ia)
                local_update = local_global_attention[:, 0:1]*Ia[:, :32]
                global_update = local_global_attention[:, 1:]*Ia[:, 32:]
                Ia = torch.cat([local_update, global_update], dim=-1)
                cur_point_bev_features = self.neighborhood_offset_block[index](Ia)

            else:
                offset = self.offset(offset_Ia)
                cur_point_bev_features = self.neighborhood_offset(Ia)

            point_bev_features[batch_mask] = cur_point_bev_features+offset

        return point_bev_features


    # def point_grid_pool(self, batch_dict):
    #     """
    #     Args:
    #         batch_dict:
    #             batch_size:
    #             rois: (B, num_rois, 7 + C)
    #             point_coords: (num_points, 4)  [bs_idx, x, y, z]
    #             point_features: (num_points, C)
    #             point_cls_scores: (N1 + N2 + N3 + ..., 1)
    #             point_part_offset: (N1 + N2 + N3 + ..., 3)
    #     Returns:
    #
    #     """
    #     batch_size = batch_dict['batch_size']
    #     point_coords = batch_dict['point_coords']
    #
    #     point_grid_coords = point_coords.clone()
    #     point_grid_coords[:, 1] = (point_grid_coords[:, 1] - self.point_cloud_range[0]) // self.voxel_size[0]
    #     point_grid_coords[:, 2] = (point_grid_coords[:, 2] - self.point_cloud_range[1]) // self.voxel_size[1]
    #     point_grid_coords[:, 3] = (point_grid_coords[:, 3] - self.point_cloud_range[2]) // self.voxel_size[2]
    #     point_grid_coords = point_grid_coords.int()
    #     point_grid_cnt = point_coords.new_zeros(batch_size).int()
    #     for i in range(batch_size):
    #         point_grid_cnt[i] = (point_grid_coords[:, 0] == i).sum()
    #     pooled_features_list = []
    #     for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
    #         pool_layer = self.roi_grid_pool_layers[k]
    #         cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
    #         cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
    #
    #         # compute voxel center xyz and batch_cnt
    #         cur_coords = cur_sp_tensors.indices
    #         cur_voxel_xyz = common_utils.get_voxel_centers(
    #             cur_coords[:, 1:4],
    #             downsample_times=cur_stride,
    #             voxel_size=self.voxel_size,
    #             point_cloud_range=self.point_cloud_range
    #         )
    #         cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
    #         for bs_idx in range(batch_size):
    #             cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
    #         # get voxel2point tensor
    #         v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
    #         # compute the grid coordinates in this scale, in [batch_idx, x y z] order
    #         cur_roi_grid_coords = point_grid_coords[:, 1:] // cur_stride
    #         cur_roi_grid_coords = torch.cat([point_grid_coords[:, 0:1], cur_roi_grid_coords], dim=-1)
    #         cur_roi_grid_coords = cur_roi_grid_coords.int()
    #         # voxel neighbor aggregation
    #         pooled_features = pool_layer(
    #             xyz=cur_voxel_xyz.contiguous(),
    #             xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
    #             new_xyz=point_coords[:, 1:].contiguous().view(-1, 3),
    #             new_xyz_batch_cnt=point_grid_cnt,
    #             new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
    #             features=cur_sp_tensors.features.contiguous(),
    #             voxel2point_indices=v2p_ind_tensor
    #         )
    #
    #         pooled_features_list.append(pooled_features)
    #
    #     pooled_features = torch.cat(pooled_features_list, dim=-1)
    #
    #     return pooled_features

    def forward(self, batch_dict):
        encoded_bev_features_init = batch_dict['spatial_features']
        # with torch.no_grad():
        #     encoded_bev_features_init_mask = torch.sum(encoded_bev_features_init, dim=1, keepdim=True)
        #     encoded_bev_features_init_mask = encoded_bev_features_init_mask != 0
        #     mask_1x = encoded_bev_features_init_mask.type_as(encoded_bev_features_init)
        #     mask_2x = torch.max_pool2d(mask_1x, 2, 2)
        # raw_points = batch_dict['points']
        # point_features = batch_dict['point_features']
        # point_coords = raw_points[:, :4]
        # batch_dict['point_coords'] = point_coords
        # point_coords = batch_dict['point_coords']
        raw_points_features = batch_dict['raw_points_features']
        point_coords = batch_dict['raw_points_bxyz']

        encoded_spconv_tensor_stride = batch_dict['encoded_spconv_tensor_stride']
        batch_size = batch_dict['batch_size']
        B, C, H, W = encoded_bev_features_init.shape
        # point_coords_list = []
        # sp_coords_1x = batch_dict['multi_scale_voxel_center']['sp_coords_1x']
        # sp_coords_2x = batch_dict['multi_scale_voxel_center']['sp_coords_2x']
        # sp_coords_4x = batch_dict['multi_scale_voxel_center']['sp_coords_4x']
        # sp_coords_8x = batch_dict['multi_scale_voxel_center']['sp_coords_8x']
        #
        # for i in range(batch_size):
        #     # cur_1x_mask = sp_coords_1x[:, 0] == i
        #     # cur_2x_mask = sp_coords_2x[:, 0] == i
        #     cur_4x_mask = sp_coords_4x[:, 0] == i
        #     cur_8x_mask = sp_coords_8x[:, 0] == i
        #     cur_raw_mask = raw_points_bxyz[:, 0] == i
        #
        #     # cur_1x_coords = sp_coords_1x[cur_1x_mask]
        #     # cur_2x_coords = sp_coords_2x[cur_2x_mask]
        #     cur_4x_coords = sp_coords_4x[cur_4x_mask]
        #     cur_8x_coords = sp_coords_8x[cur_8x_mask]
        #     cur_raw_coords = raw_points_bxyz[cur_raw_mask]
        #
        #     # point_coords_list.append(cur_1x_coords)
        #     # point_coords_list.append(cur_2x_coords)
        #     point_coords_list.append(cur_4x_coords)
        #     point_coords_list.append(cur_8x_coords)
        #     point_coords_list.append(cur_raw_coords)
        #
        # point_coords = torch.cat(point_coords_list, dim=0)
        # batch_dict['voxel_center_raw_coords'] = point_coords

        encoded_features_to_pointwise_init = self.bev_to_points(
            keypoints=point_coords,
            bev_features=encoded_bev_features_init.view(B, -1, 5, H, W),
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
            index=None,
        )
        point_features_init = torch.cat([raw_points_features, encoded_features_to_pointwise_init], dim=-1)
        point_features_init = self.point_features(point_features_init)
        fg_preds = self.fg_pred_layer(point_features_init)
        fg_preds_scores, _ = torch.max(fg_preds, dim=-1)
        coords = []
        features = []
        scores = []
        for i in range(batch_size):
            cur_mask = point_coords[:, 0] == i
            cur_fg_score = fg_preds_scores[cur_mask]
            cur_fg_preds = fg_preds[cur_mask]

            if self.training:
                _, fg_incices = torch.sort(cur_fg_score, dim=0, descending=True)
                topk_indices = fg_incices[:self.fg_points[0]]
                other_indices = fg_incices[self.fg_points[0]:]
                if self.sample_fps:
                    other_xyz = point_coords[other_indices][:, 1:].view(1, -1, 3).contiguous()
                    other_indices_selected_index = pointnet2_utils.farthest_point_sample(other_xyz, self.fg_points[1]).squeeze(0)
                    other_indices_selected = other_indices[other_indices_selected_index.long()]
                else:
                    index = [i for i in range(len(other_indices))]
                    np.random.shuffle(index)
                    other_indices_selected_index = index[:self.fg_points[1]]
                    other_indices_selected = other_indices[other_indices_selected_index]
                indices_selected = torch.cat([topk_indices, other_indices_selected], dim=0)
            else:
                topk_scores, indices_selected = cur_fg_score.topk(self.fg_points_all, dim=0, largest=True, sorted=True)

            cur_fg_preds = cur_fg_preds[indices_selected]
            cur_features = point_features_init[cur_mask]
            selected_features = cur_features[indices_selected]
            cur_coords = point_coords[cur_mask]
            selected_coords = cur_coords[indices_selected]
            coords.append(selected_coords)
            features.append(selected_features)
            scores.append(cur_fg_preds)

        point_features = torch.cat(features, dim=0)
        point_coords = torch.cat(coords, dim=0)
        scores = torch.cat(scores, dim=0)
        # point_features = torch.cat([point_features, scores], dim=-1)

        # num_voxel = batch_dict['num_voxel']
        # voxel_count = batch_dict['voxel_count']
        # mask_bool = batch_dict['mask_bool']
        batch_dict['point_coords'] = point_coords

        # point_coords_sem = batch_dict['sp_coords_2x']
        # spatial_shape_sem = batch_dict['spatial_shape_2x']
        # indices_sem = batch_dict['indices_2x']
        # point_features_sem = self.point_features_sem(batch_dict['sp_tensor_features_2x'])
        # encoded_bev_features_init = self.v_input(encoded_bev_features_init)
        encoded_bev_features = self.v_input_scale1(encoded_bev_features_init)
        scores_feature = torch.sigmoid(scores)
        point_features = torch.cat([point_features, scores_feature], dim=-1)
        point_features = self.p_input_scale1(point_features)

        for i in range(self.n_block_scale1):
            channel_wise_features = self.channel_wise_block[i](encoded_bev_features)
            aggregarion_features = self.channel_aggregation_block[i](encoded_bev_features)

            local_features = self.local_block[i](channel_wise_features)
            global_features = self.global_block[i](aggregarion_features)

            local_features_attention = self.local_se_block[i](local_features)
            global_features_sttention = self.global_se_block[i](global_features)

            local_features = local_features*local_features_attention + local_features
            global_features = global_features_sttention*global_features + global_features

            B, C, H, W = local_features.shape
            encoded_bev_features = torch.cat(
                [local_features.view(B, -1, 5, H, W), global_features.view(B, -1, 5, H, W)], dim=1)
            encoded_features_to_pointwise_1x = self.bev_to_points(
                keypoints=point_coords,
                bev_features=encoded_bev_features,
                batch_size=batch_size,
                bev_stride=encoded_spconv_tensor_stride,
                index=i,
            )
            encoded_bev_features = encoded_bev_features.view(B, 2*C, H, W)
            point_features = self.p_block[i](
                torch.cat([point_features, encoded_features_to_pointwise_1x], dim=-1))

        # v2p_features_1x = torch.cat(v2p_list_1x, dim=1)
        # all_point_coords = torch.cat([point_coords, point_coords_sem], dim=0)
        # encoded_features_to_pointwise_1x = self.bev_to_points(
        #     keypoints=point_coords,
        #     bev_features=v2p_features_1x,
        #     batch_size=batch_size,
        #     bev_stride=encoded_spconv_tensor_stride,
        # )

        # for i in range(self.n_block_scale1):
        #     point_pos = self.pos_block[i](point_coords[:, 1:])
        #     cur_encoded_features_to_pointwise_1x = encoded_features_to_pointwise_1x[:len(point_coords), i*128:(i+1)*128]
        #     point_features = self.p_block[i](torch.cat([point_features, cur_encoded_features_to_pointwise_1x+point_pos], dim=-1))

            # point_pos_sem = self.pos_block_sem[i](point_coords_sem[:, 1:])
            # cur_encoded_features_to_pointwise_1x_sem = encoded_features_to_pointwise_1x[len(point_coords):, i * 128:(i + 1) * 128]
            # point_features_sem = self.p_block_sem[i](
            #     torch.cat([point_features_sem, cur_encoded_features_to_pointwise_1x_sem + point_pos_sem], dim=-1))
        bev_list = []
        bev_list.append(encoded_bev_features)
        point_list = []
        point_list.append(point_features)

        encoded_bev_features = self.v_input_scale2(encoded_bev_features)
        point_features = torch.cat([point_features, scores_feature], dim=-1)
        point_features = self.p_input_scale2(point_features)
        for i in range(self.n_block_scale1, self.n_block):
            channel_wise_features = self.channel_wise_block[i](encoded_bev_features)
            aggregarion_features = self.channel_aggregation_block[i](encoded_bev_features)

            local_features = self.local_block[i](channel_wise_features)
            global_features = self.global_block[i](aggregarion_features)

            local_features_attention = self.local_se_block[i](local_features)
            global_features_sttention = self.global_se_block[i](global_features)

            local_features = local_features * local_features_attention + local_features
            global_features = global_features_sttention * global_features + global_features

            B, C, H, W = local_features.shape
            encoded_bev_features = torch.cat(
                [local_features.view(B, -1, 5, H, W), global_features.view(B, -1, 5, H, W)], dim=1)
            encoded_features_to_pointwise_2x = self.bev_to_points(
                keypoints=point_coords,
                bev_features=encoded_bev_features,
                batch_size=batch_size,
                bev_stride=encoded_spconv_tensor_stride*2,
                index=i,
            )

            point_features = self.p_block[i](torch.cat([point_features,
                                                        encoded_features_to_pointwise_2x], dim=-1))
            encoded_bev_features = encoded_bev_features.view(B, 2 * C, H, W)

        bev_list.append(encoded_bev_features)
        point_list.append(point_features)
        point_features_end = torch.cat(point_list, dim=-1)

        corner_preds = self.corner_preds(point_features_end)
        candidate_preds = self.candidate_preds(point_features_end)

        candidate_coords = []
        candidate_features = []
        candidate_fg_score = []
        # candidate_corner = []
        candidate_score = []
        for i in range(batch_size):
            cur_mask = point_coords[:, 0] == i
            cur_candidate_score = candidate_preds[cur_mask].squeeze()
            cur_candidate_fg_score = scores[cur_mask]
            # cur_candidate_fg_score_max = torch.sigmoid(torch.max(cur_candidate_fg_score, dim=-1)[0])
            # score_for_selected = torch.sigmoid(cur_candidate_score)*cur_candidate_fg_score_max
            if self.training:
                _, cur_candidate_incices = torch.sort(cur_candidate_score, dim=0, descending=True)
                topk_indices = cur_candidate_incices[:self.candidate_points[0]]
                other_indices = cur_candidate_incices[self.candidate_points[0]:]
                if self.sample_fps:
                    other_xyz = point_coords[other_indices][:, 1:].view(1, -1, 3).contiguous()
                    other_indices_selected_index = pointnet2_utils.farthest_point_sample(other_xyz, self.candidate_points[1]).squeeze(0)
                    other_indices_selected = other_indices[other_indices_selected_index.long()]
                else:
                    index = [i for i in range(len(other_indices))]
                    np.random.shuffle(index)
                    other_indices_selected_index = index[:self.candidate_points[1]]
                    other_indices_selected = other_indices[other_indices_selected_index]
                indices_selected = torch.cat([topk_indices, other_indices_selected.squeeze(0)], dim=0)
            else:
                topk_scores, indices_selected = cur_candidate_score.topk(self.candidate_points_all, dim=0, largest=True, sorted=True)

            cur_features = point_features_end[cur_mask]
            selected_features = cur_features[indices_selected]
            cur_coords = point_coords[cur_mask]
            selected_coords = cur_coords[indices_selected]

            selected_candidate_fg_score = cur_candidate_fg_score[indices_selected]
            selected_candidate_score = cur_candidate_score[indices_selected]
            # cur_corner_preds = corner_preds[cur_mask]
            # selected_corner_preds = cur_corner_preds[topk_indices]

            candidate_coords.append(selected_coords)
            candidate_score.append(selected_candidate_score)
            candidate_features.append(selected_features)
            # candidate_corner.append(selected_corner_preds)
            candidate_fg_score.append(selected_candidate_fg_score)

        candidate_coords = torch.cat(candidate_coords, dim=0)
        # candidate_corner = torch.cat(candidate_corner, dim=0)
        candidate_features = torch.cat(candidate_features, dim=0)
        candidate_fg_score = torch.cat(candidate_fg_score, dim=0)
        candidate_score = torch.cat(candidate_score, dim=0)

        fg_xyz = point_coords[:, 1:]
        candidate_xyz = candidate_coords[:, 1:]

        fg_xyz = fg_xyz.view(batch_size, -1, 3).contiguous()
        candidate_xyz = candidate_xyz.view(batch_size, -1, 3).contiguous()

        fg_features = point_features_end.view(batch_size, -1, point_features_end.shape[-1]).\
            permute(0, 2, 1).contiguous()

        l_xyz, l_features = [fg_xyz], [fg_features]
        for i in range(len(self.SA_modules)):
            _, local_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], new_xyz=candidate_xyz,
            )
        local_features = local_features.permute(0, 2, 1).contiguous().\
            view(batch_size*(self.candidate_points[0]+self.candidate_points[1]), -1)
        candidate_features = torch.cat([candidate_features, local_features, candidate_score.view(-1, 1)], dim=-1)
        # candidate_features = torch.cat(
        #     [candidate_features, torch.sigmoid(candidate_fg_score), torch.sigmoid(candidate_score.view(-1, 1))], dim=-1)
        candidate_features = self.candidate_features(candidate_features)
        # update_voxel_tensor = point_features.new_zeros(size=[num_voxel, voxel_count, 256])
        # update_voxel_tensor[mask_bool] = point_features_end
        # update_voxel_tensor[~mask_bool] = -999999
        # update_voxel_mean = update_voxel_tensor[:, :, :].sum(dim=1, keepdim=False)
        # cross_out_features = update_voxel_mean / normalizer
        # cross_out_features = torch.max(update_voxel_tensor, dim=1)[0]
        # score = semantic+pos
        # cross_out_features = torch.sum(update_voxel_tensor * batch_dict['scores'], dim=1).contiguous()


        # sp_bev_features_2x = self.interpolate_from_bev_features(
        #         keypoints=sp_coords_2x,
        #         bev_features=encoded_bev_features_end,
        #         batch_size=batch_size,
        #         bev_stride=encoded_spconv_tensor_stride,
        # )
        # pos_encoder_2x = self.pos_2x(sp_coords_2x[:, 1:])
        # sp_features_2x = torch.cat([sp_tensor_features_2x, sp_bev_features_2x+pos_encoder_2x], dim=-1)

        # update_sp_tensor_1x = spconv.SparseConvTensor(
        #     features=cross_out_features,
        #     indices=batch_dict['indices_1x'],
        #     spatial_shape=batch_dict['spatial_shape_1x'],
        #     batch_size=batch_size
        # )

        # update_sp_tensor_sem = spconv.SparseConvTensor(
        #     features=point_features_sem_end,
        #     indices=indices_sem,
        #     spatial_shape=spatial_shape_sem,
        #     batch_size=batch_size
        # )

        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'sp_tensor_1x': update_sp_tensor_1x,
        #         # 'sp_tensor_2x': update_sp_tensor_sem,
        #     }
        # })
        # batch_dict['multi_scale_3d_features']['sp_tensor_1x'] = update_sp_tensor_1x
        batch_dict['encoded_bev_features'] = bev_list
        batch_dict['encoded_point_features'] = point_features_end
        batch_dict['fg_preds'] = fg_preds
        batch_dict['point_corner_preds'] = corner_preds
        batch_dict['point_candidate_preds'] = candidate_preds
        batch_dict['scores_fg'] = candidate_fg_score
        batch_dict['candidate_score'] = candidate_score
        batch_dict['candidate_coords'] = candidate_coords
        batch_dict['candidate_features'] = candidate_features
        return batch_dict


