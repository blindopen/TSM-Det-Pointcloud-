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
from ...utils.spconv_utils import replace_feature, spconv


class BEVPoint(nn.Module):
    def __init__(self, model_cfg,  input_channels, voxel_size, point_cloud_range, backbone_channels):
        super(BEVPoint, self).__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.n_block_scale1 = model_cfg.N_BLOCK[0]
        self.n_block_scale2 = model_cfg.N_BLOCK[1]
        self.n_block_scale3 = model_cfg.N_BLOCK[2]
        self.n_block = self.n_block_scale1+self.n_block_scale2
        scale1_input_channels = input_channels['x_conv3']
        scale2_input_channels = input_channels['x_conv4']
        scale3_input_channels = input_channels['x_conv5']
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

        self.v_input_scale1 = nn.Sequential(
            nn.Conv2d(in_channels=scale1_input_channels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.v_input_scale2 = nn.Sequential(
            nn.Conv2d(in_channels=scale2_input_channels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.v_input_scale3 = nn.Sequential(
            nn.Conv2d(in_channels=scale3_input_channels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v_short_scale1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.v_short_scale2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # self.up2 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        # )
        # self.up3 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        # )

        self.scale1_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.scale2_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.scale3_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.point_features1 = nn.Sequential(
            nn.Linear(in_features=32, out_features=384, bias=False),
            nn.BatchNorm1d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.point_features2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=384, bias=False),
            nn.BatchNorm1d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.point_features3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=384, bias=False),
            nn.BatchNorm1d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.point_features4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=384, bias=False),
            nn.BatchNorm1d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.point_features5 = nn.Sequential(
            nn.Linear(in_features=32, out_features=384, bias=False),
            nn.BatchNorm1d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        # self.p
        v_block1_list = []
        v_block2_list = []
        v_block3_list = []
        # pos_block_list = []

        for i in range(self.n_block_scale1):
            v_block1_list.extend(
                [
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ]
            )
        for i in range(self.n_block_scale2):
            v_block2_list.extend(
                [
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ]
            )
        for i in range(self.n_block_scale3):
            v_block3_list.extend(
                [
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                ]
            )
        self.v_block1 = nn.Sequential(*v_block1_list)
        self.v_block2 = nn.Sequential(*v_block2_list)
        self.v_block3 = nn.Sequential(*v_block3_list)

        self.raw_fg_pred = nn.Sequential(
            nn.Linear(in_features=384, out_features=3, bias=False)
        )
        self.num_voxel_neck_features = 384
        self.num_point_features = 384

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
        batch_size = batch_dict['batch_size']
        # raw_points_bxyz = batch_dict['raw_points_bxyz']
        # raw_points_features = batch_dict['raw_points_features']
        encoded_spconv_tensor_stride = batch_dict['encoded_spconv_tensor_stride']
        x_conv1_features = batch_dict['multi_scale_3d_features']['x_conv1'].features
        x_conv2_features = batch_dict['multi_scale_3d_features']['x_conv2'].features

        # forward
        x_conv3_bev = batch_dict['multi_scale_2d_features']['x_conv3']
        x_conv3_features = batch_dict['multi_scale_3d_features']['x_conv3'].features
        x_conv3_bev = self.v_input_scale1(x_conv3_bev)
        x_conv3_bev = self.v_block1(x_conv3_bev)

        x_conv4_bev = batch_dict['multi_scale_2d_features']['x_conv4']
        x_conv4_features = batch_dict['multi_scale_3d_features']['x_conv4'].features
        x_conv4_bev = self.v_input_scale2(x_conv4_bev)
        x_conv4_bev = x_conv4_bev + self.v_short_scale1(x_conv3_bev)
        x_conv4_bev = self.v_block2(x_conv4_bev)

        x_conv5_bev = batch_dict['multi_scale_2d_features']['x_conv5']
        x_conv5_features = batch_dict['multi_scale_3d_features']['x_conv5'].features
        x_conv5_bev = self.v_input_scale3(x_conv5_bev)
        x_conv5_bev = x_conv5_bev + self.v_short_scale2(x_conv4_bev)
        x_conv5_bev = self.v_block3(x_conv5_bev)

        # # backward
        # x_conv3_features = batch_dict['multi_scale_3d_features']['x_conv3'].features
        # x_conv3_bev = batch_dict['multi_scale_2d_features']['x_conv3']
        # x_conv3_bev = self.v_input_scale3(x_conv3_bev)
        # x_conv3_bev = self.v_block3(x_conv3_bev)
        # x_conv3_bev_up = self.up3(x_conv3_bev)
        #
        # x_conv2_features = batch_dict['multi_scale_3d_features']['x_conv2'].features
        # x_conv2_bev = batch_dict['multi_scale_2d_features']['x_conv2']
        # x_conv2_bev = self.v_input_scale2(x_conv2_bev)
        # x_conv2_bev = x_conv2_bev + x_conv3_bev_up
        # x_conv2_bev = self.v_block2(x_conv2_bev)
        # x_conv2_bev_up = self.up2(x_conv2_bev)
        #
        # x_conv1_features = batch_dict['multi_scale_3d_features']['x_conv1'].features
        # x_conv1_bev = batch_dict['multi_scale_2d_features']['x_conv1']
        # x_conv1_bev = self.v_input_scale1(x_conv1_bev)
        # x_conv1_bev = x_conv1_bev + x_conv2_bev_up
        # x_conv1_bev = self.v_block1(x_conv1_bev)


        x_conv3_bev = self.scale1_deconv(x_conv3_bev)
        x_conv4_bev = self.scale2_deconv(x_conv4_bev)
        x_conv5_bev = self.scale3_deconv(x_conv5_bev)
        all_bev = torch.cat([x_conv3_bev, x_conv4_bev, x_conv5_bev], dim=1)

        x_conv1_coords = batch_dict['multi_scale_coords']['x_conv1']
        x_conv2_coords = batch_dict['multi_scale_coords']['x_conv2']
        x_conv3_coords = batch_dict['multi_scale_coords']['x_conv3']
        x_conv4_coords = batch_dict['multi_scale_coords']['x_conv4']
        x_conv5_coords = batch_dict['multi_scale_coords']['x_conv5']

        bev_to_pointwise_conv1 = self.interpolate_from_bev_features(
            keypoints=x_conv1_coords,
            bev_features=all_bev,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
        )
        bev_to_pointwise_conv2 = self.interpolate_from_bev_features(
            keypoints=x_conv2_coords,
            bev_features=all_bev,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
        )
        bev_to_pointwise_conv3 = self.interpolate_from_bev_features(
            keypoints=x_conv3_coords,
            bev_features=all_bev,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
        )
        bev_to_pointwise_conv4 = self.interpolate_from_bev_features(
            keypoints=x_conv4_coords,
            bev_features=all_bev,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
        )
        bev_to_pointwise_conv5 = self.interpolate_from_bev_features(
            keypoints=x_conv5_coords,
            bev_features=all_bev,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
        )
        # bev_to_pointwise_raw = self.interpolate_from_bev_features(
        #     keypoints=raw_points_bxyz,
        #     bev_features=all_bev,
        #     batch_size=batch_size,
        #     bev_stride=encoded_spconv_tensor_stride,
        # )

        # update_convraw = self.point_featuresraw(raw_points_features) + bev_to_pointwise_raw
        update_conv1 = self.point_features1(x_conv1_features) + bev_to_pointwise_conv1
        update_conv2 = self.point_features2(x_conv2_features) + bev_to_pointwise_conv2
        update_conv3 = self.point_features3(x_conv3_features) + bev_to_pointwise_conv3
        update_conv4 = self.point_features4(x_conv4_features) + bev_to_pointwise_conv4
        update_conv5 = self.point_features5(x_conv5_features) + bev_to_pointwise_conv5

        batch_dict['multi_scale_3d_features']['x_conv1'] = batch_dict['multi_scale_3d_features'][
            'x_conv1'].replace_feature(update_conv1)
        batch_dict['multi_scale_3d_features']['x_conv2'] = batch_dict['multi_scale_3d_features'][
            'x_conv2'].replace_feature(update_conv2)
        batch_dict['multi_scale_3d_features']['x_conv3'] = batch_dict['multi_scale_3d_features'][
            'x_conv3'].replace_feature(update_conv3)

        raw_coords = []
        raw_features = []
        for i in range(batch_size):
            conv1_mask = x_conv1_coords[:, 0] == i
            cur_coords = x_conv1_coords[conv1_mask]
            cur_features = update_conv1[conv1_mask]
            raw_coords.append(cur_coords)
            raw_features.append(cur_features)

            conv2_mask = x_conv2_coords[:, 0] == i
            cur_coords = x_conv2_coords[conv2_mask]
            cur_features = update_conv2[conv2_mask]
            raw_coords.append(cur_coords)
            raw_features.append(cur_features)

        raw_coords = torch.cat(raw_coords,dim=0)
        raw_features = torch.cat(raw_features, dim=0)
        batch_dict['raw_coords'] = raw_coords
        raw_fg_preds = self.raw_fg_pred(raw_features)
        raw_fg_score, _ = torch.max(raw_fg_preds, dim=-1)

        coords = []
        features = []
        for i in range(batch_size):
            # values, indices = pred.topk(2, dim=1, largest=True, sorted=True)
            conv3_mask = x_conv3_coords[:, 0] == i
            conv4_mask = x_conv4_coords[:, 0] == i
            conv5_mask = x_conv5_coords[:, 0] == i
            convraw_mask = raw_coords[:, 0] == i

            cur_raw_features = raw_features[convraw_mask]
            cur_raw_coords = raw_coords[convraw_mask]
            cur_raw_fg_score = raw_fg_score[convraw_mask]
            topk_values, topk_indices = cur_raw_fg_score.topk(1000, dim=0, largest=True, sorted=False)

            coords.append(cur_raw_coords[topk_indices])
            coords.append(x_conv3_coords[conv3_mask])
            coords.append(x_conv4_coords[conv4_mask])
            coords.append(x_conv5_coords[conv5_mask])

            features.append(cur_raw_features[topk_indices])
            features.append(update_conv3[conv3_mask])
            features.append(update_conv4[conv4_mask])
            features.append(update_conv5[conv5_mask])

        batch_dict['encoded_bev_features'] = all_bev
        batch_dict['encoded_point_features'] = torch.cat(features, dim=0)
        batch_dict['point_coords'] = torch.cat(coords, dim=0)
        batch_dict['raw_fg_preds'] = raw_fg_preds
        return batch_dict


