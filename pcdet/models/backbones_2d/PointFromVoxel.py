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


class PointFromVoxel(nn.Module):
    def __init__(self, model_cfg,  input_channels, voxel_size, point_cloud_range, backbone_channels):
        super(PointFromVoxel, self).__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        fg_candidate_points = self.model_cfg.FG_CORNER_POINTS
        self.fg_points = fg_candidate_points[0]
        self.fg_points_all = sum(self.fg_points)
        self.candidate_points = fg_candidate_points[1]
        self.candidate_points_all = sum(self.candidate_points)
        self.sample_fps = self.model_cfg.SAMPLE_FPS

        # self.SA_modules = nn.ModuleList()
        # channel_in = 384
        # for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
        #     mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
        #     channel_out = 0
        #     for idx in range(mlps.__len__()):
        #         mlps[idx] = [channel_in] + mlps[idx]
        #         channel_out += mlps[idx][-1]
        #
        #     self.SA_modules.append(
        #         pointnet2_modules.PointnetSAModuleMSG(
        #             npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
        #             radii=self.model_cfg.SA_CONFIG.RADIUS[k],
        #             nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
        #             mlps=mlps,
        #             use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
        #         )
        #     )

        self.v_input = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )

        self.local_scale = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            # # nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            # nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )
        self.global_scale = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.point_features = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.raw_point_features = nn.Sequential(
            nn.Linear(in_features=16, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        # self.offset = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        # )
        # self.pos_coder = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        # )

        # ==============foreground===============
        self.fg_pred_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3, bias=True),
        )
        self.register_buffer('object_statistic_features', torch.zeros(3, 128))

        # ==================scale1================
        self.v_input_scale1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=10),
            # nn.BatchNorm2d(320),
            nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )
        self.channel_wise_scale1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )
        self.local_scale1 = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            # # nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            # nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )
        self.global_scale1 = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # self.local_global_attention_scale1 = nn.Sequential(
        #     nn.Linear(in_features=64, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=2, bias=False),
        #     nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Sigmoid(),
        # )
        self.point_features_scale1 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        # ==============scale2=================
        self.v_input_scale2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )
        self.channel_wise_scale2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )
        self.local_scale2 = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            # # nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            # nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1, bias=False, groups=10),
            nn.BatchNorm2d(320),
            # nn.GroupNorm(num_groups=10, num_channels=320),
            nn.ReLU(),
        )
        self.global_scale2 = nn.Sequential(
            # nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.point_features_scale2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        # self.local_global_attention_scale2 = nn.Sequential(
        #     nn.Linear(in_features=64, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=2, bias=False),
        #     nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Sigmoid(),
        # )
        # ===========corner_preds and scores================
        # self.corner_preds = nn.Sequential(
        #     nn.Linear(in_features=384, out_features=64, bias=False),
        #     nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=24, bias=True),
        # )
        self.center_preds = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3, bias=True),
        )
        self.candidate_preds = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=True),
        )

        # self.candidate_features = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=256, bias=False),
        #     nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     # nn.Linear(in_features=256, out_features=256, bias=False),
        #     # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     # nn.ReLU(),
        # )
        self.center_point_features_scale1 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.center_point_features_scale2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.num_voxel_neck_features = 256
        self.num_point_features = 256

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.fg_pred_layer[3].bias, -np.log((1 - pi) / pi))
        nn.init.constant_(self.candidate_preds[3].bias, -np.log((1 - pi) / pi))
        # nn.init.normal_(self.corner_preds[3].weight, mean=0, std=0.001)
        # nn.init.constant_(self.corner_preds[3].bias, 0)
        nn.init.normal_(self.center_preds[3].weight, mean=0, std=0.01)
        nn.init.constant_(self.center_preds[3].bias, 0)

    def bev_to_points(self, keypoints, local_features, global_features, batch_size, bev_stride, z_stride):
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        batch_idxs = keypoints[:, 0]
        n_points = len(keypoints)
        z_idxs = (keypoints[:, 3] - self.point_cloud_range[2]) / self.voxel_size[2]
        z_idxs = z_idxs / z_stride

        n_local_channels = local_features.shape[1]
        n_global_channels = global_features.shape[1]
        point_local_global_features = local_features.new_zeros((n_points, n_local_channels+n_global_channels))
        for k in range(batch_size):
            batch_mask = batch_idxs == k
            cur_x_idxs = x_idxs[batch_mask]
            cur_y_idxs = y_idxs[batch_mask]
            cur_z_idxs = z_idxs[batch_mask]
            cur_point_coords = keypoints[batch_mask][:, 1:]

            cur_local_features = local_features[k].permute(2, 3, 1, 0)  # (H, W, C)
            im, x, y, z = cur_local_features, cur_x_idxs, cur_y_idxs, cur_z_idxs
            cur_global_features = global_features[k].permute(1, 2, 0)

            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            z0 = torch.floor(z).long()
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            x0 = torch.clamp(x0, 0, im.shape[1] - 1)
            x1 = torch.clamp(x1, 0, im.shape[1] - 1)
            y0 = torch.clamp(y0, 0, im.shape[0] - 1)
            y1 = torch.clamp(y1, 0, im.shape[0] - 1)
            z0 = torch.clamp(z0, 0, im.shape[2] - 1)
            z1 = torch.clamp(z1, 0, im.shape[2] - 1)

            u, v, w = (x - x0).unsqueeze(-1), (y - y0).unsqueeze(-1), (z - z0).unsqueeze(-1)

            c_000 = cur_local_features[y0, x0, z0]
            c_001 = cur_local_features[y0, x0, z1]
            c_010 = cur_local_features[y1, x0, z0]
            c_011 = cur_local_features[y1, x0, z1]
            c_100 = cur_local_features[y0, x1, z0]
            c_101 = cur_local_features[y0, x1, z1]
            c_110 = cur_local_features[y1, x1, z0]
            c_111 = cur_local_features[y1, x1, z1]
            c_xyz = (1.0 - u) * (1.0 - v) * (1.0 - w) * c_000 + \
                    (1.0 - u) * (1.0 - v) * (w) * c_001 + \
                    (1.0 - u) * (v) * (1.0 - w) * c_010 + \
                    (1.0 - u) * (v) * (w) * c_011 + \
                    (u) * (1.0 - v) * (1.0 - w) * c_100 + \
                    (u) * (1.0 - v) * (w) * c_101 + \
                    (u) * (v) * (1.0 - w) * c_110 + \
                    (u) * (v) * (w) * c_111


            Ia = cur_global_features[y0, x0]
            Ib = cur_global_features[y1, x0]
            Ic = cur_global_features[y0, x1]
            Id = cur_global_features[y1, x1]

            wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
            wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
            wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
            wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
            ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
                torch.t(Id) * wd)

            # offset_x0 = (x-x0.type_as(x)-0.5).view(-1, 1)
            # offset_y0 = (y-y0.type_as(x)-0.5).view(-1, 1)
            # offset_z0 = (z-z0.type_as(x)-0.5).view(-1, 1)

            # offset = torch.cat([offset_x0, offset_y0, offset_z0], dim=-1)
            # offset = self.offset(offset)
            # coord_features = self.pos_coder(cur_point_coords)

            point_local_global_features[batch_mask] = torch.cat([c_xyz, ans], dim=-1)

        return point_local_global_features

    def forward(self, batch_dict):
        spatial_features = batch_dict['spatial_features']
        raw_points_features = self.raw_point_features(batch_dict['point_features'])
        point_coords = batch_dict['raw_points_bxyz']

        encoded_spconv_tensor_stride = batch_dict['encoded_spconv_tensor_stride']
        batch_size = batch_dict['batch_size']
        spatial_features = self.v_input(spatial_features)
        B, C, H, W = spatial_features.shape
        local_features = self.local_scale(spatial_features)
        global_features = self.global_scale(spatial_features)
        local_global_features = self.bev_to_points(
            keypoints=point_coords,
            local_features=local_features.view(B, -1, 10, H, W),
            global_features=global_features,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
            z_stride=4,
        )
        point_features_init = torch.cat([raw_points_features, local_global_features], dim=-1)
        point_features_init = self.point_features(point_features_init)
        # features_for_fps_fg = self.features_fps_fg(local_global_features)

        fg_preds = self.fg_pred_layer(point_features_init)
        fg_preds_scores, fg_class_idx = torch.max(fg_preds, dim=-1)


        coords = []
        features = []
        scores = []
        class_idx = []
        for i in range(batch_size):
            cur_mask = point_coords[:, 0] == i
            cur_fg_score = fg_preds_scores[cur_mask]
            cur_fg_preds = fg_preds[cur_mask]
            cur_coords = point_coords[cur_mask]
            cur_class_idx = fg_class_idx[cur_mask]

            if self.training:
                _, fg_incices = torch.sort(cur_fg_score, dim=0, descending=True)
                topk_indices = fg_incices[:self.fg_points[0]]
                other_indices = fg_incices[self.fg_points[0]:]
                if self.sample_fps:
                    top_xyz = cur_coords[:, 1:][topk_indices].view(1, -1, 3).contiguous()
                    top_indices_selected_index = pointnet2_utils.farthest_point_sample(top_xyz, self.fg_points[1]).squeeze(0)
                    top_indices_selected = topk_indices[top_indices_selected_index.long()]

                    other_xyz = cur_coords[:, 1:][other_indices].view(1, -1, 3).contiguous()
                    other_indices_selected_index = pointnet2_utils.farthest_point_sample(other_xyz, self.fg_points[1]).squeeze(0)
                    other_indices_selected = other_indices[other_indices_selected_index.long()]
                else:
                    index = [i for i in range(len(other_indices))]
                    np.random.shuffle(index)
                    other_indices_selected_index = index[:self.fg_points[1]]
                    other_indices_selected = other_indices[other_indices_selected_index]
                indices_selected = torch.cat([top_indices_selected, other_indices_selected], dim=0)
            else:
                topk_scores, indices_selected = cur_fg_score.topk(self.fg_points[1]*2, dim=0, largest=True, sorted=True)


            cur_fg_preds = cur_fg_preds[indices_selected]
            cur_features = point_features_init[cur_mask]
            selected_features = cur_features[indices_selected]
            # cur_coords = point_coords[cur_mask]
            selected_coords = cur_coords[indices_selected]
            coords.append(selected_coords)
            features.append(selected_features)
            scores.append(cur_fg_preds)
            selected_class_idx = cur_class_idx[indices_selected]
            class_idx.append(selected_class_idx)

        point_features = torch.cat(features, dim=0)
        point_coords = torch.cat(coords, dim=0)
        scores = torch.cat(scores, dim=0)
        class_idx = torch.cat(class_idx, dim=0)

        batch_dict['point_coords'] = point_coords

        spatial_features = self.v_input_scale1(spatial_features)
        spatial_features = self.channel_wise_scale1(spatial_features)
        local_features1 = self.local_scale1(spatial_features)
        global_features1 = self.global_scale1(spatial_features)
        B1, C1, H1, W1 = local_features1.shape
        local_global_scale1 = self.bev_to_points(
            keypoints=point_coords,
            local_features=local_features1.view(B1, -1, 10, H1, W1),
            global_features=global_features1,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
            z_stride=4,
        )
        local_global_scale1 = self.point_features_scale1(local_global_scale1)

        bev_list = []
        # bev_features_scale1 = torch.cat([local_features1, global_features1], dim=1)
        bev_list.append(spatial_features)
        point_list = []
        # point_list.append(point_features)
        point_list.append(local_global_scale1)

        spatial_features = self.v_input_scale2(spatial_features)
        spatial_features = self.channel_wise_scale2(spatial_features)
        local_features2 = self.local_scale2(spatial_features)
        global_features2 = self.global_scale2(spatial_features)
        B2, C2, H2, W2 = local_features2.shape
        local_global_scale2 = self.bev_to_points(
            keypoints=point_coords,
            local_features=local_features2.view(B2, -1, 10, H2, W2),
            global_features=global_features2,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride*2,
            z_stride=4,
        )
        # bev_features_scale2 = torch.cat([local_features2, global_features2], dim=1)
        bev_list.append(spatial_features)
        local_global_scale2 = self.point_features_scale2(local_global_scale2)
        point_list.append(local_global_scale2)

        point_features_end = torch.cat(point_list, dim=-1)
        center_preds = self.center_preds(point_features_end)
        candidate_preds = self.candidate_preds(point_features_end)
        point_coords_with_offset = point_coords[:, 1:] + center_preds.detach()
        point_coords_with_offset = torch.cat([point_coords[:, 0:1], point_coords_with_offset], dim=-1)

        # candidate_coords = []
        # # candidate_features = []
        # candidate_fg_score = []
        # candidate_score = []
        # candidate_centers = []
        # for i in range(batch_size):
        #     cur_mask = point_coords[:, 0] == i
        #     cur_candidate_score = candidate_preds[cur_mask].squeeze()
        #     cur_candidate_fg_score = scores[cur_mask]
        #     cur_coords = point_coords_with_offset[cur_mask]
        #     # cur_candidate_fg_score_max = torch.sigmoid(torch.max(cur_candidate_fg_score, dim=-1)[0])
        #     # score_for_selected = torch.sigmoid(cur_candidate_score)*cur_candidate_fg_score_max
        #     if self.training:
        #         _, cur_candidate_incices = torch.sort(cur_candidate_score, dim=0, descending=True)
        #         topk_indices = cur_candidate_incices[:self.candidate_points[0]]
        #         other_indices = cur_candidate_incices[self.candidate_points[0]:]
        #         if self.sample_fps:
        #             top_xyz = cur_coords[:, 1:][topk_indices].view(1, -1, 3).contiguous()
        #             top_indices_selected_index = pointnet2_utils.farthest_point_sample(top_xyz, self.candidate_points[1]).squeeze(0)
        #             top_indices_selected = topk_indices[top_indices_selected_index.long()]
        #
        #             other_xyz = cur_coords[:, 1:][other_indices].view(1, -1, 3).contiguous()
        #             other_indices_selected_index = pointnet2_utils.farthest_point_sample(other_xyz, self.candidate_points[1]).squeeze(0)
        #             other_indices_selected = other_indices[other_indices_selected_index.long()]
        #         else:
        #             index = [i for i in range(len(other_indices))]
        #             np.random.shuffle(index)
        #             other_indices_selected_index = index[:self.candidate_points[1]]
        #             other_indices_selected = other_indices[other_indices_selected_index]
        #         indices_selected = torch.cat([top_indices_selected.squeeze(0), other_indices_selected.squeeze(0)], dim=0)
        #     else:
        #         topk_scores, indices_selected = cur_candidate_score.topk(self.candidate_points[1]*2, dim=0, largest=True, sorted=True)
        #
        #     # topk_scores, indices_selected = score_for_selected.topk(self.candidate_points_all, dim=0, largest=True,
        #     #                                                          sorted=True)
        #     # cur_features = point_features_end[cur_mask]
        #     # selected_features = cur_features[indices_selected]
        #
        #     selected_coords = cur_coords[indices_selected]
        #     cur_centers = center_preds[cur_mask]
        #     selected_centers = cur_centers[indices_selected]
        #
        #     selected_candidate_fg_score = cur_candidate_fg_score[indices_selected]
        #     selected_candidate_score = cur_candidate_score[indices_selected]
        #
        #     candidate_coords.append(selected_coords)
        #     candidate_score.append(selected_candidate_score)
        #     # candidate_features.append(selected_features)
        #     candidate_fg_score.append(selected_candidate_fg_score)
        #     candidate_centers.append(selected_centers)
        #
        # candidate_coords = torch.cat(candidate_coords, dim=0)
        # # candidate_features = torch.cat(candidate_features, dim=0)
        # candidate_fg_score = torch.cat(candidate_fg_score, dim=0)
        # candidate_score = torch.cat(candidate_score, dim=0)
        # candidate_centers = torch.cat(candidate_centers, dim=0)

        candidate_coords = point_coords_with_offset
        candidate_fg_score = scores
        candidate_score = candidate_preds
        candidate_centers = center_preds
        # local_global_scale_candidate = self.bev_to_points(
        #     keypoints=candidate_coords,
        #     local_features=local_features.view(B, -1, 10, H, W),
        #     global_features=global_features,
        #     batch_size=batch_size,
        #     bev_stride=encoded_spconv_tensor_stride,
        #     z_stride=4,
        # )
        local_global_scale_candidate1 = self.bev_to_points(
            keypoints=candidate_coords,
            local_features=local_features1.view(B1, -1, 10, H1, W1),
            global_features=global_features1,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
            z_stride=4,
        )
        local_global_scale_candidate2 = self.bev_to_points(
            keypoints=candidate_coords,
            local_features=local_features2.view(B2, -1, 10, H2, W2),
            global_features=global_features2,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride * 2,
            z_stride=4,
        )
        local_global_scale_candidate1 = self.center_point_features_scale1(local_global_scale_candidate1)
        local_global_scale_candidate2 = self.center_point_features_scale2(local_global_scale_candidate2)

        candidate_center_features = torch.cat([
            local_global_scale_candidate1,
            local_global_scale_candidate2], dim=-1)

        point_object_statistic_features = candidate_center_features.new_zeros(candidate_center_features.shape)
        if not self.training or (batch_dict['accumulated_iter'] is not None and batch_dict['accumulated_iter']>927):
            with torch.no_grad():
                for i in range(3):
                    cur_class_mask = class_idx==i
                    cur_class_features = candidate_center_features[cur_class_mask]
                    if len(cur_class_features)>0:
                        if self.training:
                            max_cur_class_features = torch.mean(cur_class_features, dim=0)
                            if batch_dict['accumulated_iter'] == 928:
                                self.object_statistic_features[i, :] =  max_cur_class_features
                            else:
                                temp = self.object_statistic_features[i, :]*0.7 + max_cur_class_features*0.3
                                self.object_statistic_features[i, :] = temp
                        point_object_statistic_features[cur_class_mask] =  self.object_statistic_features[i, :]

        # fg_xyz = point_coords[:, 1:]
        # candidate_xyz = candidate_coords[:, 1:]
        #
        # fg_xyz = fg_xyz.view(batch_size, -1, 3).contiguous()
        # candidate_xyz = candidate_xyz.view(batch_size, -1, 3).contiguous()
        #
        # fg_features = point_features_end.view(batch_size, -1, point_features_end.shape[-1]).\
        #     permute(0, 2, 1).contiguous()
        #
        # l_xyz, l_features = [fg_xyz], [fg_features]
        # for i in range(len(self.SA_modules)):
        #     _, local_features = self.SA_modules[i](
        #         xyz=l_xyz[i], features=l_features[i], new_xyz=candidate_xyz,
        #     )
        # local_features = local_features.permute(0, 2, 1).contiguous().\
        #     view(batch_size*(self.candidate_points[0]+self.candidate_points[1]), -1)
        # candidate_center_features = torch.cat([candidate_center_features, local_features], dim=-1)
        # candidate_center_features = self.candidate_features(candidate_center_features)
        # local_features_list = [local_features, local_features1, local_features2]
        # global_features_list  = [global_features, global_features1, global_features2]

        candidate_center_features = torch.cat([candidate_center_features, point_object_statistic_features], dim=-1)

        batch_dict['encoded_bev_features'] = bev_list
        batch_dict['encoded_point_features'] = point_features_end
        batch_dict['fg_preds'] = fg_preds
        # batch_dict['point_corner_preds'] = corner_preds
        batch_dict['point_center_preds'] = center_preds
        batch_dict['point_candidate_preds'] = candidate_preds
        batch_dict['scores_fg'] = candidate_fg_score
        batch_dict['candidate_score'] = candidate_score
        batch_dict['candidate_coords'] = candidate_coords
        batch_dict['candidate_points_all'] = self.candidate_points_all
        batch_dict['candidate_features'] = candidate_center_features
        # batch_dict['local_features_list'] = local_features_list
        # batch_dict['global_features_list'] = global_features_list
        return batch_dict


