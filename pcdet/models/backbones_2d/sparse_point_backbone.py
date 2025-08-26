# -*- coding: utf-8 -*- 
# @Time : 2021/11/29 下午3:40 
# @Author : Peng Hao 
# @File : VoxelPointCross.py
import torch
import numpy as np
from torch import nn
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
# from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...utils.spconv_utils import spconv
from ...utils import common_utils, box_utils, voxel_aggregation_utils
from functools import partial
import torch.nn.functional as F
# from mayavi import mlab
# CUDA_LAUNCH_BLOCKING=1

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, active=True):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    if active:
        m = spconv.SparseSequential(
            conv,
            norm_fn(out_channels),
            nn.ReLU(),
        )
    else:
        m = spconv.SparseSequential(
            conv,
            norm_fn(out_channels),
        )

    return m


class SparsePointBackbone(nn.Module):
    def __init__(self, model_cfg,  input_channels, voxel_size, point_cloud_range, backbone_channels):
        super(SparsePointBackbone, self).__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        fg_candidate_points = self.model_cfg.FG_CORNER_POINTS
        self.point_num_list = fg_candidate_points
        self.trans_range = self.model_cfg.MAX_TRANSLATION_RANGE
        self.pts_num_pre = self.model_cfg.PTS_NUM_PRE
        pts_num_sample = self.model_cfg.PTS_NUM_SAMPLE
        self.pts_num_sample_fir = pts_num_sample[0]
        self.pts_num_sample_sec = pts_num_sample[1]
        # self.pts_num_sample_tir = pts_num_sample[2]
        self.num_class = self.model_cfg.N_CLS

        # channel_in = 128
        # mlps = self.model_cfg.FG_SA_CONFIG.MLPS.copy()
        # channel_out = 0
        # for idx in range(mlps.__len__()):
        #     mlps[idx] = [channel_in] + mlps[idx]
        #     channel_out += mlps[idx][-1]
        #
        # self.FG_SA_layer = pointnet2_modules.PointnetSAModuleMSG(
        #     npoint=self.model_cfg.FG_SA_CONFIG.NPOINTS,
        #     radii=self.model_cfg.FG_SA_CONFIG.RADIUS,
        #     nsamples=self.model_cfg.FG_SA_CONFIG.NSAMPLE,
        #     mlps=mlps,
        #     use_xyz=self.model_cfg.FG_SA_CONFIG.get('USE_XYZ', True),
        # )
        #
        # channel_in = 128
        # mlps = self.model_cfg.C_SA_CONFIG.MLPS.copy()
        # channel_out = 0
        # for idx in range(mlps.__len__()):
        #     mlps[idx] = [channel_in] + mlps[idx]
        #     channel_out += mlps[idx][-1]
        #
        # self.C_SA_layer = pointnet2_modules.PointnetSAModuleMSG(
        #     npoint=self.model_cfg.C_SA_CONFIG.NPOINTS,
        #     radii=self.model_cfg.C_SA_CONFIG.RADIUS,
        #     nsamples=self.model_cfg.C_SA_CONFIG.NSAMPLE,
        #     mlps=mlps,
        #     use_xyz=self.model_cfg.C_SA_CONFIG.get('USE_XYZ', True),
        # )

        self.pool_cfg = model_cfg.POINT_GRID_POOL
        layer_cfg = self.pool_cfg.POOL_LAYERS
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = layer_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=layer_cfg[src_name].QUERY_RANGES,
                nsamples=layer_cfg[src_name].NSAMPLE,
                radii=layer_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=layer_cfg[src_name].POOL_METHOD,
            )
            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])

        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # block = post_act_block

        # backbone
        # self.sparse_neck = spconv.SparseSequential(
        #     block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_neck'),
        #     block(128, 256, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_neck'),
        #     block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_neck'),
        #     block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_neck'),
        # )
        # self.sparse_neck_local = spconv.SparseSequential(
        #     block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='sparse_neck_local', conv_type='subm'),
        #     block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='sparse_neck_local', conv_type='subm'),
        # )
        # self.dense_neck_global = nn.Sequential(
        #     nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )

        # self.sparse_scale1 = spconv.SparseSequential(
        #     block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale1'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale1'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale1'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale1'),
        #     block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale1'),
        # )
        # self.sparse_scale2 = spconv.SparseSequential(
        #     block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale2'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale2'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale2'),
        #     block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale2'),
        #     block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale2'),
        # )
        # self.sparse_scale_transform12 = spconv.SparseSequential(
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='scale_transform12', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale_transform12'),
        # )
        # self.sparse_scale_transform = spconv.SparseSequential(
        #     block(64, 64, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='sparse_scale_transform', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='sparse_scale_transform'),
        # )

        # features for multi

        # self.sparse_scale1_out = spconv.SparseSequential(
        #     block(64, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='scale1_fg1', conv_type='subm'),
        #     block(32, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='scale1_fg2', conv_type='subm'),
        # )
        # self.sparse_scale2_out = spconv.SparseSequential(
        #     block(64, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='scale2_fg1', conv_type='subm'),
        #     block(32, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='scale2_fg2', conv_type='subm'),
        # )
        # self.sparse_scale3_out = spconv.SparseSequential(
        #     block(64, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='scale3_fg1', conv_type='subm'),
        #     block(32, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='scale3_fg2', conv_type='subm'),
        # )

        # self.dense_scale1_global = nn.Sequential(
        #     nn.Conv2d(in_channels=640, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )
        # self.dense_scale2_global = nn.Sequential(
        #     nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )
        # self.dense_scale3_global = nn.Sequential(
        #     nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )

        # self.offset_local_weight = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=16, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=16, out_features=64, bias=False),
        #     nn.Sigmoid()
        # )
        # self.offset_global_weight = nn.Sequential(
        #     nn.Linear(in_features=2, out_features=16, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=16, out_features=64, bias=False),
        #     nn.Sigmoid()
        # )

        # ==============fg==============
        self.fg_pred_layer = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=True),
        )

        self.register_buffer('object_statistic_features', torch.zeros(3, 128))
        # self.pos_coder_for_statistic_cls = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=128, bias=False),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # )
        # self.features_for_statistic_reg = nn.Sequential(
        #     nn.Linear(in_features=32, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # )
        # self.pos_coder_for_statistic_reg = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # )

        # ===========center_preds and scores================
        self.center_pred_layer = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=True),
        )

        # multi features dim X --> 128
        self.features_raw = nn.Sequential(
            nn.Linear(in_features=192, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        self.pos_raw = nn.Sequential(
            nn.Linear(in_features=3, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.features_fg = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        self.features_center = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        # self.multi_features_center_score = nn.Sequential(
        #     nn.Linear(in_features=192, out_features=128, bias=False),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=256, bias=False),
        #     nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=128, bias=False),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        # )

        self.features_cls = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        self.features_reg = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Linear(in_features=128, out_features=128, bias=False),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        # self.cls_layers = nn.Sequential(
        #     nn.Conv1d(in_channels=128+96, out_channels=64, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=True),
        # )

        self.cls_block = nn.ModuleList()
        for i in range(self.num_class):
            self.cls_block.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, bias=True),
                )
            )

        self.temp_features = nn.Sequential(
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.Sigmoid(),
        )
        self.num_voxel_neck_features = 576
        self.num_point_features = 128

        self.init_weights()

    # def init_weights(self):
    #     pi = 0.01
    #     nn.init.normal_(self.fg_pred_layer[3].weight, mean=0, std=0.001)
    #     nn.init.constant_(self.fg_pred_layer[3].bias, -np.log((1 - pi) / pi))
    #
    #     # nn.init.normal_(self.fg_bias[2].weight, mean=0, std=0.001)
    #     # nn.init.constant_(self.fg_bias[2].bias, -np.log((1 - pi) / pi))
    #     # nn.init.normal_(self.fg_weight[2].weight, mean=0, std=0.001)
    #
    #     # nn.init.normal_(self.center_bias[2].weight, mean=0, std=0.001)
    #     # nn.init.constant_(self.center_bias[2].bias, 0)
    #     # nn.init.normal_(self.center_weight[2].weight, mean=0, std=0.001)
    #     nn.init.normal_(self.center_pred_layer[3].weight, mean=0, std=0.001)
    #     nn.init.constant_(self.center_pred_layer[3].bias, 0)
    #
    #     # nn.init.normal_(self.center_score_bias[2].weight, mean=0, std=0.001)
    #     # nn.init.constant_(self.center_score_bias[2].bias, 0)
    #     # nn.init.normal_(self.center_score_weight[2].weight, mean=0, std=0.001)
    #
    #     # nn.init.normal_(self.cls_bias[2].weight, mean=0, std=0.001)
    #     # nn.init.constant_(self.cls_bias[2].bias, -np.log((1 - pi) / pi))
    #     # nn.init.normal_(self.cls_weight[2].weight, mean=0, std=0.001)
    #     nn.init.normal_(self.cls_layers[3].weight, mean=0, std=0.001)
    #     nn.init.constant_(self.cls_layers[3].bias, -np.log((1 - pi) / pi))

    def init_weights(self, weight_init='xavier'):
        pi = 0.01
        for i in range(self.num_class):
            nn.init.constant_(self.cls_block[i][3].bias, -np.log((1 - pi) / pi))
        # nn.init.constant_(self.cls_layers[3].bias, -np.log((1 - pi) / pi))
        nn.init.constant_(self.fg_pred_layer[3].bias, -np.log((1 - pi) / pi))

        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # def bev_to_points(self, keypoints, local_features, global_features, batch_size, bev_stride, z_stride):
    #     x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
    #     y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
    #     x_idxs = x_idxs / bev_stride
    #     y_idxs = y_idxs / bev_stride
    #     batch_idxs = keypoints[:, 0]
    #     n_points = len(keypoints)
    #     z_idxs = (keypoints[:, 3] - self.point_cloud_range[2]) / self.voxel_size[2]
    #     z_idxs = z_idxs / z_stride
    #
    #     n_local_channels = local_features.shape[1]
    #     n_global_channels = global_features.shape[1]
    #     point_local_global_features = local_features.new_zeros((n_points, (n_local_channels+n_global_channels)))
    #     for k in range(batch_size):
    #         batch_mask = batch_idxs == k
    #         cur_x_idxs = x_idxs[batch_mask]
    #         cur_y_idxs = y_idxs[batch_mask]
    #         cur_z_idxs = z_idxs[batch_mask]
    #
    #         cur_local_features = local_features[k].permute(2, 3, 1, 0)  # (H, W, C)
    #         im, x, y, z = cur_local_features, cur_x_idxs, cur_y_idxs, cur_z_idxs
    #         cur_global_features = global_features[k].permute(1, 2, 0)
    #
    #         x0 = torch.floor(x).long()
    #         y0 = torch.floor(y).long()
    #         z0 = torch.floor(z).long()
    #         x1 = x0 + 1
    #         y1 = y0 + 1
    #         z1 = z0 + 1
    #
    #         x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    #         x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    #         y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    #         y1 = torch.clamp(y1, 0, im.shape[0] - 1)
    #         z0 = torch.clamp(z0, 0, im.shape[2] - 1)
    #         z1 = torch.clamp(z1, 0, im.shape[2] - 1)
    #
    #         # u, v, w = (x - x0).unsqueeze(-1), (y - y0).unsqueeze(-1), (z - z0).unsqueeze(-1)
    #
    #         c_000 = cur_local_features[y0, x0, z0]
    #         c_001 = cur_local_features[y0, x0, z1]
    #         c_010 = cur_local_features[y1, x0, z0]
    #         c_011 = cur_local_features[y1, x0, z1]
    #         c_100 = cur_local_features[y0, x1, z0]
    #         c_101 = cur_local_features[y0, x1, z1]
    #         c_110 = cur_local_features[y1, x1, z0]
    #         c_111 = cur_local_features[y1, x1, z1]
    #         # c_xyz = (1.0 - u) * (1.0 - v) * (1.0 - w) * c_000 + \
    #         #         (1.0 - u) * (1.0 - v) * (w) * c_001 + \
    #         #         (1.0 - u) * (v) * (1.0 - w) * c_010 + \
    #         #         (1.0 - u) * (v) * (w) * c_011 + \
    #         #         (u) * (1.0 - v) * (1.0 - w) * c_100 + \
    #         #         (u) * (1.0 - v) * (w) * c_101 + \
    #         #         (u) * (v) * (1.0 - w) * c_110 + \
    #         #         (u) * (v) * (w) * c_111
    #         offset_x0 = (x - x0).view(-1, 1)
    #         offset_x1 = (x - x1).view(-1, 1)
    #         offset_y0 = (y - y0).view(-1, 1)
    #         offset_y1 = (y - y1).view(-1, 1)
    #         offset_z0 = (z - z0).view(-1, 1)
    #         offset_z1 = (z - z1).view(-1, 1)
    #
    #         offset_local = [offset_x0, offset_y0, offset_z0,
    #                         offset_x0, offset_y0, offset_z1,
    #                         offset_x0, offset_y1, offset_z0,
    #                         offset_x0, offset_y1, offset_z1,
    #                         offset_x1, offset_y0, offset_z0,
    #                         offset_x1, offset_y0, offset_z1,
    #                         offset_x1, offset_y1, offset_z0,
    #                         offset_x1, offset_y1, offset_z1]
    #         offset_local = torch.cat(offset_local, dim=-1).view(-1, 3)
    #         offset_local_weight = self.offset_local_weight(offset_local).view(-1, 8, 64)
    #         # offset_local_weight = torch.softmax(offset_local_weight, dim=1)
    #         offset_local_features = torch.cat([c_000, c_001, c_010, c_011, c_100, c_101, c_110, c_111], dim=-1)
    #         offset_local_features = offset_local_features.view(-1, 8, 64)
    #         offset_local_features = offset_local_features * offset_local_weight
    #         offset_local_features = torch.sum(offset_local_features, dim=1)
    #
    #         Ia = cur_global_features[y0, x0]
    #         Ib = cur_global_features[y1, x0]
    #         Ic = cur_global_features[y0, x1]
    #         Id = cur_global_features[y1, x1]
    #
    #         offset_global = [offset_x0, offset_y0,
    #                          offset_x0, offset_y1,
    #                          offset_x1, offset_y0,
    #                          offset_x1, offset_y1]
    #         offset_global = torch.cat(offset_global, dim=-1).view(-1, 2)
    #         offset_global_weight = self.offset_global_weight(offset_global).view(-1, 4, 64)
    #         # offset_global_weight = torch.softmax(offset_global_weight, dim=1)
    #         offset_global_features = torch.cat([Ia, Ib, Ic, Id], dim=-1)
    #         offset_global_features = offset_global_features.view(-1, 4, 64)
    #         offset_global_features = offset_global_features * offset_global_weight
    #         offset_global_features = torch.sum(offset_global_features, dim=1)
    #
    #         # wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    #         # wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    #         # wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    #         # wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    #         # ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
    #         #     torch.t(Id) * wd)
    #
    #         # add max features TBD
    #         # local_max = torch.cat([c_000.unsqueeze(1), c_001.unsqueeze(1), c_010.unsqueeze(1), c_011.unsqueeze(1),
    #         #                        c_100.unsqueeze(1), c_101.unsqueeze(1), c_110.unsqueeze(1), c_111.unsqueeze(1)], dim=1)
    #         # local_max = torch.max(local_max, dim=1)[0]
    #         # global_max = torch.cat([Ia.unsqueeze(1), Ib.unsqueeze(1), Ic.unsqueeze(1), Id.unsqueeze(1)], dim=1)
    #         # global_max = torch.max(global_max, dim=1)[0]
    #
    #         cur_local_global = torch.cat([offset_local_features, offset_global_features], dim=-1)
    #         point_local_global_features[batch_mask] = cur_local_global
    #
    #     return point_local_global_features

    # def get_multi_features(self, batch_size, point_coords,
    #                        scale1, global1, bev_stride1, z_stride1,
    #                        scale2, global2, bev_stride2, z_stride2,
    #                        # scale3, global3, bev_stride3, z_stride3,
    #                        ):
    #     features_scale1 = self.bev_to_points(
    #         keypoints=point_coords,
    #         local_features=scale1,
    #         global_features=global1,
    #         batch_size=batch_size,
    #         bev_stride=bev_stride1,
    #         z_stride=z_stride1,
    #     )
    #     features_scale2 = self.bev_to_points(
    #         keypoints=point_coords,
    #         local_features=scale2,
    #         global_features=global2,
    #         batch_size=batch_size,
    #         bev_stride=bev_stride2,
    #         z_stride=z_stride2,
    #     )
    #     # features_scale3 = self.bev_to_points(
    #     #     keypoints=point_coords,
    #     #     local_features=scale3,
    #     #     global_features=global3,
    #     #     batch_size=batch_size,
    #     #     bev_stride=bev_stride3,
    #     #     z_stride=z_stride3,
    #     # )
    #
    #     return torch.cat([features_scale1, features_scale2], dim=-1)

    def point_grid_pool(self, batch_dict, centroids_all, overlapping_voxel_feature_indices_nonempty_all,
                        overlapping_voxel_feature_nonempty_mask_all):
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
        point_coords = batch_dict['pool_coords']

        point_grid_coords = point_coords.clone()
        point_grid_coords[:, 1] = (point_grid_coords[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        point_grid_coords[:, 2] = (point_grid_coords[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        point_grid_coords[:, 3] = (point_grid_coords[:, 3] - self.point_cloud_range[2]) / self.voxel_size[2]
        # point_grid_coords = point_grid_coords.int()
        point_grid_cnt = point_coords.new_zeros(batch_size).int()
        for i in range(batch_size):
            point_grid_cnt[i] = (point_grid_coords[:, 0] == i).sum()
        pooled_features_list = []
        density_list = []
        # grouped_xyz_scale_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            cur_stride = torch.tensor(cur_stride, device=point_coords.device)
            cur_centroids = centroids_all[src_name]
            cur_overlapping_indices_nonempty = overlapping_voxel_feature_indices_nonempty_all[k]
            cur_overlapping_nonempty_mask = overlapping_voxel_feature_nonempty_mask_all[k]

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
            cur_roi_grid_coords = point_grid_coords[:, 1:] / cur_stride
            cur_roi_grid_coords = torch.cat([point_grid_coords[:, 0:1], cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            # cur_voxel_xyz_masked = cur_voxel_xyz[cur_overlapping_indices_nonempty]
            cur_voxel_xyz[cur_overlapping_indices_nonempty] = cur_centroids[:, 1:][cur_overlapping_nonempty_mask]

            pooled_features, cur_scale_density = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=point_coords[:, 1:].contiguous().view(-1, 3),
                new_xyz_batch_cnt=point_grid_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )
            density_list.append(cur_scale_density)
            # grouped_xyz_scale_list.append(grouped_xyz_scale[0])

            pooled_features_list.append(pooled_features)
        pooled_features = torch.cat(pooled_features_list, dim=-1)
        density_score = torch.cat(density_list, dim=-1)
        density_score = torch.mean(density_score, dim=-1)
        return pooled_features, density_score

    def get_point_feature_based_sparse(self, sp_tensor, sp_stride, point_coords):

        sp_stride = torch.tensor(sp_stride, device=point_coords.device)
        z_size, y_size, x_size = sp_tensor.spatial_shape

        voxel2pinds = common_utils.generate_voxel2pinds(sp_tensor)
        point_grid_coords = point_coords.clone()
        point_grid_coords[:, 1] = (point_grid_coords[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        point_grid_coords[:, 2] = (point_grid_coords[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        point_grid_coords[:, 3] = (point_grid_coords[:, 3] - self.point_cloud_range[2]) / self.voxel_size[2]
        point_grid_coords[:, 1:] = point_grid_coords[:, 1:] / sp_stride

        point_grid_coords = point_grid_coords.long()
        point_grid_coords[:, 1] = torch.clamp(point_grid_coords[:, 1], min=0, max=x_size - 1)
        point_grid_coords[:, 2] = torch.clamp(point_grid_coords[:, 2], min=0, max=y_size - 1)
        point_grid_coords[:, 3] = torch.clamp(point_grid_coords[:, 3], min=0, max=z_size - 1)

        temp = point_grid_coords[:, (0, 3, 2, 1)]
        slices = [temp[:, i] for i in range(point_grid_coords.shape[-1])]
        voxel_xyz = common_utils.get_voxel_centers(
            sp_tensor.indices[:, 1:4],
            downsample_times=sp_stride,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        pindex = voxel2pinds[slices].long()
        point_features = sp_tensor.features[pindex].contiguous()
        point_voxel_center = voxel_xyz[pindex].contiguous()
        point_offset = point_coords[:, 1:].clone() - point_voxel_center
        empty_mask = pindex < 0
        if torch.any(empty_mask):
            point_features[empty_mask] = 0.0
            point_offset[empty_mask] = 0.0

        features_raw = self.features_raw(point_features.contiguous())
        pos_raw = self.pos_raw(point_offset.contiguous())
        features_raw_for_fg = self.relu(features_raw + pos_raw)

        return features_raw_for_fg, point_offset

    def show_point_score(self, gt, batch_size, point_coords, cls_scores):
        for i in range(0, batch_size):
            mlab.figure(bgcolor=(0, 0, 0))
            mask = point_coords[:, 0] == i
            cur_batch_point_coords = point_coords[mask]
            cur_x = cur_batch_point_coords[:, 1].cpu().numpy()
            cur_y = cur_batch_point_coords[:, 2].cpu().numpy()
            cur_z = cur_batch_point_coords[:, 3].cpu().numpy()
            # val_depict = (torch.sigmoid(val[mask])).detach().cpu().numpy()
            val_depict = (cls_scores[mask]).detach().cpu().numpy()
            cur_batch_gt = gt[i]
            point_corner = box_utils.boxes_to_corners_3d(boxes3d=cur_batch_gt[:, :-1])
            for index in range(len(point_corner)):
                cur_gt = point_corner[index].cpu().numpy()
                cur_gt_x = cur_gt[:, 0]
                cur_gt_y = cur_gt[:, 1]
                cur_gt_z = cur_gt[:, 2]
                mlab.plot3d(cur_gt_x, cur_gt_y, cur_gt_z,)

            mlab.points3d(cur_x, cur_y, cur_z, val_depict, scale_factor=0.25, scale_mode='none', mode='sphere',
                          line_width=1, colormap='jet')
            mlab.colorbar()
            mlab.show()

    def forward(self, batch_dict):

        point_coords = batch_dict['points'][:, :4]
        batch_size = batch_dict['batch_size']

        # features backbone
        # sparse_tensor_scale1 = batch_dict['multi_scale_3d_features']['scale2']
        # sparse_tensor_scale2 = batch_dict['multi_scale_3d_features']['scale3']
        # sparse_tensor_scale3 = batch_dict['multi_scale_3d_features']['scale4']
        # sparse_scale1_features = self.sparse_scale1(sparse_tensor_scale1)
        # sparse_scale2_features = self.sparse_scale_transform12(sparse_scale1_features)
        # sparse_scale2_features = sparse_scale2_features.replace_feature(sparse_scale2_features.features +
        #                                                                 sparse_tensor_scale2.features)
        # sparse_scale2_features = self.sparse_scale2(sparse_scale2_features)
        # sparse_scale3_features = self.sparse_scale_transform23(sparse_scale2_features)
        # sparse_scale3_features = sparse_scale3_features.replace_feature(sparse_scale3_features.features +
        #                                                                 sparse_tensor_scale3.features)
        # sparse_scale3_features = self.sparse_scale3(sparse_scale3_features)
        #
        # sparse_scale1_features = self.sparse_scale1_out(sparse_scale1_features)
        # sparse_scale2_features = self.sparse_scale2_out(sparse_scale2_features)
        # sparse_scale3_features = self.sparse_scale3_out(sparse_scale3_features)
        # sparse_scale1_features = self.sparse_scale1(sparse_tensor_scale1)
        # sparse_scale2_features = self.sparse_scale2(sparse_tensor_scale2)
        # sparse_scale3_features = self.sparse_scale3(sparse_tensor_scale3)
        # sparse_scale1_features = self.sparse_scale1_out(sparse_scale1_features)
        # sparse_scale2_features = self.sparse_scale2_out(sparse_scale2_features)
        # sparse_scale3_features = self.sparse_scale3_out(sparse_scale3_features)

        # 16384-->4096
        sp_tensor = batch_dict['multi_scale_3d_features']['multi_scale']
        sp_stride = batch_dict['multi_scale_3d_strides']['multi_scale']

        # indices downsampled 16384->4096
        point_xyz_for_downsample = point_coords[:, 1:].contiguous().view(batch_size, -1, 3)
        _, num_point_per_scene, _ = point_xyz_for_downsample.shape
        downsample_indices_selected = pointnet2_utils.farthest_point_sample(
            point_xyz_for_downsample, self.point_num_list[0])
        for i in range(batch_size):
            downsample_indices_selected[i] = downsample_indices_selected[i] + i * num_point_per_scene
        downsample_indices_selected = downsample_indices_selected.view(-1).long()

        # info update
        point_coords = point_coords[downsample_indices_selected]
        num_point = len(point_coords)
        num_point_per_scene = self.point_num_list[0]
        batch_dict['raw_points_bxyz'] = point_coords
        batch_dict['candidate_points_bxyz'] = point_coords
        batch_dict['pool_coords'] = point_coords

        centroids_all, centroid_voxel_idxs_all = voxel_aggregation_utils.get_centroids_per_voxel_layer(
            batch_dict['points'][:, :4],
            self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS,
            batch_dict['multi_scale_3d_strides'],
            self.voxel_size,
            self.point_cloud_range)
        overlapping_voxel_feature_indices_nonempty_all = []
        overlapping_voxel_feature_nonempty_mask_all = []
        for feature_location in self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS:
            # centroids = centroids_all[feature_location][:, :4]
            centroid_voxel_idxs = centroid_voxel_idxs_all[feature_location]
            x_conv = batch_dict['multi_scale_3d_features'][feature_location]
            overlapping_voxel_feature_indices_nonempty, overlapping_voxel_feature_nonempty_mask = \
                voxel_aggregation_utils.get_nonempty_voxel_feature_indices(centroid_voxel_idxs, x_conv)
            overlapping_voxel_feature_indices_nonempty_all.append(overlapping_voxel_feature_indices_nonempty)
            overlapping_voxel_feature_nonempty_mask_all.append(overlapping_voxel_feature_nonempty_mask)

            # if self.model_cfg.VOXEL_AGGREGATION.get('USE_EMPTY_VOXELS'):
            #     voxel_points = torch.zeros((x_conv.features.shape[0], centroids.shape[-1]), dtype=centroids.dtype,
            #                                device=centroids.device)
            #     voxel_points[overlapping_voxel_feature_indices_nonempty] = centroids[
            #         overlapping_voxel_feature_nonempty_mask]
            #
            #     # Set voxel center
            #     empty_mask = torch.ones((x_conv.features.shape[0]), dtype=torch.bool, device=centroids.device)
            #     empty_mask[overlapping_voxel_feature_indices_nonempty] = False
            #     cur_coords = x_conv.indices[empty_mask]
            #     xyz = common_utils.get_voxel_centers(
            #         cur_coords[:, 1:4],
            #         downsample_times=batch_dict['multi_scale_3d_strides'][feature_location],
            #         voxel_size=self.voxel_size,
            #         point_cloud_range=self.point_cloud_range
            #     )
            #     cur_coords = cur_coords.type(torch.cuda.FloatTensor)
            #     cur_coords[:, 1:4] = xyz
            #     voxel_points[empty_mask] = cur_coords
            #
            #     point_features[feature_location] = x_conv.features
            #     point_coords[feature_location] = voxel_points
            # else:
            #     x_conv_features = torch.zeros((centroids.shape[0], x_conv.features.shape[-1]),
            #                                   dtype=x_conv.features.dtype, device=centroids.device)
            #     x_conv_features[overlapping_voxel_feature_nonempty_mask] = x_conv.features[
            #         overlapping_voxel_feature_indices_nonempty]
            #
            #     point_features[feature_location] = x_conv_features[overlapping_voxel_feature_nonempty_mask]
            #     point_coords[feature_location] = centroids[overlapping_voxel_feature_nonempty_mask]
        # centroids_all, centroid_voxel_idxs_all = voxel_aggregation_utils.get_centroids_per_voxel_layer(
        #     batch_dict['points'],
        #     self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS,
        #     batch_dict['multi_scale_3d_strides'],
        #     self.voxel_size,
        #     self.point_cloud_range)

        # fg and center prediction
        features_single_raw, _ = self.get_point_feature_based_sparse(
            sp_tensor=sp_tensor, sp_stride=sp_stride, point_coords=point_coords)
        features_multi_raw, density_score_raw = self.point_grid_pool(
            batch_dict, centroids_all, overlapping_voxel_feature_indices_nonempty_all,
        overlapping_voxel_feature_nonempty_mask_all)
        features_raw = torch.cat([features_single_raw, features_multi_raw], dim=-1)
        features_raw_fg = self.features_fg(features_raw)

        features_raw_fg = features_raw_fg.view(batch_size, num_point_per_scene, 128)
        features_raw_fg = features_raw_fg.permute(0, 2, 1).contiguous()
        fg_preds = self.fg_pred_layer(features_raw_fg)
        fg_preds = fg_preds.permute(0, 2, 1).contiguous().view(num_point, 3)
        fg_preds_norm = torch.sigmoid(fg_preds)
        fg_score, fg_score_idx = torch.max(fg_preds_norm, dim=-1)
        fg_score_for_statistic, fg_score_idx_for_statistic = fg_score, fg_score_idx
        features_for_statistic = features_raw_fg.permute(0, 2, 1).contiguous().view(num_point, -1)

        # self.show_point_score(batch_size=4, gt=batch_dict['gt_boxes'], point_coords=point_coords, cls_scores=fg_score)



        # key point selection
        pts_depth = torch.norm(point_coords[:, 1:4], p=2, dim=1)
        # pts_depth = pts_depth.view(batch_size, -1)
        pts_near_mask = pts_depth < 40

        point_xyz_for_key1 = point_coords[:, 1:].contiguous().view(batch_size, -1, 3)
        select_matrix = point_xyz_for_key1.new_zeros(num_point)
        select_idx_matrix = torch.arange(0, num_point, step=1, device=select_matrix.device)

        # key_part3_indices_selected = pointnet2_utils.farthest_point_sample(
        #     point_xyz_for_key3, self.pts_num_sample_tir)
        # for i in range(batch_size):
        #     key_part3_indices_selected[i] = key_part3_indices_selected[i] + i * num_point_per_scene
        # key_part3_indices_selected = key_part3_indices_selected.view(-1).long()
        # select_matrix[key_part3_indices_selected] = 3
        # select_matrix_mask = select_matrix == 3

        # num_point_per_scene1 = num_point_per_scene - self.pts_num_sample_tir
        # point_xyz_for_key1 = point_coords[:, 1:][~select_matrix_mask].contiguous().view(
        #     batch_size, num_point_per_scene1, 3)

        weights_for_key1 = fg_score.view(batch_size, -1)
        key_part1_indices_selected = pointnet2_utils.furthest_point_sample_weights(
            point_xyz_for_key1, weights_for_key1, self.pts_num_sample_fir)
        for i in range(batch_size):
            key_part1_indices_selected[i] = key_part1_indices_selected[i] + i * num_point_per_scene
        key_part1_indices_selected = key_part1_indices_selected.view(-1).long()
        select_idx1 = select_idx_matrix[key_part1_indices_selected]
        select_matrix[select_idx1] = 1
        select_matrix_mask = select_matrix == 1
        # self.show_point_score(batch_size=4, gt=batch_dict['gt_boxes'],
        #                       point_coords=point_coords[key_part1_indices_selected],
        #                       cls_scores=fg_score[key_part1_indices_selected])

        num_point_per_scene2 = num_point_per_scene - self.pts_num_sample_fir
        point_xyz_for_key2 = point_coords[:, 1:][~select_matrix_mask].contiguous().view(batch_size, num_point_per_scene2, 3)
        weights_for_key2 = fg_score.clone()
        weights_for_key2[pts_near_mask.view(-1)] = 0.0
        weights_for_key2 = weights_for_key2[~select_matrix_mask].view(batch_size, -1)
        key_part2_indices_selected = pointnet2_utils.furthest_point_sample_weights(
            point_xyz_for_key2, weights_for_key2, self.pts_num_sample_sec)
        for i in range(batch_size):
            key_part2_indices_selected[i] = key_part2_indices_selected[i] + i * num_point_per_scene2
        key_part2_indices_selected = key_part2_indices_selected.view(-1).long()
        select_idx2 = select_idx_matrix[~select_matrix_mask][key_part2_indices_selected]
        select_matrix[select_idx2] = 2
        select_matrix_mask[select_idx2] = True

        # info update
        # self.show_point_score(batch_size=4, gt=batch_dict['gt_boxes'], point_coords=point_coords[select_idx2], cls_scores=fg_score[select_idx2])
        point_coords = point_coords[select_matrix_mask]
        num_point_per_scene = self.point_num_list[1]
        fg_preds_norm_key_points = fg_preds_norm[select_matrix_mask]

        # fg_score_idx_key_points = fg_score_idx[select_matrix_mask]
        pts_depth_key_points = pts_depth[select_matrix_mask]
        features_raw_fg_key_points = features_raw[select_matrix_mask]
        # select_matrix_key_points = select_matrix[select_matrix_mask]
        # select_matrix_key_points_sfps_mask = ~ (select_matrix_key_points==1)

        features_raw_center = self.features_center(features_raw_fg_key_points)

        features_raw_center = features_raw_center.view(batch_size, -1, features_raw_center.shape[-1])
        features_raw_center = features_raw_center.permute(0, 2, 1).contiguous()
        center_preds = self.center_pred_layer(features_raw_center)
        vote_translation_range = np.array(self.model_cfg.MAX_TRANSLATION_RANGE, dtype=np.float32)
        vote_translation_range = torch.from_numpy(vote_translation_range).cuda().unsqueeze(dim=0).unsqueeze(dim=-1)
        center_preds = torch.max(center_preds, -vote_translation_range)
        center_preds = torch.min(center_preds, vote_translation_range)
        center_preds = center_preds.permute(0, 2, 1).contiguous().view(-1, 3)

        # center_preds_key_points = center_preds

        # fg_score_key_points = fg_score[select_matrix_mask]
        batch_dict['point_coords'] = point_coords
        # self.show_point_score(batch_size=4, gt=batch_dict['gt_boxes'], point_coords=point_coords,
        #                       cls_scores=fg_score[select_matrix_mask])

        # get vote pts
        # offset_template = torch.tensor([[-1, 0, 0],
        #                                 [1, 0, 0],
        #                                 [0, -1, 0],
        #                                 [0, 1, 0],
        #                                 [0, 0, -1],
        #                                 [0, 0, 1],
        #                                 [0, 0, 0]], device=center_preds_key_points.device)
        # offset_factor = torch.tensor([[0.8, 0.8, 0.4]], device=center_preds_key_points.device)
        # offset = offset_template * offset_factor
        vote_xyz = point_coords[:, 1:] + center_preds
        # vote_xyz = vote_xyz.unsqueeze(1) + offset.unsqueeze(0)
        # batch_idx = point_coords[:, 0:1].repeat(1, 7).unsqueeze(-1)
        vote_coords = torch.cat([point_coords[:, 0:1], vote_xyz], dim=-1).view(-1, 4)
        batch_dict['vote_coords'] = vote_coords
        batch_dict['pool_coords'] = vote_coords

        vote_xyz = vote_xyz.view(-1, 3)
        num_point = len(vote_xyz)

        # get statistics
        start_iter = 464 * 0
        temp = vote_coords.new_zeros(size=(self.num_class, features_for_statistic.shape[-1]))

        if not self.training or (
                batch_dict['accumulated_iter'] is not None and batch_dict['accumulated_iter'] >= start_iter):
            for i in range(3):
                cur_class_mask = fg_score_idx_for_statistic == i
                cur_score_norm = fg_score_for_statistic[cur_class_mask]
                filter_score_mask = cur_score_norm >= 0.3
                cur_class_features = features_for_statistic[cur_class_mask][filter_score_mask]

                if len(cur_class_features) > 0:
                    max_cur_class_features = torch.max(cur_class_features, dim=0)[0]
                    mean_cur_class_features = torch.mean(cur_class_features, dim=0)
                    max_cur_class_features = max_cur_class_features - mean_cur_class_features
                    if self.training:
                        if batch_dict['accumulated_iter'] == start_iter:
                            temp[i] = max_cur_class_features
                            with torch.no_grad():
                                self.object_statistic_features[i, :] = temp[i]
                        else:
                            temp[i] = self.object_statistic_features[i, :] * 0.98 + max_cur_class_features
                            with torch.no_grad():
                                self.object_statistic_features[i, :] = temp[i]
                    else:
                        temp[i] = self.object_statistic_features[i, :] * 0.98 + max_cur_class_features

        features_single_point, vote_pos = self.get_point_feature_based_sparse(
            sp_tensor=sp_tensor, sp_stride=sp_stride, point_coords=vote_coords)
        features_multi_vote, density_score_vote = self.point_grid_pool(
            batch_dict, centroids_all, overlapping_voxel_feature_indices_nonempty_all,
            overlapping_voxel_feature_nonempty_mask_all)
        features_vote = torch.cat([features_single_point, features_multi_vote], dim=-1)
        temp_tag = self.temp_features(temp)
        features_for_cls = self.features_cls(features_vote)
        cls_res_list = []
        # temp_tag_for_cls = temp_tag.view(1, -1).repeat(num_point, 1)
        # features_for_cls_with_statistic = torch.cat([features_for_cls, temp_tag_for_cls], dim=-1)
        # features_for_cls_with_statistic = features_for_cls_with_statistic.view(
        #     batch_size, num_point_per_scene, -1).permute(0, 2, 1)
        # point_cls_preds = self.cls_layers(features_for_cls_with_statistic)
        # point_cls_preds = point_cls_preds.permute(0, 2, 1).contiguous().view(num_point, -1)
        for i in range(self.num_class):
            # cur_class_statistic = temp[i:(i+1), :].repeat(num_point, 1)
            cur_class_statistic = temp_tag[i:(i + 1), :]
            # features_for_cls_with_statistic = torch.cat([features_for_cls, cur_class_statistic], dim=-1)
            features_for_cls_with_statistic = features_for_cls*cur_class_statistic
            features_for_cls_with_statistic = features_for_cls_with_statistic.view(
                batch_size, num_point_per_scene, -1).permute(0, 2, 1)
            cur_cls_res = self.cls_block[i](features_for_cls_with_statistic)
            cur_cls_res = cur_cls_res.permute(0, 2, 1).contiguous().view(num_point, -1)
            cls_res_list.append(cur_cls_res)
        point_cls_preds = torch.cat(cls_res_list, dim=-1)

        statistic_extended_for_reg = features_for_statistic.new_zeros(num_point, temp_tag.shape[-1])
        # for i in range(3):
        #     cur_class_mask = fg_score_idx_key_points == i
        #     cur_score_norm = statistic_extended_for_cls[cur_class_mask]
        #     if len(cur_score_norm) > 0:
        #         statistic_extended_for_cls[cur_class_mask] = temp[i] / 100
        # vote features

        # pos_cls = self.pos_coder_for_statistic_cls(vote_pos)
        # statistic_extended_for_cls = self.relu(statistic_extended_for_cls + pos_cls)
        # features_for_cls = torch.cat([features_for_cls, statistic_extended_for_cls], dim=-1)

        point_cls_preds_norm = torch.sigmoid(point_cls_preds)
        point_cls_preds_score, point_cls_preds_idx = torch.max(point_cls_preds_norm, dim=-1)
        for i in range(3):
            cur_class_mask = point_cls_preds_idx == i
            cur_score_norm = statistic_extended_for_reg[cur_class_mask]
            if len(cur_score_norm) > 0:
                statistic_extended_for_reg[cur_class_mask] = temp_tag[i]

        # self.show_point_score(batch_size=4, gt=batch_dict['gt_boxes'], point_coords=vote_coords,
        #                       cls_scores=point_cls_preds_score)

        # features for fg
        # scale1_local = self.sparse_scale1_local(sparse_scale1_features)
        # dense_scale1_local = scale1_local.dense()
        # scale2_local = self.sparse_scale2_local(sparse_scale2_features)
        # dense_scale2_local = scale2_local.dense()
        # scale3_local = self.sparse_scale3_local(sparse_scale3_features)
        # dense_scale3_local = scale3_local.dense()

        # dense_scale1_features = sparse_scale1_features.dense()
        # N, C, D, H, W = dense_scale1_features.shape
        # dense_scale1_features = dense_scale1_features.view(N, C * D, H, W)
        # dense_scale2_features = sparse_scale2_features.dense()
        # N, C, D, H, W = dense_scale2_features.shape
        # dense_scale2_features = dense_scale2_features.view(N, C * D, H, W)
        # dense_scale3_features = sparse_scale3_features.dense()
        # N, C, D, H, W = dense_scale3_features.shape
        # dense_scale3_features = dense_scale3_features.view(N, C * D, H, W)

        # bev_features_list = [dense_scale1_features, dense_scale2_features]
        # dense_scale1_global = self.dense_scale1_global(dense_scale1_features)
        # dense_scale2_global = self.dense_scale2_global(dense_scale2_features)
        # dense_scale3_global = self.dense_scale3_global(dense_scale3_features)

        # features_multi_raw = self.get_multi_features(
        #     batch_size=batch_size, point_coords=batch_dict['raw_points_bxyz'],
        #     scale1=dense_scale1_local, global1=dense_scale1_global, bev_stride1=tensor_stride//2, z_stride1=4,
        #     scale2=dense_scale2_local, global2=dense_scale2_global, bev_stride2=tensor_stride, z_stride2=8,
        #     # scale3=dense_scale3_local, global3=dense_scale3_global, bev_stride3=tensor_stride*2, z_stride3=8,
        # )

        # fg_statistic_extended = features_for_fg.new_zeros(num_point, features_for_fg.shape[-1])
        # # get fg statistics
        # start_iter = 200 * 0
        # if not self.training or (
        #         batch_dict['accumulated_iter'] is not None and batch_dict['accumulated_iter'] >= start_iter):
        #     for i in range(3):
        #         cur_class_mask = fg_score_idx == i
        #         cur_score_norm = fg_score[cur_class_mask]
        #         filter_score_mask = cur_score_norm >= 0.3
        #         cur_class_features = features_for_fg[cur_class_mask][filter_score_mask]
        #         if len(cur_class_features) > 0:
        #             max_cur_class_features = torch.max(cur_class_features, dim=0)[0]
        #             mean_cur_class_features = torch.mean(cur_class_features, dim=0)
        #             max_cur_class_features = max_cur_class_features - mean_cur_class_features
        #             if self.training:
        #                 if batch_dict['accumulated_iter'] == start_iter:
        #                     temp = max_cur_class_features
        #                     with torch.no_grad():
        #                         self.object_fg_statistic_features[i, :] = temp
        #                 else:
        #                     temp = self.object_fg_statistic_features[i, :] * 0.8 + max_cur_class_features
        #                     with torch.no_grad():
        #                         self.object_fg_statistic_features[i, :] = temp
        #             else:
        #                 temp = self.object_fg_statistic_features[i, :] * 0.8 + max_cur_class_features
        #             fg_statistic_extended[cur_class_mask] = temp / 10


        # center
        # raw_cls_aware = torch.cat([features_for_fg, fg_statistic_extended], dim=-1)
        # raw_cls_aware = raw_cls_aware.view(batch_size, num_point_per_scene, -1).permute(0, 2, 1)
        # raw_pos_coder = self.pos_coder_for_fg_statistic(point_coords[:, 1:])
        # raw_fg_statistic = raw_pos_coder + fg_statistic_extended
        # vote_translation_range = np.array(self.trans_range, dtype=np.float32)
        # vote_translation_range = torch.from_numpy(vote_translation_range).cuda().unsqueeze(dim=0)
        # center_preds = torch.max(center_preds, -vote_translation_range)
        # center_preds = torch.min(center_preds, vote_translation_range)

        # center score
        # features_for_center_score = self.multi_features_center_score(features_multi_raw)
        # center_score_temp_features = self.center_score_temp_features(features_for_center_score).view(num_point, 1, -1)
        # center_score_weight = self.center_score_weight(features_for_center_score).view(num_point, -1, 1) / 64
        # center_score_bias = self.center_score_bias(features_for_center_score).view(num_point, 1)
        # center_score_preds = torch.matmul(center_score_temp_features, center_score_weight).squeeze(1) + center_score_bias
        # center_score_preds_norm = torch.sigmoid(center_score_preds)
        # if self.training:
        #     indices_selected = []
        #     for i in range(batch_size):
        #         cur_batch_mask = point_coords[:, 0] == i
        #         cur_fg_preds_scores = fg_preds_scores[cur_batch_mask]
        #         _, cur_indices_selected = torch.topk(cur_fg_preds_scores, k=self.point_num_list[1], dim=0, sorted=True)
        #         cur_indices_selected = i * self.point_num_list[0] + cur_indices_selected
        #         indices_selected.append(cur_indices_selected)
        #     indices_selected = torch.cat(indices_selected, dim=0)
        # else:
        #     point_xyz_for_sample = point_coords[:, 1:].view(batch_size, self.point_num_list[0], 3)
        #     weights = torch.sigmoid(fg_preds_scores.view(batch_size, -1))
        #     indices_selected = pointnet2_utils.furthest_point_sample_weights(point_xyz_for_sample.contiguous(), weights,
        #                                                                      self.point_num_list[1])
        #     for i in range(batch_size):
        #         indices_selected[i] = indices_selected[i] + i * self.point_num_list[0]
        #     indices_selected = indices_selected.view(-1).long()

        # n x 3
        # one_hot_cls = fg_preds.new_zeros(num_point, 3)
        # one_hot_cls.scatter_(-1, (fg_score_idx * (fg_score_idx >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        # one_hot_cls = one_hot_cls.view(batch_size, -1, 3)
        # cls_dist = torch.matmul(one_hot_cls, one_hot_cls.permute(0, 2, 1))
        # cls_dist = 1 - cls_dist
        # fg_preds_dist = fg_preds_norm.view(batch_size, -1, 3)

        # point_xyz_for_sample = point_coords[:, 1:].view(batch_size, self.point_num_list[0], 3)
        # point_xyz_for_sample_far = point_xyz_for_sample[:, :num_far_fir, :]
        # point_xyz_for_sample_near = point_xyz_for_sample[:, num_far_fir:, :]
        # point_features_for_sample = features_for_fg.contiguous().view(batch_size, self.point_num_list[0], -1)
        # weights = indices_scores.view(batch_size, -1)
        # weights_far = weights[:, :num_far_fir]
        # weights_near = weights[:, num_far_fir:]
        # dist_matrix = calc_dist_matrix_for_sampling(xyz=point_xyz_for_sample,
        #                                             # features=point_features_for_sample,
        #                                             dist_lidar=fg_preds_dist)
        # density_score_raw = torch.clamp(density_score_raw, min=0.7)
        # indices_selected = pointnet2_utils.furthest_point_sample_with_weighted_dist(dist_matrix, weights,
        #                                                                             self.point_num_list[1])
        # indices_selected = pointnet2_utils.furthest_point_sample_with_dist(dist_matrix, self.point_num_list[1])
        # indices_selected_far = pointnet2_utils.furthest_point_sample_weights(
        #     point_xyz_for_sample_far.contiguous(), weights_far, num_far_sec)
        # indices_selected_near = pointnet2_utils.furthest_point_sample_weights(
        #     point_xyz_for_sample_near.contiguous(), weights_near, num_near_sec) + num_far_fir
        # indices_selected = torch.cat([indices_selected_far, indices_selected_near], dim=-1)
        # for i in range(batch_size):
        #     indices_selected[i] = indices_selected[i] + i * self.point_num_list[0]
        # indices_selected = indices_selected.view(-1).long()

        # fg_xyz = point_coords[:, 1:]
        # new_xyz = point_coords[indices_selected][:, 1:]
        # fg_xyz = fg_xyz.view(batch_size, -1, 3).contiguous()
        # new_xyz = new_xyz.view(batch_size, -1, 3).contiguous()
        # fg_features = features_multi_raw.view(batch_size, -1, features_multi_raw.shape[-1]).\
        #     permute(0, 2, 1).contiguous()
        # _, fg_geo_features_batch = self.FG_SA_layer(
        #     xyz=fg_xyz, features=fg_features, new_xyz=new_xyz,
        # )
        #
        # fg_geo_features = fg_geo_features_batch.permute(0, 2, 1).contiguous().\
        #     view(batch_size*self.point_num_list[1], -1)

        # point_coords = point_coords[indices_selected]
        # fg_preds_norm_key_points = fg_preds_norm[indices_selected]
        # density_score_key_points = density_score_raw[indices_selected]
        # features_multi_raw = features_multi_raw.permute(0, 2, 1).contiguous().view(num_point, -1)
        # features_multi_raw_key_points = features_multi_raw[indices_selected]
        # center_preds_key_points = center_preds[indices_selected]
        # batch_dict['point_coords'] = point_coords
        # c_new_xyz = vote_xyz
        # c_new_xyz = c_new_xyz.view(batch_size, -1, 3).contiguous()
        # _, c_geo_features_batch = self.C_SA_layer(
        #     xyz=new_xyz, features=fg_geo_features_batch, new_xyz=c_new_xyz,
        # )
        # c_geo_features = c_geo_features_batch.permute(0, 2, 1).contiguous(). \
        #     view(batch_size * self.point_num_list[2], -1)

        # combine_point_vote_coords = torch.cat([point_coords, vote_coords], dim=-1).view(-1, 4)

        # features_multi_vote = self.bev_to_points(
        #     keypoints=vote_coords,
        #     local_features=dense_tensor_local,
        #     global_features=dense_tensor_global,
        #     batch_size=batch_size,
        #     bev_stride=8,
        #     z_stride=8,
        # )
        # vote_pos_coder = self.pos_coder_for_statistic(vote_point_offset)
        # vote_pos_coder = F.max_pool2d(
        #     vote_pos_coder, kernel_size=[1, vote_pos_coder.size(3)]
        # ).squeeze().permute(1, 0)
        # features_multi_vote = self.get_multi_features(
        #     batch_size=batch_size, point_coords=vote_coords,
        #     scale1=dense_scale1_local, global1=dense_scale1_global, bev_stride1=tensor_stride // 2, z_stride1=4,
        #     scale2=dense_scale1_local, global2=dense_scale1_global, bev_stride2=tensor_stride, z_stride2=8,
        #     # scale3=dense_scale2_local, global3=dense_scale2_global, bev_stride3=tensor_stride * 2, z_stride3=8,
        # )

        # features_multi_vote = torch.cat([features_multi_vote, features_multi_raw_key_points], dim=-1)

        # loss_mask = False
        # if self.training and loss_mask:
        #     dist_matrix_cls = calc_dist_matrix_for_sampling(features=features_for_cls.view(batch_size, num_point_per_scene, -1))
        #     indices_selected_cls = pointnet2_utils.furthest_point_sample_with_dist(dist_matrix_cls, self.point_num_list[2])
        #     for i in range(batch_size):
        #         indices_selected_cls[i] = indices_selected_cls[i] + i * self.point_num_list[1]
        #     indices_selected_cls = indices_selected_cls.view(-1).long()
        features_for_reg = self.features_reg(features_vote)
        # pos_reg = self.pos_coder_for_statistic_reg(vote_pos)
        # statistic_extended_for_reg = self.features_for_statistic_reg(statistic_extended_for_reg)
        # statistic_extended_for_reg = self.relu(statistic_extended_for_reg+pos_reg)
        # features_for_reg = torch.cat([features_for_reg, statistic_extended_for_reg], dim=-1)
        # features_for_reg = features_for_reg.view(batch_size, num_point_per_scene, -1).permute(0, 2, 1)
        # vote_statistic_key_points = self.relu(statistic_extended + vote_pos_coder)
        vote_cls_aware = torch.cat([features_for_cls, statistic_extended_for_reg], dim=-1)
        # vote_cls_aware = vote_cls_aware.view(batch_size, num_point_per_scene, -1).permute(0, 2, 1)
        # features_for_reg = self.multi_features_reg(features_multi_vote)

        # if self.training and loss_mask:
        #     dist_matrix_reg = calc_dist_matrix_for_sampling(features=features_for_reg.view(batch_size, num_point_per_scene, -1))
        #     indices_selected_reg = pointnet2_utils.furthest_point_sample_with_dist(dist_matrix_reg, self.point_num_list[2])
        #     for i in range(batch_size):
        #         indices_selected_reg[i] = indices_selected_reg[i] + i * self.point_num_list[1]
        #     indices_selected_reg = indices_selected_reg.view(-1).long()

        batch_dict['fg_preds'] = fg_preds
        batch_dict['point_center_preds'] = center_preds
        batch_dict['scores_fg'] = fg_preds_norm_key_points
        batch_dict['point_cls_preds'] = point_cls_preds
        batch_dict['features_for_reg'] = features_for_reg
        pts_depth_max = torch.max(pts_depth_key_points, dim=0)[0]
        pts_depth_min = torch.min(pts_depth_key_points, dim=0)[0]
        pts_depth_norm = (pts_depth_key_points - pts_depth_min) / (pts_depth_max - pts_depth_min)
        pts_depth_score = torch.pow(1.2, pts_depth_norm)
        batch_dict['pts_depth'] = pts_depth_score
        batch_dict['vote_cls_aware'] = vote_cls_aware
        # batch_dict['sfps_mask'] = select_matrix_key_points_sfps_mask
        # if self.training and loss_mask:
        #     batch_dict['cls_loss_mask'] = indices_selected_cls
        #     batch_dict['reg_loss_mask'] = indices_selected_reg
        # batch_dict['score_density'] = density_score_key_points
        # batch_dict['encoded_bev_features_list'] = bev_features_list
        # batch_dict['point_candidate_preds'] = center_score_preds
        # batch_dict['score_center'] = center_score_key_points
        return batch_dict


@torch.no_grad()
def calc_dist_matrix_for_sampling(xyz: torch.Tensor = None, features: torch.Tensor = None,
                                  dist_lidar: torch.Tensor = None,
                                  gamma: float = 1.0,
                                  beta: float = 10.0,):
    dist = 0
    if xyz is not None:
        dist = torch.cdist(xyz, xyz)

    if features is not None:
        dist_fetures = torch.cdist(features, features)
        dist += dist_fetures * gamma
        # dist = dist_fetures * gamma * dist

    if dist_lidar is not None:
        dist_dist_lidar = torch.cdist(dist_lidar, dist_lidar)
        # dist += dist_dist_lidar*beta
        dist += dist_dist_lidar * beta

    return dist
