import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
import numpy as np
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
# import matplotlib.pyplot as plt

from ...datasets.processor.data_processor import VoxelGeneratorWrapper
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
from ...utils.spconv_utils import spconv


class EPointRoIHeadV2(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, grid_size=None, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.roi_per_image = model_cfg.TARGET_CONFIG['ROI_PER_IMAGE']
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        self.grid_size = grid_size

        # self.offset_roi = nn.Sequential(
        #     nn.Linear(in_features=3, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        # )
        self.pos_coder_roi = nn.Sequential(
            nn.Linear(in_features=3, out_features=32, bias=False),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        # self.local_global = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        # )
        self.local_global1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=32, bias=False),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.local_global2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=32, bias=False),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * 64

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.roi_voxel_size = self.voxel_size
        # if self.training:
        #     mode = "train"
        # else:
        #     mode = 'test'
        # self.voxel_generator = PointToVoxel(
        #     vsize_xyz=self.roi_voxel_size,
        #     coors_range_xyz=self.point_cloud_range,
        #     num_point_features=260,
        #     max_num_points_per_voxel=self.model_cfg.POINT_TO_VOXEL.MAX_POINTS_PER_VOXEL,
        #     max_num_voxels=self.model_cfg.POINT_TO_VOXEL.MAX_NUMBER_OF_VOXELS[mode],
        #     device=torch.device("cuda:0")
        # )

        # self.point_features_transform = nn.Sequential(
        #     nn.Linear(128, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        # self.combined_propagationed_transform = nn.Sequential(
        #     nn.Linear(64, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        # self.bev_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # self.x_part_layer = nn.Sequential(
        #     nn.Linear(64, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )
        # self.y_part_layer = nn.Sequential(
        #     nn.Linear(64, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )
        # self.z_part_layer = nn.Sequential(
        #     nn.Linear(64, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )
        # self.xyz_fix_layer = nn.Sequential(
        #     nn.Linear(64, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        # self.roi_point_encoder = nn.Sequential(
        #     nn.Linear(3, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    def bev_to_points_batch(self, keypoints, local_features, global_features, batch_size, bev_stride, z_stride, local_keypoints):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        n_points = keypoints.shape[1]
        z_idxs = (keypoints[:, :, 2] - self.point_cloud_range[2]) / self.voxel_size[2]
        z_idxs = z_idxs / z_stride

        n_local_channels = local_features.shape[1]
        n_global_channels = global_features.shape[1]
        point_local_global_features = local_features.new_zeros((batch_size, n_points, n_local_channels+n_global_channels+32))
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_z_idxs = z_idxs[k]
            cur_point_coords = local_keypoints[k]

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
            #
            # offset = torch.cat([offset_x0, offset_y0, offset_z0], dim=-1)
            # offset = self.offset_roi(offset)
            coord_features = self.pos_coder_roi(cur_point_coords)

            point_local_global_features[k] = torch.cat([c_xyz, ans, coord_features], dim=-1)

        return point_local_global_features

    def get_global_grid_points_of_roi(self, rois, grid_size=None):
        grid_size = self.pool_cfg.GRID_SIZE
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    # def roi_features_propagation(self, pooled_features_list):
    #     # conv2_mean = pooled_features_list[2]
    #     # conv2_max = pooled_features_list[3]
    #     combined_mean = pooled_features_list[0]
    #     combined_max = pooled_features_list[1]
    #     # conv3_mean = pooled_features_list[4]
    #     # conv3_max = pooled_features_list[5]
    #     batch_size_rcnn = combined_max.shape[0]
    #
    #     # conv2_propagation = (conv2_max-conv2_mean).clone().detach()
    #     # conv2_propagation = conv2_propagation.view(batch_size_rcnn, 6, 6, 6, -1)
    #     # conv2_propagation = torch.flip(conv2_propagation, dims=[2])
    #     # conv2_propagationed = conv2_propagation.view(batch_size_rcnn, 216, -1) + conv2_mean
    #     #
    #     # conv3_propagation = (conv3_max - conv3_mean).clone().detach()
    #     # conv3_propagation = conv3_propagation.view(batch_size_rcnn, 6, 6, 6, -1)
    #     # conv3_propagation = torch.flip(conv3_propagation, dims=[2])
    #     # conv3_propagationed = conv3_propagation.view(batch_size_rcnn, 216, -1) + conv3_mean
    #
    #     combined_propagation = (combined_max - combined_mean).clone().detach()
    #     combined_propagation = combined_propagation.view(batch_size_rcnn, 6, 6, 6, -1)
    #     combined_propagation = torch.flip(combined_propagation, dims=[2])
    #     combined_propagationed = combined_propagation.view(batch_size_rcnn, 216, -1) + combined_mean
    #
    #     # max_show = combined_max.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
    #     # max_img = max_show[1, :, :, 1].detach().cpu().numpy()
    #     # mean_show = combined_mean.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
    #     # mean_img = mean_show[1, :, :, 1].detach().cpu().numpy()
    #     # propagation_show = combined_propagation.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
    #     # propagation_img = propagation_show[1, :, :, 1].detach().cpu().numpy()
    #     # propagationed_show = combined_propagationed.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
    #     # propagationed_img = propagationed_show[1, :, :, 1].detach().cpu().numpy()
    #     #
    #     # plt.imshow(max_img, cmap='magma')
    #     # plt.show()
    #     # plt.imshow(mean_img, cmap='magma')
    #     # plt.show()
    #     # plt.imshow(propagation_img, cmap='magma')
    #     # plt.show()
    #     # plt.imshow(propagationed_img, cmap='magma')
    #     # plt.show()
    #     combined_propagationed = self.combined_propagationed_transform(
    #         combined_propagationed.view(batch_size_rcnn * 216, -1))
    #
    #     features = torch.cat([combined_max.view(batch_size_rcnn * 216, -1), combined_propagationed], dim=-1)
    #     features = self.point_features_transform(features).view(batch_size_rcnn, -1)
    #
    #     return features
    #
    # def transform_points_to_voxels(self, data_dict=None):
    #     point_features = data_dict['encoded_point_features']
    #     points = data_dict['points']
    #     point_xyz_features = torch.cat([points, point_features], dim=-1)
    #     batch_size = data_dict['batch_size']
    #     voxel_features_list = []
    #     voxel_coords_list = []
    #     voxel_num_points_list = []
    #     for i in range(batch_size):
    #         cur_batch_mask = point_xyz_features[:, 0]==i
    #         cur_point_xyz_features = point_xyz_features[cur_batch_mask][:, 1:].contiguous()
    #         cur_voxel_features, cur_voxel_coords, cur_voxel_num_points = self.voxel_generator(cur_point_xyz_features)
    #         cur_voxel_coords_batch_idx = cur_voxel_coords.new_zeros(size=(len(cur_voxel_coords), 1)) + i
    #         cur_voxel_coords = torch.cat([cur_voxel_coords_batch_idx, cur_voxel_coords], dim=-1)
    #         voxel_coords_list.append(cur_voxel_coords)
    #         voxel_features_list.append(cur_voxel_features)
    #         voxel_num_points_list.append(cur_voxel_num_points)
    #     voxel_features = torch.cat(voxel_features_list, dim=0)
    #     voxel_coords = torch.cat(voxel_coords_list, dim=0)
    #     voxel_num_points = torch.cat(voxel_num_points_list, dim=0)
    #     points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
    #     normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
    #     points_mean = points_mean / normalizer
    #
    #     batch_size = data_dict['batch_size']
    #     grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.roi_voxel_size)
    #     grid_size = np.round(grid_size).astype(np.int64)
    #     roi_sparse_shape = grid_size[::-1] + [1, 0, 0]
    #     input_sp_tensor = spconv.SparseConvTensor(
    #         features=points_mean,
    #         indices=voxel_coords.int(),
    #         spatial_shape=roi_sparse_shape,
    #         batch_size=batch_size
    #     )
    #     return input_sp_tensor
    # def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
    #     x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
    #     y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
    #     x_idxs = x_idxs / bev_stride
    #     y_idxs = y_idxs / bev_stride
    #
    #     point_bev_features_list = []
    #     for k in range(batch_size):
    #         cur_x_idxs = x_idxs[k]
    #         cur_y_idxs = y_idxs[k]
    #         cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
    #         point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
    #         point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))
    #
    #     point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
    #     return point_bev_features
    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(batch_dict['rois'] )
        batch_size = batch_dict['batch_size']
        batch_size_roi = global_roi_grid_points.shape[0]
        encoded_spconv_tensor_stride = batch_dict['encoded_spconv_tensor_stride']
        local_features_list = batch_dict['local_features_list']
        global_features_list = batch_dict['global_features_list']

        local_features1 = local_features_list[1]
        global_features1 = global_features_list[1]
        B1, C1, H1, W1 = local_features1.shape
        local_features2 = local_features_list[2]
        global_features2 = global_features_list[2]
        B2, C2, H2, W2 = local_features2.shape

        local_global_1 = self.bev_to_points_batch(
            keypoints=global_roi_grid_points.view(B1, -1, 3),
            local_features=local_features1.view(B1, -1, 10, H1, W1),
            global_features=global_features1,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride,
            z_stride=4,
            local_keypoints=local_roi_grid_points.view(B1, -1, 3),
        ).view(batch_size_roi*216, -1)
        local_global_1 = self.local_global1(local_global_1)
        local_global_2 = self.bev_to_points_batch(
            keypoints=global_roi_grid_points.view(B2, -1, 3),
            local_features=local_features2.view(B2, -1, 10, H2, W2),
            global_features=global_features2,
            batch_size=batch_size,
            bev_stride=encoded_spconv_tensor_stride * 2,
            z_stride=4,
            local_keypoints=local_roi_grid_points.view(B2, -1, 3),
        ).view(batch_size_roi*216, -1)
        local_global_2 = self.local_global2(local_global_2)

        pooled_features = torch.cat([local_global_1, local_global_2], dim=-1)
        pooled_features = pooled_features.contiguous().view(batch_size_roi, -1)
        shared_features = self.shared_fc_layer(pooled_features)

        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict