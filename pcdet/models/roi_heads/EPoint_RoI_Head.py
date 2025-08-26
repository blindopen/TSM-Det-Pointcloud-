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


class EPointRoIHead(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, grid_size=None, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.roi_per_image = model_cfg.TARGET_CONFIG['ROI_PER_IMAGE']
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.grid_size = grid_size
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

    # def _init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
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
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.roi_voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.roi_voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.roi_voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            # cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.roi_voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE,self.pool_cfg.GRID_SIZE, self.pool_cfg.GRID_SIZE,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)

        # ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        # ms_pooled_features_mean = pooled_features_list[0]
        # ms_pooled_features_max = pooled_features_list[1]

        return pooled_features_list, roi_grid_xyz

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

        # test = dense_idx.view(batch_size_rcnn, 6, 6, 6, 3)
        # test_up = test[:, :, :3, :, :]
        # test_down = torch.flip(test[:, :, 3:, :, :], dims=[2])

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def roi_features_propagation(self, pooled_features_list):
        # conv2_mean = pooled_features_list[2]
        # conv2_max = pooled_features_list[3]
        combined_mean = pooled_features_list[0]
        combined_max = pooled_features_list[1]
        # conv3_mean = pooled_features_list[4]
        # conv3_max = pooled_features_list[5]
        batch_size_rcnn = combined_max.shape[0]

        # conv2_propagation = (conv2_max-conv2_mean).clone().detach()
        # conv2_propagation = conv2_propagation.view(batch_size_rcnn, 6, 6, 6, -1)
        # conv2_propagation = torch.flip(conv2_propagation, dims=[2])
        # conv2_propagationed = conv2_propagation.view(batch_size_rcnn, 216, -1) + conv2_mean
        #
        # conv3_propagation = (conv3_max - conv3_mean).clone().detach()
        # conv3_propagation = conv3_propagation.view(batch_size_rcnn, 6, 6, 6, -1)
        # conv3_propagation = torch.flip(conv3_propagation, dims=[2])
        # conv3_propagationed = conv3_propagation.view(batch_size_rcnn, 216, -1) + conv3_mean

        combined_propagation = (combined_max - combined_mean).clone().detach()
        combined_propagation = combined_propagation.view(batch_size_rcnn, 6, 6, 6, -1)
        combined_propagation = torch.flip(combined_propagation, dims=[2])
        combined_propagationed = combined_propagation.view(batch_size_rcnn, 216, -1) + combined_mean

        # max_show = combined_max.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
        # max_img = max_show[1, :, :, 1].detach().cpu().numpy()
        # mean_show = combined_mean.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
        # mean_img = mean_show[1, :, :, 1].detach().cpu().numpy()
        # propagation_show = combined_propagation.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
        # propagation_img = propagation_show[1, :, :, 1].detach().cpu().numpy()
        # propagationed_show = combined_propagationed.view(batch_size_rcnn, 6, 6, 6, -1)[:, :, :, 1, :]
        # propagationed_img = propagationed_show[1, :, :, 1].detach().cpu().numpy()
        #
        # plt.imshow(max_img, cmap='magma')
        # plt.show()
        # plt.imshow(mean_img, cmap='magma')
        # plt.show()
        # plt.imshow(propagation_img, cmap='magma')
        # plt.show()
        # plt.imshow(propagationed_img, cmap='magma')
        # plt.show()
        combined_propagationed = self.combined_propagationed_transform(
            combined_propagationed.view(batch_size_rcnn * 216, -1))

        features = torch.cat([combined_max.view(batch_size_rcnn * 216, -1), combined_propagationed], dim=-1)
        features = self.point_features_transform(features).view(batch_size_rcnn, -1)

        return features

    def transform_points_to_voxels(self, data_dict=None):
        point_features = data_dict['encoded_point_features']
        points = data_dict['points']
        point_xyz_features = torch.cat([points, point_features], dim=-1)
        batch_size = data_dict['batch_size']
        voxel_features_list = []
        voxel_coords_list = []
        voxel_num_points_list = []
        for i in range(batch_size):
            cur_batch_mask = point_xyz_features[:, 0]==i
            cur_point_xyz_features = point_xyz_features[cur_batch_mask][:, 1:].contiguous()
            cur_voxel_features, cur_voxel_coords, cur_voxel_num_points = self.voxel_generator(cur_point_xyz_features)
            cur_voxel_coords_batch_idx = cur_voxel_coords.new_zeros(size=(len(cur_voxel_coords), 1)) + i
            cur_voxel_coords = torch.cat([cur_voxel_coords_batch_idx, cur_voxel_coords], dim=-1)
            voxel_coords_list.append(cur_voxel_coords)
            voxel_features_list.append(cur_voxel_features)
            voxel_num_points_list.append(cur_voxel_num_points)
        voxel_features = torch.cat(voxel_features_list, dim=0)
        voxel_coords = torch.cat(voxel_coords_list, dim=0)
        voxel_num_points = torch.cat(voxel_num_points_list, dim=0)
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer

        batch_size = data_dict['batch_size']
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.roi_voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        roi_sparse_shape = grid_size[::-1] + [1, 0, 0]
        input_sp_tensor = spconv.SparseConvTensor(
            features=points_mean,
            indices=voxel_coords.int(),
            spatial_shape=roi_sparse_shape,
            batch_size=batch_size
        )
        return input_sp_tensor
    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features
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

        # roi_sparse_tensor = self.transform_points_to_voxels(batch_dict)
        # point_features = batch_dict['encoded_point_features']
        # batch_dict['multi_scale_3d_features']['x_points_max'] = batch_dict['multi_scale_3d_features']['x_points_max'].replace_feature(point_features)
        # global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(batch_dict['rois'] )
        batch_size = batch_dict['batch_size']

        # RoI aware pooling
        pooled_features_list, global_roi_grid_points = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        # pooled_bev_features = self.interpolate_from_bev_features(
        #     global_roi_grid_points, encoded_bev_features, batch_size,
        #     bev_stride=2
        # ).view(-1, 6, 6, 6, 64)
        # roi_point_encoder = self.roi_point_encoder(global_roi_grid_points.view(-1, 3)).view(-1, 6, 6, 6, 64)
        # pooled_bev_features = pooled_bev_features + roi_point_encoder
        # pooled_features = torch.cat([pooled_bev_features, pooled_features_list[0]], dim=-1)

        # propagation
        # pooled_features = self.roi_features_propagation(pooled_features_list)
        # Box Refinement
        # pooled_features = pooled_features_list[0]
        pooled_features = torch.cat(pooled_features_list, dim=-1)
        # batch_rcnn, x_n, y_n, z_n, f_n = pooled_features.shape
        # z_part = torch.mean(pooled_features.view(batch_rcnn, -1, z_n, f_n), dim=1, keepdim=True)
        # # z_part = self.z_part_layer(z_part.view(-1, f_n)).view(batch_rcnn, 1, z_n, f_n//2)
        # xy_part = torch.mean(pooled_features, dim=3)
        # x_part = torch.mean(xy_part, dim=2, keepdim=True)
        # # x_part = self.x_part_layer(x_part.view(-1, f_n)).view(batch_rcnn, x_n, 1, f_n//2)
        # y_part = torch.mean(xy_part, dim=1, keepdim=True)
        # # y_part = self.y_part_layer(y_part.view(-1, f_n)).view(batch_rcnn, 1, y_n, f_n//2)
        # xy_fix = torch.matmul(x_part.permute(0, 3, 1, 2), y_part.permute(0, 3, 1, 2))
        # xy_fix = xy_fix.view(batch_rcnn, f_n, -1, 1)
        # xyz_fix = torch.matmul(xy_fix, z_part.permute(0, 3, 1, 2)).view(batch_rcnn, f_n, x_n, y_n, z_n).permute(0, 2, 3, 4, 1)
        # xyz_fix = self.xyz_fix_layer(xyz_fix.contiguous().view(-1, f_n)).view(batch_rcnn, x_n, y_n, z_n, f_n)
        # pooled_features = xyz_fix + pooled_features
        pooled_features = pooled_features.contiguous().view(pooled_features.size(0), -1)
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