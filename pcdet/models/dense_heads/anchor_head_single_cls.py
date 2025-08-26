import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingleCls(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            160, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        # self.scale1_deconv = nn.Sequential(
        #     nn.ConvTranspose2d(320, 128, kernel_size=1, stride=1, bias=False, groups=1),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        # )
        # self.scale2_deconv = nn.Sequential(
        #     nn.ConvTranspose2d(320, 128, kernel_size=2, stride=2, bias=False, groups=1),
        #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        # )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        rpn_loss = cls_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def forward(self, data_dict):
        # spatial_features_2d = data_dict['encoded_bev_features']
        sparse_tensor_scale3 = data_dict['multi_scale_3d_features']['scale4']
        bev_features = sparse_tensor_scale3.dense()
        N, C, D, H, W = bev_features.shape
        bev_features = bev_features.view(N, C * D, H, W)
        # encoded_bev_features_list =data_dict['encoded_bev_features_list']
        # encoded_bev_scale1 = self.scale1_deconv(encoded_bev_features_list[0])
        # encoded_bev_scale2 = self.scale2_deconv(encoded_bev_features_list[-1])
        # encoded_bev_features = torch.cat([encoded_bev_scale1, encoded_bev_scale2], dim=1)
        cls_preds = self.conv_cls(bev_features)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]


        self.forward_ret_dict['cls_preds'] = cls_preds

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        return data_dict
