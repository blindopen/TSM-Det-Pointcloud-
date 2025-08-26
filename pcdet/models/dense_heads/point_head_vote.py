import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
# from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...utils import box_coder_utils, box_utils, common_utils, loss_utils
from .point_head_template import PointHeadTemplate


class PointHeadVote(PointHeadTemplate):
    """
    A simple vote-based detection head, which is used for 3DSSD.
    Reference Paper: https://arxiv.org/abs/2002.10187
    3DSSD: Point-based 3D Single Stage Object Detector
    """

    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        # use_bn = self.model_cfg.USE_BN
        self.predict_boxes_when_training = predict_boxes_when_training

        # self.cls_layers = self.make_fc_layers(
        #     input_channels=input_channels,
        #     output_channels=num_class if not self.model_cfg.LOSS_CONFIG.LOSS_CLS == 'CrossEntropy' else num_class + 1,
        #     fc_list=self.model_cfg.CLS_FC
        # )
        # self.cls_layers = nn.Linear(
        #     in_features=64,
        #     out_features=num_class if not self.model_cfg.LOSS_CONFIG.LOSS_CLS == 'CrossEntropy' else num_class + 1,
        #     bias=True)

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.reg_channel = self.box_coder.code_size

        # self.reg_layers = nn.Sequential(
        #     nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=1, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=self.reg_channel, kernel_size=1, bias=True),
        # )
        self.cls_aware_feature_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.reg_feature_layer = nn.Sequential(
            nn.Linear(in_features=input_channels, out_features=64, bias=False),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.reg_weight = nn.Parameter(torch.Tensor(1, 64, self.reg_channel))
        self.weight_gate = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64*self.reg_channel, bias=False),
            nn.Sigmoid(),
        )
        self.weight_bias = nn.Sequential(
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.reg_channel, bias=True),
        )

        self.init_weights()

    def init_weights(self):
        # pi = 0.01
        nn.init.xavier_normal_(self.reg_weight)
        # nn.init.constant_(self.reg_bias, 0)
        nn.init.constant_(self.weight_bias[2].bias, 0)

    # def init_weights(self, weight_init='xavier'):
    #     # pi = 0.01
    #     # nn.init.constant_(self.cls_layers[0].bias, -np.log((1 - pi) / pi))
    #
    #     if weight_init == 'kaiming':
    #         init_func = nn.init.kaiming_normal_
    #     elif weight_init == 'xavier':
    #         init_func = nn.init.xavier_normal_
    #     elif weight_init == 'normal':
    #         init_func = nn.init.normal_
    #     else:
    #         raise NotImplementedError
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
    #             if weight_init == 'normal':
    #                 init_func(m.weight, mean=0, std=0.001)
    #             else:
    #                 init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def build_losses(self, losses_cfg):
        self.add_module(
            'fg_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        self.add_module(
            'center_loss_func',
            loss_utils.WeightedSmoothL1Loss(
                code_weights=[1, 1, 1]
            )
        )

        # classification loss
        if losses_cfg.LOSS_CLS.startswith('WeightedBinaryCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedBinaryCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS == 'WeightedCrossEntropy':
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS == 'FocalLoss':
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(
                    **losses_cfg.get('LOSS_CLS_CONFIG', {})
                )
            )
        else:
            raise NotImplementedError

        # regression loss
        if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            )
        else:
            raise NotImplementedError

        # sasa loss
        loss_sasa_cfg = losses_cfg.get('LOSS_SASA_CONFIG', None)
        if loss_sasa_cfg is not None:
            self.enable_sasa = True
            self.add_module(
                'loss_point_sasa',
                loss_utils.PointSASALoss(**loss_sasa_cfg)
            )
        else:
            self.enable_sasa = False

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def assign_candidate_targets(self, input_dict):
        point_coords = input_dict['candidate_points_bxyz']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        point_center_preds = input_dict['point_center_preds'].detach()

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_ori_targets(
            points=point_coords,
            gt_boxes=extend_gt_boxes, extend_gt_boxes=extend_gt_boxes,
            point_cls_preds=None, point_box_preds=None,
            point_corner_preds=None,
            point_center_preds=point_center_preds,
            ret_candidate_labels=False, ret_corner_labels=False,
            ret_center_labels=True, ret_score_labels=False,
            set_ignore_flag=False, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=False
        )
        return targets_dict

    def assign_raw_fg_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['raw_points_bxyz']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
        targets_dict = self.assign_stack_ori_targets(
            points=point_coords,
            gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            point_cls_preds=None, point_box_preds=None,
            point_corner_preds=None, point_center_preds=None,
            ret_candidate_labels=False, ret_corner_labels=False,
            ret_center_labels=False, ret_score_labels=False,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=False,
            central_radius=central_radius
        )

        return targets_dict

    def assign_stack_ori_targets(self, points, gt_boxes, point_cls_preds, point_box_preds,
                             point_corner_preds=None, point_center_preds=None,
                             ret_candidate_labels=False, ret_corner_labels=False, ret_center_labels=False,
                             extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False, ret_score_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8 or gt_boxes.shape[2] == 10, \
            'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8 or \
               extend_gt_boxes.shape[2] == 10, 'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        # assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], gt_boxes.shape[2])) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        point_score_labels = gt_boxes.new_zeros((points.shape[0])) if ret_score_labels else None
        point_corner_labels = gt_boxes.new_zeros((points.shape[0], 8, 3)) if ret_corner_labels else None
        point_center_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_center_labels else None
        point_candidate_labels = gt_boxes.new_zeros((points.shape[0])) if ret_candidate_labels else None
        gt_box_of_fg_points_list = []
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                # box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                # box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                # ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                # fg_flag = box_fg_flag & ball_flag

                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
                ignore_flag = fg_flag ^ box_fg_flag
                point_cls_labels_single[ignore_flag] = -1
            else:
                fg_flag = box_fg_flag
                # raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            gt_box_of_fg_points_list.append(gt_box_of_fg_points)
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), gt_boxes.shape[2]))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_corner_labels:
                point_corner_labels_single = point_corner_labels.new_zeros((bs_mask.sum(), 8, 3))
                fg_point_corner_labels = box_utils.boxes_to_corners_3d(boxes3d=gt_box_of_fg_points[:, :-1])
                fg_point_corner_labels = fg_point_corner_labels - points_single[fg_flag].unsqueeze(1)
                point_corner_labels_single[fg_flag] = fg_point_corner_labels / 5
                point_corner_labels[bs_mask] = point_corner_labels_single

            if ret_center_labels:
                point_center_labels_single = point_center_labels.new_zeros((bs_mask.sum(), 3))
                fg_point_center_labels = gt_box_of_fg_points[:, 0:3]
                fg_point_center_labels = fg_point_center_labels - points_single[fg_flag]
                point_center_labels_single[fg_flag] = fg_point_center_labels
                point_center_labels[bs_mask] = point_center_labels_single

            if ret_candidate_labels:
                point_candidate_labels_single = point_candidate_labels.new_zeros((bs_mask.sum()))
                point_center_preds_single = point_center_preds[bs_mask]
                point_center_preds_single = point_center_preds_single.view(-1, 3)
                point_center_preds_single = point_center_preds_single[fg_flag]
                offset_center = point_center_preds_single - fg_point_center_labels
                # wlh = gt_box_of_fg_points[:, 3:6] / 2
                offset = torch.pow(offset_center, 2)
                offset = torch.sum(offset, dim=-1)
                offset = torch.sqrt(offset)

                norm = torch.pow(fg_point_center_labels, 2)
                norm = torch.sum(norm, dim=-1)
                norm = torch.sqrt(norm)

                cos_score = torch.sum(offset_center*(-fg_point_center_labels), dim=-1)/(offset*norm)
                cos_score = torch.clamp(cos_score, max=1.0, min=-1.0)
                cos_score = 1 - torch.arccos(cos_score) / np.pi

                offset_score = 2*offset / norm
                offset_score = torch.clamp(offset_score, max=1.0, min=0.0)
                offset_score = 1 - offset_score

                candidate_scores = 0.8 * offset_score + 0.2 * cos_score

                hot_up_mask = candidate_scores > 0.75
                candidate_scores[hot_up_mask] = 1
                hot_down_mask = candidate_scores < 0.25
                candidate_scores[hot_down_mask] = 0
                interval_mask = ~(hot_up_mask | hot_down_mask)
                candidate_scores[interval_mask] = candidate_scores[interval_mask] * 2 - 0.5

                point_candidate_labels_single[fg_flag] = candidate_scores
                point_candidate_labels[bs_mask] = point_candidate_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                temp = torch.abs(((torch.abs(transformed_points / gt_box_of_fg_points[:, 3:6]) * 2) - 0.5) * 2)
                hot_up_mask = temp > 0.75
                temp[hot_up_mask] = 1
                hot_down_mask = temp < 0.25
                temp[hot_down_mask] = 0
                interval_mask = ~(hot_up_mask | hot_down_mask)
                temp[interval_mask] = temp[interval_mask] * 2 - 0.5
                # temp = torch.clamp(temp, min=0.15)
                point_part_labels_single[fg_flag] = temp
                point_part_labels[bs_mask] = point_part_labels_single

            if ret_score_labels:
                _, point_box = self.generate_predicted_boxes(
                    points=points[bs_mask][:, 1:4],
                    point_cls_preds=point_cls_preds[bs_mask], point_box_preds=point_box_preds[bs_mask]
                )
                cur_gt = gt_boxes[k]
                cnt = cur_gt.__len__() - 1
                while cnt > 0 and cur_gt[cnt].sum() == 0:
                    cnt -= 1
                cur_gt = cur_gt[:cnt + 1]
                # anchor_by_gt_overlap = box_utils.boxes3d_nearest_bev_iou(point_box[:, 0:7], cur_gt[:, 0:7])
                anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(point_box[:, 0:7], cur_gt[:, 0:7])
                anchor_wise_bev_iou = torch.max(anchor_by_gt_overlap, dim=-1)[0]

                iou_up_mask = anchor_wise_bev_iou > 0.75
                anchor_wise_bev_iou[iou_up_mask] = 1
                iou_down_mask = anchor_wise_bev_iou < 0.25
                anchor_wise_bev_iou[iou_down_mask] = 0
                interval_mask = ~(iou_up_mask | iou_down_mask)
                anchor_wise_bev_iou[interval_mask] = anchor_wise_bev_iou[interval_mask] * 2 - 0.5

                point_score_labels[bs_mask] = anchor_wise_bev_iou

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels,
            'point_score_labels': point_score_labels,
            'point_candidate_labels': point_candidate_labels,
            'point_corner_labels': point_corner_labels,
            'point_center_labels': point_center_labels,
            'gt_box_of_fg_points_list': gt_box_of_fg_points_list
        }
        return targets_dict

    def assign_stack_targets_simple(self, points, gt_boxes, extend_gt_boxes=None, set_ignore_flag=True):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: (B, M, 8), required if set ignore flag
            set_ignore_flag:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignore
            point_reg_labels: (N1 + N2 + N3 + ..., 3), corresponding object centroid
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert not set_ignore_flag or extend_gt_boxes is not None
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((points.shape[0], 3))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if extend_gt_boxes is not None:
                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idx_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[box_fg_flag]]
            point_cls_labels_single[box_fg_flag] = 1
            point_cls_labels[bs_mask] = point_cls_labels_single

            point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), 3))
            point_reg_labels_single[box_fg_flag] = gt_box_of_fg_points[:, 0:3]
            point_reg_labels[bs_mask] = point_reg_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
        }
        return targets_dict

    def assign_targets_simple(self, points, gt_boxes, extra_width=None, set_ignore_flag=True):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extra_width: (dx, dy, dz) extra width applied to gt boxes
            assign_method: binary or distance
            set_ignore_flag:
        Returns:
            point_vote_labels: (N1 + N2 + N3 + ..., 3)
        """
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert points.shape.__len__() in [2], 'points.shape=%s' % str(points.shape)
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) \
            if extra_width is not None else gt_boxes
        if set_ignore_flag:
            targets_dict = self.assign_stack_targets_simple(points=points, gt_boxes=gt_boxes,
                                                            extend_gt_boxes=extend_gt_boxes,
                                                            set_ignore_flag=set_ignore_flag)
        else:
            targets_dict = self.assign_stack_targets_simple(points=points, gt_boxes=extend_gt_boxes,
                                                            set_ignore_flag=set_ignore_flag)
        return targets_dict

    def assign_stack_targets_mask(self, points, gt_boxes, extend_gt_boxes=None,
                                  set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            set_ignore_flag:
            use_ball_constraint:
            central_radius:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_reg_labels: (N1 + N2 + N3 + ..., code_size)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = gt_boxes.new_zeros(points.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((points.shape[0], self.box_coder.code_size))
        point_box_labels = gt_boxes.new_zeros((points.shape[0], gt_boxes.size(2) - 1))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
                ignore_flag = fg_flag ^ box_fg_flag
                point_cls_labels_single[ignore_flag] = -1
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if gt_box_of_fg_points.shape[0] > 0:
                point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), self.box_coder.code_size))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_reg_labels_single[fg_flag] = fg_point_box_labels
                point_reg_labels[bs_mask] = point_reg_labels_single

                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), gt_boxes.size(2) - 1))
                point_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                point_box_labels[bs_mask] = point_box_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
            'point_box_labels': point_box_labels
        }
        return targets_dict

    def assign_stack_targets_iou(self, points, pred_boxes, gt_boxes,
                                 pos_iou_threshold=0.5, neg_iou_threshold=0.35):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            pred_boxes: (N, 7/8)
            gt_boxes: (B, M, 8)
            pos_iou_threshold:
            neg_iou_threshold:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_reg_labels: (N1 + N2 + N3 + ..., code_size)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(pred_boxes.shape) == 2 and pred_boxes.shape[1] >= 7, 'pred_boxes.shape=%s' % str(pred_boxes.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = gt_boxes.new_zeros(pred_boxes.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((pred_boxes.shape[0], self.box_coder.code_size))
        point_box_labels = gt_boxes.new_zeros((pred_boxes.shape[0], 7))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            pred_boxes_single = pred_boxes[bs_mask]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            pred_boxes_iou = iou3d_nms_utils.boxes_iou3d_gpu(
                pred_boxes_single,
                gt_boxes[k][:, :7]
            )
            pred_boxes_iou, box_idxs_of_pts = torch.max(pred_boxes_iou, dim=-1)
            fg_flag = pred_boxes_iou > pos_iou_threshold
            ignore_flag = (pred_boxes_iou > neg_iou_threshold) ^ fg_flag
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels_single[ignore_flag] = -1
            point_cls_labels[bs_mask] = point_cls_labels_single

            if gt_box_of_fg_points.shape[0] > 0:
                point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), self.box_coder.code_size))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_reg_labels_single[fg_flag] = fg_point_box_labels
                point_reg_labels[bs_mask] = point_reg_labels_single

                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 7))
                point_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                point_box_labels[bs_mask] = point_box_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
            'point_box_labels': point_box_labels
        }
        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        assign_method = self.model_cfg.TARGET_CONFIG.ASSIGN_METHOD  # mask or iou
        if assign_method == 'mask':
            points = input_dict['point_vote_coords']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
            targets_dict = self.assign_stack_targets_mask(
                points=points, gt_boxes=gt_boxes,
                set_ignore_flag=False, use_ball_constraint=True, central_radius=central_radius
            )
        elif assign_method == 'iou':
            points = input_dict['point_vote_coords']
            pred_boxes = input_dict['point_box_preds']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert pred_boxes.shape.__len__() == 2, 'pred_boxes.shape=%s' % str(pred_boxes.shape)
            pos_iou_threshold = self.model_cfg.TARGET_CONFIG.POS_IOU_THRESHOLD
            neg_iou_threshold = self.model_cfg.TARGET_CONFIG.NEG_IOU_THRESHOLD
            targets_dict = self.assign_stack_targets_iou(
                points=points, pred_boxes=pred_boxes, gt_boxes=gt_boxes,
                pos_iou_threshold=pos_iou_threshold, neg_iou_threshold=neg_iou_threshold
            )
        else:
            raise NotImplementedError

        return targets_dict

    def get_vote_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        vote_reg_labels = self.forward_ret_dict['vote_reg_labels']
        vote_reg_preds = self.forward_ret_dict['point_vote_coords']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        vote_loss_reg_src = self.reg_loss_func(
            vote_reg_preds[None, ...],
            vote_reg_labels[None, ...],
            weights=reg_weights[None, ...])
        vote_loss_reg = vote_loss_reg_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        vote_loss_reg = vote_loss_reg * loss_weights_dict['vote_reg_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss_reg': vote_loss_reg.item()})
        return vote_loss_reg, tb_dict

    @torch.no_grad()
    def generate_centerness_label(self, point_base, point_box_labels, pos_mask, epsilon=1e-6):
        """
        Args:
            point_base: (N1 + N2 + N3 + ..., 3)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
            pos_mask: (N1 + N2 + N3 + ...)
            epsilon:
        Returns:
            centerness_label: (N1 + N2 + N3 + ...)
        """
        centerness = point_box_labels.new_zeros(pos_mask.shape)

        point_box_labels = point_box_labels[pos_mask, :]
        canonical_xyz = point_base[pos_mask, :] - point_box_labels[:, :3]
        rys = point_box_labels[:, -1]
        canonical_xyz = common_utils.rotate_points_along_z(
            canonical_xyz.unsqueeze(dim=1), -rys
        ).squeeze(dim=1)

        distance_front = point_box_labels[:, 3] / 2 - canonical_xyz[:, 0]
        distance_back = point_box_labels[:, 3] / 2 + canonical_xyz[:, 0]
        distance_left = point_box_labels[:, 4] / 2 - canonical_xyz[:, 1]
        distance_right = point_box_labels[:, 4] / 2 + canonical_xyz[:, 1]
        distance_top = point_box_labels[:, 5] / 2 - canonical_xyz[:, 2]
        distance_bottom = point_box_labels[:, 5] / 2 + canonical_xyz[:, 2]

        centerness_l = torch.min(distance_front, distance_back) / torch.max(distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(distance_left, distance_right)
        centerness_h = torch.min(distance_top, distance_bottom) / torch.max(distance_top, distance_bottom)
        centerness_pos = torch.clamp(centerness_l * centerness_w * centerness_h, min=epsilon) ** (1 / 3.0)

        centerness[pos_mask] = centerness_pos

        return centerness

    def get_axis_aligned_iou_loss_lidar(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            pred_boxes: (N, 7) float Tensor.
            gt_boxes: (N, 7) float Tensor.
        Returns:
            iou_loss: (N) float Tensor.
        """
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        pos_p, len_p, *cps = torch.split(pred_boxes, 3, dim=-1)
        pos_g, len_g, *cgs = torch.split(gt_boxes, 3, dim=-1)

        len_p = torch.clamp(len_p, min=1e-5)
        len_g = torch.clamp(len_g, min=1e-5)
        vol_p = len_p.prod(dim=-1)
        vol_g = len_g.prod(dim=-1)

        min_p, max_p = pos_p - len_p / 2, pos_p + len_p / 2
        min_g, max_g = pos_g - len_g / 2, pos_g + len_g / 2

        min_max = torch.min(max_p, max_g)
        max_min = torch.max(min_p, min_g)
        diff = torch.clamp(min_max - max_min, min=0)
        intersection = diff.prod(dim=-1)
        union = vol_p + vol_g - intersection
        iou_axis_aligned = intersection / torch.clamp(union, min=1e-5)

        iou_loss = 1 - iou_axis_aligned
        return iou_loss

    def get_corner_loss_lidar(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            pred_boxes: (N, 7) float Tensor.
            gt_boxes: (N, 7) float Tensor.
        Returns:
            corner_loss: (N) float Tensor.
        """
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        pred_box_corners = box_utils.boxes_to_corners_3d(pred_boxes)
        gt_box_corners = box_utils.boxes_to_corners_3d(gt_boxes)

        gt_boxes_flip = gt_boxes.clone()
        gt_boxes_flip[:, 6] += np.pi
        gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_boxes_flip)
        # (N, 8, 3)
        corner_loss = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(pred_box_corners - gt_box_corners, 1.0)
        corner_loss_flip = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(pred_box_corners - gt_box_corners_flip, 1.0)
        corner_loss = torch.min(corner_loss.sum(dim=2), corner_loss_flip.sum(dim=2))

        return corner_loss.mean(dim=1)

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)
        loss_mask = self.forward_ret_dict['loss_mask']

        if loss_mask:
            cls_loss_mask = self.forward_ret_dict['cls_loss_mask']
            point_cls_labels = point_cls_labels[cls_loss_mask]
            point_cls_preds = point_cls_preds[cls_loss_mask]

        positives = point_cls_labels > 0
        negatives = point_cls_labels == 0
        cls_weights = positives * 1.0 + negatives * 1.0

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        self.forward_ret_dict['point_cls_labels_onehot'] = one_hot_targets

        loss_cfgs = self.model_cfg.LOSS_CONFIG
        if 'WithCenterness' in loss_cfgs.LOSS_CLS:
            point_base = self.forward_ret_dict['point_vote_coords']
            point_box_labels = self.forward_ret_dict['point_box_labels']
            if loss_mask:
                point_base = point_base[cls_loss_mask]
                point_box_labels = point_box_labels[cls_loss_mask]
            centerness_label = self.generate_centerness_label(point_base, point_box_labels, positives)

            loss_cls_cfg = loss_cfgs.get('LOSS_CLS_CONFIG', None)
            centerness_min = loss_cls_cfg['centerness_min'] if loss_cls_cfg is not None else 0.0
            centerness_max = loss_cls_cfg['centerness_max'] if loss_cls_cfg is not None else 1.0
            centerness_label = centerness_min + (centerness_max - centerness_min) * centerness_label

            one_hot_targets *= centerness_label.unsqueeze(dim=-1)

        point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets[..., 1:], weights=cls_weights)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_pos_num': positives.sum().item()
        })
        # cls_loss_mask = self.forward_ret_dict['cls_loss_mask']
        # point_loss_cls = point_loss_cls[cls_loss_mask]
        # cls_weights = cls_weights[cls_loss_mask]
        return point_loss_cls, cls_weights, tb_dict  # point_loss_cls: (N)

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_reg_preds = self.forward_ret_dict['point_reg_preds']
        point_reg_labels = self.forward_ret_dict['point_reg_labels']
        loss_mask = self.forward_ret_dict['loss_mask']

        if loss_mask:
            reg_loss_mask = self.forward_ret_dict['reg_loss_mask']
            pos_mask = pos_mask[reg_loss_mask]
            point_reg_preds = point_reg_preds[reg_loss_mask]
            point_reg_labels = point_reg_labels[reg_loss_mask]

        reg_weights = pos_mask.float()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        if tb_dict is None:
            tb_dict = {}

        point_loss_offset_reg = self.reg_loss_func(
            point_reg_preds[None, :, :6],
            point_reg_labels[None, :, :6],
            weights=reg_weights[None, ...]
        )
        point_loss_offset_reg = point_loss_offset_reg.sum(dim=-1).squeeze()

        if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
            point_loss_velo_reg = self.reg_loss_func(
                point_reg_preds[None, :, 6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                point_reg_labels[None, :, 6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                weights=reg_weights[None, ...]
            )
            point_loss_velo_reg = point_loss_velo_reg.sum(dim=-1).squeeze()
            point_loss_offset_reg = point_loss_offset_reg + point_loss_velo_reg

        point_loss_offset_reg *= loss_weights_dict['point_offset_reg_weight']

        if isinstance(self.box_coder, box_coder_utils.PointBinResidualCoder):
            point_angle_cls_labels = \
                point_reg_labels[:, 6:6 + self.box_coder.angle_bin_num]
            point_loss_angle_cls = F.cross_entropy(  # angle bin cls
                point_reg_preds[:, 6:6 + self.box_coder.angle_bin_num],
                point_angle_cls_labels.argmax(dim=-1), reduction='none') * reg_weights

            point_angle_reg_preds = point_reg_preds[:,
                                    6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
            point_angle_reg_labels = point_reg_labels[:,
                                     6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
            point_angle_reg_preds = (point_angle_reg_preds * point_angle_cls_labels).sum(dim=-1, keepdim=True)
            point_angle_reg_labels = (point_angle_reg_labels * point_angle_cls_labels).sum(dim=-1, keepdim=True)
            point_loss_angle_reg = self.reg_loss_func(
                point_angle_reg_preds[None, ...],
                point_angle_reg_labels[None, ...],
                weights=reg_weights[None, ...]
            )
            point_loss_angle_reg = point_loss_angle_reg.squeeze()

            point_loss_angle_cls *= loss_weights_dict['point_angle_cls_weight']
            point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']

            point_loss_box = point_loss_offset_reg + point_loss_angle_cls + point_loss_angle_reg  # (N)
        else:
            point_angle_reg_preds = point_reg_preds[:, 6:]
            point_angle_reg_labels = point_reg_labels[:, 6:]
            point_loss_angle_reg = self.reg_loss_func(
                point_angle_reg_preds[None, ...],
                point_angle_reg_labels[None, ...],
                weights=reg_weights[None, ...]
            )
            point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']
            point_loss_box = point_loss_offset_reg + point_loss_angle_reg

        if reg_weights.sum() > 0:
            point_box_preds = self.forward_ret_dict['point_box_preds']
            point_box_labels = self.forward_ret_dict['point_box_labels']
            point_loss_box_aux = 0

            if loss_mask:
                point_box_preds = point_box_preds[reg_loss_mask]
                point_box_labels = point_box_labels[reg_loss_mask]

            if self.model_cfg.LOSS_CONFIG.get('AXIS_ALIGNED_IOU_LOSS_REGULARIZATION', False):
                point_loss_iou = self.get_axis_aligned_iou_loss_lidar(
                    point_box_preds[pos_mask, :],
                    point_box_labels[pos_mask, :]
                )
                point_loss_iou *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_iou_weight']
                point_loss_box_aux = point_loss_box_aux + point_loss_iou

            if self.model_cfg.LOSS_CONFIG.get('CORNER_LOSS_REGULARIZATION', False):
                point_loss_corner = self.get_corner_loss_lidar(
                    point_box_preds[pos_mask, 0:7],
                    point_box_labels[pos_mask, 0:7]
                )
                point_loss_corner *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_corner_weight']
                point_loss_box_aux = point_loss_box_aux + point_loss_corner

            point_loss_box[pos_mask] = point_loss_box[pos_mask] + point_loss_box_aux

        return point_loss_box, reg_weights, tb_dict  # point_loss_box: (N)

    def get_sasa_layer_loss(self, tb_dict=None):
        if self.enable_sasa:
            point_loss_sasa_list = self.loss_point_sasa.loss_forward(
                self.forward_ret_dict['point_sasa_preds'],
                self.forward_ret_dict['point_sasa_labels']
            )
            point_loss_sasa = 0
            tb_dict = dict()
            for i in range(len(point_loss_sasa_list)):
                cur_point_loss_sasa = point_loss_sasa_list[i]
                if cur_point_loss_sasa is None:
                    continue
                point_loss_sasa = point_loss_sasa + cur_point_loss_sasa
                tb_dict['point_loss_sasa_layer_%d' % i] = point_loss_sasa_list[i].item()
            tb_dict['point_loss_sasa'] = point_loss_sasa.item()
            return point_loss_sasa, tb_dict
        else:
            return None, None

    def get_raw_fg_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_raw_fg_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_raw_fg_cls_preds'].view(-1, self.num_class)
        # point_cls_preds = self.forward_ret_dict['point_raw_fg_cls_preds'].view(-1, 1)
        # weight_pos_cos = self.3['weight_pos_cos']

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        # only set when predict fg
        if point_cls_preds.shape[-1] == 1:
            point_cls_labels[positives] = 1

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        # one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), 1 + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.fg_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        # car_mask = point_cls_labels == 1
        # cls_loss_src[car_mask] = cls_loss_src[car_mask] * (1+weight_pos_cos)
        # non_car_mask = ((point_cls_labels == 2) | (point_cls_labels == 3))
        # cls_loss_src[non_car_mask] = cls_loss_src[non_car_mask] * (1 - weight_pos_cos)

        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_raw_fg_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'rawfg': point_loss_cls.item(),
            'rawfgpos': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_candidate_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_candidate_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_candidate_labels = self.forward_ret_dict['point_candidate_labels']
        point_candidate_preds = self.forward_ret_dict['point_candidate_preds']
        point_loss_candidate = F.binary_cross_entropy(torch.sigmoid(point_candidate_preds), point_candidate_labels.unsqueeze(1), reduction='none')
        point_loss_candidate = (point_loss_candidate.sum(dim=-1) * pos_mask.float()).sum() / (pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_candidate = point_loss_candidate * loss_weights_dict['point_candidate_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'candidate': point_loss_candidate.item(),
            'candidate_pos': pos_normalizer})
        return point_loss_candidate, tb_dict

    def get_center_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_center_cls_label'] > 0
        point_box_labels = self.forward_ret_dict['point_center_labels'].view(-1, 3)
        point_box_preds = self.forward_ret_dict['point_center_preds']


        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.center_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        ).squeeze(0)
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_center_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center': point_loss_box.item()})
        return point_loss_box, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_raw_fg_cls, tb_dict = self.get_raw_fg_cls_layer_loss(tb_dict)
        # point_loss_candidate, tb_dict = self.get_candidate_loss(tb_dict)
        # point_loss_center, tb_dict = self.get_center_loss(tb_dict)
        point_loss_vote, tb_dict = self.get_vote_layer_loss(tb_dict)
        pts_depth = self.forward_ret_dict['pts_depth']

        point_loss_cls, cls_weights, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, box_weights, tb_dict_2 = self.get_box_layer_loss()

        point_loss_cls = point_loss_cls * pts_depth
        point_loss_box = point_loss_box * pts_depth

        point_loss_cls = point_loss_cls.sum() / torch.clamp(cls_weights.sum(), min=1.0)
        point_loss_box = point_loss_box.sum() / torch.clamp(box_weights.sum(), min=1.0)
        tb_dict.update({
            # 'point_loss_vote': point_loss_vote.item(),
            'point_loss_cls': point_loss_cls.item(),
            'point_loss_box': point_loss_box.item()
        })

        point_loss = point_loss_cls + point_loss_box + point_loss_vote + point_loss_raw_fg_cls
        # tb_dict.update(tb_dict_0)
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)

        # point_loss_sasa, tb_dict_3 = self.get_sasa_layer_loss()
        # if point_loss_sasa is not None:
        #     tb_dict.update(tb_dict_3)
        #     point_loss += point_loss_sasa
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_scores (optional): (B, N)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        batch_size = batch_dict['batch_size']
        candidate_coords = batch_dict['point_coords']
        vote_coords = batch_dict['vote_coords']
        batch_dict['point_candidate_coords'] = candidate_coords
        batch_dict['point_vote_coords'] = vote_coords
        point_cls_preds = batch_dict['point_cls_preds']
        vote_cls_aware = batch_dict['vote_cls_aware']

        ret_dict = {'batch_size': batch_size,
                    'point_candidate_coords': candidate_coords[:, 1:].contiguous(),
                    'point_vote_coords': vote_coords[:, 1:].contiguous()}

        if self.training:  # assign targets for vote loss
            extra_width = self.model_cfg.TARGET_CONFIG.get('VOTE_EXTRA_WIDTH', None)
            targets_dict = self.assign_targets_simple(batch_dict['point_candidate_coords'],
                                                      batch_dict['gt_boxes'],
                                                      extra_width=extra_width,
                                                      set_ignore_flag=False)
            ret_dict['vote_cls_labels'] = targets_dict['point_cls_labels']  # (N)
            ret_dict['vote_reg_labels'] = targets_dict['point_reg_labels']  # (N, 3)

            # candidate_targets_dict = self.assign_candidate_targets(batch_dict)
            # ret_dict['point_candidate_labels'] = candidate_targets_dict['point_candidate_labels']
            # ret_dict['point_center_labels'] = candidate_targets_dict['point_center_labels']
            # ret_dict['point_center_cls_label'] = candidate_targets_dict['point_cls_labels']

        # point_features_cls = batch_dict['features_for_cls']
        point_features_reg = batch_dict['features_for_reg']
        # batch_size, _, num_point_per_scene = point_features_reg.shape
        # num_point = batch_size*num_point_per_scene
        num_point, _ = point_features_reg.shape

        # point_cls_preds = self.cls_layers(point_features_cls)
        # reg_temp_features = self.reg_temp_features(point_features_reg).view(num_point, 1, 64)
        # reg_weight = self.reg_weight(vote_cls_aware).view(num_point, 64, 64)
        # reg_temp_features = torch.matmul(reg_temp_features, reg_weight).squeeze(1)
        # reg_temp_features = reg_temp_features.view(batch_size, -1, 64).permute(0, 2, 1)

        point_features_reg = self.reg_feature_layer(point_features_reg)
        vote_cls_aware = self.cls_aware_feature_layer(vote_cls_aware)
        point_features_reg = point_features_reg.unsqueeze(1)
        weight_gate = self.weight_gate(vote_cls_aware).view(num_point, -1, self.reg_channel)
        weight_bias = self.weight_bias(vote_cls_aware)
        reg_weight = self.reg_weight*weight_gate
        # reg_bias = self.reg_bias+weight_bias
        point_reg_preds = torch.matmul(point_features_reg, reg_weight).squeeze(1)
        point_reg_preds = point_reg_preds + weight_bias
        # point_reg_preds = self.reg_layers(point_features_reg)
        # point_reg_preds = point_reg_preds.permute(0, 2, 1).contiguous().view(num_point, self.reg_channel)

        # max, _ = torch.max(point_reg_preds, dim=0)

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'] = point_cls_scores

        # _, point_box_preds = self.generate_predicted_boxes(
        #     points=batch_dict['point_vote_coords'][:, 1:4],
        #     point_cls_preds=point_cls_preds, point_box_preds=point_reg_preds
        # )
        point_box_preds = self.box_coder.decode_torch(point_reg_preds,
                                                      ret_dict['point_vote_coords'])
        # box_max, _ = torch.max(point_box_preds, dim=0)
        batch_dict['point_box_preds'] = point_box_preds

        ret_dict.update({'point_cls_preds': point_cls_preds,
                         'point_reg_preds': point_reg_preds,
                         'point_box_preds': point_box_preds,
                         'point_cls_scores': point_cls_scores})

        ret_dict['point_raw_fg_cls_preds'] = batch_dict['fg_preds']
        # ret_dict['point_candidate_preds'] = batch_dict['point_candidate_preds']
        ret_dict['point_center_preds'] = batch_dict['point_center_preds']
        ret_dict['pts_depth'] = batch_dict['pts_depth']
        # ret_dict['sfps_mask'] = batch_dict['sfps_mask']


        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_reg_labels'] = targets_dict['point_reg_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            raw_fg_targets_dict = self.assign_raw_fg_targets(batch_dict)
            ret_dict['point_raw_fg_cls_labels'] = raw_fg_targets_dict['point_cls_labels']
            loss_mask = False
            ret_dict['loss_mask'] = loss_mask
            if loss_mask:
                ret_dict['cls_loss_mask'] = batch_dict['cls_loss_mask']
                ret_dict['reg_loss_mask'] = batch_dict['reg_loss_mask']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_vote_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_reg_preds
            )
            # sfps_mask = batch_dict['sfps_mask']
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['cls_preds_normalized'] = False
            batch_dict['batch_index'] = batch_dict['vote_coords'][:, 0]

        self.forward_ret_dict = ret_dict
        return batch_dict
