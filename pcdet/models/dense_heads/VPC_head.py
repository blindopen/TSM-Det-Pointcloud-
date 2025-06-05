import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils, common_utils, loss_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.iou3d_nms import iou3d_nms_utils
from .point_head_template import PointHeadTemplate
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
# from mayavi import mlab


class VPCNetHead(PointHeadTemplate):
    """
    Point-based head for predicting the intra-object part locations.
    Reference Paper: https://arxiv.org/abs/1907.03670
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, voxel_size, point_cloud_range, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training

        # self.cls_layers = self.make_fc_layers_v1(
        #     fc_cfg=self.model_cfg.CLS_FC,
        #     input_channels=384,
        #     output_channels=num_class
        # )
        # self.part_reg_layers = self.make_fc_layers_v1(
        #     fc_cfg=self.model_cfg.PART_FC,
        #     input_channels=384,
        #     output_channels=1
        # )
        target_cfg = self.model_cfg.TARGET_CONFIG
        if target_cfg.get('BOX_CODER', None) is not None:
            self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
                **target_cfg.BOX_CODER_CONFIG
            )
            self.box_layers = self.make_fc_layers_v1(
                fc_cfg=self.model_cfg.REG_FC,
                input_channels=512,
                output_channels=self.box_coder.code_size
            )
        else:
            self.box_layers = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        # nn.init.constant_(self.cls_layers[3].bias, -np.log((1 - pi) / pi))
        # nn.init.constant_(self.part_reg_layers[3].bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.box_layers[3].weight, mean=0, std=0.001)
        nn.init.constant_(self.box_layers[3].bias, 0)

    # def assign_targets(self, input_dict):
    #     """
    #     Args:
    #         input_dict:
    #             point_features: (N1 + N2 + N3 + ..., C)
    #             batch_size:
    #             point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
    #             gt_boxes (optional): (B, M, 8)
    #     Returns:
    #         point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
    #         point_part_labels: (N1 + N2 + N3 + ..., 3)
    #     """
    #     point_coords = input_dict['point_coords']
    #     # end_point_coords = input_dict['end_point_coords']
    #     gt_boxes = input_dict['gt_boxes']
    #     assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
    #     assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
    #
    #     batch_size = gt_boxes.shape[0]
    #     extend_gt_boxes = box_utils.enlarge_box3d(
    #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
    #     ).view(batch_size, -1, gt_boxes.shape[-1])
    #     targets_dict = self.assign_stack_targets(
    #         points=point_coords,
    #         gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
    #         set_ignore_flag=True, use_ball_constraint=False,
    #         ret_part_labels=False, ret_box_labels=(self.box_layers is not None)
    #     )
    #
    #     return targets_dict

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

        # point_box_preds = input_dict['point_box_preds'].detach()
        # point_cls_preds = input_dict['point_cls_preds'].detach()

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords,
            gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            point_cls_preds=None, point_box_preds=None,
            point_corner_preds=None, point_center_preds=None,
            ret_candidate_labels=False, ret_corner_labels=False,
            ret_center_labels=False, ret_score_labels=False,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=False
        )

        return targets_dict

    def assign_candidate_targets(self, input_dict):
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        # point_box_preds = input_dict['point_box_preds'].detach()
        # point_cls_preds = input_dict['point_cls_preds'].detach()
        # point_corner_preds = input_dict['point_corner_preds'].detach()
        point_center_preds = input_dict['point_center_preds'].detach()

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords,
            gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            point_cls_preds=None, point_box_preds=None,
            point_corner_preds=None,
            point_center_preds=point_center_preds,
            ret_candidate_labels=True, ret_corner_labels=False,
            ret_center_labels=True, ret_score_labels=False,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=False
        )

        return targets_dict

    def assign_targets(self, input_dict):
        point_coords = input_dict['candidate_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        point_box_preds = input_dict['point_box_preds'].detach()
        point_cls_preds = input_dict['point_cls_preds'].detach()

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords,
            gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            point_cls_preds=point_cls_preds, point_box_preds=point_box_preds,
            point_corner_preds=None, point_center_preds=None,
            ret_candidate_labels=False, ret_corner_labels=False,
            ret_center_labels=False, ret_score_labels=False,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=(self.box_layers is not None)
        )

        return targets_dict

    def assign_stack_targets(self, points, gt_boxes, point_cls_preds, point_box_preds,
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
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
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
                points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
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
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

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
                # testtemp = cos_score.sum()
                # cos_score = torch.arccos(cos_score)
                # testtemp = cos_score.sum()
                cos_score = 1 - torch.arccos(cos_score) / np.pi

                # norm = torch.abs(fg_point_center_labels)
                # fenmu = torch.pow(wlh/10, 2).unsqueeze(1)
                # candidate_scores = torch.exp(-torch.pow(offset_corner, 2)/fenmu)
                # candidate_scores = 1-torch.abs(offset_center)/(norm)
                # candidate_scores = torch.clamp(candidate_scores, max=1.0, min=0.0)
                # candidate_scores = candidate_scores.view(-1, 3)
                # candidate_scores = torch.mean(candidate_scores, dim=-1)

                candidate_scores = 1 - offset / norm
                candidate_scores = 0.7*candidate_scores + 0.3* cos_score
                # candidate_scores = torch.clamp(candidate_scores, max=1.0, min=0.0)

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

                # score_labels_single = point_box_labels.new_zeros((bs_mask.sum()))
                # score_labels_single[anchor_mask] = selected_anchor_iou
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

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_raw_fg_cls, tb_dict = self.get_raw_fg_cls_layer_loss(tb_dict)
        point_loss_candidate, tb_dict = self.get_candidate_loss(tb_dict)
        point_loss_center, tb_dict = self.get_center_loss(tb_dict)
        # point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)
        # point_loss_score, tb_dict = self.get_score_layer_loss(tb_dict)
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss = point_loss_cls + point_loss_raw_fg_cls + point_loss_candidate + point_loss_center

        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
        return point_loss, tb_dict

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        batch_idxs = keypoints[:, 0]

        point_bev_features_list = []
        for k in range(batch_size):
            batch_mask = batch_idxs == k
            cur_x_idxs = x_idxs[batch_mask]
            cur_y_idxs = y_idxs[batch_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    # def show_point_score(self, batch_size, point_coords, cls_scores):
    #     # point_coords = common_utils.get_voxel_centers(
    #     #     sp_tensor.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
    #     #     point_cloud_range=self.point_cloud_range
    #     # )
    #     # point_coords = torch.cat((sp_tensor.indices[:, 0:1].float(), point_coords), dim=1)
    #     # cls_scores = cls_scores[mask]
    #     # point_part_scores = point_part_scores[mask]
    #     for i in range(0, batch_size):
    #         mlab.figure(bgcolor=(0, 0, 0))
    #         mask = point_coords[:, 0] == i
    #         cur_batch_point_coords = point_coords[mask]
    #         cur_x = cur_batch_point_coords[:, 1].cpu().numpy()
    #         cur_y = cur_batch_point_coords[:, 2].cpu().numpy()
    #         cur_z = cur_batch_point_coords[:, 3].cpu().numpy()
    #         # val_depict = (torch.sigmoid(val[mask])).detach().cpu().numpy()
    #         val_depict = (cls_scores[mask]).detach().cpu().numpy()
    #
    #         mlab.points3d(cur_x, cur_y, cur_z, val_depict, scale_factor=0.25, scale_mode='none', mode='sphere',
    #                       line_width=1, colormap='jet')
    #         mlab.colorbar()
    #         mlab.show()

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['candidate_features']

        # point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_cls_preds = batch_dict['point_cls_preds']
        # point_part_preds = self.part_reg_layers(point_features)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            # 'point_part_preds': point_part_preds,
        }

        ret_dict['point_raw_fg_cls_preds'] = batch_dict['fg_preds']
        ret_dict['point_candidate_preds'] = batch_dict['point_candidate_preds']
        ret_dict['point_center_preds'] = batch_dict['point_center_preds']

        if self.box_layers is not None:
            point_box_preds = self.box_layers(point_features)
            ret_dict['point_box_preds'] = point_box_preds

        point_cls_scores = torch.softmax(point_cls_preds, dim=-1)
        # point_cls_scores = torch.sigmoid(point_cls_preds)
        # point_part_offset = torch.sigmoid(point_part_preds)
        # score_for_show = point_part_offset.mean(dim=1)*point_cls_scores.max(dim=-1)[0]
        # self.show_point_score(
        #     batch_size=2,
        #     point_coords=point_coords,
        #     cls_scores=score_for_show
        # )
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        # batch_dict['point_part_offset'] = point_part_offset
        batch_dict['point_box_preds'] = point_box_preds
        batch_dict['point_cls_preds'] = point_cls_preds

        if self.training:
            candidate_targets_dict = self.assign_candidate_targets(batch_dict)
            ret_dict['point_candidate_labels'] = candidate_targets_dict['point_candidate_labels']
            ret_dict['point_center_labels'] = candidate_targets_dict['point_center_labels']
            ret_dict['candidate_points_all'] = batch_dict['candidate_points_all']
            targets_dict = self.assign_targets(batch_dict)
            raw_fg_targets_dict = self.assign_raw_fg_targets(batch_dict)
            ret_dict['point_raw_fg_cls_labels'] = raw_fg_targets_dict['point_cls_labels']
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['gt_box_of_fg_points_end'] = targets_dict['gt_box_of_fg_points_list']
            ret_dict['candidate_coords'] = batch_dict['candidate_coords'][:, 1:4]
            # ret_dict['point_part_labels'] = targets_dict.get('point_part_labels')
            # ret_dict['point_score_labels'] = targets_dict.get('point_score_labels')
            ret_dict['point_box_labels'] = targets_dict.get('point_box_labels')

            total_cos_iter = 928*15
            weight_pos_cos = np.cos((np.pi*batch_dict['accumulated_iter'])/(2*total_cos_iter))
            if batch_dict['accumulated_iter'] >= total_cos_iter:
                weight_pos_cos = 0
            ret_dict['weight_pos_cos'] = 0

        if self.box_layers is not None and (not self.training or self.predict_boxes_when_training):
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['candidate_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=ret_dict['point_box_preds']
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['candidate_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict
        return batch_dict

    def get_raw_fg_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_raw_fg_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_raw_fg_cls_preds'].view(-1, self.num_class)
        weight_pos_cos = self.forward_ret_dict['weight_pos_cos']

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.fg_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        car_mask = point_cls_labels == 1
        cls_loss_src[car_mask] = cls_loss_src[car_mask] * (1+weight_pos_cos)
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

    def get_score_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_score_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_score_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels.unsqueeze(1), reduction='none')

        # candidate_points_all = self.forward_ret_dict['candidate_points_all']
        # point_loss_part = point_loss_part.view(-1, candidate_points_all)
        # point_loss_part[:, 1000:2000] = point_loss_part[:, 1000:2000] * 0.8
        # point_loss_part[:, 2000:3000] = point_loss_part[:, 1000:2000] * 0.6
        point_loss_part = point_loss_part.view(-1, 1)

        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'p_part': point_loss_part.item()})
        return point_loss_part, tb_dict

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
        tb_dict.update({'candidate': point_loss_candidate.item(),
                        'candidate_pos': pos_normalizer})
        return point_loss_candidate, tb_dict

    def get_corner_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_candidate_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_corner_labels'].view(-1, 24)
        point_box_preds = self.forward_ret_dict['point_corner_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.corner_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner': point_loss_box.item()})
        return point_loss_box, tb_dict

    def get_center_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_candidate_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_center_labels'].view(-1, 3)
        point_box_preds = self.forward_ret_dict['point_center_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.center_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_center_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center': point_loss_box.item()})
        return point_loss_box, tb_dict

    # @staticmethod
    # def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
    #     batch_size = reg_targets.shape[0]
    #     anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    #     rot_gt = reg_targets[..., 6] + anchors[..., 6]
    #     offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    #     dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    #     dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    #
    #     if one_hot:
    #         dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
    #                                   device=dir_cls_targets.device)
    #         dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
    #         dir_cls_targets = dir_targets
    #     return dir_cls_targets

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']
        gt_box_of_fg_points_list = self.forward_ret_dict['gt_box_of_fg_points_end']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        ).squeeze(0)

        # candidate_points_all = self.forward_ret_dict['candidate_points_all']
        # point_loss_box_src = point_loss_box_src.view(-1, candidate_points_all)
        # point_loss_box_src[:, 1500:3000] = point_loss_box_src[:, 1500:3000] * 1.5
        # point_loss_box_src[:, 2000:3000] = point_loss_box_src[:, 1000:2000] * 0.6
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        car_mask = point_cls_labels == 1
        weight_pos_cos = self.forward_ret_dict['weight_pos_cos']
        point_loss_box_src[car_mask] = point_loss_box_src[car_mask] * (1+weight_pos_cos)
        # non_car_mask = ((point_cls_labels == 2) | (point_cls_labels == 3))
        # point_loss_box_src[non_car_mask] = point_loss_box_src[non_car_mask] * (1 - weight_pos_cos)

        # point_loss_box_src = point_loss_box_src.view(-1, 1)

        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'pbox': point_loss_box.item()})

        # if box_dir_cls_preds is not None:
        #     dir_targets = self.get_direction_target(
        #         anchors, box_reg_targets,
        #         dir_offset=self.model_cfg.DIR_OFFSET,
        #         num_bins=self.model_cfg.NUM_DIR_BINS
        #     )
        #
        #     dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
        #     weights = positives.type_as(dir_logits)
        #     weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
        #     dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
        #     dir_loss = dir_loss.sum() / batch_size
        #     dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
        #     box_loss += dir_loss
        #     tb_dict['rpn_loss_dir'] = dir_loss.item()
        if pos_normalizer > 0:
            # TODO: NEED to BE CHECK
            fg_candidate_coords = self.forward_ret_dict['candidate_coords'][pos_mask]
            fg_point_box_preds = point_box_preds[pos_mask]
            fg_point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)[pos_mask]
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=fg_candidate_coords,
                point_cls_preds=fg_point_cls_preds, point_box_preds=fg_point_box_preds
            )
            gt_box_of_fg_points_end = torch.cat(gt_box_of_fg_points_list, dim=0)[:, :-1]

            loss_corner = loss_utils.get_corner_loss_lidar(
                point_box_preds[:, 0:7],
                gt_box_of_fg_points_end[:, 0:7]
            )
            loss_corner = loss_corner.mean()
            loss_corner = loss_corner * loss_weights_dict['point_corner_weight']
            point_loss_box += loss_corner
            tb_dict['pcorner'] = loss_corner.item()

        return point_loss_box, tb_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)

        # candidate_points_all = self.forward_ret_dict['candidate_points_all']
        car_mask = point_cls_labels == 1

        weight_pos_cos = self.forward_ret_dict['weight_pos_cos']
        cls_loss_src[car_mask] = cls_loss_src[car_mask]*(1+weight_pos_cos)
        # non_car_mask = ((point_cls_labels == 2) | (point_cls_labels == 3))
        # cls_loss_src[non_car_mask] = cls_loss_src[non_car_mask] * (1 - weight_pos_cos)
        cls_loss_src = cls_loss_src.view(-1, 1)

        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'pcls': point_loss_cls.item(),
            'ppos': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict