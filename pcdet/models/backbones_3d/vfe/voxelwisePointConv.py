import torch
import torch.nn as nn

from .vfe_template import VFETemplate


class VPCVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.linear_relative = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # self.point_score = nn.Sequential(
        #     nn.Linear(6, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16, bias=False),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1, bias=False),
        # )
        # self.sorted_features = nn.Sequential(
        #     nn.Linear(320, 128, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )

    def get_output_feature_dim(self):
        return 64

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        num_voxel, voxel_count, num_channel = voxel_features.shape
        mask_bool = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask_bool, -1).type_as(voxel_features)

        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        #
        # f_center = torch.zeros_like(voxel_features[:, :, :3])
        # f_center[:, :, 0] = voxel_features[:, :, 0] - (
        #             coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        # f_center[:, :, 1] = voxel_features[:, :, 1] - (
        #             coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        # f_center[:, :, 2] = voxel_features[:, :, 2] - (
        #             coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        # f_center *= mask

        mean_offset = voxel_features-points_mean.unsqueeze(1)
        mean_offset *= mask
        voxel_features = self.linear(voxel_features.permute(0, 2, 1)).permute(0, 2, 1)
        voxel_features_relaive = self.linear_relative(mean_offset.permute(0, 2, 1)).permute(0, 2, 1)
        voxel_features = torch.cat([voxel_features, voxel_features_relaive], dim=-1)
        # voxel_features[~mask_bool] = -999999
        voxel_features *= mask

        # semantic_features + center_offset = score
        # features_for_score = torch.cat([f_center, torch.abs(f_center)], dim=-1).view(-1, 6)
        # scores = self.point_score(features_for_score).view(-1, voxel_count)
        # scores[~mask_bool] = -99999
        # scores = torch.softmax(scores, dim=-1).unsqueeze(-1)
        #
        # indices = torch.argsort(scores.squeeze(-1), dim=1, descending=True)
        # add = torch.linspace(0, num_voxel - 1, num_voxel, device=indices.device) * indices.size(1)
        # indices = (indices + add.long().view(-1, 1)).view(-1)
        #
        # f_sorted = (voxel_features.view(num_voxel*voxel_count, -1)[indices]).view(num_voxel, -1).contiguous()
        # f_sorted = self.sorted_features(f_sorted)

        # voxel_features_scored = voxel_features * scores
        # out_features = torch.sum(voxel_features_scored, dim=1).contiguous()
        # out_features = torch.cat([out_features, f_sorted], dim=-1)
        # out_features = f_sorted

        voxel_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        out_features = voxel_mean / normalizer
        # out_features = torch.cat([mean_features, f_sorted], dim=-1)

        # voxel_max = torch.max(voxel_features, dim=1)[0]
        # out_features = voxel_max

        raw_points_features = voxel_features[mask_bool]
        raw_points_batch_idx = coords[:, :1].repeat(1, voxel_count).unsqueeze(-1)
        raw_points_all = torch.cat([raw_points_batch_idx, batch_dict['voxels'][:, :, :3]], dim=-1)
        raw_points_bxyz = raw_points_all[mask_bool]

        batch_dict['voxel_features'] = out_features
        # batch_dict['center_offset'] = f_center[mask_bool]
        batch_dict['raw_points_features'] = raw_points_features
        batch_dict['raw_points_bxyz'] = raw_points_bxyz
        # batch_dict['num_voxel'] = num_voxel
        # batch_dict['voxel_count'] = voxel_count
        # batch_dict['mask_bool'] = mask_bool
        # batch_dict['mask'] = mask
        # batch_dict['f_center_abs'] = f_center_abs
        # batch_dict['normalizer'] = normalizer
        # batch_dict['scores'] = scores
        return batch_dict
