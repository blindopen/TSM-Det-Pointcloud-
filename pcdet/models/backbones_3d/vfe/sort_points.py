import torch
import torch.nn as nn

from .vfe_template import VFETemplate


class SPVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # self.point_score = nn.Sequential(
        #     nn.Linear(6, 16, bias=False),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1, bias=False),
        # )
        # self.point_features = nn.Sequential(
        #     nn.Linear(10, 16, bias=False),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )
        # self.sorted_features = nn.Sequential(
        #     nn.Linear(320, 64, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        self.relative_weight = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Linear(32, 32, bias=False),
        )
        self.relative_feature = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.relative_nonlinear = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Linear(32, 32, bias=False),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
        )
        self.relative_point = nn.Sequential(
            nn.Linear(4, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.center_weight = nn.Sequential(
            nn.Linear(6, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Linear(32, 32, bias=False),
        )
        self.center_nonlinear = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Linear(32, 32, bias=False),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
        )

    def get_output_feature_dim(self):
        return 32

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

        relative_point = self.relative_point(voxel_features.view(-1 ,num_channel)).view(num_voxel, voxel_count, 1, -1)
        relative_point = relative_point.repeat(1, 1, voxel_count, 1)
        # Mx1xNx4-->MxNxNx4
        partl = voxel_features.unsqueeze(1).repeat(1, voxel_count, 1, 1)
        # MxNx1x4-->MxNxNx4
        partr = voxel_features.unsqueeze(2).repeat(1, 1, voxel_count, 1)
        relative = partl-partr
        relative_val = torch.cat([relative, torch.abs(relative)], dim=-1)
        relative_val[~mask_bool, :, :]=0
        relative_val = relative_val.permute(0, 2, 1, 3)
        relative_val[~mask_bool, :, :] = 0
        relative_val = relative_val.permute(0, 2, 1, 3)
        # weight
        relative_weight = self.relative_weight(relative_val.view(-1, 8)).view(num_voxel, voxel_count, voxel_count, -1)
        # relative_weight = torch.softmax(relative_weight, dim=1)
        # relative_feature
        relative_feature = self.relative_feature(relative_val.view(-1, 8))
        relative_feature = relative_feature.view(num_voxel, voxel_count, voxel_count, -1)
        relative_feature = torch.cat([relative_point, relative_feature], dim=-1)
        relative_feature = relative_feature*relative_weight

        relative_feature = torch.sum(relative_feature, dim=1)
        relative_feature *= mask
        relative_feature = self.relative_nonlinear(relative_feature.view(num_voxel*voxel_count, -1))
        relative_feature = relative_feature.view(num_voxel, voxel_count, -1)

        raw_points_features = relative_feature[mask_bool]
        raw_points_batch_idx = coords[:, :1].repeat(1, voxel_count).unsqueeze(-1)
        raw_points_all = torch.cat([raw_points_batch_idx, voxel_features[:, :, :3]], dim=-1)
        raw_points_bxyz = raw_points_all[mask_bool]

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        f_center *= mask

        f_center_abs = torch.cat([f_center, torch.abs(f_center)], dim=-1).view(-1, 6)
        f_center_weight = self.center_weight(f_center_abs).view(num_voxel, voxel_count, -1)
        # f_center_weight = torch.softmax(f_center_weight, dim=1)
        f_center_out = f_center_weight*relative_feature
        out_features = torch.sum(f_center_out, dim=1)
        out_features = self.center_nonlinear(out_features)
        # scores = self.point_score(f_center_abs).view(-1, voxel_count)
        # scores[~mask_bool] = -99999
        # scores = torch.softmax(scores, dim=-1).unsqueeze(-1)
        #
        # f_mix = self.point_features(f_mix).view(len(scores), voxel_count, -1)
        # f_mix = torch.cat([relative_feature, f_mix], dim=-1)
        # f_sum = torch.sum(f_mix*scores, dim=1).contiguous()
        #
        # indices = torch.argsort(scores.squeeze(-1), dim=1, descending=True)
        # add = torch.linspace(0, num_voxel - 1, num_voxel, device=indices.device) * indices.size(1)
        # indices = (indices + add.long().view(-1, 1)).view(-1)
        #
        # f_sorted = (f_mix.view(num_voxel*voxel_count, -1)[indices]).view(num_voxel, -1).contiguous()
        # f_sorted = self.sorted_features(f_sorted)
        # out_features = torch.cat([f_sum, f_sorted], dim=-1)

        batch_dict['voxel_features'] = out_features
        batch_dict['raw_points_features'] = raw_points_features
        batch_dict['raw_points_bxyz'] = raw_points_bxyz
        # batch_dict['f_center_abs'] = f_center_abs
        # batch_dict['num_voxel'] = num_voxel
        # batch_dict['voxel_count'] = voxel_count
        # batch_dict['mask_bool'] = mask_bool
        return batch_dict
