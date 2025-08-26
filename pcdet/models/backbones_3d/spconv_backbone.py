from functools import partial

import torch.nn as nn
import torch

from ...utils import common_utils
from ...utils.spconv_utils import replace_feature, spconv


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


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64,
            'x_points_mean': 32,
            'x_points_max': 32,
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_points_mean': x_conv2,
                'x_points_max': x_conv2,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_points_mean': 2,
                'x_points_max': 2,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict


class DSASNetVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        # self.convz = spconv.SparseSequential(
        #     # block(16, 32, (3, 1, 1), norm_fn=norm_fn, padding=(1, 0, 0), indice_key='subm1', conv_type='spconv'),
        #     block(32, 64, (1, 3, 3), norm_fn=norm_fn, padding=(0, 1, 1), indice_key='submxy'),
        #     block(64, 64, (1, 3, 3), norm_fn=norm_fn, padding=(0, 1, 1), indice_key='submxy'),
        # )
        # self.convx = spconv.SparseSequential(
        #     # block(16, 32, (1, 1, 3), norm_fn=norm_fn, padding=(0, 0, 1), indice_key='submx', conv_type='spconv'),
        #     block(32, 64, (3, 3, 1), norm_fn=norm_fn, padding=(1, 1, 0), indice_key='submzy'),
        #     block(64, 64, (3, 3, 1), norm_fn=norm_fn, padding=(1, 1, 0), indice_key='submzy', ),
        # )
        # self.convy = spconv.SparseSequential(
        #     # block(16, 32, (1, 3, 1), norm_fn=norm_fn, padding=(0, 1, 0), indice_key='submy', conv_type='spconv'),
        #     block(32, 64, (3, 1, 3), norm_fn=norm_fn, padding=(1, 0, 1), indice_key='submzx'),
        #     block(64, 64, (3, 1, 3), norm_fn=norm_fn, padding=(1, 0, 1), indice_key='submzx', ),
        # )

        self.conv_points = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submxyz'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submxyz'),
        )
        # self.score = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=64, bias=False),
        #     nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=64, out_features=32, bias=False),
        #     nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=4, bias=True),
        # )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 64
        self.backbone_channels = {
            # 'x_conv1': 256,
            # 'x_conv2': 32,
            # 'x_conv3': 64,
            # 'x_conv4': 64,
            # 'x_points': 256
            # 'x_conv2_mean': 32,
            # 'x_conv2_max': 32,
            'x_points_mean': 256,
            'x_points_max': 256,
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_points = self.conv_points(x_conv2)
        # x_convx = self.convx(x_conv2)
        # x_convy = self.convy(x_conv2)
        # x_convz = self.convz(x_conv2)
        # feature_base = torch.cat([x_points.features, x_convx.features, x_convy.features, x_convz.features], dim=-1)
        # score = torch.softmax(self.score(feature_base), dim=-1)
        # xyz_features = score[:, 0:1] * x_points.features
        # xy_features = score[:, 1:2] * x_convz.features
        # yz_features = score[:, 2:3] * x_convx.features
        # xz_features = score[:, 3:] * x_convy.features
        # x_points = x_points.replace_feature(xyz_features + xy_features + yz_features + xz_features)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_points_mean': x_points,
                'x_points_max': x_points,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_points_mean': 2,
                'x_points_max': 2,
            }
        })

        batch_dict['point_features'] = x_points.features
        point_coords = common_utils.get_voxel_centers(
            x_points.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x_points.indices[:, 0:1].float(), point_coords), dim=1)
        batch_dict['point_indices'] = x_points.indices
        return batch_dict


class SpaceVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subminput'),
            norm_fn(16),
            nn.ReLU()
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv1', active=True),
        )
        self.conv2 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2', active=True),
        )
        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(32, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2', active=True),
        #
        # )
        # self.conv3 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(32, 32, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv3'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv3', active=True),
        #
        # )
        self.conv_points = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submxyz'),
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submxyz', active=True),
        )
        self.conv_out = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submxyz'),
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submxyz', active=True),
        )

        # self.input_transform = spconv.SparseSequential(
        #     block(3, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', active=True),
        # )
        self.conv1_transform = spconv.SparseSequential(
            block(3, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subminput'),
            block(16, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv1', active=True),
        )
        self.conv2_transform = spconv.SparseSequential(
            block(3, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submconv1'),
            block(16, 16, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2', active=True),
        )
        # self.conv2_transform = spconv.SparseSequential(
        #     block(3, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submconv1'),
        #     block(16, 16, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2', active=True),
        # )
        # self.conv3_transform = spconv.SparseSequential(
        #     block(3, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2'),
        #     block(16, 16, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv3', active=True),
        # )
        # self.pointout_transform = spconv.SparseSequential(
        #     block(3, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submconv3'),
        #     block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv3', active=True),
        # )

        # last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(32, 32, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(32),
        #     nn.ReLU()
        # )
        # self.out_transform = spconv.SparseSequential(
        #     block(3, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        #     block(16, 16, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=last_pad, indice_key='spconv_down2', conv_type='spconv'),
        #     block(16, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submout', active=True),
        # )
        # self.relu = nn.ReLU()
        self.num_point_features = 32
        self.backbone_channels = {
            # 'x_conv1': 256,
            # 'x_conv2': 32,
            # 'x_conv3': 64,
            # 'x_conv4': 64,
            'x_point': 256,
            # 'x_conv2_mean': 32,
            # 'x_conv2_max': 32,
            # 'x_points_mean': 256,
            # 'x_points_max': 256,
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_input_coords = common_utils.get_voxel_centers(
            input_sp_tensor.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        x_input_coords_sp = spconv.SparseConvTensor(
            features=x_input_coords,
            indices=input_sp_tensor.indices,
            spatial_shape=input_sp_tensor.spatial_shape,
            batch_size=batch_size
        )
        # x_input_spatial_features = self.input_transform(x_input_coords_sp)
        x = self.conv_input(input_sp_tensor)
        # x = x.replace_feature(x.features + x_input_spatial_features.features)

        x_conv1 = self.conv1(x)
        x_conv1_spatial_features = self.conv1_transform(x_input_coords_sp)
        x_conv1 = x_conv1.replace_feature(x_conv1.features + x_conv1_spatial_features.features)

        x_conv1_coords = common_utils.get_voxel_centers(
            x_conv1.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        x_conv1_coords_sp = spconv.SparseConvTensor(
            features=x_conv1_coords,
            indices=x_conv1.indices,
            spatial_shape=x_conv1.spatial_shape,
            batch_size=batch_size
        )
        x_conv2 = self.conv2(x_conv1)
        x_conv2_spatial_features = self.conv2_transform(x_conv1_coords_sp)
        x_conv2 = x_conv2.replace_feature(x_conv2.features + x_conv2_spatial_features.features)

        # x_conv2_coords = common_utils.get_voxel_centers(
        #     x_conv2.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # x_conv2_sp = spconv.SparseConvTensor(
        #     features=x_conv2_coords,
        #     indices=x_conv2.indices,
        #     spatial_shape=x_conv2.spatial_shape,
        #     batch_size=batch_size
        # )
        # x_conv3_spatial_features = self.conv3_transform(x_conv2_sp)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv3 = x_conv3.replace_feature(x_conv3.features + x_conv3_spatial_features.features)

        x_point = self.conv_points(x_conv1)
        # x_point = x_point.replace_feature(x_point.features + x_pointout_spatial_features.features)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv2)
        # x_out_spatial_features = self.out_transform(x_conv2_sp)
        # out = out.replace_feature(out.features + x_out_spatial_features.features)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 4
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_point': x_point,
                # 'x_points_mean': x_point,
                # 'x_points_max': x_point,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_point': 2,
                # 'x_points_mean': 2,
                # 'x_points_max': 1,
            }
        })

        batch_dict['point_features'] = x_point.features
        x_point_coords = common_utils.get_voxel_centers(
            x_point.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_dict['point_coords'] = torch.cat((x_point.indices[:, 0:1].float(), x_point_coords), dim=1)
        batch_dict['point_indices'] = x_point.indices
        return batch_dict

class SparseTensor(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subminput'),
            norm_fn(16),
            nn.ReLU()
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv1', active=True),
        )
        self.conv2 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv2', active=True),
        )
        self.conv3 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv3'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv3', active=True),
        )
        self.conv4 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv4'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv4', active=True),
        )
        self.conv5 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv5'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submconv5', active=True),
        )


        self.num_point_features = 128
        self.num_bev_features = {
            'x_conv3': 192,
            'x_conv4': 96,
            'x_conv5': 64,
        }
        self.backbone_channels = {
            'x_conv1': 384,
            'x_conv2': 384,
            'x_conv3': 384,
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_point = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x_point)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        x_conv3_features = x_conv3.dense()
        N, C, D, H, W = x_conv3_features.shape
        x_conv3_features = x_conv3_features.view(N, C * D, H, W)

        x_conv4_features = x_conv4.dense()
        N, C, D, H, W = x_conv4_features.shape
        x_conv4_features = x_conv4_features.view(N, C * D, H, W)

        x_conv5_features = x_conv5.dense()
        N, C, D, H, W = x_conv5_features.shape
        x_conv5_features = x_conv5_features.view(N, C * D, H, W)

        x_conv1_coords = common_utils.get_voxel_centers(
            x_conv1.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        x_conv1_coords = torch.cat((x_conv1.indices[:, 0:1].float(), x_conv1_coords), dim=1)

        x_conv2_coords = common_utils.get_voxel_centers(
            x_conv2.indices[:, 1:], downsample_times=4, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        x_conv2_coords = torch.cat((x_conv2.indices[:, 0:1].float(), x_conv2_coords), dim=1)

        x_conv3_coords = common_utils.get_voxel_centers(
                x_conv3.indices[:, 1:], downsample_times=8, voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
        x_conv3_coords = torch.cat((x_conv3.indices[:, 0:1].float(), x_conv3_coords), dim=1)

        x_conv4_coords = common_utils.get_voxel_centers(
            x_conv4.indices[:, 1:], downsample_times=16, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        x_conv4_coords = torch.cat((x_conv4.indices[:, 0:1].float(), x_conv4_coords), dim=1)

        x_conv5_coords = common_utils.get_voxel_centers(
            x_conv5.indices[:, 1:], downsample_times=32, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        x_conv5_coords = torch.cat((x_conv5.indices[:, 0:1].float(), x_conv5_coords), dim=1)


        batch_dict.update({
            # 'encoded_spconv_tensor': x_point,
            'encoded_spconv_tensor_stride': 8
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 2,
                'x_conv2': 4,
                'x_conv3': 8,
                'x_conv4': 16,
                'x_conv5': 32,
            }
        })
        batch_dict.update({
            'multi_scale_2d_features': {
                # 'x_conv1': x_conv1_features,
                # 'x_conv2': x_conv2_features,
                'x_conv3': x_conv3_features,
                'x_conv4': x_conv4_features,
                'x_conv5': x_conv5_features,

            }
        })
        batch_dict.update({
            'multi_scale_coords': {
                'x_conv1': x_conv1_coords,
                'x_conv2': x_conv2_coords,
                'x_conv3': x_conv3_coords,
                'x_conv4': x_conv4_coords,
                'x_conv5': x_conv5_coords,
            }
        })

        # batch_dict['point_features'] = x_point.features
        # x_point_coords = common_utils.get_voxel_centers(
        #     x_point.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_point.indices[:, 0:1].float(), x_point_coords), dim=1)
        # batch_dict['point_indices'] = x_point.indices

        # batch_dict['spatial_shape'] =self.sparse_shape
        # batch_dict['indices'] = input_sp_tensor.indices
        return batch_dict

class TransformToSparseTensor(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.sparse_shape = grid_size[::-1]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(32, 32, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='spconv4', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        # self.conv5 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(32, 32, (1, 3, 3), norm_fn=norm_fn, stride=1, padding=(0, 1, 1), indice_key='spconv5', conv_type='spconv'),
        #     block(32, 32, 1, norm_fn=norm_fn, padding=0, indice_key='subm5'),
        #     block(32, 32, (1, 3, 3), norm_fn=norm_fn, stride=1, padding=(0, 1, 1), indice_key='spconv6', conv_type='spconv'),
        #     block(32, 32, 1, norm_fn=norm_fn, padding=0, indice_key='subm6'),
        #     block(32, 32, (1, 3, 3), norm_fn=norm_fn, stride=1, padding=(0, 1, 1), indice_key='spconv7', conv_type='spconv'),
        #     block(32, 32, 1, norm_fn=norm_fn, padding=0, indice_key='subm7'),
        #     block(32, 32, (1, 3, 3), norm_fn=norm_fn, stride=1, padding=(0, 1, 1), indice_key='spconv8', conv_type='spconv'),
        #     block(32, 32, 1, norm_fn=norm_fn, padding=0, indice_key='subm8'),
        # )

        # last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(64, 64, 1, stride=(1, 1, 1), padding=0,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(64),
        #     nn.ReLU(),
        # )

        self.num_point_features = 128
        self.num_bev_features = {
            'x_conv3': 192,
            'x_conv4': 96,
            'x_conv5': 64,
        }
        self.backbone_channels = {
            'sp_tensor_1x': 256,
            # 'sp_tensor_2x': 32,
            # 'x_conv3': 384,
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        # x_conv6 = self.conv6(x_conv5)
        # x_out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8
        })

        # batch_dict['spatial_shape_1x'] = self.sparse_shape
        # batch_dict['indices_1x'] = sp_tensor.indices

        # batch_dict['spatial_shape_2x'] = conv1.spatial_shape
        # batch_dict['indices_2x'] = conv1.indices
        # batch_dict['sp_tensor_features_2x'] = conv1.features
        conv1_coords = common_utils.get_voxel_centers(
            x_conv1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        sp_coords_1x = torch.cat((x_conv1.indices[:, 0:1].float(), conv1_coords), dim=1)
        #
        # conv2_coords = common_utils.get_voxel_centers(
        #     x_conv2.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # sp_coords_2x = torch.cat((x_conv2.indices[:, 0:1].float(), conv2_coords), dim=1)
        # batch_dict['raw_points_features'] = x_conv2.features
        # batch_dict['raw_points_bxyz'] = sp_coords_2x
        # batch_dict['raw_points_bxyz'] = batch_dict['points'][:, :-1]
        #
        # conv3_coords = common_utils.get_voxel_centers(
        #     x_conv3.indices[:, 1:], downsample_times=4, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # sp_coords_4x = torch.cat((x_conv3.indices[:, 0:1].float(), conv3_coords), dim=1)
        #
        # conv4_coords = common_utils.get_voxel_centers(
        #     x_conv4.indices[:, 1:], downsample_times=8, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # sp_coords_8x = torch.cat((x_conv4.indices[:, 0:1].float(), conv4_coords), dim=1)

        # point_coords_list = []
        # for i in range(batch_size):
        #     # cur_1x_mask = sp_coords_1x[:, 0] == i
        #     cur_2x_mask = sp_coords_2x[:, 0] == i
        #     cur_8x_mask = sp_coords_8x[:, 0] == i
        #
        #     # cur_1x_coords = sp_coords_1x[cur_1x_mask]
        #     cur_2x_coords = sp_coords_2x[cur_2x_mask]
        #     cur_8x_coords = sp_coords_8x[cur_8x_mask]
        #
        #     # point_coords_list.append(cur_1x_coords)
        #     point_coords_list.append(cur_2x_coords)
        #     point_coords_list.append(cur_8x_coords)
        #
        # point_coords = torch.cat(point_coords_list, dim=0)
        batch_dict['raw_points_bxyz'] = sp_coords_1x
        batch_dict['point_features'] = x_conv1.features
        return batch_dict


class Point2Sparse(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv_scale1 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.conv_scale2 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        self.conv5 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='spconv5', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
        )
        self.conv_scale3 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
            block(128, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
            block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
        )

        self.backward_scale1 = spconv.SparseSequential(
            block(128+3, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.backward_scale2 = spconv.SparseSequential(
            block(128+3, 64, 3, norm_fn=norm_fn, indice_key='spconv5', conv_type='inverseconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        self.backward_scale3 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
        )

        self.forward_scale1 = spconv.SparseSequential(
            block(128+3, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.forward_scale2 = spconv.SparseSequential(
            block(192+3, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='forward_scale2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        self.forward_scale3 = spconv.SparseSequential(
            block(192+3, 64, 3, norm_fn=norm_fn, stride=(1, 2, 2), padding=1, indice_key='forward_scale3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm5'),
        )

        self.scale1_trans = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(192+3, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='scale1_trans', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        self.scale2_trans = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(192+3, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        self.scale3_trans = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(192+3, 64, 3, norm_fn=norm_fn, indice_key='spconv5', conv_type='inverseconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.num_point_features = 128
        self.backbone_channels = {
            # 'x_conv2': 32,
            # 'x_conv3': 64,
            # 'x_conv4': 64,
            'multi_scale': 192,
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_scale1 = self.conv_scale1(x_conv3)
        x_conv4 = self.conv4(x_scale1)
        x_scale2 = self.conv_scale2(x_conv4)
        x_conv5 = self.conv5(x_scale2)
        x_scale3 = self.conv_scale3(x_conv5)

        downsample_times_scale1 = torch.tensor([4, 4, 40/11], device=voxel_features.device)
        xyz_scale1 = common_utils.get_voxel_centers(
            x_scale1.indices[:, 1:], downsample_times=downsample_times_scale1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        downsample_times_scale2 = torch.tensor([8, 8, 8], device=voxel_features.device)
        xyz_scale2 = common_utils.get_voxel_centers(
            x_scale2.indices[:, 1:], downsample_times=downsample_times_scale2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        downsample_times_scale3 = torch.tensor([16, 16, 8], device=voxel_features.device)
        xyz_scale3 = common_utils.get_voxel_centers(
            x_scale3.indices[:, 1:], downsample_times=downsample_times_scale3, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        # backward scale
        x_scale3_back = self.backward_scale3(x_scale3)
        x_scale3_back_temp = torch.cat([xyz_scale3, x_scale3.features, x_scale3_back.features], dim=-1)
        x_scale3_back = replace_feature(x_scale3_back, x_scale3_back_temp)

        x_scale2_back = self.backward_scale2(x_scale3_back)
        x_scale2_back_temp = torch.cat([xyz_scale2, x_scale2.features, x_scale2_back.features], dim=-1)
        x_scale2_back = replace_feature(x_scale2_back, x_scale2_back_temp)

        x_scale1_back = self.backward_scale1(x_scale2_back)
        x_scale1_back_temp = torch.cat([xyz_scale1, x_scale1.features, x_scale1_back.features], dim=-1)
        x_scale1_back = replace_feature(x_scale1_back, x_scale1_back_temp)

        # foreward scale
        x_scale1_forward = self.forward_scale1(x_scale1_back)
        x_scale1_forward_temp = torch.cat([x_scale1_back.features, x_scale1_forward.features], dim=-1)
        x_scale1_forward = replace_feature(x_scale1_forward, x_scale1_forward_temp)

        x_scale2_forward = self.forward_scale2(x_scale1_forward)
        x_scale2_forward_temp = torch.cat([x_scale2_back.features, x_scale2_forward.features], dim=-1)
        x_scale2_forward = replace_feature(x_scale2_forward, x_scale2_forward_temp)

        x_scale3_forward = self.forward_scale3(x_scale2_forward)
        x_scale3_forward_temp = torch.cat([x_scale3_back.features, x_scale3_forward.features], dim=-1)
        x_scale3_forward = replace_feature(x_scale3_forward, x_scale3_forward_temp)

        x_scale1_norm = self.scale1_trans(x_scale1_forward)
        x_scale2_norm = self.scale2_trans(x_scale2_forward)
        x_scale3_norm = self.scale3_trans(x_scale3_forward)

        scale_features = torch.cat([x_scale1_norm.features,
                                    x_scale2_norm.features,
                                    x_scale3_norm.features], dim=-1)
        x_scale2_norm = replace_feature(x_scale2_norm, scale_features)

        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                # 'x_conv2': x_conv2,
                # 'x_conv3': x_conv3,
                # 'x_conv4': x_conv_scale1,
                'multi_scale': x_scale2_norm,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                # 'x_conv2': [2, 2, 40 / 21],
                # 'x_conv3': [4, 4, 40 / 11],
                # 'x_conv4': [8, 8, 8],
                'multi_scale': [8, 8, 8]
            }
        })


        # downsample_times = torch.tensor([1, 1, 1], device=voxel_features.device)
        # conv1_coords = common_utils.get_voxel_centers(
        #     x_conv1.indices[:, 1:], downsample_times=downsample_times, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # sp_coords_1x = torch.cat((x_conv1.indices[:, 0:1].float(), conv1_coords), dim=1)
        # batch_dict['raw_points_bxyz'] = sp_coords_1x
        # batch_dict['raw_points_features'] = x_conv1.features

        return batch_dict