from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate
from .DSASNet_RoI_Head import DSASNetRoIHead
from .EPoint_RoI_Head import EPointRoIHead
from .EPoint_RoI_Head_V2 import EPointRoIHeadV2


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'DSASNetRoIHead': DSASNetRoIHead,
    'EPointRoIHead': EPointRoIHead,
    'EPointRoIHeadV2': EPointRoIHeadV2
}
