from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .PV_SSD_A_head import PVSSDAHead
from .anchor_head_single_cls import AnchorHeadSingleCls
from .DSASNet_head import DSASNetHead
from .anchor_head_multi_cls import AnchorHeadMultiCls
from .VPC_head import VPCNetHead
from .point_head_vote import PointHeadVote
# from .point_head_vote_sasa import PointHeadVoteSASA
from .point_head_vote_sasa_statistic import PointHeadVoteSASAStatistic
from .point_head_vote_sasa_statistic_distillation import PointHeadVoteSASAStatisticDistillation

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'PVSSDAHead': PVSSDAHead,
    'AnchorHeadSingleCls': AnchorHeadSingleCls,
    'DSASNetHead': DSASNetHead,
    'AnchorHeadMultiCls': AnchorHeadMultiCls,
    'VPCNetHead': VPCNetHead,
    'PointHeadVote': PointHeadVote,
    # 'PointHeadVoteSASA': PointHeadVoteSASA,
    'PointHeadVoteSASAStatistic':PointHeadVoteSASAStatistic,
    'PointHeadVoteSASAStatisticDistillation': PointHeadVoteSASAStatisticDistillation
}
