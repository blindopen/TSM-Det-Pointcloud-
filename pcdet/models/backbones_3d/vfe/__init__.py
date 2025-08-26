from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .mean_density_vfe import MeanDensityVFE
from .sort_points import SPVFE
from .voxelwisePointConv import VPCVFE

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'MeanDensityVFE': MeanDensityVFE,
    'SPVFE': SPVFE,
    'VPCVFE': VPCVFE
}
