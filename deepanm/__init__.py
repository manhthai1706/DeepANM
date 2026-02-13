"""
DeepANM: Deep Additive Noise Model for Causal Discovery
"""

__version__ = '1.0.0'

# Export main classes to top-level
from .models.deepanm import DeepANM
from .models.analysis import ANMMM_cd, ANMMM_clu
from .models.trainer import DeepANMTrainer

# Export core components for advanced users
from .core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC
from .core.hsic import hsic_gam


__all__ = [
    'DeepANM',
    'ANMMM_cd',
    'ANMMM_clu',
    'DeepANMTrainer',
    'GPPOMC_lnhsic_Core',
    'FastHSIC',
    'hsic_gam'
]
