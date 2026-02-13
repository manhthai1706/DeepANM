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
from . import datasets

# Backward compatibility alias
CausalFlow = DeepANM
CausalFlowTrainer = DeepANMTrainer

__all__ = [
    'DeepANM',
    'CausalFlow',
    'ANMMM_cd',
    'ANMMM_clu',
    'DeepANMTrainer',
    'CausalFlowTrainer',
    'GPPOMC_lnhsic_Core',
    'FastHSIC',
    'hsic_gam',
    'datasets'
]
