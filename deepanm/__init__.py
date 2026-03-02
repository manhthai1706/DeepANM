"""
DeepANM: Deep Additive Noise Model for Causal Discovery
"""

__version__ = '1.0.0'

# Export main classes to top-level
from .models.deepanm import DeepANM
from .utils.trainer import DeepANMTrainer

# Export core components for advanced users
from .core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC

# Utilities
from .utils.visualize import plot_dag


__all__ = [
    'DeepANM',
    'DeepANMTrainer',
    'GPPOMC_lnhsic_Core',
    'FastHSIC',
    'plot_dag'
]
