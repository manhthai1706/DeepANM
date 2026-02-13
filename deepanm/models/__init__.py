"""
DeepANM Models Module
Contains high-level models and training utilities
"""

from .deepanm import DeepANM
from deepanm.core.mlp import MLP
from .trainer import DeepANMTrainer
from .analysis import ANMMM_cd, ANMMM_clu

# Backward compatibility
CausalFlow = DeepANM
CausalFlowTrainer = DeepANMTrainer

__all__ = [
    'DeepANM',
    'CausalFlow',
    'MLP',
    'DeepANMTrainer',
    'CausalFlowTrainer',
    'ANMMM_cd',
    'ANMMM_clu'
]
