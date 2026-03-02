"""
DeepANM Models Module
Contains high-level models and training utilities
"""

from .deepanm import DeepANM
from .fast_baseline import FastANM
from .lite_baseline import LiteANM
from deepanm.core.mlp import MLP
from deepanm.utils.trainer import DeepANMTrainer

# Backward compatibility
CausalFlow = DeepANM
CausalFlowTrainer = DeepANMTrainer

__all__ = [
    'DeepANM',
    'CausalFlow',
    'MLP',
    'DeepANMTrainer',
    'CausalFlowTrainer',
    'FastANM',
    'LiteANM'
]
