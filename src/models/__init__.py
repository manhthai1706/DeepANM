"""
DeepANM Models Module
Contains high-level models and training utilities
"""

from .deepanm import DeepANM
from .fast_baseline import FastANM
from src.core.mlp import MLP
from src.utils.trainer import DeepANMTrainer

# Backward compatibility
CausalFlow = DeepANM
CausalFlowTrainer = DeepANMTrainer

__all__ = [
    'DeepANM',
    'CausalFlow',
    'MLP',
    'DeepANMTrainer',
    'CausalFlowTrainer',
    'FastANM'
]
