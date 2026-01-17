# cdt/models/__init__.py
"""Model components for CDT - causal inference from text."""

from .components import CrossAttentionAggregator
from .dragonnet import DragonNet
from .uplift import UpliftNet
from .outcome_heads import OutcomeHeadsOnly, UpliftHeadsOnly
from .cnn_extractor import CNNFeatureExtractor, WordTokenizer
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor, AttentionPooling
from .causal_cnn import CausalCNNText
from .propensity_model import PropensityOnlyModel, PropensityNet, create_propensity_model_from_config

__all__ = [
    'CrossAttentionAggregator',
    'DragonNet',
    'UpliftNet',
    'OutcomeHeadsOnly',
    'UpliftHeadsOnly',
    'CNNFeatureExtractor',
    'WordTokenizer',
    'BertFeatureExtractor',
    'GRUFeatureExtractor',
    'AttentionPooling',
    'CausalCNNText',
    'PropensityOnlyModel',
    'PropensityNet',
    'create_propensity_model_from_config',
]