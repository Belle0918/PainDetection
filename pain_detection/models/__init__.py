"""
Pain detection models.

Exports all model builders so existing code continues to work:
    from pain_detection.models import build_rf, build_svm, CNN1D
    from pain_detection.models import AttentionFusionModel, TransferModel
"""
from .classical import build_rf, build_svm
from .cnn import CNN1D
from .attention_fusion import AttentionFusionModel
from .transfer import TransferModel
from .llm_baseline import LLMBaseline
from .crossmod_transformer import CrossModTransformer

__all__ = [
    "build_rf",
    "build_svm",
    "CNN1D",
    "AttentionFusionModel",
    "TransferModel",
    "LLMBaseline",
    "CrossModTransformer",
]
