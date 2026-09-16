"""Explainability module for GNN-based dementia risk prediction."""

from .gradient_explainer import GradientExplainer, Explanation as GradientExplanation, explain_with_gradients
from .guided_backprop_explainer import GuidedBackpropExplainer, explain_with_guided_backprop

__all__ = [
    'GradientExplainer',
    'GuidedBackpropExplainer',
    'GradientExplanation',
    'explain_with_gradients',
    'explain_with_guided_backprop',
]
