"""
Advanced interpretability analyses for EmberFormer-DINO
Phase 1: Low-hanging fruit implementations with thesis-ready outputs
"""

from .thesis_utils import ThesisOutputManager, format_p_value, compute_cohens_d

__all__ = [
    'ThesisOutputManager',
    'format_p_value', 
    'compute_cohens_d',
]
