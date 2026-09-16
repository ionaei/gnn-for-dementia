"""GNN module for dementia risk prediction."""

from .model import PatientICDGNN_BioBERT, get_pool

__all__ = ['PatientICDGNN_BioBERT', 'get_pool']
