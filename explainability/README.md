# GNN Explainability for Dementia Risk Prediction

This module provides explainability methods for the `PatientICDGNN_BioBERT` model that predicts dementia risk from longitudinal EHR data represented as knowledge graphs.

## Overview

The paper employs two gradient-based explainability methods to understand which diagnoses (ICD-10 codes) and temporal patterns drive the model's predictions:

1. **Gradient Explainer** — Vanilla gradient saliency: attributes importance to nodes/edges by summing gradients of the loss with respect to input features.
2. **Guided BackPropagation Explainer** — Modified ReLU-gated backprop: only positive gradients propagate, highlighting features that actively contribute to the prediction.

## Files

### Core Explainers

- **`gradient_explainer.py`** — Vanilla gradient-based explainer
  - Class: `GradientExplainer`
  - Convenience function: `explain_with_gradients(model, data, target_class)`
  - Supports single-graph and batch explanations
  
- **`guided_backprop_explainer.py`** — Guided backpropagation explainer
  - Class: `GuidedBackpropExplainer`
  - Convenience function: `explain_with_guided_backprop(model, data, target_class)`
  - Registers hooks on ReLU layers to gate gradients during backward pass
  - Produces non-negative saliencies (only beneficial features highlighted)

### Utilities

- **`test_explainers.py`** — Comprehensive smoke test
  - Creates synthetic star graphs matching the paper's format
  - Verifies model forward pass
  - Tests gradient and guided backprop explainers end-to-end
  - Confirms top-k diagnosis extraction works
  - **Run with:** `python3 test_explainers.py`

## Model Architecture (for reference)

The model being explained is `PatientICDGNN_BioBERT`, which:

- Represents each patient as a **complete bipartite star graph**:
  - 1 central patient node with features `[age, sex, PRS]`
  - 1 leaf node per ICD-10 block diagnosis
  - Edges carry temporal attributes: `[years_since_diagnosis, log1p(days), exp(-days/180)]`

- Uses **2 layers of GINEConv** (Graph Isomorphism Networks with Edge attributes)
- Applies **global pooling** (mean, add, or max)
- Final **linear classification head** (2 classes: dementia vs. control)

## Usage

### Basic Example

```python
import torch
from torch_geometric.data import Data
from gnn.model import PatientICDGNN_BioBERT
from explainability.gradient_explainer import GradientExplainer

# Load or create your model
model = PatientICDGNN_BioBERT(...)
model.load_state_dict(torch.load('path/to/checkpoint.pt'))
model.eval()

# Load a patient graph (Data object with x, edge_index, edge_attr, batch, code_ids, is_patient)
patient_graph = ...  # from your data loader

# Create explainer
explainer = GradientExplainer(model)

# Generate explanation
explanation = explainer.explain_graph(patient_graph, target_class=0)

# Get top diagnoses
top_diagnoses = explanation.get_top_nodes(k=5, exclude_patient=True)
for node_idx, importance_score in top_diagnoses:
    print(f"Node {node_idx}: importance = {importance_score:.4f}")

# Get top temporal relationships
top_edges = explanation.get_top_edges(k=5)
for (src, dst), importance_score in top_edges:
    print(f"Edge {src}->{dst}: importance = {importance_score:.4f}")
```

### Batch Explanation

```python
data_list = [patient_graph_1, patient_graph_2, patient_graph_3]

# Single explainer for multiple graphs
explanations = explainer.explain_batch(data_list)

for i, exp in enumerate(explanations):
    print(f"Patient {i}:")
    top_nodes = exp.get_top_nodes(k=3)
    print(f"  Top diagnoses: {top_nodes}")
```

### Guided BackPropagation

```python
from explainability.guided_backprop_explainer import GuidedBackpropExplainer

gbp_explainer = GuidedBackpropExplainer(model)
explanation = gbp_explainer.explain_graph(patient_graph, target_class=0)

# Guided backprop produces non-negative saliencies
print(f"Min importance: {explanation.node_imp.min()}")  # >= 0
```

## Interface

### `Explanation` Class

Both explainers return an `Explanation` object with:

```python
class Explanation:
    node_imp: Optional[torch.Tensor]       # [num_nodes] importance scores
    edge_imp: Optional[torch.Tensor]       # [num_edges] importance scores
    graph_data: Optional[Data]              # Original torch_geometric.Data
    node_ids: Optional[torch.Tensor]        # [num_codes] diagnosis code IDs

    def get_top_nodes(k=5, exclude_patient=True) -> List[Tuple[int, float]]
        """Returns [(node_idx, importance_score), ...] sorted descending"""

    def get_top_edges(k=5) -> List[Tuple[Tuple[int,int], float]]
        """Returns [((src, dst), importance_score), ...] sorted descending"""
```

### Gradient Explainer

```python
class GradientExplainer:
    def __init__(model, criterion=None):
        """criterion defaults to F.cross_entropy"""

    def explain_graph(data, target_class=None, aggregate_node_imp=torch.sum):
        """
        Generate explanation for single graph.
        
        Args:
            data: torch_geometric.data.Data
            target_class: int or None (auto-detect from prediction)
            aggregate_node_imp: Function to aggregate feature gradients per node
        
        Returns:
            Explanation object
        """

    def explain_batch(data_list, target_classes=None, aggregate_node_imp=torch.sum):
        """Generate explanations for list of graphs"""
```

### Guided BackPropagation Explainer

```python
class GuidedBackpropExplainer:
    def __init__(model, criterion=None):
        """criterion defaults to F.cross_entropy"""

    def explain_graph(data, target_class=None, aggregate_node_imp=torch.sum):
        """Same interface as GradientExplainer"""

    def explain_batch(data_list, target_classes=None, aggregate_node_imp=torch.sum):
        """Same interface as GradientExplainer"""
```

## Data Format

The explainers expect `torch_geometric.data.Data` objects with:

```python
Data(
    x=torch.Tensor,              # [num_nodes, num_features] node features
    edge_index=torch.LongTensor, # [2, num_edges] edge connectivity
    edge_attr=torch.Tensor,      # [num_edges, edge_dim] edge attributes
    y=torch.Tensor,              # [batch_size] labels
    batch=torch.Tensor,          # [num_nodes] batch indices for pooling
    code_ids=torch.LongTensor,   # [num_diagnosis_nodes] ICD-10 code indices
    is_patient=torch.BoolTensor, # [num_nodes] mask for patient vs diagnosis nodes
)
```

For the dementia/EHR model:
- **Patient node (index 0)**: `x = [age, sex, PRS]` (all normalized)
- **Diagnosis nodes (indices 1..N)**: `x = [0, 0, 0]` (features loaded from embeddings via `code_ids`)
- **Edges**: patient node → each diagnosis node (complete bipartite star)
- **Edge attributes**: `[years_since_diagnosis, log1p(days_since_diagnosis), exp(-days/180)]`

## Interpretation

### Gradient Explainer
- Positive gradient → feature increases predicted probability of the class
- Negative gradient → feature decreases predicted probability of the class
- Larger magnitude → stronger influence on prediction

### Guided BackPropagation
- Only non-negative saliencies (ReLU-gated)
- Highlights features that **actively supported** the prediction
- Useful for "what made the model predict this class"
- Less useful for understanding preventive factors


## References

- **Gradient Explainability**: Simonyan et al., "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" (ICLR 2014)
- **Guided Backpropagation**: Bach et al., "Deep Inside Convolutional Networks" (2013)
- **Original GraphXAI Repo**: https://github.com/mims-harvard/GraphXAI

## File Provenance

| File | Source | Status |
|------|--------|--------|
| `gradient_explainer.py` | Adapted from GraphXAI + custom | ✓ New, dementia-specific |
| `guided_backprop_explainer.py` | Adapted from GraphXAI `guided_bp.py` | ✓ New, dementia-specific |
| `test_explainers.py` | Custom | ✓ New, comprehensive test suite |

**Excluded sMRI-related files from original Explainers/ folder:**
- `model_sMRI.py`, `train_sMRI.py` — Different model architecture for brain imaging
- `dataset_split.py` — Hard-coded sMRI multimodal loading
- `generate_graphs_func_val.py`, `generate_node_features_func.py` — Brain region correlation matrices, not patient ICD graphs
- `AS_best_mod_exp*.ipynb`, `DT_best_mod_exp*.ipynb` — Notebooks for sMRI experiments
- `pgm_*.py` — PGM explainer (generic, but only needed if paper uses it)

