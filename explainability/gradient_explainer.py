"""
Gradient-based explainer for GNN models predicting dementia risk from EHR graphs.

Explanation method: Vanilla gradient wrt node features and edge attributes.
Attribution: sum of feature gradients per node, per edge.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, List, Tuple
import warnings


class Explanation:
    """Container for explanation outputs."""

    def __init__(self,
                 node_imp: Optional[torch.Tensor] = None,
                 edge_imp: Optional[torch.Tensor] = None,
                 graph_data: Optional[object] = None,
                 node_ids: Optional[torch.Tensor] = None):
        """
        Args:
            node_imp: [num_nodes] importance scores per node
            edge_imp: [num_edges] importance scores per edge
            graph_data: torch_geometric.data.Data object
            node_ids: [num_codes] diagnosis code indices for diagnosis nodes
        """
        self.node_imp = node_imp
        self.edge_imp = edge_imp
        self.graph_data = graph_data
        self.node_ids = node_ids

    def get_top_nodes(self, k: int = 5, exclude_patient: bool = True) -> List[Tuple[int, float]]:
        """
        Get top-k nodes by importance.

        Args:
            k: Number of top nodes
            exclude_patient: If True, exclude patient node (index 0)

        Returns:
            List of (node_idx, importance) tuples, sorted by importance descending
        """
        if self.node_imp is None:
            return []

        node_imp = self.node_imp.detach().cpu()
        start_idx = 1 if exclude_patient else 0
        top_k_vals, top_k_indices = torch.topk(node_imp[start_idx:], k=min(k, node_imp[start_idx:].numel()))
        top_k_indices = top_k_indices + start_idx

        return list(zip(top_k_indices.tolist(), top_k_vals.tolist()))

    def get_top_edges(self, k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get top-k edges by importance.

        Args:
            k: Number of top edges

        Returns:
            List of ((src, dst), importance) tuples, sorted by importance descending
        """
        if self.edge_imp is None or self.graph_data is None:
            return []

        edge_imp = self.edge_imp.detach().cpu()
        edge_index = self.graph_data.edge_index.detach().cpu()

        top_k_vals, top_k_indices = torch.topk(edge_imp, k=min(k, edge_imp.numel()))

        result = []
        for idx, val in zip(top_k_indices.tolist(), top_k_vals.tolist()):
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()
            result.append(((src, dst), val))

        return result


class GradientExplainer:
    """
    Gradient-based explainer for whole-graph GNN predictions.

    Computes gradients of model output wrt input features and edge attributes.
    Node importance = sum of feature gradients per node.
    Edge importance = sum of edge attribute gradients per edge.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 criterion: Optional[Callable] = None):
        """
        Args:
            model: GNN model to explain
            criterion: Loss function. If None, uses cross-entropy for classification.
        """
        self.model = model
        self.criterion = criterion or F.cross_entropy
        self.model.eval()

    def explain_graph(self,
                     data: object,
                     target_class: Optional[int] = None,
                     aggregate_node_imp: Callable = torch.sum) -> Explanation:
        """
        Generate explanation for a single graph.

        Args:
            data: torch_geometric.data.Data with forward_compatible attributes
                  (x, edge_index, edge_attr, batch, code_ids, is_patient)
            target_class: Target class for gradient. If None, uses predicted class.
            aggregate_node_imp: Function to aggregate gradients per node

        Returns:
            Explanation object with node_imp and edge_imp
        """
        # Clone data and move to model device
        data = data.clone()
        device = next(self.model.parameters()).device
        data = data.to(device)

        # Enable gradients
        x = data.x.clone().detach().requires_grad_(True)
        edge_attr = None
        if data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_attr = data.edge_attr.clone().detach().requires_grad_(True)

        # `PatientICDGNN_BioBERT.forward` only reads `x` for the *patient*
        # node -- diagnosis-leaf-node features are looked up separately
        # from the code-embedding table via `code_ids` (see `gnn/model.py`,
        # `self.lookup_code_vecs(code_ids)`), so `x[code_node_rows]` is an
        # unused placeholder and its gradient is always exactly zero. An
        # earlier version of this explainer only ever read `x.grad`, which
        # meant every diagnosis node was reported with importance 0 no
        # matter what actually drove the prediction -- the one thing this
        # explainer exists to surface. Fix: monkey-patch
        # `model.lookup_code_vecs` for the duration of this call to capture
        # the actual embedding-lookup tensor and retain its gradient.
        code_grad_holder = {}
        orig_lookup = self.model.lookup_code_vecs

        def _capturing_lookup(code_ids):
            vecs = orig_lookup(code_ids)
            if not vecs.requires_grad:
                vecs = vecs.detach().requires_grad_(True)
            vecs.retain_grad()
            code_grad_holder["vecs"] = vecs
            return vecs

        self.model.lookup_code_vecs = _capturing_lookup

        # Forward pass
        self.model.zero_grad()
        data_copy = data.clone()
        data_copy.x = x
        if edge_attr is not None:
            data_copy.edge_attr = edge_attr

        try:
            with torch.enable_grad():
                logits = self.model(data_copy)

            # Determine target class if not specified
            if target_class is None:
                target_class = logits.argmax(dim=1, keepdim=True)
            else:
                target_class = torch.tensor([target_class], device=device).view(-1, 1)

            # Compute loss and gradients
            loss = self.criterion(logits, target_class.view(-1))
            loss.backward()

            # Combine patient-node importance (from x.grad) with
            # diagnosis-node importance (from the captured code-embedding
            # lookup gradient) into one [num_nodes] tensor aligned with the
            # graph's node ordering -- `graph_construction.row_to_graph`
            # always places the patient at node 0 and diagnosis leaves at
            # nodes 1..N in the same order as `code_ids`, and PyG's default
            # batch collation preserves that per-graph order, so `is_patient`
            # / `code_ids` line up correctly here even for batched graphs.
            node_imp = None
            if x.grad is not None:
                is_patient = data.is_patient if hasattr(data, "is_patient") else None
                node_imp = torch.zeros(x.size(0), device=x.device, dtype=x.grad.dtype)
                if is_patient is not None:
                    p_idx = torch.nonzero(is_patient, as_tuple=False).view(-1)
                    c_idx = torch.nonzero(~is_patient, as_tuple=False).view(-1)
                    node_imp[p_idx] = aggregate_node_imp(x.grad[p_idx], dim=1)
                    code_vecs = code_grad_holder.get("vecs")
                    if code_vecs is not None and code_vecs.grad is not None:
                        node_imp[c_idx] = aggregate_node_imp(code_vecs.grad, dim=1)
                else:
                    node_imp = aggregate_node_imp(x.grad, dim=1)

            # Aggregate gradients per edge
            edge_imp = None
            if edge_attr is not None and edge_attr.grad is not None:
                edge_imp = aggregate_node_imp(edge_attr.grad, dim=1)

            exp = Explanation(
                node_imp=node_imp,
                edge_imp=edge_imp,
                graph_data=data,
                node_ids=data.code_ids if hasattr(data, 'code_ids') else None
            )
        finally:
            self.model.lookup_code_vecs = orig_lookup

        return exp

    def explain_batch(self,
                     data_list: List[object],
                     target_classes: Optional[List[int]] = None,
                     aggregate_node_imp: Callable = torch.sum) -> List[Explanation]:
        """
        Generate explanations for multiple graphs.

        Args:
            data_list: List of torch_geometric.data.Data objects
            target_classes: List of target classes. If None, uses predicted classes.
            aggregate_node_imp: Function to aggregate gradients per node

        Returns:
            List of Explanation objects
        """
        explanations = []
        for i, data in enumerate(data_list):
            target_class = target_classes[i] if target_classes is not None else None
            exp = self.explain_graph(data, target_class=target_class,
                                   aggregate_node_imp=aggregate_node_imp)
            explanations.append(exp)

        return explanations


class SaliencyExplainer(GradientExplainer):
    """Alias for GradientExplainer using standard saliency computation."""

    def __init__(self, model, criterion=None):
        super().__init__(model, criterion)


# Convenience function for one-off explanations
def explain_with_gradients(model: torch.nn.Module,
                          data: object,
                          target_class: Optional[int] = None,
                          criterion: Optional[Callable] = None) -> Explanation:
    """
    Quick interface to generate gradient-based explanation.

    Args:
        model: GNN model
        data: torch_geometric.data.Data
        target_class: Target class. If None, uses predicted class.
        criterion: Loss function. If None, uses cross-entropy.

    Returns:
        Explanation object
    """
    explainer = GradientExplainer(model, criterion=criterion)
    return explainer.explain_graph(data, target_class=target_class)
