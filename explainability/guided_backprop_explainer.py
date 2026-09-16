"""
Guided BackPropagation explainer for GNN models predicting dementia risk from EHR graphs.

Explanation method: Modified backpropagation with ReLU gating on gradients.
Only positive gradients (saliency) are propagated backward through ReLU layers.
Attribution: sum of modified gradients per node, per edge.

Reference: Bach et al. "Deep Inside Convolutional Networks" (2013)

Implementation note: `PatientICDGNN_BioBERT` (the ground-truth model this
explainer targets -- see `gnn/model.py`, ported verbatim from the original
notebooks) applies its two post-GINEConv ReLUs as a functional tensor method
call (`h = self.conv1(...).relu()`), not as an `nn.ReLU` submodule, so they
cannot be intercepted with `register_full_backward_hook` on `nn.ReLU`
instances (an earlier version of this file only hooked `nn.ReLU` submodules,
which correctly gated the ReLUs *inside* each GINEConv's internal edge-MLP
but silently missed these two outer ones -- verified empirically: node
importances could come out slightly negative, e.g. -0.029, violating the
"guided backprop output is elementwise non-negative" invariant asserted in
`test_explainers.py`). Rather than alter the ground-truth model to swap in
`nn.ReLU` submodules purely for explainability's convenience, this version
temporarily monkey-patches `torch.Tensor.relu` / `torch.relu` / `F.relu`
(all three call sites `.relu()` could resolve to) to route through a custom
autograd Function implementing the guided-backprop gradient rule, for the
duration of a single `explain_graph()` call only; the original functions are
always restored in a `finally` block, even for `nn.ReLU` (whose `forward()`
itself calls `F.relu` internally), which lets this single mechanism replace
the previous split hook-based approach entirely.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, List, Tuple
import warnings


class _GuidedReLUFunction(torch.autograd.Function):
    """
    Autograd Function implementing the guided-backprop ReLU rule: gradient
    flows backward only where the *input* was positive (standard ReLU
    backward) AND where the *incoming* gradient itself is positive (the
    extra "guided" clipping on top of standard backward). Standard autograd
    already gives the first condition for free once the second is applied,
    since `grad_input` inherits the input mask by construction below.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[grad_output < 0] = 0  # guided: block negative incoming gradient
        grad_input[input < 0] = 0  # standard ReLU backward: block inactive units
        return grad_input


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


class GuidedBackpropExplainer:
    """
    Guided BackPropagation explainer for whole-graph GNN predictions.

    During backward pass, ReLU activations gate gradients:
    - Gradient only flows backward if both the activation AND gradient are positive
    - This highlights features that actively contribute to the prediction

    Attributes:
        model: GNN model to explain
        criterion: Loss function (default: cross-entropy)
        hooks: List of registered backward hooks
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

    def _patch_relu(self):
        """
        Monkey-patch every entry point `.relu()` could resolve to
        (`torch.Tensor.relu`, `torch.relu`, `torch.nn.functional.relu`) so
        that, for the duration of one explanation, all ReLUs in the model
        -- whether called as a tensor method, a bare function, or via an
        `nn.ReLU` submodule (whose `forward()` itself delegates to
        `F.relu`) -- route through `_GuidedReLUFunction` instead. Returns
        the originals so `_unpatch_relu` can restore them exactly.
        """
        originals = (torch.Tensor.relu, torch.relu, F.relu)

        def _guided_relu(input, *args, **kwargs):
            return _GuidedReLUFunction.apply(input)

        torch.Tensor.relu = _guided_relu
        torch.relu = _guided_relu
        F.relu = _guided_relu
        return originals

    def _unpatch_relu(self, originals):
        """Restore the real `relu` implementations patched by `_patch_relu`."""
        torch.Tensor.relu, torch.relu, F.relu = originals

    def explain_graph(self,
                     data: object,
                     target_class: Optional[int] = None,
                     aggregate_node_imp: Callable = torch.sum) -> Explanation:
        """
        Generate guided backprop explanation for a single graph.

        Args:
            data: torch_geometric.data.Data with attributes
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

        # Patch ReLU (tensor-method, functional, and nn.ReLU-via-F.relu call
        # sites all at once) so gradients are guided-clipped end to end.
        relu_originals = self._patch_relu()

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

            # Compute loss and gradients (with ReLUs guided-gated)
            loss = self.criterion(logits, target_class.view(-1))
            loss.backward()

            # Combine patient-node importance (from x.grad) with
            # diagnosis-node importance (from the captured code-embedding
            # lookup gradient) into one [num_nodes] tensor aligned with the
            # graph's node ordering -- see the matching comment in
            # `gradient_explainer.py` for why this alignment holds even for
            # batched graphs.
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
            # Always restore the real relu implementations and lookup fn,
            # even on error.
            self._unpatch_relu(relu_originals)
            self.model.lookup_code_vecs = orig_lookup

        return exp

    def explain_batch(self,
                     data_list: List[object],
                     target_classes: Optional[List[int]] = None,
                     aggregate_node_imp: Callable = torch.sum) -> List[Explanation]:
        """
        Generate guided backprop explanations for multiple graphs.

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


# Convenience function for one-off explanations
def explain_with_guided_backprop(model: torch.nn.Module,
                                data: object,
                                target_class: Optional[int] = None,
                                criterion: Optional[Callable] = None) -> Explanation:
    """
    Quick interface to generate guided backprop explanation.

    Args:
        model: GNN model
        data: torch_geometric.data.Data
        target_class: Target class. If None, uses predicted class.
        criterion: Loss function. If None, uses cross-entropy.

    Returns:
        Explanation object
    """
    explainer = GuidedBackpropExplainer(model, criterion=criterion)
    return explainer.explain_graph(data, target_class=target_class)
