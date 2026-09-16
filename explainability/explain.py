#!/usr/bin/env python3
"""
Command-line interface for generating explanations for dementia risk predictions.

Usage:
    python3 explain.py --model /path/to/model.pt --graphs /path/to/graphs.pkl --method gradient --output results.csv
    python3 explain.py --model /path/to/model.pt --graphs /path/to/graphs.pkl --method guided_bp --output results.csv
"""

import argparse
import sys
import pickle
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch_geometric.data import Data

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'gnn'))
sys.path.insert(0, str(Path(__file__).parent))

from model import PatientICDGNN_BioBERT
from gradient_explainer import GradientExplainer, Explanation as GradExplanation
from guided_backprop_explainer import GuidedBackpropExplainer, Explanation as GBPExplanation


def load_model(model_path: str, device: str = 'cpu',
               pool: Optional[str] = None, head: Optional[str] = None) -> torch.nn.Module:
    """
    Load a trained PatientICDGNN_BioBERT model.

    Args:
        model_path: Path to checkpoint (either a raw state_dict, or a dict
            with a 'state_dict' key and optionally a 'config' key).
        device: Device to load onto.
        pool: Pooling strategy ('mean' | 'add' | 'max'). IMPORTANT: pooling
            has no learnable parameters, so it leaves no trace in a raw
            state_dict -- it CANNOT be inferred from checkpoint weights
            alone. If the checkpoint dict doesn't carry a 'config' with a
            'pool' entry (e.g. it's a bare state_dict, as `train.py
            --no-wandb` saves), you MUST pass this explicitly if the run
            didn't use the default 'mean', or the loaded model will
            silently behave like the wrong architecture (same numbers of
            parameters, different predictions).
        head: Classification head ('linear' | 'mlp'). Unlike pooling, an
            'mlp' head DOES have distinguishing keys in the state_dict
            (e.g. 'head.0.weight' / 'head.2.weight' vs. a single
            'head.weight'), so this is auto-detected from the checkpoint
            when not explicitly overridden.
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Assume checkpoint is either:
    # 1. A state dict (dict of tensors)
    # 2. A full model object

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Checkpoint with metadata
        state = checkpoint['state_dict']
        config = checkpoint.get('config', {})
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        # Pure state dict - need to infer config from checkpoint keys
        state = checkpoint
        config = _infer_config_from_state(state)
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    # CLI-provided --pool/--head always win: pooling in particular cannot
    # be recovered from the state_dict, so silently trusting a guessed
    # default here would build a model that loads without error but
    # computes the wrong thing.
    if pool is not None:
        config['pool'] = pool
    elif 'pool' not in config:
        warnings.warn(
            "Checkpoint does not record its pooling strategy and --pool "
            "was not given; defaulting to 'mean'. Pooling has no learnable "
            "weights, so this CANNOT be verified from the checkpoint -- if "
            "the original run used pool='add' or pool='max', predictions "
            "and explanations from this loaded model will be silently "
            "wrong. Pass --pool explicitly if you know the run's config."
        )
    if head is not None:
        config['head'] = head

    # Create model with resolved config
    model = _build_model_from_config(config, state)
    model.to(device)
    model.eval()

    return model


def _infer_config_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Infer model config from state dict keys/shapes where possible.

    `head` can be detected this way (an 'mlp' head has an extra Linear
    layer's worth of keys that a 'linear' head doesn't). `pool` cannot --
    pooling ops (mean/add/max) have no parameters -- so it is left for the
    caller to override via `load_model(..., pool=...)` if it isn't 'mean'.
    """
    head = 'mlp' if any(k.startswith('head.0.') or k.startswith('head.2.') for k in state) else 'linear'
    config = {
        'num_codes': state.get('code_emb.weight', torch.zeros(10, 64)).shape[0],
        'emb_dim': state.get('code_emb.weight', torch.zeros(10, 64)).shape[1],
        'hidden': state.get('proj_patient.weight', torch.zeros(32, 3)).shape[0],
        'edge_dim': 3,
        'out_classes': 2,
        'dropout': 0.0,
        'train_eps': False,
        'trainable': True,
        'pool': 'mean',
        'head': head,
    }
    return config


def _build_model_from_config(config: Dict[str, Any], state: Optional[Dict] = None) -> PatientICDGNN_BioBERT:
    """Build model from config dict."""
    code_emb = torch.randn(config.get('num_codes', 10), config.get('emb_dim', 64))

    model = PatientICDGNN_BioBERT(
        code_emb_matrix=code_emb,
        edge_dim=config.get('edge_dim', 3),
        hidden=config.get('hidden', 64),
        out_classes=config.get('out_classes', 2),
        dropout=config.get('dropout', 0.0),
        train_eps=config.get('train_eps', False),
        trainable=config.get('trainable', True),
        pool=config.get('pool', 'mean'),
        head=config.get('head', 'linear'),
    )

    if state is not None:
        model.load_state_dict(state, strict=False)

    return model


def load_graphs(graphs_path: str) -> List[Data]:
    """Load graphs from pickle file."""
    with open(graphs_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list):
        return data
    elif hasattr(data, '__iter__'):
        return list(data)
    else:
        return [data]


def explain_graphs(model: torch.nn.Module,
                   graphs: List[Data],
                   method: str = 'gradient',
                   target_class: Optional[int] = None,
                   device: str = 'cpu',
                   top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Generate explanations for a list of graphs.

    Args:
        model: GNN model
        graphs: List of torch_geometric.data.Data objects
        method: 'gradient' or 'guided_bp'
        target_class: Target class for explanation (if None, uses predicted class)
        device: Device to run on
        top_k: Number of top nodes/edges to extract

    Returns:
        List of explanation dicts
    """
    model.to(device)
    model.eval()

    if method == 'gradient':
        explainer = GradientExplainer(model)
    elif method == 'guided_bp':
        explainer = GuidedBackpropExplainer(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    results = []

    for i, graph in enumerate(graphs):
        graph = graph.to(device)

        # Determine target class if not provided
        if target_class is None:
            with torch.no_grad():
                logits = model(graph)
            pred_class = logits.argmax(dim=1).item()
        else:
            pred_class = target_class

        # Generate explanation
        exp = explainer.explain_graph(graph, target_class=pred_class)

        # Extract top-k nodes and edges
        top_nodes = exp.get_top_nodes(k=top_k, exclude_patient=True)
        top_edges = exp.get_top_edges(k=top_k)

        result = {
            'graph_id': i,
            'predicted_class': pred_class,
            'explanation_method': method,
            'top_diagnoses': [
                {'node_idx': int(idx), 'importance': float(imp)}
                for idx, imp in top_nodes
            ],
            'top_edges': [
                {'src': int(src), 'dst': int(dst), 'importance': float(imp)}
                for (src, dst), imp in top_edges
            ],
            'node_importances': (
                exp.node_imp.detach().cpu().tolist()
                if exp.node_imp is not None else None
            ),
            'edge_importances': (
                exp.edge_imp.detach().cpu().tolist()
                if exp.edge_imp is not None else None
            )
        }

        results.append(result)

    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate explainability attributions for dementia risk GNN.'
    )
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--graphs', required=True, help='Path to pickle file containing graphs')
    parser.add_argument('--method', default='gradient', choices=['gradient', 'guided_bp'],
                       help='Explanation method')
    parser.add_argument('--output', required=True, help='Path to save results (JSON)')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top nodes/edges to extract')
    parser.add_argument('--target_class', type=int, default=None,
                       help='Target class for explanation (if None, use predicted class)')
    parser.add_argument('--pool', default=None, choices=['mean', 'add', 'max'],
                       help="Pooling used by the checkpoint's training run. Pooling has no "
                            "learnable weights so it CANNOT be inferred from the checkpoint "
                            "-- pass this explicitly if the run did not use the default "
                            "'mean' (see gnn/train.py / risk_stratification's --pool), "
                            "otherwise the loaded model will silently use the wrong pooling.")
    parser.add_argument('--head', default=None, choices=['linear', 'mlp'],
                       help="Classification head used by the checkpoint's training run. "
                            "Usually auto-detected from the checkpoint's state_dict keys; "
                            "pass this to override if auto-detection seems wrong.")

    args = parser.parse_args()

    try:
        print(f"Loading model from {args.model}...")
        model = load_model(args.model, device=args.device, pool=args.pool, head=args.head)
        print("✓ Model loaded")

        print(f"Loading graphs from {args.graphs}...")
        graphs = load_graphs(args.graphs)
        print(f"✓ Loaded {len(graphs)} graphs")

        print(f"Generating {args.method} explanations...")
        results = explain_graphs(
            model, graphs,
            method=args.method,
            target_class=args.target_class,
            device=args.device,
            top_k=args.top_k
        )
        print(f"✓ Generated {len(results)} explanations")

        save_results(results, args.output)

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
