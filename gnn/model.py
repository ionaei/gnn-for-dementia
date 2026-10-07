"""
GNN model for dementia risk prediction from patient star-graphs.

This module implements `PatientICDGNN_BioBERT`, supporting two classification
head variants used for different paper results:

  - A single `nn.Linear(hidden, 2)` head, with pooling passed in as a
    constructor argument (tuned as part of the sweep). This is the variant
    that won and is reported as the paper's main GNN result.
  - An MLP classification head (`Linear -> ReLU -> Dropout -> Linear`)
    instead of a single linear layer, with pooling hardcoded to 'add' rather
    than tuned. This corresponds to the paper's GNN+MLP+RV ablation variant.

Rather than keep two near-duplicate class definitions, this module merges
them into one class with a `head` argument (`"linear"` or `"mlp"`), so both
paper variants can be trained from the same code path. The star-graph
message passing, the patient/code node split via `is_patient`, the
code-embedding lookup, and the GINEConv layers are shared across both.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool


def get_pool(name):
    """Get a PyTorch Geometric global-pooling function by name."""
    return {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}[name]


class PatientICDGNN_BioBERT(nn.Module):
    """
    Star-graph GNN for patient outcome prediction.

    Each patient is a complete bipartite star graph:
      - One central "patient" node with features [age, sex, PRS] (already
        standardized upstream).
      - One leaf "diagnosis" node per ICD-10 block the patient has, with
        node features looked up from `code_emb_matrix` (either randomly
        initialised & trainable, or precomputed BioClinicalBERT embeddings
        -- see `code_embeddings_bert.py`).
      - Edges patient -> each diagnosis leaf, carrying temporal attributes
        [years_since_diagnosis, log1p(days), exp(-days/180)].

    Args:
        code_emb_matrix (torch.Tensor): [num_codes, emb_dim] embedding
            matrix for diagnosis codes (Xavier-random or BioClinicalBERT).
        edge_dim (int): Dimension of edge attributes (3 in the paper).
        hidden (int): Hidden dimension for the GINEConv layers / projections.
        out_classes (int): Number of output classes (2: dementia vs control).
        dropout (float): Dropout rate used after each GINEConv layer and
            inside the MLP head (if `head="mlp"`).
        train_eps (bool): Whether GINEConv's epsilon is trainable.
        trainable (bool): If True, `code_emb_matrix` is wrapped in a
            trainable `nn.Embedding`; if False, it is frozen (registered as
            a buffer).
        pool (str): Graph pooling method: "mean", "add", or "max". The
            winning sweep run used a tuned value here; the risk-stratification
            pipeline hardcodes "add" to match its specific best run -- see
            `risk_stratification/`.
        head (str): "linear" (single `nn.Linear(hidden, out_classes)`,
            the paper's best-performing GNN configuration) or "mlp"
            (`Linear -> ReLU -> Dropout -> Linear`, the GNN+MLP ablation).
    """

    def __init__(
        self,
        code_emb_matrix,
        edge_dim,
        hidden,
        out_classes=2,
        dropout=0.0,
        train_eps=False,
        trainable=True,
        pool="mean",
        head="linear",
    ):
        super().__init__()
        num_codes, dim = code_emb_matrix.shape

        if trainable:
            self.code_emb = nn.Embedding.from_pretrained(code_emb_matrix, freeze=False)
        else:
            self.register_buffer("code_emb_static", code_emb_matrix)
            self.code_emb = None

        self.proj_code = nn.Linear(dim, hidden)
        self.proj_patient = nn.Linear(3, hidden)  # age, sex, PRS

        def mlp():
            return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.conv1 = GINEConv(mlp(), edge_dim=edge_dim, train_eps=train_eps)
        self.conv2 = GINEConv(mlp(), edge_dim=edge_dim, train_eps=train_eps)

        self.dropout = nn.Dropout(dropout)

        if head == "linear":
            self.classifier = nn.Linear(hidden, out_classes)
        elif head == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {head!r} (expected 'linear' or 'mlp')")
        self.head = head

        self.pool = get_pool(pool)

    def lookup_code_vecs(self, code_ids):
        """Look up diagnosis-code embeddings, trainable or frozen."""
        return self.code_emb(code_ids) if self.code_emb is not None else self.code_emb_static[code_ids]

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data or Batch): graph(s) with
                - x: node features [n, 3] (only the first 3 cols of patient
                  rows are used; code-node rows are placeholders, real
                  features come from `code_emb_matrix` via `code_ids`)
                - edge_index: [2, m]
                - edge_attr: [m, edge_dim]
                - batch: [n] graph index per node (added by PyG's DataLoader)
                - code_ids: [num_present_codes] indices into code_emb_matrix
                - is_patient: [n] bool mask, True for the one patient node
                  per graph

        Returns:
            torch.Tensor: [batch_size, out_classes] classification logits.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        is_patient = data.is_patient
        p_idx = torch.nonzero(is_patient, as_tuple=False).view(-1)
        c_idx = torch.nonzero(~is_patient, as_tuple=False).view(-1)

        # Project first, to lock dtype under autocast (matches original).
        p_feats = self.proj_patient(x[p_idx, :3])
        code_ids = data.code_ids.to(x.device)
        c_feats = self.proj_code(self.lookup_code_vecs(code_ids))

        H = p_feats.size(-1)
        h = torch.zeros(x.size(0), H, device=x.device, dtype=p_feats.dtype)
        h[p_idx] = p_feats
        h[c_idx] = c_feats

        edge_index = edge_index.to(h.device)
        if edge_attr.numel() > 0:
            edge_attr = edge_attr.to(h.device, dtype=h.dtype)

        h = self.conv1(h, edge_index, edge_attr).relu()
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr).relu()
        h = self.dropout(h)

        g = self.pool(h, batch)
        return self.classifier(g)


def build_model_from_cfg(cfg, code_emb_matrix, edge_dim=3, out_classes=2, head="linear"):
    """
    Construct a `PatientICDGNN_BioBERT` from a WandB run config dict.

    `gnn/evaluate_best_run.py` uses this helper to rebuild a model from a
    saved sweep config.

    Args:
        cfg (dict or wandb.sdk.wandb_config.Config): must provide "hidden",
            "dropout", "train_eps"; "pool" is used if present, else falls
            back to the `pool` kwarg default. "trainable" defaults to True
            (as in every sweep run actually launched).
        code_emb_matrix (torch.Tensor): [num_codes, emb_dim] embedding
            matrix matching the run's EMB_DIM.
        edge_dim (int): edge attribute dimension (3 in the paper).
        out_classes (int): number of output classes.
        head (str): "linear" or "mlp" -- pick based on which sweep config
            produced `cfg` (see module docstring).
    """
    get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: cfg[k] if k in cfg else d)
    return PatientICDGNN_BioBERT(
        code_emb_matrix=code_emb_matrix,
        edge_dim=edge_dim,
        hidden=get("hidden"),
        out_classes=out_classes,
        dropout=get("dropout", 0.0),
        train_eps=get("train_eps", False),
        trainable=get("trainable", True),
        pool=get("pool", "mean"),
        head=head,
    )
