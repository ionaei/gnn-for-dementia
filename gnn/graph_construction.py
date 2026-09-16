"""
Build per-patient star graphs from the flat CSV schema (see
`data_prep/SCHEMA.md`), for the GNN pipeline.

Recovered verbatim (with only cosmetic renaming) from `neurips_ad_graphs.ipynb`
cells 1-8 and confirmed identical in `traffic_light.ipynb` cell 1. This is
the code that determines, for every patient row, which ICD-10 blocks they
have (`{block}_present == 1`), and builds a `torch_geometric.data.Data`
object: one central patient node [age, sex, PRS] connected to one leaf node
per diagnosis, with edge attributes encoding how long ago each diagnosis was.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

CLASS_COL = "label"  # 0 = Dementia, 1 = Control (see label_map_idx below)
SEX_COL = "Sex"  # 0 = female, 1 = male
AGE_COL = "Age"  # standardized (StandardScaler fit on train only)
PRS_COL = "PRS"  # standardized

LABEL_MAP_IDX = {"Dementia": 0, "Control": 1}


def code_to_text(name: str) -> str:
    """Turn a `_`-joined block-derived code name back into readable text."""
    return name.replace("_", " ").strip()


def build_code_vocab(df):
    """
    Derive the diagnosis-code vocabulary from a dataframe's `_present`
    columns.

    Returns:
        codes (list[str]): ICD-10 block descriptions, in column order.
        code2idx (dict[str, int]): block description -> embedding row index.
        desc_to_time (dict[str, str]): block description -> its `_time`
            column name.
        code_texts (list[str]): human-readable text per code (spaces instead
            of underscores), in the same order as `codes` -- this is what
            gets fed to BioClinicalBERT in `code_embeddings_bert.py`.
    """
    present_cols = [c for c in df.columns if c.endswith("_present")]
    codes = [c.replace("_present", "") for c in present_cols]
    code2idx = {code: i for i, code in enumerate(codes)}
    desc_to_time = {code: f"{code}_time" for code in codes}
    code_texts = [code_to_text(c) for c in codes]
    return codes, code2idx, desc_to_time, code_texts


def make_random_code_embeddings(num_codes: int, emb_dim: int = 128, seed: int = None) -> torch.Tensor:
    """
    Randomly-initialised (RV) diagnosis-code embeddings: Xavier-uniform init,
    row-wise L2-normalized. This is the `code_emb_simple` matrix from the
    original notebook -- the "randomly initialised vectors" baseline the
    paper compares against BioClinicalBERT embeddings.
    """
    if seed is not None:
        # nn.init.xavier_uniform_ doesn't take a generator directly, so seed
        # the global RNG state instead (matches the original notebook, which
        # didn't seed this at all -- seeding here is an improvement for
        # reproducibility, documented as a deliberate deviation).
        torch.manual_seed(seed)
    code_emb = torch.empty(num_codes, emb_dim, dtype=torch.float32)
    nn.init.xavier_uniform_(code_emb)
    code_emb = F.normalize(code_emb, p=2, dim=1)
    return code_emb


def row_to_graph(row, codes, code2idx, desc_to_time) -> Data:
    """
    Build one patient's star graph.

    Args:
        row: a pandas Series (one row of the preprocessed dataframe) with at
            least AGE_COL, SEX_COL, PRS_COL, CLASS_COL, and the
            `{code}_present` / `{code}_time` columns for every code in
            `codes`.
        codes, code2idx, desc_to_time: as returned by `build_code_vocab`.

    Returns:
        torch_geometric.data.Data with:
            x: [1+N, 3] node features (patient row is real, code rows are
               zero placeholders -- the model looks up real code features
               via `code_ids` and `code_emb_matrix` at forward time)
            edge_index: [2, N] patient(0) -> code(1..N)
            edge_attr: [N, 3] = [years_since_diagnosis, log1p(days),
               exp(-days/180)]
            y: [1] graph label (0=Dementia, 1=Control)
            code_ids: [N] indices into the code embedding matrix
            is_patient: [1+N] bool mask, True only for node 0
    """
    age_val = float(row[AGE_COL])
    prs_val = float(row[PRS_COL])
    sex_val = float(row[SEX_COL])
    x_patient = torch.tensor([[age_val, sex_val, prs_val]], dtype=torch.float)  # [1, 3]

    present_descs = [d for d in codes if row.get(f"{d}_present", 0) == 1]
    num_codes = len(present_descs)

    code_ids = torch.tensor([code2idx[d] for d in present_descs], dtype=torch.long)

    # Placeholder node features for code nodes; real features are pulled
    # from the code embedding matrix inside the model's forward pass.
    x_codes = torch.zeros((num_codes, 3), dtype=torch.float)
    x = torch.cat([x_patient, x_codes], dim=0)  # [1+N, 3]

    if num_codes > 0:
        src = torch.zeros(num_codes, dtype=torch.long)
        dst = torch.arange(1, num_codes + 1, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)

        days = []
        for d in present_descs:
            tcol = desc_to_time[d]
            val = row.get(tcol, np.nan)
            if (val is None) or (isinstance(val, float) and np.isnan(val)):
                days.append(0.0)
            else:
                days.append(float(val))

        days = torch.tensor(days, dtype=torch.float)
        years = days / 365.0
        edge_attr = torch.stack(
            [
                years,  # years ago
                torch.log1p(days),  # log days
                torch.exp(-days / 180.0),  # ~6-month decay
            ],
            dim=1,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    y = torch.tensor([row[CLASS_COL]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.code_ids = code_ids
    is_patient = torch.zeros(x.size(0), dtype=torch.bool)
    is_patient[0] = True
    data.is_patient = is_patient
    return data


def build_graphs(df, codes, code2idx, desc_to_time):
    """Convenience wrapper: build a star graph for every row of `df`."""
    return [row_to_graph(row, codes, code2idx, desc_to_time) for _, row in df.iterrows()]
