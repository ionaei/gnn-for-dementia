"""
Build the per-diagnosis-code BioClinicalBERT embedding matrix (`code_emb_bert`).

This is used as `code_emb_matrix` in `PatientICDGNN_BioBERT`, in place of the
Xavier-random `code_emb_simple` baseline, for the GNN+BioClinical paper
variant. The GNN needs one embedding PER DIAGNOSIS CODE (`[NUM_CODES, 768]`),
as distinct from the per-patient embeddings used by the BERT-family models in
`bert_models/` (one mean-pooled BioClinicalBERT embedding of a patient's full
`icd10_sequence` string).

This module builds that per-code matrix in the most direct way consistent
with how the rest of the pipeline uses BioClinicalBERT: encode each unique
diagnosis code's human-readable text (`code_texts`, from
`graph_construction.build_code_vocab`) with `emilyalsentzer/Bio_ClinicalBERT`
and mean-pool over tokens.
"""

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counted = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counted


@torch.no_grad()
def build_code_embedding_matrix(code_texts, batch_size=32, max_length=32, device=None):
    """
    Encode each diagnosis code's text with BioClinicalBERT and mean-pool to
    get a fixed-size vector per code.

    Args:
        code_texts (list[str]): human-readable diagnosis block descriptions,
            in the same order as `codes`/`code2idx` from
            `graph_construction.build_code_vocab` -- row i of the returned
            matrix corresponds to `code_texts[i]`.
        batch_size (int): encoding batch size.
        max_length (int): max token length; block descriptions are short
            phrases, so 32 is generous (vs. 512 used for full patient
            sequences elsewhere in this repo).
        device (str or torch.device, optional): defaults to CUDA if
            available, else CPU.

    Returns:
        torch.Tensor: [len(code_texts), 768] embedding matrix, usable
        directly as `code_emb_matrix` in `PatientICDGNN_BioBERT` (in place
        of the Xavier-random `code_emb_simple`).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)

    all_embs = []
    for i in range(0, len(code_texts), batch_size):
        batch = code_texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def main():
    """CLI smoke test: build embeddings for a handful of ICD-10 block names."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="code_emb_bert.pt", help="Output tensor path")
    args = parser.parse_args()

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
    from icd10_blocks import ICD10_BLOCKS  # noqa: E402

    code_texts = [b.replace("_", " ").strip() for b in ICD10_BLOCKS]
    emb = build_code_embedding_matrix(code_texts)
    torch.save(emb, args.out)
    print(f"Built code embedding matrix of shape {tuple(emb.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
