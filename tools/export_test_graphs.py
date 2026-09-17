#!/usr/bin/env python3
"""
Export a data split's star graphs to a pickle file consumable by
`explainability/explain.py --graphs`.

Why this exists: `explain.py` requires a pre-pickled `List[torch_geometric.
data.Data]`, and nothing in this package ever produced one from real data --
`explainability/test_explainers.py` only builds synthetic in-memory graphs
for its own smoke test, so the CLI explainers (`explain.py`) were never
actually runnable against real patients.

This script reuses the *exact same* pipeline `gnn/train.py` uses at training
time -- `load_and_split()` (the 70/10/20 stratified split) followed by
`build_code_vocab()` + `build_graphs()` (both from `gnn/graph_construction.py`)
-- so the exported graphs' `code_ids` line up with whatever checkpoint you
point `explain.py --model` at, *as long as* you export with the SAME
--data-path and --seed used to train that checkpoint. The vocabulary
(`codes`/`code2idx`) is always derived from the TRAIN split's `_present`
columns (`build_code_vocab(df_train)`), exactly as `train.py` does, even when
you export the val or test split -- this is what keeps a diagnosis code's
`code_id` consistent between training and explanation.

Usage:
    # Export the held-out test split (default), matching a checkpoint trained
    # with `python gnn/train.py --data-path data_prep/five_updated_synthetic.csv`
    # (i.e. default --seed 42):
    python tools/export_test_graphs.py \\
        --data-path data_prep/five_updated_synthetic.csv \\
        --out explainability/test_graphs.pkl

    # Then run the CLI explainers against real held-out patients:
    python explainability/explain.py \\
        --model gnn/checkpoints_gnn/best_local.pt \\
        --graphs explainability/test_graphs.pkl \\
        --method gradient --output explainability/gradient_results.json
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

from graph_construction import build_code_vocab, build_graphs  # noqa: E402
from train import load_and_split  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=str, default="data_prep/five_updated_synthetic.csv",
                         help="MUST match the --data-path used to train the checkpoint you plan to run "
                              "explain.py against -- the diagnosis-code vocabulary (and therefore every "
                              "code_id in the exported graphs) is derived from this CSV's train split.")
    parser.add_argument("--seed", type=int, default=42,
                         help="MUST match the --seed used at train time (gnn/train.py also defaults to 42) "
                              "-- this determines which rows land in train/val/test.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                         help="Which split to export. 'test' (the default) is the held-out split that "
                              "evaluate_best_run.py / run_stratification.py also score on, so exporting it "
                              "gives you explanations for the same patients your reported metrics cover.")
    parser.add_argument("--out", type=str, default="test_graphs.pkl")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optionally export only the first N graphs of the chosen split (explaining "
                              "graphs one at a time is not fast; useful for a quick smoke test).")
    args = parser.parse_args()

    df_train, df_val, df_test = load_and_split(args.data_path, seed=args.seed)
    df_by_split = {"train": df_train, "val": df_val, "test": df_test}
    df = df_by_split[args.split]

    codes, code2idx, desc_to_time, code_texts = build_code_vocab(df_train)
    graphs = build_graphs(df, codes, code2idx, desc_to_time)

    if args.limit is not None:
        graphs = graphs[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f)

    print(
        f"Exported {len(graphs)} graph(s) from the '{args.split}' split "
        f"({len(codes)} diagnosis codes in vocab, from --data-path {args.data_path} --seed {args.seed}) "
        f"to {out_path}"
    )
    print(
        "Point explainability/explain.py --graphs at this file. --model must be a checkpoint trained on "
        f"the SAME --data-path ({args.data_path}) with the SAME --seed ({args.seed}), or the code_ids in "
        "these graphs won't line up with that checkpoint's code_emb weights."
    )


if __name__ == "__main__":
    main()
