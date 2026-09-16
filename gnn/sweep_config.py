"""
WandB Bayesian hyperparameter sweep configuration for the GNN.

Recovered verbatim from `neurips_ad_graphs.ipynb` cell 13. Matches the
search space reported in the paper's Appendix A2.
"""

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "epochs": {"value": 50},
        "batch_size": {"values": [8, 16, 32]},
        "hidden": {"values": [96, 128, 192, 256]},
        "EMB_DIM": {"values": [96, 128, 192, 256]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
        "lr_main": {"distribution": "log_uniform_values", "min": 3e-4, "max": 3e-3},
        "lr_emb": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},
        "warmup_pct": {"values": [0.0, 0.06, 0.1]},
        "train_eps": {"values": [True, False]},
        "pool": {"values": ["mean", "add", "max"]},
        "trainable": {"values": [True]},
    },
}
