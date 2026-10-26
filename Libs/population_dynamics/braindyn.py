"""BrainDyn author model for deterministic source-indexed field forecasts.

This adapter preserves the pinned author LSTM, sheaf operator and ODE solver.
It supplies no spike observation law or unsupported future-control interface.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from external.BrainDyn.model.braindyn import BrainDyn, BrainDynConfig
from .provenance import sha256_file


SOURCE_COMMIT = "048ba11f75dab38099e7a0f4a588d5a8fdb0a04c"


def source_files():
    root = Path(__file__).resolve().parents[1] / "external/BrainDyn"
    import json
    proof = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
    if proof["commit"] != SOURCE_COMMIT or proof["source_modified"] is not False:
        raise ValueError("BRAINDYN_PINNED_AUTHOR_SOURCE_REQUIRED")
    for row in proof["files"]:
        path = root / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("BRAINDYN_AUTHOR_SOURCE_CHANGED:" + row["path"])
    return proof


def initialize(field_count: int, history_bins: int, seed: int, *, hidden_dim=8,
               map_hidden_dim=4, vf_hidden_dim=32):
    if (type(seed) is not int or seed not in (41, 42, 43)
            or any(type(x) is not int or x < 1 for x in
                   (field_count, history_bins, hidden_dim, map_hidden_dim, vf_hidden_dim))):
        raise ValueError("BRAINDYN_SOURCE_DIMENSIONS_AND_SEED_41_42_43_REQUIRED")
    source_files()
    config = BrainDynConfig(signal_dim=1, hidden_dim=hidden_dim, num_nodes=field_count,
        window_size=history_bins, map_hidden_dim=map_hidden_dim, vf_hidden_dim=vf_hidden_dim,
        ode_method="rk4", use_gat=False, use_lstm_encoder=True, precompute_lap_h=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BrainDyn(config)
    return model


def forecast(model, rates_hz, edge_index, horizon_bins: int, bin_seconds: float):
    rates = torch.as_tensor(rates_hz, dtype=next(model.parameters()).dtype,
                            device=next(model.parameters()).device)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=rates.device)
    if (rates.ndim != 2 or rates.shape != (model.config.window_size, model.config.num_nodes)
            or not torch.isfinite(rates).all() or edges.ndim != 2 or edges.shape[0] != 2
            or not edges.shape[1] or torch.any(edges < 0) or torch.any(edges >= model.config.num_nodes)
            or type(horizon_bins) is not int or horizon_bins < 1
            or not np.isfinite(bin_seconds) or bin_seconds <= 0):
        raise ValueError("BRAINDYN_COMPLETE_SOURCE_FIELD_HISTORY_GRAPH_AND_CLOCK_REQUIRED")
    history = rates.T[None, :, :, None]
    result = model(history, edges, horizon_bins, dt=bin_seconds)["x_pred"]
    if result.shape != (horizon_bins, 1, model.config.num_nodes, 1) or not torch.isfinite(result).all():
        raise ValueError("BRAINDYN_FINITE_COMPLETE_FIELD_FORECAST_REQUIRED")
    return result[:, 0, :, 0]


def configuration(model):
    return asdict(model.config)
