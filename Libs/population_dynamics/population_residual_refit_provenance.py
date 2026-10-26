"""Bind all six native residual refits to one comparison campaign."""
from __future__ import annotations

import json
from pathlib import Path

from .provenance import sha256_file

FIT_SCHEMA = "native_residual_observation_fit_v1"
FAMILY_SCHEMA = "native_residual_observation_refits_v1"
SEEDS = (41, 42, 43)
MODES = ("current", "history")
SHARED_BINDING_KEYS = ("input_manifest_sha256", "config_sha256", "sources")


def refit_identity(report: dict, digest: str) -> dict:
    """Read completed refit metadata; runtime/predictor admission stays in the loader."""
    binding = report.get("binding", {})
    config = binding.get("config", {})
    epoch = report.get("completed_epochs")
    if (report.get("schema") != FIT_SCHEMA or report.get("status") != "FITTED_DEVELOPMENT"
            or report.get("phase") != "refit" or binding.get("phase") != "refit"
            or binding.get("seed") not in SEEDS or binding.get("mode") not in MODES
            or config.get("training_seeds") != list(SEEDS)
            or type(epoch) is not int or epoch != report.get("selected_epoch")
            or not 0 <= epoch <= config.get("max_epochs", -1)
            or not binding.get("selection_sha256") or binding.get("validation_cases") != []
            or report.get("parent_unchanged") is not True
            or report.get("scientific_claim_qualified") is not False
            or not report.get("predictor_sha256") or not binding.get("parent_checkpoint_sha256")
            or not all(binding.get(key) for key in SHARED_BINDING_KEYS)
            or not all(isinstance(report.get(key), str) and report[key] for key in ("run_id", "trial_id"))):
        raise ValueError("complete Merlin native observation refit metadata required")
    return dict(schema=FIT_SCHEMA, fit_sha256=digest,
                predictor_sha256=report["predictor_sha256"], binding=binding)


def build_refit_receipt(refits: Path) -> dict:
    """Build the shared receipt consumed by every seed's native comparison delivery.

    ``refits`` contains ``seed_41/current/fit.json`` and the other five cells.
    The generation caller copies the reports to the returned delivery paths.
    """
    rows, shared, configuration = [], None, None
    for seed in SEEDS:
        for mode in MODES:
            path = Path(refits) / f"seed_{seed}" / mode / "fit.json"
            report = json.loads(path.read_text())
            identity = refit_identity(report, sha256_file(path))
            binding = identity["binding"]
            if binding["seed"] != seed or binding["mode"] != mode:
                raise ValueError("native refit path and seed/mode differ")
            common = {key: binding[key] for key in SHARED_BINDING_KEYS}
            if shared is not None and (shared != common or configuration != binding["config"]):
                raise ValueError("native refits must share inputs, configuration and fitting sources")
            shared, configuration = common, binding["config"]
            rows.append(dict(seed=seed, mode=mode,
                path=f"provenance/fits/seed_{seed}/{mode}/fit.json",
                sha256=identity["fit_sha256"], predictor_sha256=identity["predictor_sha256"],
                parent_checkpoint_sha256=binding["parent_checkpoint_sha256"],
                run_id=report["run_id"], trial_id=report["trial_id"]))
    return dict(schema=FAMILY_SCHEMA, status="COMPLETE_NATIVE_RESIDUAL_REFITS", phase="refit",
                parent_training=False, training_seeds=list(SEEDS), modes=list(MODES),
                config=configuration, shared_binding=shared, fits=rows)
