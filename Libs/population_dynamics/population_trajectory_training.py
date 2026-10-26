"""Fit the population model through the complete-episode trainer.

The recurrent and fixed representations use the same native observation law,
cohort gate, optimizer, validation draws and committed-epoch resume protocol.
Development fitting does not establish source or generative qualification.
PFC source-reproduction inputs use shared parameters and a declared whole-circuit
development partition. V1 keeps original catalog train/validation membership.
The native Mesocircuit campaign keeps its frozen protocol.
"""
from __future__ import annotations

from pathlib import Path

from . import renewal_trajectory_training as shared

FIXED_BUDGET = shared.POPULATION_FIXED_BUDGET


def validate_config(config: dict, seed: int) -> None:
    shared.validate_config(config, seed, backend="population")


def train(acquisition: Path, fields: Path | None, config_path: Path, output: Path, seed: int,
          resume: bool = False, *, stop_after_epochs: int | None = None,
          resume_from: Path | None = None, total_epochs: int | None = None,
          parent_checkpoint_sha256: str | None = None, prior_parent: Path | None = None,
          prior_cache_source: Path | None = None):
    return shared.train(
        acquisition, fields, config_path, output, seed, resume, backend="population",
        stop_after_epochs=stop_after_epochs, resume_from=resume_from, total_epochs=total_epochs,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        **({"prior_parent": prior_parent} if prior_parent is not None else {}),
        **({"prior_cache_source": prior_cache_source} if prior_cache_source is not None else {}))


def main():
    shared.main(backend="population")


if __name__ == "__main__":
    main()
