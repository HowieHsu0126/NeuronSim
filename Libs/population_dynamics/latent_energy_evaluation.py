"""Matched complete-unit EAG evaluation using the shared count/replay contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .joint_count_evaluation import evaluate_counts, fitting_scale
from .latent_energy_training import _model, _sources, validate_config
from .latent_spike_diffusion import SOURCE_REVISION
from .native_trajectory_cohort import TrajectoryCohort
from .native_renewal_data import NativeRenewalEpisode
from .provenance import sha256_file


def _load(acquisition, fields, training, config_path, scale_path, seed):
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    cohort = TrajectoryCohort(acquisition, fields, config)
    pointer = json.loads((training / "committed.json").read_text())
    if sha256_file(training / pointer["checkpoint"]) != pointer["sha256"]:
        raise ValueError("EAG terminal checkpoint changed")
    payload = torch.load(training / pointer["checkpoint"], map_location=config["device"], weights_only=False)
    state = payload["state"]
    expected = dict(config=config, seed=seed, cohort=cohort.binding, sources=_sources(config),
                    source_revision=SOURCE_REVISION, torch_version=torch.__version__)
    presentations = len(cohort.fit) * len(config["forecast_origins_bins"]) * config["energy"]["maximum_epochs"]
    if (state["binding"] != expected or state["stage"] != "complete" or pointer["stage"] != "complete"
            or state["autoencoder_completed_epochs"] != config["autoencoder"]["maximum_epochs"]
            or state["energy_sampler"]["presentations"] != presentations
            or state["steps"] != (presentations + config["energy"]["source_batch_size"] - 1) // config["energy"]["source_batch_size"]
            or json.loads((training / "run.json").read_text())["status"] != "FITTED_DEVELOPMENT"):
        raise ValueError("complete EAG training and unchanged data/software/protocol required")
    for name, digest in ((state["selected_path"], state["selected_digest"]),
                         (state["autoencoder_selected_path"], state["autoencoder_selected_digest"]),
                         ("latent_cache.pt", state["cache_digest"])):
        if sha256_file(training / name) != digest:
            raise ValueError("EAG selected checkpoint or training cache changed")
    model = _model(cohort, config)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cohort, config, *fitting_scale(cohort, scale_path)


def _infer_state(model, counts, observed_controls):
    return model.infer_state(counts, observed_controls)


def _simulate(model, state, stimulus, config, **streams):
    sampling = config["sampling"]
    return model.simulate(state, stimulus, steps=sampling["steps"],
                          guidance=sampling["guidance"], temperature=sampling["temperature"], **streams)


def evaluate(acquisition, fields, training, config_path, scale_path, output, seed, *, replay=False):
    return evaluate_counts(acquisition, fields, training, config_path, scale_path, output, seed,
        replay=replay, load=_load, method="eag", infer_state=_infer_state, simulate=_simulate,
        evaluator_path=Path(__file__), native_episode_factory=NativeRenewalEpisode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "training", "config", "fitting-scale", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.acquisition, args.fields, args.training, args.config,
                             args.fitting_scale, args.output, args.seed, replay=args.replay)))


if __name__ == "__main__":
    main()
