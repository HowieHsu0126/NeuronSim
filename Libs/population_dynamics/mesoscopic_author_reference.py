"""Execute the pinned author mesoscopic Potjans reference without equation edits.

This reference emits eight population rates, not identified unit trajectories.
Its finite-size population count does not establish a production coupled-core
dataset or matched spatial-circuit comparison. License permission is unresolved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import resource
import time
import traceback

import numpy as np
import yaml

from .provenance import sha256_file

AUTHOR_COMMIT = "934d0d94bba75e62d853aef2c82f16e6d5de9f4c"
AUTHOR_REPOSITORY = "https://github.com/schwalger/mesopopdyn_microcircuit"
AUTHOR_DOI = "10.1371/journal.pcbi.1005507"
AUTHOR_FILES = {'README.md': '458664813e6b8d453bd0f54fb1e4174965a655df32e4eeb68eff89adee501dd3', 'single_run/sim4.py': '00cb39ce0a0d0e5c64e5da87f64295ca06147b82a22436e2c477958d9f0be010', 'single_run/potjans.py': '1ff2878b861aa14ec2eb7760f5c133100a3c9c626631af86190e04c946a65b9a', 'single_run/PDmodel_mesopython.py': 'ca7d3de6a5a36b8d951d39c6dbcc981eeaded761a482b0b8eaddcf347197e060'}
FROZEN = dict(
    schema="neural_population_mesoscopic_author_reference_v1", represented_units=100000,
    duration_seconds=2., dt_seconds=.0001, output_bin_seconds=.005,
    relaxation_seconds=1., stimulus_delay_seconds=.06, stimulus_duration_seconds=.03,
    stimulus_amplitude_mv=15., finite_size_noise_seed=12345, distinct_noise_seed=12346,
    population_order=["L23E", "L4E", "L5E", "L6E", "L23I", "L4I", "L5I", "L6I"],
    coupling_convention="author_original_N_effective_weights_held_under_finite_size_scaling",
    observation="population_mean_rate_Abar_and_sampled_activity_rate_A_hz",
    formal_comparison=False, source_fidelity_qualified=False, unit_outputs_available=False,
)


def source_binding(root):
    observed = {name: sha256_file(root / name) for name in AUTHOR_FILES}
    if observed != AUTHOR_FILES:
        raise ValueError("pinned author source changed")
    return dict(repository=AUTHOR_REPOSITORY, commit=AUTHOR_COMMIT, paper_doi=AUTHOR_DOI,
                author_file_sha256=observed, runner_sha256=sha256_file(Path(__file__)),
                license="not_declared_by_source_repository; internal_evaluation_only")


def load_module(root, name):
    spec = importlib.util.spec_from_file_location("mesoscopic_author_" + name, root / "single_run" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameters(root, config, condition):
    """Reproduce the author's parameter assembly and finite-size scaling rule."""
    if condition not in ("coupled", "uncoupled"):
        raise ValueError("registered author coupling condition required")
    original = load_module(root, "potjans")
    layers = ("L23", "L4", "L5", "L6")
    source_n = np.array([original.N_full[layer][kind] for kind in ("E", "I") for layer in layers])
    fractions = source_n / source_n.sum() * config["represented_units"]
    sizes = np.floor(fractions).astype(np.int64)
    remainder = config["represented_units"] - int(sizes.sum())
    order = np.argsort(-(fractions - sizes), kind="stable")
    sizes[order[:remainder]] += 1
    if int(sizes.sum()) != 100000 or np.any(sizes <= 0):
        raise ValueError("exact100000 finite-size represented population total required")
    reorder = [0, 2, 4, 6, 1, 3, 5, 7]
    weights = (original.weight_matrix * np.asarray(original.conn_probs))[np.ix_(reorder, reorder)].copy()
    neuron = original.neuron_params
    weights[:, :4] *= neuron["tau_syn_E"] * 1e-3 / neuron["cm"] * 1e3
    weights[:, 4:] *= neuron["tau_syn_I"] * 1e-3 / neuron["cm"] * 1e3
    weights *= source_n[None]
    if condition == "uncoupled":
        weights.fill(0)
    bins = round(config["duration_seconds"] / config["output_bin_seconds"])
    step = np.zeros(8)
    step[[1, 5, 3, 7]] = original.tha_drive_vec[[2, 3, 6, 7]]
    step = step / step.max() * config["stimulus_amplitude_mv"]
    controls = np.zeros((bins, 8), dtype=np.float32)
    onset = config["relaxation_seconds"] + config["stimulus_delay_seconds"]
    begin = int(onset / config["output_bin_seconds"])
    end = min(int((onset + config["stimulus_duration_seconds"]) / config["output_bin_seconds"]), bins)
    controls[begin:end] = step
    values = dict(
        M=8, Me=4, N=sizes,
        mu=np.array([19.14942021, 30.80455243, 29.43680143, 34.93225564,
                     20.36247192, 28.06868188, 29.32954448, 32.08123423]),
        hazard_Delta_u=np.full(8, 5.), hazard_c=10.,
        vth=neuron["v_thresh"] - neuron["v_rest"], vreset=neuron["v_reset"] - neuron["v_rest"],
        taum_e=neuron["tau_m"] * 1e-3, taum_i=neuron["tau_m"] * 1e-3,
        delay=original.d_mean["E"] * 1e-3, tref=neuron["tau_refrac"] * 1e-3,
        taus_e=neuron["tau_syn_E"] * 1e-3, taus_i=neuron["tau_syn_I"] * 1e-3,
        weights=weights, Iext=controls)
    return values, source_n


def execute(source, config_path, output, condition, seed, reference=None):
    if output.exists():
        raise FileExistsError("existing author-reference evidence will not be overwritten")
    config = yaml.safe_load(config_path.read_text())
    if config != FROZEN or seed not in (config["finite_size_noise_seed"], config["distinct_noise_seed"]):
        raise ValueError("frozen author-reference configuration and finite-size noise seed required")
    binding = dict(source_binding(source), config=config, config_sha256=sha256_file(config_path))
    params, source_sizes = parameters(source, config, condition)
    module = load_module(source, "sim4")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status="RUNNING", binding=binding, condition=condition, finite_size_noise_seed=seed,
                  started_utc=datetime.now(timezone.utc).isoformat(), population_sizes=params["N"].tolist(),
                  source_population_sizes=source_sizes.tolist(), represented_units=int(params["N"].sum()),
                  population_outputs_only=True, unit_outputs_available=False, production_episodes_counted=0,
                  source_fidelity_qualified=False, scientific_claim_qualified=False)
    try:
        expected, activity = module.simulate(config["duration_seconds"], config["dt_seconds"],
                                             config["output_bin_seconds"], params, 8, seed)
        shape = (8, round(config["duration_seconds"] / config["output_bin_seconds"]))
        if (expected.shape != shape or activity.shape != shape
                or not np.isfinite(expected).all() or not np.isfinite(activity).all()
                or np.any(expected < 0) or np.any(activity < 0)):
            raise ValueError("finite nonnegative complete author population outputs required")
        arrays = dict(Abar_hz=expected, A_hz=activity, Iext_mv=params["Iext"],
                      population_sizes=params["N"], effective_weights=params["weights"],
                      time_seconds=np.arange(shape[1]) * config["output_bin_seconds"],
                      population_names=np.asarray(config["population_order"]))
        np.savez(output / "population_outputs.npz", **arrays)
        report.update(status="EXECUTED_AUTHOR_REFERENCE", output_values=expected.size + activity.size,
                      mean_Abar_hz=expected.mean(axis=1).tolist(), mean_A_hz=activity.mean(axis=1).tolist())
        if reference is not None:
            parent = json.loads((reference / "run.json").read_text())
            if (parent["status"] != "EXECUTED_AUTHOR_REFERENCE" or parent["binding"] != binding
                    or sha256_file(reference / "population_outputs.npz") != parent["artifacts"]["population_outputs.npz"]):
                raise ValueError("author reference source, configuration or output binding changed")
            with np.load(reference / "population_outputs.npz", allow_pickle=False) as previous:
                same_noise = seed == parent["finite_size_noise_seed"]
                same_condition = condition == parent["condition"]
                exact = all(previous[key].dtype == value.dtype and np.array_equal(previous[key], value) for key, value in arrays.items())
                controls_equal = np.array_equal(previous["Iext_mv"], arrays["Iext_mv"])
                if not controls_equal:
                    raise ValueError("reference diagnostics changed original external controls")
                if same_noise and same_condition and not exact:
                    raise ValueError("fresh-process author array replay differs")
                if not (same_noise and same_condition) and np.array_equal(previous["A_hz"], activity):
                    raise ValueError("changed noise/coupling diagnostic produced identical activity")
                report["reference_check"] = dict(
                    status="PASS_EXACT_ARRAY_REPLAY" if same_noise and same_condition else "PASS_CHANGED_DIAGNOSTIC",
                    reference_sha256=sha256_file(reference / "run.json"), exact_arrays=exact,
                    same_noise_seed=same_noise, same_coupling_condition=same_condition, original_controls_equal=True,
                    maximum_A_hz_difference=float(np.max(np.abs(previous["A_hz"].astype(np.float64) - activity))))
        import numba
        report["environment"] = dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__)
    except BaseException as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      artifacts={p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "run.json"})
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--condition", choices=("coupled", "uncoupled"), default="coupled")
    parser.add_argument("--noise-seed", type=int, choices=(12345, 12346), default=12345)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(execute(args.source, args.config, args.output, args.condition, args.noise_seed, args.reference)))


if __name__ == "__main__":
    main()
