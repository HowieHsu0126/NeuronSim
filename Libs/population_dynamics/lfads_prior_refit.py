"""Fit the source AR prior to complete ancestral paths from fitting episodes.

The selected LFADS networks stay frozen. Full-response variational paths are
fitting data, not causal forecasts or observed mechanistic states. Selection
blocks never enter the prior fit, even though their source split is also train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .lfads import LFADSConditionalSimulator, SOURCE_REVISION
from .lfads_stationary_prior import (
    StationaryARFitError, StationaryARMoments, _float32_source_range,
    fit_stationary_ar_prior,
)
from .lfads_trajectory_evaluation import _verified_training
from .lfads_trajectory_training import _write_json
from .provenance import sha256_file


PRIOR_VARIABLES = {key: f"LFADS/z/u_prior_ar1/{key}:0"
                   for key in ("logatau", "logevars")}
SCHEMA = "lfads_complete_fitting_posterior_cache_v1"
EPISODE_PROOF_PROTOCOL = "lfads_frozen_globals_after_each_episode_before_commit_v1"
JOINT_SAMPLES = 16
COMPLETE_BINS = 1200
POSTERIOR_DRAW_BATCH_SIZE = 1


def _training_update_count(value):
    if type(value) is not int or value < 0:
        raise ValueError("SELECTED_CHECKPOINT_TRAINING_UPDATE_COUNT_REQUIRED")
    return value


def _proof_training_updates(binding):
    if binding.get("frozen_network_proof_protocol") != EPISODE_PROOF_PROTOCOL:
        raise ValueError("UNQUALIFIED_FROZEN_EPISODE_PROOF_PROTOCOL: fresh extraction required")
    return _training_update_count(binding.get("selected_training_updates"))


def _episode_proof(values, expected_updates):
    """Read the same strictly typed proof from a ledger row or scalar NPZ fields."""
    proof = {}
    for key in ("frozen_global_variables_verified", "training_updates_before_and_after"):
        if key not in values:
            raise ValueError("UNQUALIFIED_FROZEN_EPISODE_PROOF: fresh extraction required")
        value = values[key]
        if isinstance(value, np.ndarray):
            if value.shape != ():
                raise ValueError("INVALID_FROZEN_EPISODE_PROOF")
            value = value.item()
        if type(value) is not int:
            raise ValueError("INVALID_FROZEN_EPISODE_PROOF")
        proof[key] = value
    if (proof["frozen_global_variables_verified"] <= 0
            or proof["training_updates_before_and_after"] != expected_updates):
        raise ValueError("INVALID_FROZEN_EPISODE_PROOF")
    return proof


def _complete_frozen_proof(rows, expected_updates):
    if len(rows) != 56:
        raise ValueError("COMPLETE_56_EPISODE_FROZEN_PROOFS_REQUIRED")
    proof = _episode_proof(rows[0], expected_updates)
    if any(_episode_proof(row, expected_updates) != proof for row in rows[1:]):
        raise ValueError("FITTING_CACHE_FROZEN_PROOF_CHANGED")
    return proof


def fitting_cases(cohort):
    """Check the complete independent block partition before reading counts."""
    expected = {f"train_block_{block:03d}_familiar_{template}"
                for block in range(14) for template in range(4)}
    actual = [case["case_id"] for case in cohort.fit]
    if len(actual) != 56 or set(actual) != expected:
        raise ValueError("COMPLETE_56_FITTING_EPISODES_REQUIRED")
    for case in cohort.fit:
        if (type(case["block"]) is not int or case["block"] not in range(14)
                or case["template_id"] not in {f"familiar_{i}" for i in range(4)}
                or case["case_id"] != f"train_block_{case['block']:03d}_{case['template_id']}"):
            raise ValueError("FITTING_EPISODE_IDENTITY_OR_BLOCK_CHANGED")
    validation = [case["case_id"] for case in cohort.validation]
    expected_validation = {f"train_block_{block:03d}_familiar_{template}"
                           for block in range(14, 18) for template in range(4)}
    if len(validation) != 16 or set(validation) != expected_validation:
        raise ValueError("SEPARATE_16_SELECTION_EPISODES_REQUIRED")
    return sorted(cohort.fit, key=lambda case: (case["block"], case["template_id"]))


def sampling_seeds(training_seed: int, case_id: str) -> tuple[int, int]:
    if type(training_seed) is not int or training_seed not in (41, 42, 43):
        raise ValueError("REGISTERED_TRAINING_SEED_REQUIRED")
    result = []
    for stream in ("initial", "controller"):
        value = json.dumps([SCHEMA, training_seed, case_id, stream], separators=(",", ":"))
        result.append(int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little"))
    return tuple(result)


def sample_complete_posterior(model, counts, stimulus, *, draws: int,
                              seeds: tuple[int, int], draw_batch_size=None) -> dict[str, np.ndarray]:
    """Run the native full graph with explicit independent ancestral noises.

    No posterior mean, latent sample, encoder value or generator state is fed
    back into the graph. Each sample's controller therefore keeps its own
    generator-feedback ancestry. Native encoder arithmetic uses the draw batch.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("POSITIVE_POSTERIOR_DRAW_COUNT_REQUIRED")
    batch_size = draws if draw_batch_size is None else draw_batch_size
    if type(batch_size) is not int or not 1 <= batch_size <= draws:
        raise ValueError("VALID_POSTERIOR_DRAW_BATCH_SIZE_REQUIRED")
    if (len(seeds) != 2 or any(type(seed) is not int or seed < 0 for seed in seeds)
            or seeds[0] == seeds[1]):
        raise ValueError("TWO_DISTINCT_NONNEGATIVE_POSTERIOR_SEEDS_REQUIRED")
    bins = model.history_bins + model.horizon_bins
    y = model._counts(counts, bins)
    if len(y) != 1:
        raise ValueError("ONE_COMPLETE_EPISODE_PER_POSTERIOR_BATCH_REQUIRED")
    u = model._stimulus(stimulus, 1, bins)
    source = model._full.model
    if len(source.posterior_zs_co) != bins:
        raise ValueError("COMPLETE_SOURCE_POSTERIOR_GRAPH_REQUIRED")
    initial_rng, controller_rng = (np.random.default_rng(seed) for seed in seeds)
    initial_noise = initial_rng.standard_normal(
        (draws, model.model_config["ic_dim"])).astype(np.float32)
    controller_noise = controller_rng.standard_normal(
        (bins, draws, model.model_config["co_dim"])).astype(np.float32)
    fetches = dict(
        paths=[p.sample for p in source.posterior_zs_co],
        means=[p.mean for p in source.posterior_zs_co],
        log_variances=[p.logvar for p in source.posterior_zs_co])
    chunks = {name: [] for name in fetches}
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        feed = source.build_feed_dict("recording",
            np.broadcast_to(y, (stop - start, bins, len(model.unit_ids))),
            np.broadcast_to(u, (stop - start, bins, model.input_size)), keep_prob=1.)
        feed[source.posterior_zs_g0.noise_bxn] = initial_noise[start:stop]
        for tick, posterior in enumerate(source.posterior_zs_co):
            feed[posterior.noise_bxn] = controller_noise[tick, start:stop]
        values = model._full.session.run(fetches, feed_dict=feed)
        for name, value in values.items():
            chunks[name].append(np.stack(value, axis=1))
    result = {name: np.concatenate(value) for name, value in chunks.items()}
    shape = (draws, bins, model.model_config["co_dim"])
    if any(value.shape != shape or value.dtype != np.float32
           or not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("INVALID_COMPLETE_NATIVE_POSTERIOR_PATH")
    with np.errstate(over="ignore", under="ignore"):
        scales = np.exp(.5 * result["log_variances"])
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise FloatingPointError("INVALID_NATIVE_POSTERIOR_VARIANCE")
    return result


def apply_prior_parameters(model, parameters):
    """Assign only the two source [1,4] variables; keep network/optimizer fixed."""
    arrays = {}
    for key, name in PRIOR_VARIABLES.items():
        value = np.asarray(parameters[key])
        if value.shape != (1, 4) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("FINITE_FLOAT32_FOUR_CHANNEL_PRIOR_REQUIRED")
        arrays[name] = value
    # Check the actual source float32 transformations before any assignment.
    _float32_source_range(parameters["logatau"], parameters["logevars"])
    view = model._full
    names = list(view.variables)
    if not set(arrays) <= set(names):
        raise ValueError("SOURCE_AR_PRIOR_VARIABLES_MISSING")
    for name in arrays:
        variable = view.variables[name]
        if variable.shape.as_list() != [1, 4] or variable.dtype.as_numpy_dtype != np.float32:
            raise ValueError("SOURCE_AR_PRIOR_VARIABLE_SHAPE_OR_DTYPE_CHANGED")
    before = view.values()
    updates = model.training_updates
    view.session.run([view.assign_ops[names.index(name)] for name in arrays],
        feed_dict={view.assign_inputs[name]: value for name, value in arrays.items()})
    after = view.values()
    for name, value in before.items():
        expected = arrays[name] if name in arrays else value
        if expected.dtype != after[name].dtype or not np.array_equal(expected, after[name]):
            raise ValueError("PRIOR_REFIT_CHANGED_UNAUTHORIZED_GLOBAL_VARIABLE:" + name)
    if model.training_updates != updates:
        raise ValueError("PRIOR_REFIT_CHANGED_TRAINING_UPDATE_COUNTER")
    model._prefix_dirty = True
    return dict(assigned_scalars=8, unchanged_global_variables=len(before) - 2,
                unchanged_scalar_values=sum(value.size for name, value in before.items() if name not in arrays),
                training_updates=updates, nonprior_weights_and_optimizer_state_exact=True)


def load_prior_parameters(directory, *, run, selected, cohort_binding, training_run_sha256):
    """Bind the eight-float inference patch to its exact network and fit cohort."""
    directory = Path(directory)
    record_path = directory / "posterior_cache.json"
    record = json.loads(record_path.read_text())
    binding = record["binding"]
    updates = _proof_training_updates(binding)
    selected_updates = _training_update_count(selected.get("training_updates"))
    cases = [dict(case_id=f"train_block_{block:03d}_familiar_{template}",
                  block=block, template_id=f"familiar_{template}")
             for block in range(14) for template in range(4)]
    expected_ids = [case["case_id"] for case in cases]
    if (record.get("status") != "FITTED_FROZEN_NETWORK_AR_PRIOR"
            or record.get("fitting_cohort_extracted") is not True
            or record.get("formal_test_accessed") is not False
            or record.get("complete_episodes") != 56 or record.get("posterior_paths") != 896
            or binding.get("schema") != SCHEMA or binding.get("training_binding") != run["binding"]
            or binding.get("training_run_sha256") != training_run_sha256
            or binding.get("selected_model_artifacts") != selected["model_artifacts"]
            or binding.get("selected_epoch") != selected["next_epoch"]
            or updates != selected_updates
            or binding.get("cohort_binding") != cohort_binding
            or binding.get("case_ids") != expected_ids
            or binding.get("fitting_blocks") != list(range(14))
            or binding.get("excluded_selection_blocks") != list(range(14, 18))
            or binding.get("joint_samples") != JOINT_SAMPLES
            or binding.get("complete_bins") != COMPLETE_BINS
            or [row["case_id"] for row in record["episodes"]] != expected_ids):
        raise ValueError("PRIOR_REFIT_NETWORK_OR_FITTING_COHORT_BINDING_CHANGED")
    for case, row in zip(cases, record["episodes"]):
        _cached_paths(directory, case, row, _cache_identity(binding, case))
    if _episode_proof(record, updates) != _complete_frozen_proof(record["episodes"], updates):
        raise ValueError("FITTING_CACHE_FROZEN_PROOF_CHANGED")
    path = directory / "prior_parameters.npz"
    digest = sha256_file(path)
    if digest != record["prior_parameters_sha256"]:
        raise ValueError("FITTED_PRIOR_PARAMETERS_CHANGED")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(PRIOR_VARIABLES):
            raise ValueError("EXACT_TWO_PRIOR_TENSORS_REQUIRED")
        parameters = {name: archive[name] for name in archive.files}
    for value in parameters.values():
        if value.shape != (1, 4) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("FINITE_FLOAT32_FOUR_CHANNEL_PRIOR_REQUIRED")
    _float32_source_range(parameters["logatau"], parameters["logevars"])
    return parameters, dict(posterior_cache_sha256=sha256_file(record_path),
        prior_parameters_sha256=digest, fitted_parameters=8, fitting_episodes=56,
        source="full-response ancestral variational posterior on fitting blocks0-13",
        training_seed=run["binding"]["seed"], selected_epoch=selected["next_epoch"])


def _cache_identity(binding, case):
    updates = _proof_training_updates(binding)
    return dict(case_id=case["case_id"], block=case["block"], template_id=case["template_id"],
        training_seed=binding["training_seed"], selected_training_updates=updates,
        frozen_network_proof_protocol=EPISODE_PROOF_PROTOCOL,
        sampling_seeds=list(sampling_seeds(binding["training_seed"], case["case_id"])),
        binding_sha256=hashlib.sha256(json.dumps(binding, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest())


def _cached_paths(output, case, row, identity, *, draws=JOINT_SAMPLES, bins=COMPLETE_BINS):
    filename = case["case_id"] + ".npz"
    updates = _proof_training_updates(identity)
    proof = _episode_proof(row, updates)
    seeds = list(sampling_seeds(row["training_seed"], case["case_id"]))
    if (row["case_id"] != case["case_id"] or row["block"] != case["block"]
            or row["template_id"] != case["template_id"] or row["path"] != filename
            or row["sampling_seeds"] != seeds
            or any(row.get(key) != value for key, value in identity.items())
            or sha256_file(output / filename) != row["sha256"]):
        raise ValueError("FITTING_POSTERIOR_CACHE_IDENTITY_CHANGED")
    with np.load(output / filename, allow_pickle=False) as archive:
        if json.loads(str(archive["cache_identity"])) != identity:
            raise ValueError("FITTING_POSTERIOR_CACHE_BINDING_CHANGED")
        if _episode_proof(archive, updates) != proof:
            raise ValueError("FITTING_CACHE_FROZEN_PROOF_CHANGED")
        paths = archive["controller_paths"]
    if paths.shape != (draws, bins, 4) or paths.dtype != np.float32 or not np.isfinite(paths).all():
        raise ValueError("COMPLETE_FINITE_FITTING_CONTROLLER_PATHS_REQUIRED")
    return paths


def _committed_row(path, identity):
    with np.load(path, allow_pickle=False) as archive:
        proof = _episode_proof(archive, _proof_training_updates(identity))
        elapsed = float(archive["elapsed_seconds"])
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("INVALID_POSTERIOR_EXTRACTION_COST")
    return dict(identity, path=path.name, sha256=sha256_file(path),
                elapsed_seconds=elapsed, **proof)


def _commit_paths(path, paths, identity, elapsed, proof):
    # Only an episode checked against the loaded checkpoint may reach this commit.
    # The payload carries its proof across an interrupted later ledger write.
    proof = _episode_proof(proof, _proof_training_updates(identity))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".",
                                         suffix=".writing", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, controller_paths=paths,
                cache_identity=np.asarray(json.dumps(identity, sort_keys=True)),
                elapsed_seconds=np.asarray(elapsed),
                **{key: np.asarray(value) for key, value in proof.items()})
        if path.exists():
            raise FileExistsError(path)
        temporary.rename(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def extract_and_fit(acquisition, fields, training, output, *, device="cuda",
                    inference_migration=None, continue_cache=False):
    acquisition, fields, training, output = map(Path, (acquisition, fields, training, output))
    cohort, run, selected_directory, selected, migration = _verified_training(
        acquisition, fields, training, inference_migration)
    cases = fitting_cases(cohort)
    seed = run["binding"]["seed"]
    updates = _training_update_count(selected.get("training_updates"))
    binding = dict(schema=SCHEMA, training_seed=seed, selected_epoch=selected["next_epoch"],
        frozen_network_proof_protocol=EPISODE_PROOF_PROTOCOL, selected_training_updates=updates,
        selected_model_artifacts=selected["model_artifacts"], source_revision=SOURCE_REVISION,
        training_binding=run["binding"], training_run_sha256=sha256_file(training / "run.json"),
        inference_migration=migration, cohort_binding=cohort.binding,
        joint_samples=JOINT_SAMPLES, complete_bins=COMPLETE_BINS,
        native_posterior_draw_batch_size=POSTERIOR_DRAW_BATCH_SIZE,
        case_ids=[case["case_id"] for case in cases],
        fitting_blocks=list(range(14)), excluded_selection_blocks=list(range(14, 18)),
        sampling="native full-response ancestral variational posterior; keep_prob=1",
        sampling_rng="NumPy default_rng PCG64; independent initial/controller streams",
        numpy_version=np.__version__, device=device,
        implementation_sha256=sha256_file(Path(__file__)),
        fitter_sha256=sha256_file(Path(__file__).with_name("lfads_stationary_prior.py")))
    record_path = output / "posterior_cache.json"
    if output.exists():
        if not continue_cache:
            raise FileExistsError(output)
        record = json.loads(record_path.read_text())
        _proof_training_updates(record["binding"])
        if record["binding"] != binding:
            raise ValueError("FROZEN_POSTERIOR_CACHE_BINDING_CHANGED")
    else:
        output.mkdir(parents=True)
        record = dict(status="EXTRACTING_FITTING_POSTERIOR", binding=binding, episodes=[],
            fitting_cohort_extracted=False, model_parameters_updated=False,
            formal_test_accessed=False, scientific_claim_qualified=False)
        _write_json(record_path, record)
    rows = record["episodes"]
    if [row["case_id"] for row in rows] != binding["case_ids"][:len(rows)]:
        raise ValueError("FITTING_CACHE_MUST_BE_AN_ORDERED_COMPLETE_EPISODE_PREFIX")
    moments = StationaryARMoments(sequence_length=COMPLETE_BINS)
    for case, row in zip(cases, rows):
        if row["training_seed"] != seed:
            raise ValueError("FITTING_CACHE_TRAINING_SEED_CHANGED")
        moments.update(_cached_paths(output, case, row, _cache_identity(binding, case)))
    if record["status"] == "FITTED_FROZEN_NETWORK_AR_PRIOR":
        proof = _complete_frozen_proof(rows, updates)
        if _episode_proof(record, updates) != proof:
            raise ValueError("FITTING_CACHE_FROZEN_PROOF_CHANGED")
        if sha256_file(output / "prior_parameters.npz") != record["prior_parameters_sha256"]:
            raise ValueError("COMPLETE_FITTED_PRIOR_ARTIFACT_REQUIRED")
        return record
    # Recover committed payloads before loading the network. In particular, a
    # last-episode ledger interruption needs no further source sampling or load.
    for case in cases[len(rows):]:
        path = output / (case["case_id"] + ".npz")
        if not path.exists():
            break
        identity = _cache_identity(binding, case)
        row = _committed_row(path, identity)
        moments.update(_cached_paths(output, case, row, identity))
        rows.append(row)
        _write_json(record_path, record)
    if len(rows) != 56:
        _extract_remaining(cohort, selected_directory, output, device, cases, rows,
                           moments, record, record_path, seed)
    original = {key: np.asarray(value, dtype=np.float32)
                for key, value in record["original_prior_parameters"].items()}
    proof = _complete_frozen_proof(rows, updates)
    record.update(status="COMPLETE_FITTING_POSTERIOR_CACHE", fitting_cohort_extracted=True,
                  complete_episodes=len(rows), posterior_paths=moments.sequence_count, **proof)
    _write_json(record_path, record)
    try:
        fitted = fit_stationary_ar_prior(moments, original["logatau"], original["logevars"])
    except StationaryARFitError as error:
        record.update(status="COMPLETE_CACHE_PRIOR_FIT_FAILED", fit_diagnostics=error.diagnostics)
        _write_json(record_path, record)
        raise
    path = output / "prior_parameters.npz"
    parameters = {key: fitted[key] for key in PRIOR_VARIABLES}
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(parameters) or any(
                    not np.array_equal(archive[key], value) for key, value in parameters.items()):
                raise ValueError("UNCOMMITTED_PRIOR_PARAMETERS_DIFFER_FROM_COMPLETE_CACHE_FIT")
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output, prefix="prior_parameters.",
                                             suffix=".writing", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, **parameters)
            temporary.rename(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    record.update(status="FITTED_FROZEN_NETWORK_AR_PRIOR", prior_parameters_sha256=sha256_file(path),
        fit_diagnostics=fitted["diagnostics"], causal_generation_executed=False)
    _write_json(record_path, record)
    return record


def _extract_remaining(cohort, selected_directory, output, device, cases, rows,
                       moments, record, record_path, seed):
    with LFADSConditionalSimulator.load(selected_directory, device=device) as model:
        cohort.verify_model(model)
        if model.history_bins + model.horizon_bins != COMPLETE_BINS or model.model_config["co_dim"] != 4:
            raise ValueError("SOURCE_FULL_EPISODE_FOUR_CHANNEL_MODEL_REQUIRED")
        before = {name: value.copy() for name, value in model._full.values().items()}
        updates = model.training_updates
        if (type(updates) is not int
                or updates != _proof_training_updates(record["binding"])):
            raise ValueError("POSTERIOR_EXTRACTION_CHECKPOINT_UPDATE_COUNT_CHANGED")
        if not before:
            raise ValueError("FROZEN_GLOBAL_VARIABLES_REQUIRED")
        original = model._full.session.run({key: model._full.variables[name]
                                            for key, name in PRIOR_VARIABLES.items()})
        record.update(original_prior_parameters={key: value.tolist() for key, value in original.items()},
                      execution_device=model.execution_device)
        _write_json(record_path, record)
        for case in cases[len(rows):]:
            filename = case["case_id"] + ".npz"
            path = output / filename
            identity = _cache_identity(record["binding"], case)
            if not path.exists():
                start = time.perf_counter()
                counts, stimulus = cohort.window(case)
                paths = sample_complete_posterior(model, counts[None], stimulus[None],
                    draws=JOINT_SAMPLES, seeds=tuple(identity["sampling_seeds"]),
                    draw_batch_size=POSTERIOR_DRAW_BATCH_SIZE)["paths"]
                del counts, stimulus
                after = model._full.values()
                if (set(after) != set(before)
                        or any(value.dtype != after[name].dtype
                               or not np.array_equal(value, after[name])
                               for name, value in before.items())
                        or type(model.training_updates) is not int
                        or model.training_updates != updates):
                    raise ValueError("POSTERIOR_EXTRACTION_CHANGED_FROZEN_NETWORK_OR_OPTIMIZER")
                proof = dict(frozen_global_variables_verified=len(before),
                             training_updates_before_and_after=updates)
                _commit_paths(path, paths, identity, time.perf_counter() - start, proof)
            row = _committed_row(path, identity)
            moments.update(_cached_paths(output, case, row, identity))
            rows.append(row)
            _write_json(record_path, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--inference-migration", type=Path)
    parser.add_argument("--continue-cache", action="store_true")
    args = parser.parse_args(argv)
    report = extract_and_fit(args.acquisition, args.fields, args.training, args.output,
        device=args.device, inference_migration=args.inference_migration,
        continue_cache=args.continue_cache)
    print(json.dumps(dict(status=report["status"], episodes=len(report["episodes"]),
        scientific_claim_qualified=False)))


if __name__ == "__main__":
    main()
