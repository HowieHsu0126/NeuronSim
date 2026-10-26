"""Canonical metadata counters and immutable batch validation; no source work."""
import hashlib
import json
from pathlib import Path

import pytest

from . import pfc_merlin_canonical_acceptance as merger


BASE = "hdfs://haruna/checkpoints/20261022/original-run/pfc_intake_continuation"
EXECUTION = dict(backend="merlin", run_id="original-run", trial_id="original-trial", hostname="source-host",
                 implementation_sha256="a" * 64, numpy_version="2.4.2", recorded_utc="2026-10-24T12:00:00Z")


def write(path, value):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def scoped(**values):
    return dict(source_simulations=0, production_episodes=0, failures=[], source_qualification_granted=False,
                scaling_qualified=False, scientific_parameter_domain_qualified=False, **values)


def fixture(tmp_path, initial=None, reference_new=None, later=None):
    initial = {0, 1, 41, 42, 43} if initial is None else set(initial)
    reference_new = {2} if reference_new is None else set(reference_new)
    later = [{3, 4}] if later is None else [set(value) for value in later]
    directory = tmp_path / "batches"
    directory.mkdir()
    rows = []
    accepted = set(initial)
    for number, new in enumerate([reference_new, *later], 1):
        accepted |= new
        checkpoint = BASE + f"/batch_{number:04d}/pfc_scientific_intake"
        completed = [dict(index=index, acceptance_sha256=hashlib.sha256(f"task-{index}".encode()).hexdigest(),
                         acceptance_uri=checkpoint + f"/task_{index:04d}_acceptance.json") for index in sorted(new)]
        value = scoped(schema=merger.BATCH_SCHEMA, status="PASS", execution=dict(EXECUTION),
            config_sha256="b" * 64, control_validation="frozen/control/acceptance.json", control_validation_sha256="c" * 64,
            checkpoint_path=checkpoint, completed=completed, requested_task_indices=sorted(new),
            accepted_task_indices=sorted(accepted), accepted_episode_count=len(accepted), new_accepted_episode_count=len(new))
        path = directory / f"batch_{number:04d}_acceptance_package_bound.json"
        digest = write(path, value)
        rows.append(dict(acceptance=str(path), acceptance_sha256=digest, acceptance_uri=checkpoint + "/acceptance.json",
            batch_checkpoint_path=BASE + f"/batch_{number:04d}", completed=completed,
            new_accepted_episode_count=len(new), run_id=EXECUTION["run_id"], trial_id=EXECUTION["trial_id"], failures=[]))
    prior_original = initial | reference_new
    prior_row = dict(rows[0], batch=1)
    prior = scoped(schema=merger.SCHEMA, status="PASS", accepted_task_indices=sorted(prior_original | {1906}),
        accepted_episode_count=len(prior_original) + 1, original_continuous_accepted_episode_count=len(prior_original),
        continuous_intake_verified_batches=1, independent_recovery_indices=[1906], batches=[prior_row],
        prior_accepted_episode_count=len(initial) + 1, new_accepted_episode_count=len(reference_new),
        validator_source="preserved/old-validator.py", validator_source_sha256="d" * 64)
    progress = scoped(schema=merger.SCHEMA, status="WAITING_FOR_PERSISTED_SOURCE", target=3000,
        accepted_task_indices=sorted(accepted), accepted_episode_count=len(accepted), checkpoint_path=BASE,
        prior_accepted_episode_count=len(initial), new_accepted_episode_count=len(accepted - initial), batches=rows,
        execution={key:EXECUTION[key] for key in ("backend", "run_id", "trial_id", "hostname")})
    prior_path, progress_path = tmp_path / "prior.json", tmp_path / "progress.json"
    write(prior_path, prior)
    write(progress_path, dict(uri=BASE + "/progress.json", observed_utc="2026-10-24T12:10:00Z", progress=progress))
    return prior_path, progress_path, directory / "batch_0001_acceptance_package_bound.json", directory, tmp_path / "canonical.json"


def call(arguments):
    return merger.merge(*arguments)


def change_batch(arguments, mutate):
    _, progress_path, _, directory, _ = arguments
    wrapper = json.loads(progress_path.read_bytes())
    path = directory / "batch_0002_acceptance_package_bound.json"
    value = json.loads(path.read_bytes())
    mutate(value)
    wrapper["progress"]["batches"][1]["acceptance_sha256"] = write(path, value)
    write(progress_path, wrapper)


def test_cli_updates_all_counters_and_keeps_recovery_once(tmp_path, capsys):
    args = fixture(tmp_path)
    original_prior, original_progress = args[0].read_bytes(), args[1].read_bytes()
    names = ("--prior", "--progress", "--reference", "--batch-directory", "--output")
    argv = [item for name,path in zip(names,args) for item in (name,str(path))]
    merger.main(argv)
    printed = json.loads(capsys.readouterr().out)
    result = json.loads(args[-1].read_bytes())
    assert printed["accepted_episode_count"] == result["accepted_episode_count"] == 9
    assert result["original_continuous_accepted_episode_count"] == 8
    assert result["continuous_intake_verified_batches"] == 2
    assert result["prior_accepted_episode_count"] == 7 and result["new_accepted_episode_count"] == 2
    assert result["accepted_task_indices"].count(1906) == 1
    assert {41, 42, 43} <= set(result["accepted_task_indices"])
    assert result["independent_recovery_indices"] == [1906]
    assert result["batches"][0]["batch"] == 2
    assert result["validator_source"] == str(Path(merger.__file__).resolve())
    assert result["validator_source_sha256"] == hashlib.sha256(Path(merger.__file__).read_bytes()).hexdigest()
    assert result["progress_snapshot_sha256"] == hashlib.sha256(original_progress).hexdigest()
    assert result["source_simulations"] == result["production_episodes"] == 0
    assert all(result[field] is False for field in merger.FLAGS)
    assert args[0].read_bytes() == original_prior and args[1].read_bytes() == original_progress


def test_original2999_plus_independent1906_produces_exact3000(tmp_path):
    original_prefix = set(range(2999)) - {1906}
    args = fixture(tmp_path, initial=original_prefix - {2998}, reference_new={2998}, later=[{2999}])
    result = call(args)
    assert result["original_continuous_accepted_episode_count"] == 2999
    assert result["accepted_episode_count"] == 3000
    assert result["accepted_task_indices"] == list(range(3000))
    assert result["independent_recovery_indices"] == [1906]
    assert result["new_accepted_episode_count"] == 1
    assert result["source_qualification_granted"] is False


@pytest.mark.parametrize("field", ["original_continuous_accepted_episode_count", "accepted_episode_count", "continuous_intake_verified_batches"])
def test_inconsistent_prior_counters_are_rejected_without_output(tmp_path, field):
    args = fixture(tmp_path)
    prior = json.loads(args[0].read_bytes())
    prior[field] += 1
    write(args[0], prior)
    with pytest.raises(ValueError):
        call(args)
    assert not args[-1].exists()


def test_changed_immutable_package_hash_is_rejected(tmp_path):
    args = fixture(tmp_path)
    path = args[3] / "batch_0002_acceptance_package_bound.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="IMMUTABLE_BATCH_HASH_CHANGED"):
        call(args)
    assert not args[-1].exists()


@pytest.mark.parametrize("change", ["execution", "control", "source_scope", "qualification", "requested_identity", "cumulative_identity"])
def test_rehashed_but_invalid_batch_evidence_is_rejected(tmp_path, change):
    args = fixture(tmp_path)
    def mutate(value):
        if change == "execution":
            value["execution"]["implementation_sha256"] = "e" * 64
        elif change == "control":
            value["control_validation_sha256"] = "e" * 64
        elif change == "source_scope":
            value["source_simulations"] = 1
        elif change == "qualification":
            value["scaling_qualified"] = True
        elif change == "requested_identity":
            value["requested_task_indices"] = [3]
        else:
            value["accepted_task_indices"].remove(0)
            value["accepted_task_indices"].append(2998)
    change_batch(args, mutate)
    with pytest.raises(ValueError):
        call(args)
    assert not args[-1].exists()


@pytest.mark.parametrize("change", ["overlap", "gap", "count", "recovery_overlap", "bad_task_hash"])
def test_progress_overlap_gap_count_and_binding_rejections(tmp_path, change):
    args = fixture(tmp_path)
    wrapper = json.loads(args[1].read_bytes())
    progress = wrapper["progress"]
    row = progress["batches"][1]
    if change == "overlap":
        row["completed"][0]["index"] = 2
        row["completed"][0]["acceptance_uri"] = BASE + "/batch_0002/pfc_scientific_intake/task_0002_acceptance.json"
    elif change == "gap":
        row["batch_checkpoint_path"] = BASE + "/batch_0003"
    elif change == "count":
        progress["accepted_episode_count"] += 1
    elif change == "recovery_overlap":
        progress["accepted_task_indices"].append(1906)
        progress["accepted_episode_count"] += 1
    else:
        row["completed"][0]["acceptance_sha256"] = "invalid"
    write(args[1], wrapper)
    with pytest.raises(ValueError):
        call(args)
    assert not args[-1].exists()


def test_changed_reference_hash_is_rejected(tmp_path):
    args = fixture(tmp_path)
    args[2].write_bytes(args[2].read_bytes() + b" ")
    with pytest.raises(ValueError, match="REFERENCE_HASH_CHANGED"):
        call(args)
    assert not args[-1].exists()


def test_existing_output_is_never_overwritten(tmp_path):
    args = fixture(tmp_path)
    args[-1].write_bytes(b"preserved authoritative artifact")
    with pytest.raises(ValueError, match="FRESH_OUTPUT_REQUIRED"):
        call(args)
    assert args[-1].read_bytes() == b"preserved authoritative artifact"


# These tests exercise metadata transport/identity. They run on Merlin CPU;
# no mechanistic simulation, observation generation or scientific gate changes.
def accepted_task(index, *, identity=None, bundle_hash="e" * 64, observation_hash="f" * 64):
    checkpoint = f"hdfs://haruna/source/source-run/pfc/task_{index:04d}"
    receipt = dict(index=index, run_id="source-run", trial_id="source-trial", xie_job=f"source-{index}",
                   status="COMPLETE", exit_code=0, persisted=True, checkpoint_path=checkpoint,
                   result_bytes=86010908, result_sha256=bundle_hash)
    return scoped(status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE", index=index,
        identity=identity or dict(episode_id=f"episode-{index}", circuit_id=23, block_id=7, condition_id=index),
        run_id="source-run", trial_id="source-trial", remote_receipt=receipt,
        receipt_uri=f"hdfs://haruna/recovery/task_{index:04d}/receipt.json",
        source_bundle_uri=checkpoint + "/result.tar.gz", source_bundle_sha256=bundle_hash,
        observation_entry=dict(path=f"canonical/observations/task_{index:04d}",
            development_report_sha256="d" * 64, observation_report_sha256=observation_hash))


def store_task(directory, value, checkpoint):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    (directory / (digest + ".json")).write_bytes(data)
    return dict(index=value["index"], acceptance_sha256=digest,
                acceptance_uri=checkpoint + f"/task_{value['index']:04d}_acceptance.json")


def recovery_report(arguments, directory, values):
    prior = json.loads(arguments[0].read_bytes())
    checkpoint = "hdfs://haruna/new-validation/pfc_scientific_intake"
    completed = [store_task(directory, value, checkpoint) for value in values]
    new = {item["index"] for item in completed}
    cumulative = set(prior["accepted_task_indices"]) | new
    report = scoped(schema=merger.BATCH_SCHEMA, status="PASS", execution=dict(EXECUTION, run_id="new-validation"),
        prior_acceptance_sha256=hashlib.sha256(arguments[0].read_bytes()).hexdigest(), config_sha256="b" * 64,
        control_validation="frozen/control/acceptance.json", control_validation_sha256="c" * 64,
        checkpoint_path=checkpoint, completed=completed, requested_task_indices=sorted(new),
        accepted_task_indices=sorted(cumulative), accepted_episode_count=len(cumulative),
        new_accepted_episode_count=len(new))
    path = directory.parent / "recovery.json"
    write(path, report)
    return path


def bind_original_task(arguments, directory, value, number=2):
    wrapper = json.loads(arguments[1].read_bytes())
    path = arguments[3] / f"batch_{number:04d}_acceptance_package_bound.json"
    batch = json.loads(path.read_bytes())
    checkpoint = BASE + f"/batch_{number:04d}/pfc_scientific_intake"
    record = store_task(directory, value, checkpoint)
    batch["completed"] = [record if item["index"] == value["index"] else item for item in batch["completed"]]
    row = wrapper["progress"]["batches"][number - 1]
    row["completed"] = batch["completed"]
    row["acceptance_sha256"] = write(path, batch)
    write(arguments[1], wrapper)
    return record


def test_independent_recovery_adds_only_unique_ids_and_keeps_scientific_scope(tmp_path):
    args = fixture(tmp_path)
    directory = tmp_path / "tasks"
    directory.mkdir()
    report = recovery_report(args, directory, [accepted_task(10), accepted_task(11)])
    prior_bytes, progress_bytes, report_bytes = args[0].read_bytes(), args[1].read_bytes(), report.read_bytes()
    result = merger.merge(*args, recovery_reports=[report], task_acceptance_directory=directory)
    assert result["accepted_episode_count"] == 11
    assert result["new_accepted_episode_count"] == 4
    assert result["new_original_continuous_episode_count"] == 2
    assert result["original_continuous_accepted_episode_count"] == 8
    assert result["independent_recovery_indices"] == [10, 11, 1906]
    assert result["independent_recovery_bindings"][0]["source_binding"]["source_bundle_bytes"] == 86010908
    assert all(result[field] is False for field in merger.FLAGS)
    assert result["source_simulations"] == result["production_episodes"] == 0
    assert (args[0].read_bytes(), args[1].read_bytes(), report.read_bytes()) == (prior_bytes, progress_bytes, report_bytes)


def test_recovery_only_then_future_original_batch_counts_overlap_once(tmp_path):
    args = fixture(tmp_path, later=[])
    directory = tmp_path / "tasks"
    directory.mkdir()
    task = accepted_task(10)
    report = recovery_report(args, directory, [task])
    first = merger.merge(*args, recovery_reports=[report], task_acceptance_directory=directory)
    assert first["new_accepted_episode_count"] == 1 and first["new_original_continuous_episode_count"] == 0
    assert first["batches"] == []
    checkpoint = BASE + "/batch_0002/pfc_scientific_intake"
    # Original and recovered validation JSONs have different transport metadata.
    original_task = json.loads(json.dumps(task))
    original_task["receipt_uri"] = "hdfs://haruna/source/task_0010/receipt.json"
    original_task["validation_execution"] = EXECUTION
    record = store_task(directory, original_task, checkpoint)
    original = set(first["original_continuous_task_indices"]) | {10}
    batch = scoped(schema=merger.BATCH_SCHEMA, status="PASS", execution=dict(EXECUTION),
        config_sha256="b" * 64, control_validation="frozen/control/acceptance.json", control_validation_sha256="c" * 64,
        checkpoint_path=checkpoint, completed=[record], requested_task_indices=[10],
        accepted_task_indices=sorted(original), accepted_episode_count=len(original), new_accepted_episode_count=1)
    path = args[3] / "batch_0002_acceptance_package_bound.json"
    wrapper = json.loads(args[1].read_bytes())
    wrapper["progress"]["batches"].append(dict(acceptance_sha256=write(path, batch),
        acceptance_uri=checkpoint + "/acceptance.json", completed=[record], new_accepted_episode_count=1,
        run_id=EXECUTION["run_id"], trial_id=EXECUTION["trial_id"]))
    wrapper["progress"].update(accepted_task_indices=sorted(original), accepted_episode_count=len(original),
                               new_accepted_episode_count=2)
    write(args[1], wrapper)
    output = tmp_path / "next_canonical.json"
    second = merger.merge(args[-1], args[1], args[2], args[3], output, task_acceptance_directory=directory)
    assert second["accepted_episode_count"] == first["accepted_episode_count"]
    assert second["new_accepted_episode_count"] == 0
    assert second["new_original_continuous_episode_count"] == 1
    assert second["accepted_task_indices"].count(10) == 1
    assert second["independent_recovery_indices"] == [10, 1906]


def test_same_source_recovery_already_in_original_progress_is_counted_once(tmp_path):
    args = fixture(tmp_path)
    directory = tmp_path / "tasks"
    directory.mkdir()
    task = accepted_task(3)
    report = recovery_report(args, directory, [task])
    bind_original_task(args, directory, dict(task, validation_execution=EXECUTION))
    result = merger.merge(*args, recovery_reports=[report], task_acceptance_directory=directory)
    assert result["accepted_episode_count"] == 9 and result["new_accepted_episode_count"] == 2
    assert result["independent_recovery_indices"] == [3, 1906]


@pytest.mark.parametrize("change", ["identity", "bundle", "observation", "source_trial", "source_job"])
def test_original_recovery_source_conflicts_are_rejected(tmp_path, change):
    args = fixture(tmp_path)
    directory = tmp_path / "tasks"
    directory.mkdir()
    task = accepted_task(3)
    report = recovery_report(args, directory, [task])
    changed = json.loads(json.dumps(task))
    if change == "identity":
        changed["identity"]["block_id"] += 1
    elif change == "bundle":
        changed["source_bundle_sha256"] = changed["remote_receipt"]["result_sha256"] = "a" * 64
    elif change == "observation":
        changed["observation_entry"]["observation_report_sha256"] = "a" * 64
    elif change == "source_trial":
        changed["trial_id"] = changed["remote_receipt"]["trial_id"] = "different-source-trial"
    else:
        changed["remote_receipt"]["xie_job"] = "different-source-job"
    bind_original_task(args, directory, changed)
    with pytest.raises(ValueError, match="RECOVERY_SOURCE_CONFLICT"):
        merger.merge(*args, recovery_reports=[report], task_acceptance_directory=directory)
    assert not args[-1].exists()


@pytest.mark.parametrize("change", ["status", "control", "qualification", "source_scope", "requested", "task_hash", "cumulative"])
def test_independent_recovery_invalid_evidence_is_rejected(tmp_path, change):
    args = fixture(tmp_path)
    directory = tmp_path / "tasks"
    directory.mkdir()
    report_path = recovery_report(args, directory, [accepted_task(10)])
    report = json.loads(report_path.read_bytes())
    if change == "status":
        report["status"] = "RUNNING"
    elif change == "control":
        report["control_validation_sha256"] = "a" * 64
    elif change == "qualification":
        report["scaling_qualified"] = True
    elif change == "source_scope":
        report["source_simulations"] = 1
    elif change == "requested":
        report["requested_task_indices"] = [11]
    elif change == "task_hash":
        path = directory / (report["completed"][0]["acceptance_sha256"] + ".json")
        path.write_bytes(path.read_bytes() + b" ")
    else:
        report["accepted_task_indices"].append(2999)
        report["accepted_episode_count"] += 1
    write(report_path, report)
    with pytest.raises(ValueError):
        merger.merge(*args, recovery_reports=[report_path], task_acceptance_directory=directory)
    assert not args[-1].exists()


def test_legacy1906_can_be_hydrated_without_top_level_bundle_bytes(tmp_path):
    args = fixture(tmp_path)
    directory = tmp_path / "tasks"
    directory.mkdir()
    task = accepted_task(1906)
    assert "source_bundle_bytes" not in task
    report = recovery_report(args, directory, [task])
    result = merger.merge(*args, recovery_reports=[report], task_acceptance_directory=directory)
    assert result["accepted_episode_count"] == 9
    assert result["independent_recovery_bindings"][0]["source_binding"]["source_bundle_bytes"] == 86010908
