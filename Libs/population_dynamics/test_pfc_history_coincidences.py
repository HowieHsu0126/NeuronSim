"""Sparse matched-time covariance must agree with full binary unit counts."""
import numpy as np
import pytest

from .pfc_history_coincidences import binary_event_keys, matched_history_variance


def keys_from_counts(counts):
    return [np.flatnonzero(block.ravel()).astype(np.int64) for block in counts]


@pytest.mark.parametrize('counts', [
    [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
    [[[1, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]],
    [[[1, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 0]]],
    [[[1, 0]], [[0, 1]], [[0, 0]]],
])
def test_sparse_diagonal_and_signed_covariance_equal_dense_oracle(counts):
    counts = np.asarray(counts, dtype=np.int64)
    rows, pairs = matched_history_variance(keys_from_counts(counts), 2, 1,
        np.asarray([0, counts.shape[1] * 80], dtype=np.int64))
    assert len(pairs) == 3
    for row in rows:
        columns = [0] if row['population'] == 'excitatory' else [1] if row['population'] == 'inhibitory' else [0, 1]
        x = counts[:, :, columns]
        units = len(columns)
        expected_diagonal = x.var(axis=0, ddof=1).mean(axis=0).sum() / units**2 / .001**2
        expected_variance = x.mean(axis=2).var(axis=0, ddof=1).mean() / .001**2
        assert row['diagonal_contribution_rate_Hz2'] == pytest.approx(expected_diagonal)
        assert row['population_mean_variance_rate_Hz2'] == pytest.approx(expected_variance)
        assert row['ordered_offdiagonal_contribution_rate_Hz2'] == pytest.approx(expected_variance - expected_diagonal)


def test_same_phase_totals_do_not_replace_time_and_unit_coincidences():
    a = np.array([[[1, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]])
    b = a.copy(); b[1] = b[1, ::-1]
    first, _ = matched_history_variance(keys_from_counts(a), 2, 1, np.array([0, 160]))
    shifted, _ = matched_history_variance(keys_from_counts(b), 2, 1, np.array([0, 160]))
    assert first[0]['unordered_history_pair_unit_bin_coincidences'] == 1
    assert shifted[0]['unordered_history_pair_unit_bin_coincidences'] == 0
    assert shifted[0]['diagonal_contribution_rate_Hz2'] == pytest.approx(2 * first[0]['diagonal_contribution_rate_Hz2'])


def test_negative_offdiagonal_and_half_open_boundaries_remain_visible():
    counts = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 0]]])
    rows, _ = matched_history_variance(keys_from_counts(counts), 2, 1, np.array([0, 80, 160]))
    assert rows[2]['ordered_offdiagonal_contribution_rate_Hz2'] < 0
    assert all(row['population_mean_variance_rate_Hz2'] == 0 for row in rows[3:])
    keys, terminal = binary_event_keys(np.array([0, 79, 80, 160]), np.array([0, 1, 0, 1]), 2, 160)
    assert keys.tolist() == [0, 1, 2] and terminal == 1


def test_nonbinary_or_unsorted_event_keys_are_rejected():
    with pytest.raises(ValueError, match='NONBINARY1MS'):
        binary_event_keys(np.array([0, 79]), np.array([0, 0]), 2, 160)
    with pytest.raises(ValueError, match='SORTED_UNIQUE'):
        matched_history_variance([np.array([0, 0]), np.array([], dtype=np.int64)], 2, 1, np.array([0, 80]))
    with pytest.raises(ValueError, match='EVENT_IDENTITY'):
        binary_event_keys(np.array([161]), np.array([0]), 2, 160)


def test_cohort_readiness_requires_all_three_histories_and_reuses_complete_groups():
    from .pfc_history_coincidences import ready_groups
    accepted = [0, 10, 20, 30, 40, 50, 60, 70, 80, 690, 700, 710]
    # Initial35event gap blocks circuit0/condition00 andcircuit1/condition00.
    assert ready_groups(accepted, [690], set(range(35))) == [60]
    assert ready_groups(accepted[:-4], [], set(range(35))) == []


def slurm_transfer_fixture(tmp_path, monkeypatch):
    import json, hashlib, shutil
    from . import pfc_history_coincidences as reader
    source = tmp_path / "persisted"; source.mkdir()
    identity = dict(episode_id="pfc_working_memory_circuit_001_block_006_condition_00",
        circuit_index=1, block_index=6, condition_index=0, split="train")
    raw = source / "raw.npz"; raw.write_bytes(b"exact original event bytes")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = source / "report.json"
    report.write_text(json.dumps(dict(config_sha256="c"*64,
        outputs={"raw.npz":dict(sha256=digest(raw),bytes=raw.stat().st_size)})))
    delivery = source / "delivery.json"
    delivery.write_text(json.dumps(dict(status="ARCHIVED_PFC_SOURCE_EXECUTION",job_id="643777",
        exit_code=0,source_execution_complete=True,case_id=identity["episode_id"])))
    receipt = source / "receipt.json"
    receipt.write_text(json.dumps(dict(index=30,status="VERIFIED_ORIGINAL_XIE_NATIVE_EVENTS_TRANSFERRED",
        source_execution_backend="slurm",source_job_id="643777",identity=identity,
        source_simulations=0,production_episodes=0,raw_uri="hdfs://original/raw.npz",raw_sha256=digest(raw),raw_bytes=raw.stat().st_size,
        source_report_uri="hdfs://original/report.json",source_report_sha256=digest(report),
        source_delivery_uri="hdfs://original/delivery.json",source_delivery_sha256=digest(delivery),
        transfer_run_id="merlin-transfer",transfer_trial_id="trial-transfer")))
    mapping={"hdfs://original/"+p.name:p for p in (raw,report,delivery,receipt)}
    monkeypatch.setattr(reader,"hdfs",lambda op,uri,destination:shutil.copyfile(mapping[uri],destination))
    task=dict(index=30,identity=identity,transport="original_slurm_events",source_job_id="643777",
        config_sha256="c"*64,receipt_uri="hdfs://original/receipt.json",receipt_sha256=digest(receipt))
    return reader,task,source,digest


def test_slurm_event_adapter_preserves_original_execution_and_transfer_identity(tmp_path, monkeypatch):
    reader,task,source,digest=slurm_transfer_fixture(tmp_path,monkeypatch)
    destination=tmp_path/"received";destination.mkdir()
    binding=reader.receive_slurm_source(task,destination)
    assert binding["source_execution_backend"]=="slurm"
    assert binding["source_run"]=="643777" and binding["source_trial"] is None
    assert binding["transfer_run"]=="merlin-transfer"
    assert binding["raw_sha256"]==digest(source/"raw.npz")
    assert (destination/"raw.npz").read_bytes()==(source/"raw.npz").read_bytes()


@pytest.mark.parametrize("change",["source_job","raw_bytes","config_hash"])
def test_slurm_event_adapter_rejects_changed_original_bindings(tmp_path,monkeypatch,change):
    reader,task,source,_=slurm_transfer_fixture(tmp_path,monkeypatch)
    if change=="source_job":task["source_job_id"]="999999"
    elif change=="raw_bytes":(source/"raw.npz").write_bytes(b"changed source")
    else:task["config_sha256"]="d"*64
    destination=tmp_path/"received";destination.mkdir()
    with pytest.raises(ValueError,match="IDENTITY_CHANGED|FILE_CHANGED|BINDING_CHANGED"):
        reader.receive_slurm_source(task,destination)
