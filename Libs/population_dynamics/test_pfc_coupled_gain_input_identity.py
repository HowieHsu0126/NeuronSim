"""Stored-input identity regressions; fixtures contain no simulator/RNG execution."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).with_name("pfc_coupled_gain_input_identity.py")
SPEC = importlib.util.spec_from_file_location("coupled_gain_input_identity_under_test", MODULE_PATH)
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


def rng_arrays():
    result = {}
    for prefix in ("coarse", "refinement_1", "refinement_2"):
        for key, value in dict(
            algorithm=np.array("MT19937"), key=np.arange(624, dtype=np.uint32),
            position=np.array(17, dtype=np.int64), has_gauss=np.array(0, dtype=np.int64),
            cached_gaussian=np.array(0.0),
        ).items():
            result[prefix + "__" + key] = value.copy()
    return result


def test_auxiliary_rng_member_mismatch_is_rejected_when_coarse_rng_is_identical(tmp_path):
    left, right = rng_arrays(), rng_arrays()
    right["refinement_2__key"][-1] += 1
    np.savez(tmp_path / "left.npz", **left)
    np.savez(tmp_path / "right.npz", **right)
    result = identity.compare_stored_arrays(tmp_path / "left.npz", tmp_path / "right.npz", "final_rng.npz")
    assert result["complete_key_set_equal"]
    assert result["complete_coarse_and_refinement_rng_members"]
    assert not result["passed"]
    assert all(row["equal"] for row in result["members"] if row["member"].startswith("coarse__"))
    differing = [row for row in result["members"] if not row["equal"]]
    assert len(differing) == 1 and differing[0]["member"] == "refinement_2__key"
    assert differing[0]["differing_elements"] == 1


def test_missing_auxiliary_member_on_both_sides_is_not_a_complete_rng_comparison(tmp_path):
    arrays = rng_arrays()
    arrays.pop("refinement_1__cached_gaussian")
    np.savez(tmp_path / "rng.npz", **arrays)
    result = identity.compare_stored_arrays(tmp_path / "rng.npz", tmp_path / "rng.npz", "final_rng.npz")
    assert result["complete_key_set_equal"]
    assert not result["complete_coarse_and_refinement_rng_members"]
    assert not result["passed"]


@pytest.mark.parametrize("name", ["../outside.json", "/absolute.json", "source_cases/../report.json"])
def test_manifest_paths_cannot_escape_archive(name):
    with pytest.raises(identity.IdentityError, match="UNSAFE_MANIFEST_PATH"):
        identity._manifest({"files": [dict(path=name, bytes=0, sha256="a" * 64)]})


def write_archive(root, role, *, mismatch_stimulus=False, mismatch_unit_semantics=False):
    root.mkdir()
    files = []
    seed = 95004
    for dt, condition, relative in identity._case_specs(seed):
        folder = root / Path(relative).parent
        folder.mkdir(parents=True)
        ticks = int(round(5000.0 / dt))
        np.save(folder / "initial_voltage.npy", np.full(100000, -60.0, dtype=np.float64))
        np.savez(folder / "final_rng.npz", **rng_arrays())
        current = np.zeros((1, 80000), dtype=np.float64)
        if mismatch_stimulus and dt == 0.025 and condition == "cue":
            current[0, 9] = 1e-12
        np.savez(folder / "applied_current_observation.npz",
                 current_A=current, archive_unit_indices=np.arange(80000, dtype=np.int64))
        np.savez(folder / "units.npz", unit_ids=np.arange(100000, dtype=np.int64))
        (folder / "applied_current_metadata.json").write_text(
            json.dumps(dict(schema="synthetic_current", units="A")))
        (folder / "unit_metadata.json").write_text(
            json.dumps(dict(schema="synthetic_units",
                            initial_voltage_units="V" if mismatch_unit_semantics else "mV",
                            coupling=dict(circuit_provenance=dict(role=role)))))
        outputs = {}
        for name in identity.INPUT_FILES:
            path = folder / name
            binding = dict(bytes=path.stat().st_size, sha256=identity.sha256_file(path))
            outputs[name] = binding
            files.append(dict(path=str(path.relative_to(root)), **binding))
        selected = dict(case_id=f"pfc_{seed}_{condition}", source_rng_seed=seed,
                        stimulus_strength_nA=identity.STRENGTHS[condition], distractor_strength_nA=0.0)
        config = dict(scope="source_reproduction", excitatory_neurons=80000,
                      inhibitory_neurons=20000, dt_ms=dt, duration_ms=5000.0,
                      purpose="synthetic_" + role, cases=[selected], source_rng_seed=seed)
        if role == "candidate":
            config.update(sigma_weight_profile=20.0, Jpos_excit2excit=2.3)
        coarse = dict(random_stream_policy="source_global", seeds={"source_rng_seed": seed},
                      trajectory_retained=False, n_e=80000, n_i=20000,
                      dt_seconds=0.05 * 1e-3, ticks_generated=100000, total_ticks=100000,
                      numpy_version=np.__version__)
        drive = dict(
            input_mode="coupled_binomial_clock_refinement_stream", n_e=80000, n_i=20000,
            dt_seconds=dt * 1e-3, coarse_dt_seconds=0.05 * 1e-3,
            ticks_generated=ticks, total_ticks=ticks, trajectory_retained=False,
            refinement_factor=int(round(0.05 / dt)), all_refinement_levels_generated=True,
            buffered_fine_rows=0, external_event_totals={"E": 0, "I": 0},
            coupling=dict(coupling="binomial_parent_refinement",
                          refinement_seeds=[95417, 95418], coarse_source=coarse),
        )
        used = dict(input_mode=drive["input_mode"], ticks_consumed=ticks,
                    E=dict(values=ticks * 80000, events=0),
                    I=dict(values=ticks * 20000, events=0))
        code = {name: "a" * 64 for name in identity.INPUT_CODE}
        code.update(pfc_deterministic_replay=("b" if role == "baseline" else "c") * 64,
                    pfc_source_execution=("d" if role == "baseline" else "e") * 64)
        report = dict(
            status="COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE", source_execution_complete=True,
            scope=("development_paired_clock_calibration" if role == "baseline"
                   else "development_coupled_clock_recurrent_gain_replicates"),
            case_id=selected["case_id"], selected_case=selected, condition=condition,
            source_rng_seed=seed, refinement_seeds=[95417, 95418], dt_ms=dt,
            coarse_dt_ms=0.05, duration_ms=5000.0, coupled_core_units=100000,
            completed_duration_ticks=ticks, native_fine_seed_sequence_preserved=False,
            resolved_config=config, drive=drive, inputs_used=used, code=code,
            coarse_final_rng_matches_archived_episode=True,
            coarse_current_agreement={"synthetic_current_binding": True},
            validations={"synthetic_fixture_identity": {"passed": True}},
            outputs=outputs, fixture="synthetic_no_simulation",
        )
        path = folder / "report.json"
        path.write_text(json.dumps(report))
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=identity.sha256_file(path)))
    delivery = dict(
        status=("COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK" if role == "baseline"
                else "COMPLETE_COUPLED_CLOCK_RECURRENT_GAIN_BLOCK"),
        verified=True, job_id="synthetic_" + role, files=files,
    )
    (root / "delivery.json").write_text(json.dumps(delivery))
    return identity.sha256_file(root / "delivery.json")


def test_complete_archive_check_preserves_unit_provenance_differences(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    left = write_archive(baseline, "baseline")
    right = write_archive(candidate, "candidate")
    result = identity.check_archives(baseline, candidate, left, right, 95004, tmp_path / "result")
    assert result["status"] == identity.PASS_STATUS
    assert result["passed_pairs"] == result["unit_metadata_difference_pairs"] == 6
    assert result["source_simulations"] == result["new_rng_draws"] == 0
    assert result["full_external_streams_compared"] is False
    for pair in result["pairs"]:
        assert pair["unit_metadata"]["required_equal"] is False
        assert pair["unit_metadata"]["common_unit_semantics_equal"] is True
        assert pair["unit_metadata"]["differences"]
        assert pair["preserved_provenance_differences"]
        assert len(pair["array_checks"]) == 4


def test_cli_rejects_different_delivered_stimulus_after_valid_binding_checks(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    left = write_archive(baseline, "baseline")
    right = write_archive(candidate, "candidate", mismatch_stimulus=True)
    output = tmp_path / "result"
    code = identity.main([
        "--baseline-archive", str(baseline), "--candidate-archive", str(candidate),
        "--baseline-delivery-sha256", left, "--candidate-delivery-sha256", right,
        "--source-rng-seed", "95004", "--output", str(output),
    ])
    result = json.loads((output / "report.json").read_text())
    assert code == 1 and result["status"] == identity.FAIL_STATUS
    assert not result["failures"] and result["passed_pairs"] == 5
    failed = next(pair for pair in result["pairs"] if not pair["passed"])
    assert not failed["failures"]
    assert all(check["passed"] for check in failed["metadata_checks"])
    stimulus = next(check for check in failed["array_checks"]
                    if check["artifact"] == "applied_current_observation.npz")
    assert not stimulus["passed"]
    changed = next(row for row in stimulus["members"] if row["member"] == "current_A")
    assert changed["differing_elements"] == 1


def test_changed_unit_semantics_are_rejected_with_identical_stored_arrays(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    left = write_archive(baseline, "baseline")
    right = write_archive(candidate, "candidate", mismatch_unit_semantics=True)
    result = identity.check_archives(baseline, candidate, left, right, 95004, tmp_path / "result")
    assert result["status"] == identity.FAIL_STATUS
    assert result["passed_pairs"] == 0
    for pair in result["pairs"]:
        assert all(check["passed"] for check in pair["array_checks"])
        assert pair["unit_metadata"]["common_unit_semantics_equal"] is False
