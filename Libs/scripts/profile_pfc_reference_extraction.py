"""Validate and time PFC state callbacks against a frozen 100k source prefix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def worker(args):
    import numpy as np
    from Libs.population_dynamics import pfc_first_passage_reference as reference
    from Libs.population_dynamics import pfc_first_passage_history as history
    from Libs.population_dynamics import pfc_clock_coupling as coupling

    source = json.loads((args.source_archive / "report.json").read_text())
    config, case = source["resolved_config"], source["selected_case"]
    with np.load(args.source_archive / "raw.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in ("spike_ticks", "spike_unit_indices", "native_spike_counts", "spike_seconds")}
    tape = reference.validate_tape(config, raw)
    del raw
    stop = 8192
    selected = np.arange(64)
    captures = list(range(stop - 256, stop))
    results = []
    if args.history_frame:
        frame = json.loads(args.history_frame.read_text())
        rows = [dict(row) for row in frame['rows'] if row['window_stop_tick'] <= stop]
        for index, row in enumerate(rows):
            row['sample_index'] = index
        rows = history.classify(config, tape, rows)
        _, drive = history.create_input_drive(config, case, source, 'coupled_binomial_clock_refinement_stream')
        started = time.perf_counter()
        refs, reconstruction = history.extract(config, case, tape, rows, drive=drive)
        extraction_seconds = time.perf_counter() - started
        signatures = {}
        for index, (initial, forcing) in refs.items():
            signatures[str(index)] = dict(initial=initial, forcing={key: hashlib.sha256(value.tobytes()).hexdigest()
                                                                    for key, value in forcing.items()})
        import yaml
        settings = yaml.safe_load(args.history_settings.read_text())
        settings.update(source_rng_seed=case['source_rng_seed'], field_index=frame['frame']['field_index'],
                        source={'archive': str(args.source_archive), 'delivery_sha256': args.source_delivery_sha256},
                        integration_seed_context=[1, 1, 2])
        destination = args.result.parent / (args.worker + '_integration')
        destination.mkdir()
        started = time.perf_counter()
        values, variances = history.integrate_rows(config, settings, rows, refs, destination)
        integration_seconds = time.perf_counter() - started
        rng = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in
               history.prefix_state_arrays(drive, 'coupled_binomial_clock_refinement_stream').items()}
        write_json(args.result, dict(mode=args.worker, n_e=tape.n_e, n_i=tape.n_i,
                   configured_duration_ms=config['duration_ms'], dt_ms=config['dt_ms'],
                   real_selected_rows=len(rows), eligible_integrated_histories=len(refs),
                   rows=rows, history_signatures=signatures, reconstruction=reconstruction,
                   integrated_values=np.asarray(values).tolist(), signed_inner_variances=np.asarray(variances).tolist(),
                   prefix_rng_states=rng, extraction_seconds=extraction_seconds, integration_seconds=integration_seconds))
        return 0
    for _ in range(3):
        _, drive = history.create_input_drive(config, case, source, "coupled_binomial_clock_refinement_stream")
        signature = hashlib.sha256()
        values = 0

        def observe(stage, tick, state):
            nonlocal values
            if stage not in ("pre_rk2", "pre_input"):
                return
            signature.update(stage.encode())
            signature.update(str(tick).encode())
            for key in reference.FIELDS + reference.E_FIELDS:
                array = state["E__" + key][selected]
                signature.update(array.tobytes())
                values += array.size

        callback = {"on_state_view" if args.worker == "candidate" else "on_snapshot": observe}
        started = time.perf_counter()
        result = reference.reconstruct(config, case, tape, stop_tick=stop,
                                       drive=drive, capture_ticks=captures, **callback)
        elapsed = time.perf_counter() - started
        terminal = {key: hashlib.sha256(value.tobytes()).hexdigest()
                    for key, value in result["terminal_state"].items()}
        rng = {key: hashlib.sha256(value.tobytes()).hexdigest()
               for key, value in history.prefix_state_arrays(drive, "coupled_binomial_clock_refinement_stream").items()}
        results.append(dict(wall_seconds=elapsed, observed_values=values,
                            observation_sha256=signature.hexdigest(), terminal_states=terminal,
                            prefix_rng_states=rng, source_events_consumed=result["source_events_consumed"]))
    write_json(args.result, dict(mode=args.worker, n_e=tape.n_e, n_i=tape.n_i,
                                configured_duration_ms=config["duration_ms"], dt_ms=config["dt_ms"],
                                prefix_ticks=stop, capture_ticks=len(captures), results=results))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--worker", choices=("legacy", "candidate"))
    parser.add_argument("--result", type=Path)
    parser.add_argument('--history-frame', type=Path)
    parser.add_argument('--history-settings', type=Path)
    parser.add_argument('--verified-tests', type=Path)
    parser.add_argument('--source-delivery-sha256')
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    assert socket.gethostname().split(".")[0] == "cpu04" and not os.environ.get("SLURM_JOB_GPUS")
    assert (args.project_root / "PROJECT_PREFIX").read_text().strip() == "NrnSim"
    job = os.environ["SLURM_JOB_ID"]
    archive = args.output_parent / ("run_" + job)
    assert not archive.exists()
    report = dict(status="FAILED_PFC_REFERENCE_PERFORMANCE", job_id=job,
                  source_archive=str(args.source_archive), created_utc=datetime.now(timezone.utc).isoformat(),
                  scientific_fidelity_qualified=False, production_episodes=0, new_source_episodes=0,
                  changed_factor="Native dense refinement and copied snapshots versus cached small-count inversion and read-only views",
                  unchanged_factors="Archived tape,100k units,full-duration config,source and refinement seeds,clock and arithmetic",
                  inputs=[])
    exit_code = 1
    with tempfile.TemporaryDirectory(prefix="neuronsim-reference-performance-" + job + ".", dir="/tmp") as temporary:
        work = Path(temporary)
        try:
            def copy(source, relative, expected=None):
                target = work / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                sha = digest(target)
                if expected is not None and sha != expected:
                    raise ValueError("PFC_PERFORMANCE_INPUT_CHANGED:" + str(relative))
                report["inputs"].append(dict(source=str(source), path=str(relative), sha256=sha))
                return target

            copy(Path(__file__), Path("profile_pfc_reference_extraction.py"))
            copy(Path(os.environ["NRNSIM_PERFORMANCE_BATCH"]), Path("launcher.sbatch"))
            prep = json.loads((args.preparation / "delivery.json").read_text())
            assert prep["verified"] and prep["status"] == "COMPLETE_PFC_PROFILE_HISTORY_PREPARE"
            report["preparation_delivery_sha256"] = digest(args.preparation / "delivery.json")
            entries = [entry for entry in prep["files"] if entry["path"].startswith("Libs/") and entry["path"].endswith(".py")]
            for mode in ("legacy", "candidate"):
                for entry in entries:
                    copy(args.preparation / entry["path"], Path(mode) / entry["path"], entry["sha256"])
            for name in ("pfc_first_passage_reference.py", "pfc_first_passage_history.py", "test_pfc_first_passage_reference.py", "pfc_clock_coupling.py", "test_pfc_clock_coupling.py", "test_pfc_coupled_drive.py"):
                relative = Path("Libs/population_dynamics") / name
                copy(args.project_root / relative, Path("candidate") / relative)
            if args.history_frame:
                copy(args.history_frame, Path('selected_frame.json'))
                copy(args.preparation / 'Libs/configs/neural_population_pfc_profile_history.yaml', Path('history_settings.yaml'))
            source = json.loads((args.source_archive / "delivery.json").read_text())
            assert source["verified"] and source["status"] == "COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK"
            report["source_delivery_sha256"] = digest(args.source_archive / "delivery.json")
            manifest = {entry["path"]: entry for entry in source["files"]}
            case = Path("source_cases/dt_0p025/pfc_95008_strong_cue")
            for name in ("report.json", "raw.npz"):
                relative = case / name
                copy(args.source_archive / relative, Path("source") / name, manifest[str(relative)]["sha256"])
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", CUDA_VISIBLE_DEVICES="")
            env.pop("PYTHONPATH", None)
            sys.path.insert(0, str(work / "candidate"))
            from Libs.population_dynamics.resource_probe import probe
            health = probe(work / "resource_preflight.json", 4)
            assert health["status"] == "PASS", health["reasons"]
            tests = ["test_pfc_first_passage_reference.py", "test_pfc_first_passage_history.py", "test_pfc_first_passage_fine_history.py", "test_pfc_profile_history.py", "test_pfc_clock_coupling.py", "test_pfc_coupled_drive.py"]
            command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--import-mode=importlib", *["Libs/population_dynamics/" + name for name in tests], "--basetemp=" + str(work / "tests_tmp"), "--junitxml=" + str(work / "tests.xml")]
            if args.verified_tests:
                prior_delivery = json.loads((args.verified_tests / 'delivery.json').read_text())
                prior = json.loads((args.verified_tests / 'report.json').read_text())
                assert prior_delivery['verified'] and prior['exit_code'] == 0
                assert prior['prefix_state_and_rng_exact']
                prior_entries = {entry['path']: entry for entry in prior_delivery['files']}
                assert digest(args.verified_tests / 'report.json') == prior_entries['report.json']['sha256']
                for name in ('pfc_first_passage_reference.py', 'pfc_first_passage_history.py', 'pfc_clock_coupling.py',
                             'test_pfc_first_passage_reference.py', 'test_pfc_clock_coupling.py', 'test_pfc_coupled_drive.py'):
                    relative = Path('candidate/Libs/population_dynamics') / name
                    assert digest(work / relative) == prior_entries[str(relative)]['sha256']
                report['tests'] = prior['tests']
                report['test_exit_code'] = 0
                report['tests_reused_archive'] = str(args.verified_tests)
                report['test_command'] = prior['test_command']
            else:
                with (work / "tests.stdout").open("w") as out, (work / "tests.stderr").open("w") as err:
                    tested = subprocess.run(command, cwd=work / "candidate", env=env, stdout=out, stderr=err)
                report["test_command"] = command
                report["test_exit_code"] = tested.returncode
                if (work / "tests.xml").exists():
                    suites = ET.parse(work / "tests.xml").getroot()
                    report["tests"] = [{key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")} for suite in suites.iter("testsuite")]
                assert tested.returncode == 0, "PFC_REFERENCE_TESTS_FAILED"
            for mode in ("legacy", "candidate"):
                command = [sys.executable, "-B", str(work / "profile_pfc_reference_extraction.py"), "--worker", mode, "--source-archive", str(work / "source"), "--result", str(work / (mode + ".json"))]
                if args.history_frame:
                    command += ['--history-frame', str(work / 'selected_frame.json'), '--history-settings', str(work / 'history_settings.yaml'), '--source-delivery-sha256', report['source_delivery_sha256']]
                worker_env = dict(env, PYTHONPATH=str(work / mode))
                with (work / (mode + ".stdout")).open("w") as out, (work / (mode + ".stderr")).open("w") as err:
                    completed = subprocess.run(command, cwd=work / mode, env=worker_env, stdout=out, stderr=err)
                assert completed.returncode == 0, "PFC_PERFORMANCE_WORKER_FAILED:" + mode
            baseline = json.loads((work / "legacy.json").read_text())
            candidate = json.loads((work / "candidate.json").read_text())
            if args.history_frame:
                excluded = ('mode', 'extraction_seconds', 'integration_seconds')
                assert {k: v for k, v in baseline.items() if k not in excluded} == {k: v for k, v in candidate.items() if k not in excluded}
                report['performance_scope'] = 'Real selected source histories in8192fine-tick prefix plus10000inner draws per eligible history; not full-episode speedup'
                report['history_forcing_and_integrated_scores_exact'] = True
            else:
                for left, right in zip(baseline["results"], candidate["results"]):
                    assert {key: value for key, value in left.items() if key != "wall_seconds"} == {key: value for key, value in right.items() if key != "wall_seconds"}
            report["baseline"] = baseline
            report["candidate"] = candidate
            report["prefix_state_and_rng_exact"] = True
            report["status"] = "PASS_PFC_REFERENCE_PERFORMANCE_VALIDATION"
            if not args.history_frame:
                report["performance_scope"] = "8192 fine ticks from one complete100k archived source, including cue onset; not full-episode speedup"
            exit_code = 0
        except Exception:
            report["error"] = traceback.format_exc()
        finally:
            report.update(exit_code=exit_code, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(work / "report.json", report)
            archive.mkdir(parents=True, exist_ok=False)
            names = ("report.json", "launcher.sbatch", "profile_pfc_reference_extraction.py", "resource_preflight.json", "tests.xml", "tests.stdout", "tests.stderr", "legacy.json", "candidate.json", "legacy.stderr", "candidate.stderr", "selected_frame.json", "history_settings.yaml")
            kept = [work / name for name in names]
            kept.extend(work / "candidate" / "Libs/population_dynamics" / name for name in ("pfc_first_passage_reference.py", "pfc_first_passage_history.py", "test_pfc_first_passage_reference.py", "pfc_clock_coupling.py", "test_pfc_clock_coupling.py", "test_pfc_coupled_drive.py"))
            files = []
            for path in kept:
                if path.is_file():
                    relative = path.relative_to(work)
                    target = archive / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    assert digest(target) == digest(path)
                    files.append(dict(path=str(relative), bytes=target.stat().st_size, sha256=digest(target)))
            write_json(archive / "delivery.json", dict(status=report["status"], job_id=job, verified=True, files=files))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
