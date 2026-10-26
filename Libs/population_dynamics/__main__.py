"""Real-data cohort, intake and single-session qualification commands."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import time
import json
from pathlib import Path
import re
import sys
from typing import Any

from .cohort import cohort_session, fetch_metadata, freeze_cohort, read_metadata, resolve_source
from .config import load_config, project_path
from .download import download_nwb, open_remote_nwb, verify_local_nwb
from .provenance import canonical_json, freeze_json, runtime_manifest, sha256_file


def _write_csv(path: Path, frame: Any) -> None:
    from .qualification import _json_value
    serial = frame.copy()
    for name in serial.columns:
        if serial[name].dtype == object:
            serial[name] = serial[name].map(lambda value: json.dumps(_json_value(value), sort_keys=True)
                                            if isinstance(value, (list, tuple, dict)) or hasattr(value, "shape")
                                            else value)
    payload = serial.to_csv(index=False)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"FROZEN_TABLE_CHANGE_REJECTED:{path}")
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", type=Path, required=True)
    root.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = root.add_subparsers(dest="operation", required=True)
    cohort = commands.add_parser("cohort", help="freeze all matching sessions with animal-level splits")
    cohort.add_argument("--metadata-json", type=Path, help="existing complete RMA JSON or session metadata CSV")
    cohort.add_argument("--output-dir", type=Path)
    opto = commands.add_parser("opto-source-read", help="bounded native read of explicitly staged Opto source arrays")
    opto.add_argument("--source-dir", type=Path, required=True)
    opto.add_argument("--diagnostic-unit-ids", nargs="+", type=int, required=True,
                      help="explicit staged source unit IDs; always partial engineering qualification")
    opto.add_argument("--output-dir", type=Path, required=True)
    opto_windows = commands.add_parser("opto-window-check", help="engineering replay of Opto counts and nominal pulse supports")
    opto_windows.add_argument("--source-dir", type=Path, required=True)
    opto_windows.add_argument("--control-dir", type=Path, required=True)
    opto_windows.add_argument("--trial-ids", nargs="+", type=int, required=True,
                              help="explicit source trial IDs; always partial engineering qualification")
    opto_windows.add_argument("--output-dir", type=Path, required=True)
    field = commands.add_parser("population-field-check", help="source-indexed population readout engineering replay")
    field.add_argument("--source-dir", type=Path, required=True)
    field.add_argument("--window-dir", type=Path, required=True)
    field.add_argument("--output-dir", type=Path, required=True)
    spatial = commands.add_parser("mesocircuit-field-check", help="native periodic population-field engineering replay")
    spatial.add_argument("--source-dir", type=Path, required=True)
    spatial.add_argument("--output-dir", type=Path, required=True)
    windows = commands.add_parser("window-check", help="engineering replay of training-anchor count windows")
    windows.add_argument("--recording-dir", type=Path, required=True)
    windows.add_argument("--output-dir", type=Path, required=True)
    baseline = commands.add_parser("baseline-check", help="bounded engineering fit and replay of an explicit baseline")
    baseline.add_argument("--recording-dir", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    baseline.add_argument("--model", choices=("poisson-autoregression", "reduced-rank", "poisson-state-space", "lfads", "braid"),
                          default="poisson-autoregression")
    for name in ("intake", "qualify"):
        command = commands.add_parser(name)
        command.add_argument("--session-id", type=int, required=True)
        command.add_argument("--cohort", type=Path)
        if name == "qualify":
            inputs = command.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--local-nwb", "--nwb", dest="local_nwb", type=Path)
            inputs.add_argument("--source-url", help="configured, version-bound official NWB mirror")
            command.add_argument("--diagnostic-unit-ids", nargs="+", type=int,
                                 help="explicit QC-eligible training-anchor units; always partial qualification")
        else:
            command.add_argument("--local-nwb", "--nwb", dest="local_nwb", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--run-id", default="source_qualification")
        if name == "intake":
            command.add_argument("--cache-dir", type=Path, help="required node-local scratch for downloads")
    return root


def run(args: argparse.Namespace, command: list[str]) -> tuple[dict[str, Any], int]:
    project = args.project_root.resolve()
    config = load_config(project_path(project, args.config))
    output = project_path(project, args.output_dir or config["output_dir"])
    if args.operation == "opto-source-read":
        if "opto_source" not in config:
            raise ValueError("OPTO_SOURCE_CONFIG_REQUIRED")
        from .opto_source import check_opto_source
        return check_opto_source(project_path(project, args.source_dir), output, config, command,
                                 args.diagnostic_unit_ids)
    if args.operation == "opto-window-check":
        if "opto_window_check" not in config:
            raise ValueError("OPTO_WINDOW_CHECK_CONFIG_REQUIRED")
        from .opto_windows import check_opto_windows
        return check_opto_windows(project_path(project, args.source_dir),
                                  project_path(project, args.control_dir), output, config, command,
                                  args.trial_ids)
    if args.operation == "population-field-check":
        if "population_field_check" not in config:
            raise ValueError("POPULATION_FIELD_CHECK_CONFIG_REQUIRED")
        from .population_field_check import check_population_field
        return check_population_field(project_path(project, args.source_dir),
                                      project_path(project, args.window_dir), output, config, command)
    if args.operation == "mesocircuit-field-check":
        if "mesocircuit_field_check" not in config:
            raise ValueError("MESOCIRCUIT_FIELD_CHECK_CONFIG_REQUIRED")
        from .mesocircuit_field import check_mesocircuit_field
        return check_mesocircuit_field(project_path(project, args.source_dir), output, config, command)
    if args.operation == "window-check":
        from .spike_windows import check_recording_windows
        return check_recording_windows(project_path(project, args.recording_dir), output, config, command)
    if args.operation == "baseline-check":
        if args.model == "braid":
            from .braid_check import check_braid
            return check_braid(project_path(project, args.recording_dir), output, config, command, args.seed)
        if args.model == "lfads":
            from .lfads_check import check_lfads
            return check_lfads(project_path(project, args.recording_dir), output, config, command, args.seed)
        if args.model == "poisson-state-space":
            from .poisson_state_space_check import check_poisson_state_space
            return check_poisson_state_space(project_path(project, args.recording_dir), output, config, command, args.seed)
        if args.model == "reduced-rank":
            from .reduced_rank import check_reduced_rank
            return check_reduced_rank(project_path(project, args.recording_dir), output, config, command, args.seed)
        from .count_autoregression import check_count_baseline
        return check_count_baseline(project_path(project, args.recording_dir), output, config, command, args.seed)
    if args.operation == "cohort":
        rows = read_metadata(project_path(project, args.metadata_json)) if args.metadata_json else fetch_metadata(config["network"])
        result = freeze_cohort(output / "cohort.json", rows, config)
        source_path = project_path(project, args.metadata_json).resolve() if args.metadata_json else None
        source_digest = sha256_file(source_path) if source_path else hashlib.sha256(canonical_json(rows).encode()).hexdigest()
        cohort_digest = sha256_file(output / "cohort.json")
        manifest_path = output / "cohort_generation_manifest.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior["cohort_sha256"] != cohort_digest or prior["source_metadata_sha256"] != source_digest:
                raise ValueError("COHORT_SOURCE_OR_FROZEN_OUTPUT_CHANGED")
        else:
            manifest = {"schema": "neural_population_cohort_generation_v1", "status": "METADATA_COHORT_FROZEN",
                        "source_metadata": str(source_path) if source_path else "Allen RMA complete session query",
                        "source_metadata_sha256": source_digest, "cohort_sha256": cohort_digest,
                        "metadata_session_count": len(rows), "included_session_count": result["included_session_count"],
                        "included_split_counts": {split: len({row["animal_id"] for row in result["sessions"]
                            if row["included"] and row["split"] == split}) for split in ("train", "validation", "test")},
                        "dataset_ready": False, "biological_activity_used_for_selection": False,
                        **runtime_manifest(config, command, 0)}
            freeze_json(manifest_path, manifest)
        return {"status": "METADATA_COHORT_FROZEN", "cohort": str(output / "cohort.json"),
                "manifest": str(manifest_path), "included_session_count": result["included_session_count"], "dataset_ready": False}, 0
    cohort_path = project_path(project, args.cohort or Path(config["output_dir"]) / "cohort.json")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("training_seeds") != config["training_seeds"]:
        raise ValueError("COHORT_TRAINING_SEED_AUTHORITY_MISMATCH")
    row = cohort_session(cohort, args.session_id)
    diagnostic_ids = getattr(args, "diagnostic_unit_ids", None)
    if diagnostic_ids is not None:
        from .qualification import diagnostic_selection
        diagnostic_selection(diagnostic_ids, row, config)
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", args.run_id) is None:
        raise ValueError("SEMANTIC_RUN_ID_REQUIRED")
    result_dir = output / f"session_{args.session_id}" / args.run_id
    source = resolve_source(row, config["network"])
    if (result_dir / "manifest.json").exists():
        raise FileExistsError(f"QUALIFICATION_ALREADY_RECORDED_REUSE_SELECTED_ARTIFACT:{result_dir}")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    def progress(message):
        print(json.dumps({"timestamp_utc":datetime.now(timezone.utc).isoformat(),
                          "elapsed_seconds":round(time.monotonic()-started,3), **message}),file=sys.stderr,flush=True)
    if args.operation == "qualify" and args.source_url:
        known = config["cohort"].get("known_sources", {}).get(args.session_id, {})
        if args.source_url != known.get("verified_mirror_url") or not known.get("expected_etag"):
            raise ValueError("SOURCE_URL_NOT_BOUND_TO_VERIFIED_SESSION_MIRROR")
        from .qualification import qualify_session, read_nwb_handle
        progress({"event":"OPEN_REMOTE_SOURCE", "session_id":args.session_id})
        with open_remote_nwb(args.source_url, source["expected_size_bytes"], known["expected_etag"], config["network"], progress=progress) as (handle, identity, etag_reader):
            progress({"event":"REMOTE_NWB_OPENED", **identity})
            data = read_nwb_handle(handle, args.session_id, source_etag=identity["etag"],
                etag_reader=etag_reader, retain_spikes=True, progress=progress)
            qualification = qualify_session(data, row, config, progress=progress, diagnostic_unit_ids=diagnostic_ids)
            source_record = {**source, **identity, "session_id":args.session_id,
                             "cohort_sha256":sha256_file(cohort_path)}
    else:
        if args.local_nwb:
            nwb_path = project_path(project, args.local_nwb).resolve()
            checked = verify_local_nwb(nwb_path, source["expected_size_bytes"])
        else:
            if args.cache_dir is None:
                raise ValueError("NODE_LOCAL_CACHE_DIR_REQUIRED")
            cache = args.cache_dir.resolve()
            if not any(cache.is_relative_to(prefix) for prefix in (Path("/tmp"), Path("/var/tmp"), Path("/scratch"))):
                raise ValueError("CACHE_MUST_BE_NODE_LOCAL_SCRATCH")
            nwb_path, checked = download_nwb(source, cache, config["network"])
        source_record = {**source, **checked, "local_path":str(nwb_path),
                         "session_id":args.session_id,"cohort_sha256":sha256_file(cohort_path),
                         "source_full_asset_downloaded":True}
        if args.operation == "intake":
            result = {"schema":"neural_population_nwb_intake_v1","status":"SOURCE_BYTES_COMPLETE",
                      "source":source_record,"dataset_ready":False,"model_ready":False,
                      **runtime_manifest(config,command,0)}
            freeze_json(result_dir / "intake_manifest.json",result)
            return {"status":result["status"],"local_nwb":str(nwb_path),
                    "manifest":str(result_dir / "intake_manifest.json"),"dataset_ready":False},0
        from .qualification import qualify_session, read_local_nwb
        with read_local_nwb(nwb_path,args.session_id,source_full_asset_downloaded=True,retain_spikes=True,progress=progress) as data:
            qualification = qualify_session(data,row,config,progress=progress,diagnostic_unit_ids=diagnostic_ids)
    from .qualification import write_selected_recording
    progress({"event":"SOURCE_COLUMN_VALIDATION_COMPLETE", "status":qualification["summary"]["status"]})
    selected = write_selected_recording(result_dir / "selected_recording.h5",data,qualification,source_record)
    selected["path"] = "selected_recording.h5"
    selected["sha256"] = sha256_file(result_dir / "selected_recording.h5")
    progress({"event":"SELECTED_RECORDING_WRITTEN_AND_REPLAYED", **selected})
    result_dir.mkdir(parents=True, exist_ok=True)
    for name in ("units", "trials", "trial_members", "condition_counts"):
        _write_csv(result_dir / f"{name}.csv", qualification[name])
    presentations = data.presentations.reset_index().rename(columns={data.presentations.index.name or "index": "presentation_id"})
    _write_csv(result_dir / "stimulus_presentations.csv", presentations)
    import pandas as pd
    from .qualification import condition_parameters
    first_members = qualification["trial_members"].drop_duplicates("trial_id").set_index("trial_id")
    condition_rows = []
    for _, trial in qualification["trials"].drop_duplicates("condition_id").iterrows():
        presentation_id = first_members.loc[trial["trial_id"],"presentation_id"]
        parameters = condition_parameters(data.presentations.loc[presentation_id],trial["trial_kind"] == "movie_repeat")
        condition_rows.append({"condition_id":trial["condition_id"],"stimulus_name":trial["stimulus_name"],
            "parameters_json":json.dumps(parameters,sort_keys=True,allow_nan=False)})
    conditions = pd.DataFrame(condition_rows)
    _write_csv(result_dir / "stimulus_conditions.csv", conditions)
    _write_csv(result_dir / "invalid_times.csv", data.invalid_times.reset_index())
    _write_csv(result_dir / "optotagging.csv", data.optotagging.reset_index())
    if data.electrodes is not None:
        _write_csv(result_dir / "electrodes.csv",data.electrodes)
    for name, frame in (data.covariates or {}).items():
        _write_csv(result_dir / f"{name}.csv", frame)
    summary = qualification["summary"]
    code = 2 if summary["status"] == "PARTIAL" else 0
    freeze_json(result_dir / "summary.json", summary)
    manifest = {"schema": "neural_population_intake_manifest_v1", "source": source_record,
                "qualification_scope": summary["qualification_scope"], "diagnostic_unit_ids": summary["diagnostic_unit_ids"], "status": summary["status"],
                "required_column_evidence": qualification["column_evidence"],
                "consumed_spike_slice_evidence": qualification["spike_slice_evidence"],
                "artifacts": {path.name: sha256_file(path) for path in sorted(result_dir.glob("*.csv"))},
                "summary_sha256": sha256_file(result_dir / "summary.json"),
                "selected_recording": selected, "started_at_utc":started_at,
                "ended_at_utc":datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds":time.monotonic()-started,
                **runtime_manifest(config, command, code)}
    freeze_json(result_dir / "manifest.json", manifest)
    return {**summary, "manifest": str(result_dir / "manifest.json")}, code


def main(argv: list[str] | None = None) -> int:
    command = [sys.executable, "-m", "Libs.population_dynamics", *(sys.argv[1:] if argv is None else argv)]
    args = parser().parse_args(argv)
    try:
        result, exit_code = run(args, command)
    except Exception as exc:
        print(canonical_json({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
                              "operation": args.operation, "dataset_ready": False}), file=sys.stderr, end="")
        return 1
    print(canonical_json(result), end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
