"""Run one full-parameter development source and its frozen observation.

The source and observation use their existing software snapshots. A failed
attempt is retained. An explicit completed source can feed a later readout
attempt without another simulation. This runner grants no scientific PASS.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .pfc_parameter_domain_quality import load_task, qualified_mapping
from .provenance import freeze_json, sha256_file


def run_task(*, source_project: Path, observation_project: Path, prepared: Path,
             qualifications: Path, index: int, source_output: Path,
             observation_output: Path, campaign: Path, output: Path,
             source_archive: Path | None = None) -> dict:
    """Execute one recipe; reject duplicate attempts and retain nonzero exits."""
    qualified_mapping(qualifications)
    _, row, _ = load_task(prepared, index)
    job = os.environ.get("SLURM_JOB_ID")
    if (not job or os.environ.get("SLURMD_NODENAME") not in
            {"cpu03", "cpu04", "cpu05", "cpu06"}
            or os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
        raise ValueError("PFC_DOMAIN_EXECUTION_ALLOCATED_CPU_REQUIRED")
    if output.exists():
        raise FileExistsError("PFC_DOMAIN_EXECUTION_ATTEMPT_EXISTS:" + str(output))
    if source_archive is None and (source_output.exists() or observation_output.exists()):
        raise FileExistsError("PFC_DOMAIN_EXECUTION_EXISTING_SOURCE_OR_OBSERVATION:" + str(source_output))
    archive = (source_archive if source_archive is not None
               else source_output / ("run_" + job)).resolve()
    if source_archive is not None and not (archive / "delivery.json").is_file():
        raise ValueError("PFC_DOMAIN_EXECUTION_COMPLETE_SOURCE_ARCHIVE_REQUIRED")
    output.mkdir(parents=True)
    request = dict(schema="pfc_parameter_domain_execution_v1", task_index=index,
                   identity=row["identity"], config_sha256=row["config_sha256"],
                   job_id=job, source_archive=str(archive),
                   source_reused=source_archive is not None,
                   prepared=str(prepared.resolve()),
                   qualifications=str(qualifications.resolve()),
                   source_project=str(source_project.resolve()),
                   observation_project=str(observation_project.resolve()),
                   observation_output=str(observation_output.resolve()),
                   campaign=str(campaign.resolve()), campaign_sha256=sha256_file(campaign),
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   scope="development", production_episodes=0,
                   source_qualification_granted=False, scientific_claim_qualified=False)
    freeze_json(output / "request.json", request)
    report = dict(request, status="RUNNING", source_exit_code=None,
                  observation_exit_code=None, new_source_calls=0, completed_episodes=0,
                  source_wrapper_wall_seconds=None, observation_wall_seconds=None)
    try:
        if source_archive is None:
            command = [sys.executable, "-B", "-m", "population_dynamics.pfc_parameter_domain_quality",
                       "source", "--project", str(source_project.resolve()),
                       "--prepared", str(prepared.resolve()), "--index", str(index),
                       "--qualifications", str(qualifications.resolve()),
                       "--output", str(source_output.resolve())]
            environment = dict(os.environ, PYTHONPATH=str(source_project.resolve() / "Libs"),
                               NEURONSIM_PFC_PYTHON=sys.executable)
            report.update(source_command=command, new_source_calls=1)
            start = time.perf_counter()
            code = subprocess.run(command, cwd="/tmp", env=environment).returncode
            report.update(source_exit_code=code, source_wrapper_wall_seconds=time.perf_counter()-start)
            if code:
                raise RuntimeError("PFC_DOMAIN_SOURCE_EXIT:" + str(code))
        command = [sys.executable, "-B", "-m", "population_dynamics.pfc_parameter_domain_quality",
                   "readout", "--prepared", str(prepared.resolve()), "--index", str(index),
                   "--source", str(archive), "--output", str(observation_output.resolve()),
                   "--observation-bin-ms", "1", "--campaign", str(campaign.resolve())]
        report["observation_command"] = command
        start = time.perf_counter()
        code = subprocess.run(command, cwd="/tmp", env=dict(os.environ,
                              PYTHONPATH=str(observation_project.resolve() / "Libs"))).returncode
        report.update(observation_exit_code=code, observation_wall_seconds=time.perf_counter()-start)
        if code:
            raise RuntimeError("PFC_DOMAIN_OBSERVATION_EXIT:" + str(code))
        report.update(status="COMPLETE_DEVELOPMENT_SOURCE_AND_OBSERVATION", completed_episodes=1)
        return report
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-project", "observation-project", "prepared", "qualifications",
                 "source-output", "observation-output", "campaign", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--source-archive", type=Path)
    arguments = vars(parser.parse_args())
    print(json.dumps(run_task(**arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
