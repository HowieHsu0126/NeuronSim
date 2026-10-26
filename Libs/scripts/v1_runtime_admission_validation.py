#!/usr/bin/env python3
"""Test V1 runtime admission and qualify the already completed native family."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import traceback
import xml.etree.ElementTree as ET

import ca3_merlin_remote as transport


def restore_family(receipt: dict, manifest_path: Path, work: Path) -> Path:
    """Restore the published family; the artifact builder verifies each file."""
    bundle = work / "native_family.tar.gz"
    subprocess.run(["hdfs", "dfs", "-get", receipt["checkpoint_path"] + "/result.tar.gz",
                    str(bundle)], check=True)
    if (bundle.stat().st_size != receipt["result_bytes"]
            or transport.digest(bundle) != receipt["result_sha256"]):
        raise ValueError("FROZEN_FAMILY_BUNDLE_DIFFERS")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    root = Path(receipt["output"])
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(manifest) | {"merlin_return_manifest.json"}:
            raise ValueError("FROZEN_FAMILY_ARCHIVE_MEMBERS_DIFFER")
        if archive.extractfile("merlin_return_manifest.json").read() != manifest_bytes:
            raise ValueError("FROZEN_FAMILY_MANIFEST_DIFFERS")
        for member in members:
            if member.name == "merlin_return_manifest.json":
                continue
            target = Path("/") / member.name
            if (not member.isfile() or Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts or root not in target.parents
                    or target.resolve() != target or member.size != manifest[member.name]["bytes"]):
                raise ValueError("FROZEN_FAMILY_MEMBER_INVALID")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job_json.read_text())
    run, trial = os.environ["RH2_JOB_RUN_ID"], os.environ["ARNOLD_TRIAL_ID"]
    work = args.job_json.parent
    output = work / "results"
    output.mkdir(exist_ok=False)
    report = dict(status="RUNNING", run_sid=run, trial_sid=trial, hostname=socket.gethostname(),
                  execution_backend="merlin", started_utc=datetime.now(timezone.utc).isoformat(),
                  source_simulations=0, new_benchmark_episodes=0, production_qualified=False,
                  input_files=job["input_files"], input_archive_sha256=job["upload"]["sha256"],
                  launcher_sha256=transport.digest(Path(__file__)),
                  transport_helper_sha256=transport.digest(Path(transport.__file__)),
                  commands=[])
    code = 1
    try:
        if int(os.environ.get("ARNOLD_APPLIED_GPU_NUM", "0")) != 0 or job["resource_shape"]["gpu"] != 0:
            raise ValueError("RUNTIME_ADMISSION_VALIDATION_REQUIRES_CPU_ONLY")
        report["memory_evidence"] = transport.memory_evidence(job["resource_shape"], 8)
        archive = transport.download_inputs(job, work)
        transport.stage_inputs(archive, job["input_files"])
        receipt_path = Path(job["family_receipt"])
        receipt = json.loads(receipt_path.read_text())
        if (receipt["status"] != "PASS_FROZEN_C0_B2_NATIVE_BATCH_AND_INDEPENDENT_REPLAY"
                or receipt["exit_code"] != 0 or receipt["persisted"] is not True):
            raise ValueError("PASSED_PERSISTED_FROZEN_FAMILY_REQUIRED")
        manifest = Path(job["family_manifest"])
        bundle = (None if job.get("runtime_qualification_source")
                  else restore_family(receipt, manifest, work))
        repo = Path(job["repo"])
        environment = dict(os.environ, PYTHONPATH=str(repo / "Libs"),
                           PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
                           NEURONSIM_VALIDATE_MERLIN_CPU="1")
        command = [sys.executable, "-B", "-m", "pytest", "-q", *job["tests"],
                   "--junitxml", str(output / "tests.xml"), "-p", "no:cacheprovider"]
        with (output / "tests.log").open("x") as log:
            tests = subprocess.run(command, cwd=repo, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT)
        report["commands"].append(dict(stage="tests", command=command, exit_code=tests.returncode))
        if (output / "tests.xml").exists():
            xml = ET.parse(output / "tests.xml").getroot()
            suites = [xml] if xml.tag == "testsuite" else list(xml.iter("testsuite"))
            report["tests"] = {key: sum(int(suite.get(key, "0")) for suite in suites)
                               for key in ("tests", "failures", "errors", "skipped")}
            cases = list(xml.iter("testcase"))
            report["skipped_tests"] = [dict(name=case.get("name"), reason=case.find("skipped").get("message"))
                                       for case in cases if case.find("skipped") is not None]
            actual = [case for case in cases if case.get("name") ==
                      "test_prepared_source_block_uses_actual_merlin_cpu_allocation"]
            report["actual_merlin_cpu_test_passed"] = (len(actual) == 1 and
                all(actual[0].find(tag) is None for tag in ("failure", "error", "skipped")))
        if job.get("runtime_qualification_source"):
            # Reuse the unchanged real artifact whose prior builder completed.
            program = ("import json,sys;from pathlib import Path;"
                       "from population_dynamics import v1_runtime_qualification as r;"
                       "p,q,o=map(Path,sys.argv[1:]);"
                       "document,backend=r.load(p,r.digest_bytes(q.read_bytes()),"
                       "json.loads(q.read_text())['source_recipes']['glif_cuda']);"
                       "f=o.open('xb');f.write(p.read_bytes());f.close();"
                       "print(json.dumps(dict(status='PASS_REUSED_VERIFIED_RUNTIME_ARTIFACT',"
                       "sha256=document['sha256'])))")
            command = [sys.executable, "-B", "-c", program, job["runtime_qualification_source"],
                       job["original_qualification"], str(output / "runtime_qualification.json")]
            report["runtime_artifact_reused"] = True
        else:
            command = [sys.executable, "-B", "-m", "population_dynamics.v1_runtime_qualification",
                       "--original-qualification", job["original_qualification"],
                       "--result-root", receipt["output"], "--receipt", str(receipt_path),
                       "--bundle", str(bundle), "--return-manifest", str(manifest),
                       "--output", str(output / "runtime_qualification.json")]
        with (output / "artifact_builder.log").open("x") as log:
            artifact = subprocess.run(command, cwd=repo, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT)
        report["commands"].append(dict(stage="artifact_builder", command=command,
                                       exit_code=artifact.returncode))
        if artifact.returncode == 0:
            report["runtime_qualification_sha256"] = transport.digest(output / "runtime_qualification.json")
        report["package_versions"] = {name: importlib.metadata.version(name) for name in
                                      ("numpy", "pandas", "scipy", "h5py", "PyYAML", "pytest", "torch")}
        counts = report.get("tests", {})
        if (tests.returncode == artifact.returncode == 0 and counts.get("tests", 0) > 0
                and all(counts.get(key) == 0 for key in ("failures", "errors"))
                and report.get("actual_merlin_cpu_test_passed") is True):
            code = 0
            report["status"] = "PASS_V1_RUNTIME_ADMISSION_TESTS_AND_REAL_ARTIFACT"
        else:
            report["status"] = "FAILED_V1_RUNTIME_ADMISSION_VALIDATION"
    except BaseException as error:
        report.update(status="FAILED_V1_RUNTIME_ADMISSION_VALIDATION", error=str(error),
                      traceback=traceback.format_exc())
    finally:
        report.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
        transport.write_json(output / "report.json", report)
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ("input_files", "memory_evidence")}), flush=True)
        target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/v1_runtime_admission"
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        transport.hdfs_put(output / "report.json", target + "/report.json")
        for name in ("tests.log", "tests.xml", "artifact_builder.log"):
            if (output / name).is_file():
                transport.hdfs_put(output / name, target + "/" + name)
        transport.persist(output, work, target, report, chunk_bytes=256 * 1024)
        print(json.dumps(report), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
