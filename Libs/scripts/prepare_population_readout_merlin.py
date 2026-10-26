"""Prepare the tested readout diagnostic request after all inputs reach HDFS.

This command prepares a request only. Use the installed Merlin CLI schema and
precheck before submission. Reconcile a create error by exact run name before
retrying: an EOF can occur after the service has accepted a job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root
    if not args.name.startswith("mrln_") or len(args.name) > 90:
        raise ValueError("Merlin run name requires the mrln_ prefix and at most90characters")
    for directory, expected in (("merlin_readout_checks", 6), ("merlin_readout_runtime_checks", 7)):
        report = json.loads((root / directory / "result.json").read_text())
        if (report["status"] != "PASS" or report["exit_code"] != 0
                or report["tests_run"] != expected or any(report[key] for key in ("failures", "errors", "skipped"))):
            raise ValueError("accepted math and runtime checks are required")
    source = json.loads((root / "merlin_readout_runtime_checks/source_upload.json").read_text())
    inputs = json.loads((root / "merlin_readout_inputs/upload.json").read_text())
    wrapper = json.loads((root / "merlin_readout_analysis/wrapper_upload.json").read_text())
    if not all(item.get("complete") is True for item in (source, inputs, wrapper)):
        raise ValueError("complete verified HDFS inputs are required before preparing allocation")
    if sum(row["bytes"] for row in inputs["chunks"]) != inputs["bytes"]:
        raise ValueError("input chunks do not cover the complete archive")
    prepare = (
        "import hashlib,json,pathlib\n"
        "root=pathlib.Path('/tmp/neuronsim-readout-launch')\n"
        "data=(root/'run.py').read_bytes()\n"
        f"assert len(data)=={wrapper['bytes']!r} and hashlib.sha256(data).hexdigest()=={wrapper['sha256']!r}\n"
        f"(root/'source_upload.json').write_text({json.dumps(source)!r})\n"
        f"(root/'input_upload.json').write_text({json.dumps(inputs)!r})\n"
    )
    entry = (
        "#!/bin/bash\nset -euo pipefail\n"
        "export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1\n"
        "export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8\n"
        "python3 -m pip install --no-deps scipy==1.16.2 PyYAML==6.0.3\n"
        "mkdir /tmp/neuronsim-readout-launch\n"
        f"hdfs dfs -get {shlex.quote(wrapper['hdfs_path'])} /tmp/neuronsim-readout-launch/run.py\n"
        "python3 - <<'NRNSIM_PREPARE'\n" + prepare + "NRNSIM_PREPARE\n"
        "exec python3 /tmp/neuronsim-readout-launch/run.py "
        "--source-upload /tmp/neuronsim-readout-launch/source_upload.json "
        "--input-upload /tmp/neuronsim-readout-launch/input_upload.json "
        "--work /tmp/neuronsim-native-readout --cpu-threads 8\n"
    )
    request = json.loads((root / "merlin_readout_runtime_checks/request.json").read_text())
    request.pop("request_context", None)
    request.update(name=args.name, description="Posthoc readout decomposition of existing native trajectories; no training, sampling or source simulation.")
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entry
    role = request["resource_config"]["arnold_resource_config"]["roles"][0]
    role.update(cpu=8, memory_mb=16384, gpu=0, gpu_type="CPU_ONLY")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(request, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status="REQUEST_PREPARED", request=str(args.output), name=args.name, cpu=8, gpu=0)))


if __name__ == "__main__":
    main()
