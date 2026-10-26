#!/bin/bash
#SBATCH --job-name=NrnSim_MatchedClockPanel
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_MatchedClockPanel-%j.log
set -euo pipefail
[[ $# == 4 ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NRNSIM_BENCHMARK_PYTHON" -B - "$@" <<'PYTHON'
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,traceback
software,base,extension,destination=map(Path,sys.argv[1:])
if destination.exists():
    raise FileExistsError("MATCHED_PANEL_DESTINATION_EXISTS")
if os.environ.get("SLURMD_NODENAME") not in ("cpu03","cpu04") or os.environ.get("SLURM_JOB_GPUS"):
    raise ValueError("MATCHED_PANEL_CPU_ALLOCATION_REQUIRED")
receipt=dict(status="FAILED_MATCHED_CLOCK_PANEL",exit_code=1,
             job_id=int(os.environ["SLURM_JOB_ID"]),software_source=str(software),
             base_panel=str(base),extension=str(extension),new_source_simulations=0,
             scientific_qualification=False,production_episodes=0)
with tempfile.TemporaryDirectory(prefix="neuronsim-matched-clock-panel-") as temporary:
    work=Path(temporary);package=work/"Libs/population_dynamics"
    package.parent.mkdir()
    try:
        shutil.copytree(software,package)
        command=[sys.executable,"-B","-m","population_dynamics.v1_matched_clock_panel",
                 "--base-panel",str(base),"--extension",str(extension),"--output",str(work/"panel")]
        receipt["command"]=command
        env=dict(os.environ,PYTHONPATH=str(work/"Libs"),TMPDIR=str(work))
        with (work/"run.log").open("w") as log:
            result=subprocess.run(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt["command_exit_code"]=result.returncode
        if result.returncode:
            raise RuntimeError("MATCHED_PANEL_COMMAND_FAILED:"+str(result.returncode))
        path=work/"panel/panel_acceptance.json"
        acceptance=json.loads(path.read_text())
        if acceptance["status"]!="PASS_COMPLETE_CLOCK_PANEL_DELIVERY":
            raise ValueError("MATCHED_PANEL_ACCEPTANCE_REQUIRED")
        receipt.update(status="PASS_MATCHED_CLOCK_PANEL_DELIVERY",exit_code=0,
            verified_pairs=acceptance["verified_pairs"],
            panel_acceptance_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error))
        (work/"error.log").write_text(traceback.format_exc())
    finally:
        (work/"delivery.json").write_text(json.dumps(receipt,indent=2)+"\n")
        shutil.copytree(work,destination,symlinks=True)
print(json.dumps({key:receipt[key] for key in ("status","exit_code","job_id")}))
raise SystemExit(receipt["exit_code"])
PYTHON
