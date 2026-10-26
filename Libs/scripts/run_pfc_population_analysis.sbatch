#!/bin/bash
#SBATCH --job-name=NrnSim_PfcPopulationAnalysis
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-pfc-analysis-%j.log
set -euo pipefail
# Preserve the original five arguments; optional cells follow the output.
if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
    printf 'Usage: %s SNAPSHOT CAMPAIGN PREPARATION SEED41_CELL OUTPUT [SEED42_CELL] [SEED43_CELL]\n' "$0" >&2
    exit 2
fi
snapshot="$1"
campaign="$2"
prepared="$3"
seed41="$4"
output="$5"
seed42="${6:-}"
seed43="${7:-}"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun /opt/conda/bin/python -B - "$snapshot" "$campaign" "$prepared" "$seed41" "$output" "$seed42" "$seed43" <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
snapshot,campaign,prepared,seed41,output=map(Path,sys.argv[1:6])
seed42,seed43=sys.argv[6:8]
placement=(os.environ.get("SLURMD_NODENAME"),os.environ.get("SLURM_JOB_PARTITION"))
if (placement not in {("cpu03","gpu"),("cpu04","cpu")}
        or any(os.environ.get(name) for name in ("SLURM_JOB_GPUS","SLURM_STEP_GPUS"))):
    raise ValueError("PFC_ANALYSIS_CPU_ONLY_PLACEMENT_REQUIRED")
if output.exists():
    raise FileExistsError("Existing PFC analysis must be consumed before repeating")
proof=json.loads((snapshot/"delivery.json").read_text())
if proof["status"]!="FROZEN_TESTED_PFC_THREE_SEED_ANALYSIS" or proof["tests_passed"]!=15:
    raise ValueError("PFC_ANALYSIS_TESTED_SNAPSHOT_REQUIRED")
with tempfile.TemporaryDirectory(prefix="neuronsim-pfc-analysis-",dir="/tmp/neuronsim") as temporary:
    work=Path(temporary)
    for row in proof["files"]:
        relative=Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("PFC_ANALYSIS_SAFE_SNAPSHOT_PATH_REQUIRED")
        source=snapshot/relative;target=work/relative
        if source.stat().st_size!=row["bytes"] or hashlib.sha256(source.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("PFC_ANALYSIS_FROZEN_SOURCE_CHANGED")
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row["sha256"]:
            raise ValueError("PFC_ANALYSIS_STAGED_SOURCE_CHANGED")
    environment=dict(os.environ,PYTHONPATH=str(work/"Libs"),TMPDIR=str(work))
    command=[sys.executable,"-m","population_dynamics.pfc_population_forecast_analysis",
        "--campaign",str(campaign),"--preparation",str(prepared),"--seed41-cell",str(seed41),"--output",str(output)]
    for seed,cell in ((42,seed42),(43,seed43)):
        if cell:
            command.extend([f"--seed{seed}-cell",str(Path(cell))])
    process=subprocess.run(command,cwd=work,env=environment)
    if process.returncode:
        raise SystemExit(process.returncode)
    report=json.loads((output/"report.json").read_text())
    if report["status"]!="COMPLETE_DESCRIPTIVE_PFC_POPULATION_FORECAST_ANALYSIS" or report["score_rows"]!=1080 or report["origin_cost_rows"]!=360:
        raise ValueError("PFC_ANALYSIS_COMPLETE_RESULT_REQUIRED")
    delivery=dict(status="PASS",job_id=os.environ["SLURM_JOB_ID"],source_snapshot=str(snapshot),
        source_snapshot_sha256=hashlib.sha256((snapshot/"delivery.json").read_bytes()).hexdigest(),
        files=[],scientific_claim_qualified=False,recorded_utc=datetime.now(timezone.utc).isoformat())
    for name in ["report.json",*report["artifacts"]]:
        path=output/name
        delivery["files"].append(dict(path=name,bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (output/"delivery.json").write_text(json.dumps(delivery,indent=2)+"\n")
PY
