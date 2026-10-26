#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_MultiSpikePrep
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-multispike-%j.out
#SBATCH --error=/tmp/neuronsim-multispike-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_MULTISPIKE_BATCH="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_MULTISPIKE'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import yaml

repo=Path(os.environ["NEURONSIM_PROJECT_ROOT"]).resolve()
mode=os.environ.get("NEURONSIM_MULTISPIKE_STAGE","prepare")
assert mode in ("prepare","execute")
assert socket.gethostname().split(".")[0]==("cpu04" if mode=="prepare" else "cpu03")
assert not os.environ.get("SLURM_JOB_GPUS")
assert (repo/"PROJECT_PREFIX").read_text().strip()=="NrnSim"
job=os.environ["SLURM_JOB_ID"]
family=Path("Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/multispike_compensator")
archive=repo/family/mode/("run_"+job)
assert not archive.exists()
work=Path(tempfile.mkdtemp(prefix="neuronsim-multispike-"+job+".",dir="/tmp"))
(work/"scratch").mkdir()
env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(work),str(work/"Libs"))),TMPDIR=str(work/"scratch"))
config=Path("Libs/configs/neural_population_pfc_multispike_compensator.yaml")
module=Path("Libs/population_dynamics/pfc_multispike_compensator.py")
test=Path("Libs/population_dynamics/test_pfc_multispike_compensator.py")
inputs,commands=[],[]
started=time.perf_counter();status="FAILED_PFC_MULTISPIKE_"+mode.upper();exit_code=1;error=None;delivered=False

def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def copy(source,relative,expected=None):
 target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 actual=digest(target)
 if expected is not None and actual!=expected:raise RuntimeError("MULTISPIKE_STAGED_INPUT_CHANGED:"+str(relative))
 inputs.append(dict(source=str(source),path=str(relative),sha256=actual))

def manifest(archive,expected,name):
 copy(archive/"delivery.json",Path(name),expected)
 d=json.loads((work/name).read_text())
 assert d.get("verified") is True
 return {r["path"]:r for r in d["files"]}

def phase(name,command):
 begin=time.perf_counter()
 with (work/(name+".stdout")).open("w") as out,(work/(name+".stderr")).open("w") as err:
  result=subprocess.run(command,cwd=work,env=env,stdout=out,stderr=err)
 commands.append(dict(phase=name,command=command,exit_code=result.returncode,wall_seconds=time.perf_counter()-begin))
 print(json.dumps(commands[-1]),flush=True)
 if result.returncode:raise RuntimeError("MULTISPIKE_"+name.upper()+"_FAILED:"+str(result.returncode))

try:
 if mode=="prepare":
  snapshot=repo/"Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/conditional_spike_kernel/run_602381"
  entries=manifest(snapshot,"b2aeacffdd7d39cc12662556fbec63717eb0112f8e852a7ebff4fbfb951ff0a1","dependency_delivery.json")
  for name,entry in entries.items():
   relative=Path(name)
   if relative.parts[0]=="Libs" and relative.suffix==".py" and not relative.name.startswith("test_"):
    copy(snapshot/relative,relative,entry["sha256"])
  for relative in (module,test,config):copy(repo/relative,relative)
  settings=yaml.safe_load((work/config).read_text())
 else:
  prepared=Path(os.environ["NEURONSIM_MULTISPIKE_PREPARED_ARCHIVE"])
  entries=manifest(prepared,os.environ["NEURONSIM_MULTISPIKE_PREPARED_DELIVERY_SHA256"],"preparation_delivery.json")
  for name,entry in entries.items():
   relative=Path(name)
   if relative.parts[0]=="Libs" or name in ("prepared/plan.json","report.json"):
    target=Path("preparation_report.json") if name=="report.json" else relative
    copy(prepared/relative,target,entry["sha256"])
  receipt=json.loads((work/"preparation_report.json").read_text())
  assert receipt["exit_code"]==0 and any(c["phase"]=="tests" and c["exit_code"]==0 for c in receipt["commands"])
  settings=json.loads((work/"prepared/plan.json").read_text())["settings"]
  commands.append(dict(phase="tests_reused",archive=str(prepared),exit_code=0))
 copy(Path(os.environ["NEURONSIM_MULTISPIKE_BATCH"]),Path("launcher.sbatch"))
 sys.path.insert(0,str(work))
 from Libs.population_dynamics.resource_probe import probe
 health=probe(work/"resource_preflight.json",4)
 if health["status"]!="PASS":raise RuntimeError("RESOURCE_BLOCKED:"+str(health["reasons"]))
 source=Path(settings["source"]["archive"])
 copy(repo/source/"delivery.json",source/"delivery.json",settings["source"]["delivery_sha256"])
 source_delivery=json.loads((work/source/"delivery.json").read_text())
 assert source_delivery["source_execution_complete"] and source_delivery["exit_code"]==0
 source_entries={r["path"]:r for r in source_delivery["files"]}
 names=["report.json","config.yaml","initial_voltage.npy"]
 if mode=="execute":names += ["raw.npz","terminal_state.npz","final_rng.npz"]
 for name in names:
  relative=Path("source_run")/name
  copy(repo/source/relative,source/relative,source_entries[str(relative)]["sha256"])
 if mode=="prepare":
  for name in ("pfc_deterministic_replay.py","pfc_native_drive.py"):
   relative=Path("Libs/population_dynamics")/name
   copy(repo/source/relative,relative,source_entries[str(relative)]["sha256"])
  phase("tests",[sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider",str(test),
    "--basetemp="+str(work/"test_tmp"),"--junitxml="+str(work/"tests.xml")])
  phase("prepare",[sys.executable,"-B","-m","Libs.population_dynamics.pfc_multispike_compensator","prepare",
    "--project-root",str(work),"--config",str(work/config),"--output",str(work/"prepared")])
 else:
  phase("execute",[sys.executable,"-B","-m","Libs.population_dynamics.pfc_multispike_compensator","execute",
    "--project-root",str(work),"--prepared",str(work/"prepared"),"--output",str(work/"analysis")])
 status="COMPLETE_PFC_MULTISPIKE_"+mode.upper();exit_code=0
except Exception as exc:
 error=dict(type=type(exc).__name__,message=str(exc));traceback.print_exc()
finally:
 (work/"report.json").write_text(json.dumps(dict(status=status,exit_code=exit_code,error=error,job_id=job,
  mode=mode,node=socket.gethostname(),inputs=inputs,commands=commands,
  source_replays=int(mode=="execute" and any(c["phase"]=="execute" for c in commands)),
  new_full_episodes=0,new_independent_blocks=0,confirmation_execution_released=False,production_episodes=0,
  wall_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")
 try:
  archive.mkdir(parents=True,exist_ok=False);files=[]
  for source in sorted(work.rglob("*")):
   if not source.is_file():continue
   relative=source.relative_to(work)
   if relative.parts[0] in ("Output","scratch","test_tmp") or "__pycache__" in relative.parts:continue
   target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
   checksum=digest(source)
   if digest(target)!=checksum:raise RuntimeError("MULTISPIKE_COPYBACK_CHANGED:"+str(relative))
   files.append(dict(path=str(relative),bytes=source.stat().st_size,sha256=checksum))
  (archive/"delivery.json").write_text(json.dumps(dict(status=status,verified=True,job_id=job,files=files),indent=2)+"\n")
  delivered=True
  print(json.dumps(dict(status=status,archive=str(archive),exit_code=exit_code)),flush=True)
 finally:
  if delivered:shutil.rmtree(work)
  else:print(json.dumps(dict(status="COPYBACK_FAILED_SCRATCH_PRESERVED",work=str(work))),file=sys.stderr,flush=True)
raise SystemExit(exit_code)
PY_MULTISPIKE
