#!/bin/bash
#SBATCH --job-name=NrnSim_V1ClockComparison
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_V1ClockComparison-%j.log
set -euo pipefail
[[ $# == 4 ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${NRNSIM_BENCHMARK_PYTHON:?Set benchmark Python}" -B - "$@" <<'PYTHON'
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile,traceback
repo,reference,fine,destination=map(Path,sys.argv[1:])
if destination.exists():raise ValueError('CLOCK_COMPARISON_DESTINATION_EXISTS')
if os.environ['SLURMD_NODENAME'] not in ('cpu03','cpu04') or os.environ.get('SLURM_JOB_GPUS'):
 raise ValueError('CLOCK_COMPARISON_CPU_PLACEMENT_REQUIRED')
with tempfile.TemporaryDirectory(prefix='neuronsim-clock-comparison-') as temporary:
 work=Path(temporary);package=work/'Libs/population_dynamics';package.mkdir(parents=True)
 (package/'__init__.py').write_text('')
 for name in ('v1_clock_comparison','provenance'):
  shutil.copy2(repo/'Libs/population_dynamics'/(name+'.py'),package/(name+'.py'))
 sys.path.insert(0,str(work/'Libs'))
 from population_dynamics.v1_clock_comparison import required_files
 receipt=dict(job_id=os.environ['SLURM_JOB_ID'],reference_archive=str(reference),fine_archive=str(fine),exit_code=1)
 code=1
 try:
  for label,archive in [('reference',reference),('fine',fine)]:
   report=json.loads((archive/'primary/report.json').read_text())
   for name in required_files(report):
    target=work/label/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archive/name,target)
  command=[sys.executable,'-B','-m','population_dynamics.v1_clock_comparison','--reference-archive',str(work/'reference'),'--fine-archive',str(work/'fine'),'--output',str(work/'analysis')]
  with (work/'analysis.log').open('w') as log:
   code=subprocess.run(command,cwd=work,env=dict(os.environ,PYTHONPATH=str(work/'Libs')),stdout=log,stderr=subprocess.STDOUT).returncode
 except BaseException as error:
  (work/'failure.txt').write_text(traceback.format_exc());receipt.update(error_type=type(error).__name__,error=str(error))
 finally:
  destination.mkdir(parents=True,exist_ok=False)
  for name in ('analysis','analysis.log','failure.txt','Libs'):
   source=work/name
   if source.is_dir():shutil.copytree(source,destination/name)
   elif source.is_file():shutil.copy2(source,destination/name)
  receipt['exit_code']=code
  if (destination/'analysis/report.json').exists():
   receipt['report_sha256']=hashlib.sha256((destination/'analysis/report.json').read_bytes()).hexdigest()
  (destination/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps(receipt),flush=True)
sys.exit(code)
PYTHON
