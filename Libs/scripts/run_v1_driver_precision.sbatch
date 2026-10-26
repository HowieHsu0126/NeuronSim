#!/bin/bash
#SBATCH --job-name=NrnSim_V1DriverPrecision
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_V1DriverPrecision-%j.log
set -euo pipefail
[[ $# == 3 ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${NRNSIM_BENCHMARK_PYTHON:?Set benchmark Python}" -B - "$@" <<'PYTHON'
from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,sys,tempfile,time,traceback,xml.etree.ElementTree as ET
import yaml
repo,config_path,destination=map(Path,sys.argv[1:])
if destination.exists():raise ValueError('DRIVER_PRECISION_DESTINATION_EXISTS')
if os.environ['SLURMD_NODENAME']!='cpu03' or os.environ.get('SLURM_JOB_GPUS'):
 raise ValueError('DRIVER_PRECISION_CPU03_REQUIRED')
started=time.monotonic();code=1
with tempfile.TemporaryDirectory(prefix='neuronsim-driver-precision-') as temporary:
 work=Path(temporary);package=work/'Libs/population_dynamics';package.mkdir(parents=True);(package/'__init__.py').write_text('')
 receipt=dict(job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],exit_code=1,source_spike_sampler_calls=0,cortical_simulations=0,production_episodes=0,scientific_qualification=False)
 try:
  for name in ('v1_driver_joint_precision','test_v1_driver_joint_precision','provenance','resource_probe'):
   shutil.copy2(repo/'Libs/population_dynamics'/(name+'.py'),package/(name+'.py'))
  config=yaml.safe_load(config_path.read_text());source=Path(config['temporal_archive']);acceptance=Path(config['temporal_acceptance'])
  if not source.is_absolute():source=repo/source
  if not acceptance.is_absolute():acceptance=repo/acceptance
  receipt.update(original_temporal_archive=str(source),original_temporal_acceptance=str(acceptance),original_config=config)
  for name in ('report.json','analysis/report.json','analysis/temporal_summary.npz','analysis/finite_temporal_counts.h5'):
   target=work/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
  shutil.copy2(acceptance,work/'acceptance.json');shutil.copy2(config_path,work/'original_config.yaml')
  config.update(temporal_archive=str(work/'source'),temporal_acceptance=str(work/'acceptance.json'))
  (work/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.resource_probe import probe
  if probe(work/'resources.json',required_memory_gib=4)['status']!='PASS':raise RuntimeError('DRIVER_PRECISION_RESOURCE_STOP')
  env=dict(os.environ,PYTHONPATH=str(work/'Libs'))
  phases=[('tests',[sys.executable,'-B','-m','pytest',str(package/'test_v1_driver_joint_precision.py'),'-q','-p','no:cacheprovider','--junitxml='+str(work/'tests.xml')]),
          ('analysis',[sys.executable,'-B','-m','population_dynamics.v1_driver_joint_precision','--config',str(work/'config.yaml'),'--output',str(work/'analysis')])]
  receipt['commands']=[cmd for _,cmd in phases]
  for phase,cmd in phases:
   print(json.dumps(dict(phase=phase,status='STARTED')),flush=True)
   with (work/(phase+'.log')).open('w') as stream:code=subprocess.run(cmd,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
   print(json.dumps(dict(phase=phase,exit_code=code)),flush=True)
   if code:break
  if (work/'tests.xml').exists():
   suites=list(ET.parse(work/'tests.xml').getroot().iter('testsuite'));receipt['tests']={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
 except BaseException as error:
  receipt.update(error_type=type(error).__name__,error=str(error));(work/'failure.txt').write_text(traceback.format_exc());code=1
 finally:
  destination.mkdir(parents=True,exist_ok=False)
  for name in ('Libs','analysis','analysis.log','tests.log','tests.xml','failure.txt','resources.json','original_config.yaml','config.yaml','acceptance.json'):
   path=work/name
   if path.is_dir():shutil.copytree(path,destination/name)
   elif path.is_file():shutil.copy2(path,destination/name)
  receipt.update(exit_code=code,elapsed_seconds=time.monotonic()-started,child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
  if (destination/'analysis/report.json').exists():receipt['analysis_report_sha256']=hashlib.sha256((destination/'analysis/report.json').read_bytes()).hexdigest()
  (destination/'report.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('commands','original_config')}),flush=True)
sys.exit(code)
PYTHON
