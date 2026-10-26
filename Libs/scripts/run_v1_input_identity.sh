#!/bin/bash
#SBATCH --job-name=NrnSim_v1_inputs
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_v1_inputs-%j.log
set -euo pipefail
if [[ $# != 5 ]]; then
  printf 'Expected repo acquisition_archive graph_archive software_archive output_family\n' >&2
  exit 2
fi
[[ ( "${SLURMD_NODENAME:-}" == cpu04 || "${SLURMD_NODENAME:-}" == cpu05 ) && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
repo=$1
acquisition=$2
graph=$3
software=$4
family=$5
interpreter=${NRNSIM_V1_PYTHON:?Set NRNSIM_V1_PYTHON to the validated compute interpreter}
work=$(mktemp -d /tmp/neuronsim-v1-inputs-${SLURM_JOB_ID}.XXXXXX)
export NRNSIM_V1_WORK="$work" NRNSIM_V1_REPO="$repo"
export NRNSIM_V1_ACQUISITION="$acquisition" NRNSIM_V1_GRAPH="$graph"
export NRNSIM_V1_SOFTWARE="$software" NRNSIM_V1_FAMILY="$family"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p "$work/Libs/population_dynamics" "$work/phases" "$work/scratch"
: > "$work/Libs/population_dynamics/__init__.py"
for name in provenance resource_probe v1_cell_parameters v1_graph_identity v1_parameter_reconciliation v1_input_identity test_v1_input_identity; do
  cp "$repo/Libs/population_dynamics/$name.py" "$work/Libs/population_dynamics/"
done
cp "$repo/Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/native_formation_retrieval/analysis/observed_reconciliation/run_579393/run_guarded.py" "$work/run_guarded.py"
cp "${BASH_SOURCE[0]}" "$work/launcher.sh"
export PYTHONPATH="$work/Libs" TMPDIR="$work/scratch"
cd "$work"
"$interpreter" - <<'PY'
from pathlib import Path
import json,os
from population_dynamics.provenance import sha256_file
work=Path(os.environ['NRNSIM_V1_WORK']);archive=Path(os.environ['NRNSIM_V1_SOFTWARE'])
delivery=json.loads((archive/'delivery.json').read_text())
assert delivery['status']=='PASS' and delivery['exit_code']==0, 'V1_INPUT_SOFTWARE_NOT_VALIDATED'
rows={r['path']:r for r in delivery['files']}
receipt='phases/tests/receipt.json'
assert sha256_file(archive/receipt)==rows[receipt]['sha256'], 'V1_INPUT_TEST_RECEIPT_CHANGED'
phase=json.loads((archive/receipt).read_text())
assert phase['status']=='PASS' and phase['exit_code']==0, 'V1_INPUT_TESTS_NOT_PASSED'
for path in sorted((work/'Libs/population_dynamics').glob('*.py')):
    relative=str(path.relative_to(work))
    assert sha256_file(path)==rows[relative]['sha256'], 'V1_INPUT_VALIDATED_SOURCE_CHANGED:'+relative
assert sha256_file(work/'run_guarded.py')==rows['run_guarded.py']['sha256'], 'V1_INPUT_RESOURCE_GUARD_CHANGED'
(work/'reused_tests.json').write_text(json.dumps(dict(status='PASS_REUSED_NATIVE_TESTS',
    archive=str(archive),validation_job=delivery['job_id']),indent=2)+'\n')
PY
cat > "$work/stage_inputs.py" <<'PY'
from pathlib import Path
import json,os,shutil
work=Path(os.environ['NRNSIM_V1_WORK']);root=work/'scratch'
acquisition=Path(os.environ['NRNSIM_V1_ACQUISITION']);graph=Path(os.environ['NRNSIM_V1_GRAPH'])
source=json.loads((acquisition/'asset/source.json').read_text())
name=source['source']['artifact']
if Path(name).name!=name:
    raise ValueError('V1_INPUT_ARCHIVE_NAME_INVALID')
declared=[
    (acquisition/'delivery.json',root/'acquisition/delivery.json'),
    (acquisition/'asset/source.json',root/'acquisition/asset/source.json'),
    (acquisition/'asset'/name,root/'acquisition/asset'/name),
    (graph/'delivery.json',root/'graph/delivery.json'),
    (graph/'graph_identity.json',root/'graph/graph_identity.json'),
]
config=Path(os.environ['NRNSIM_V1_REPO'])/'Output/neural_population_dynamics/multicircuit/v1_asset_intake/point_config'
declared.extend([(config/'source_asset.txt',root/'point_config.json'),
                 (config/'source.json',root/'point_config_source.json')])
for original,destination in declared:
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original,destination)
print(json.dumps(dict(staged_files=len(declared),source_payload_bytes=(root/'acquisition/asset'/name).stat().st_size)),flush=True)
PY
set +e
srun --ntasks=1 "$interpreter" -B "$work/run_guarded.py" stage_inputs "$interpreter" -B "$work/stage_inputs.py"
code=$?
if [[ $code == 0 ]]; then
  srun --ntasks=1 "$interpreter" -B "$work/run_guarded.py" input_identity "$interpreter" -B -m population_dynamics.v1_input_identity \
    --source-acquisition-archive "$work/scratch/acquisition" \
    --graph-inspection-archive "$work/scratch/graph" \
    --point-config "$work/scratch/point_config.json" \
    --point-config-source "$work/scratch/point_config_source.json" \
    --output "$work/input_identity.json"
  code=$?
fi
export NRNSIM_V1_EXIT_CODE="$code"
"$interpreter" - <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import json,os,shutil
from population_dynamics.provenance import sha256_file
work=Path(os.environ['NRNSIM_V1_WORK'])
target=Path(os.environ['NRNSIM_V1_FAMILY'])/('run_'+os.environ['SLURM_JOB_ID'])
target.mkdir(parents=True,exist_ok=False)
files=[]
for path in sorted(work.rglob('*')):
    if path.is_relative_to(work/'scratch') or not path.is_file():
        continue
    rel=path.relative_to(work);destination=target/rel
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,destination)
    digest=sha256_file(path)
    if sha256_file(destination)!=digest:
        raise ValueError('V1_INPUT_ARCHIVE_COPY_CHANGED:'+str(rel))
    files.append(dict(path=str(rel),bytes=destination.stat().st_size,sha256=digest))
code=int(os.environ['NRNSIM_V1_EXIT_CODE'])
receipt=dict(status='INSPECTED' if code==0 else 'INPUT_CHECK_FAILED',
    exit_code=code,job_id=os.environ['SLURM_JOB_ID'],recorded_utc=datetime.now(timezone.utc).isoformat(),
    original_inputs=dict(acquisition_archive=os.environ['NRNSIM_V1_ACQUISITION'],
        graph_inspection_archive=os.environ['NRNSIM_V1_GRAPH'],
        point_config=str(Path(os.environ['NRNSIM_V1_REPO'])/'Output/neural_population_dynamics/multicircuit/v1_asset_intake/point_config')),
    software_validation_archive=os.environ['NRNSIM_V1_SOFTWARE'],files=files,
    native_dynamics_qualified=False,source_scientific_fidelity_qualified=False,production_episodes=0)
(target/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
shutil.rmtree(work)
PY
archive_code=$?
set -e
if [[ $archive_code != 0 ]]; then exit "$archive_code"; fi
exit "$code"
