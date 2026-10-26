"""Execute and publish a preregistered PFC native reference; preflight runs no simulation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

from . import pfc_scale_analysis as source
from . import pfc_source_comparison as comparison
from .provenance import freeze_json, sha256_file
from scripts.run_pfc_source_cases import archive_tree, write_json

NATIVE_MODULES = ('pfc_reference', 'pfc_factorized_source', 'pfc_initial_rng',
                  'pfc_kernel_parameters', 'circuit_data', 'circuit_campaign', 'config', 'provenance')
SOURCE_FILES = ('source.json', 'neurodynex3/__init__.py', 'neurodynex3/tools/__init__.py',
                'neurodynex3/tools/plot_tools.py', 'neurodynex3/tools/spike_tools.py',
                'neurodynex3/working_memory_network/wm_model.py')
RUNTIME_PREFIX = 'PFC_NATIVE_RUNTIME_JSON:'
REQUIRED_MEMORY_GIB = 8
QUALIFICATION_FLAGS = dict(full_scale_qualified=False, scientific_fidelity_qualified=False,
    working_memory_qualified=False, fidelity_at_requested_scale_qualified=False,
    scientific_claim_qualified=False, production_episodes=0)

# The same child records its runtime before invoking the existing CLI.
# The import/help branch explicitly forbids any native simulation call.
NATIVE_COMMAND_SOURCE = r"""import hashlib,importlib,json,runpy,sys
from pathlib import Path
import brian2,numpy,scipy
runtime=dict(python=sys.version,interpreter=sys.executable,brian2=brian2.__version__,
             numpy=numpy.__version__,scipy=scipy.__version__)
if sys.argv[1:]==['--preflight-help']:
    def prohibit_simulation(frame,event,arg):
        if event=='call':
            name=frame.f_code.co_name
            path=Path(frame.f_code.co_filename).resolve()
            root=Path(__file__).parent
            forbidden=(name=='simulate_wm' and path==(root/'source/neurodynex3/working_memory_network/wm_model.py').resolve())
            forbidden=forbidden or (name in ('execute','profiled_execute') and path==(root/'Libs/population_dynamics/pfc_reference.py').resolve())
            if forbidden: raise RuntimeError('NATIVE_SIMULATION_FORBIDDEN_IN_PREFLIGHT')
    sys.setprofile(prohibit_simulation)
    module=importlib.import_module('population_dynamics.pfc_reference')
    wm=importlib.import_module('neurodynex3.working_memory_network.wm_model')
    expected=Path(__file__).parent/'source/neurodynex3/working_memory_network/wm_model.py'
    if Path(wm.__file__).resolve()!=expected.resolve():
        raise RuntimeError('PREFLIGHT_IMPORTED_UNSTAGED_SOURCE')
    sys.argv=['pfc_reference','--help']
    try:
        module.main()
    except SystemExit as error:
        if error.code not in (0,None): raise
    else:
        raise RuntimeError('NATIVE_HELP_DID_NOT_TERMINATE')
    runtime.update(native_cli_import_help=True,
                   source_entry_sha256=hashlib.sha256(expected.read_bytes()).hexdigest())
    print('PFC_NATIVE_RUNTIME_JSON:'+json.dumps(runtime),flush=True)
else:
    print('PFC_NATIVE_RUNTIME_JSON:'+json.dumps(runtime),flush=True)
    sys.argv=['pfc_reference',*sys.argv[1:]]
    runpy.run_module('population_dynamics.pfc_reference',run_name='__main__')
"""


@dataclass(frozen=True)
class ExecutionInputs:
    preparation: Path
    canonical_config: Path
    source_runs: list[Path]
    factorization_proof: Path
    pinned_source: Path
    native_python: Path
    work: Path
    output: Path
    native_matplotlib_cache: Path
    project_root: Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _row(path):
    return dict(path=Path(path), bytes=Path(path).stat().st_size, sha256=sha256_file(path))


def _native_environment(inputs, work):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(work/'Libs'), str(work/'source'))),
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', MPLCONFIGDIR=str(inputs.native_matplotlib_cache),
        TMPDIR=str(work))
    env.pop('PYTHONHOME', None)
    return env


def _copy_bound(row, target):
    _require(row['path'].stat().st_size == row['bytes'] and sha256_file(row['path']) == row['sha256'],
             'native staging input changed: '+str(row['path']))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(row['path'], target)
    _require(target.stat().st_size == row['bytes'] and sha256_file(target) == row['sha256'],
             'native staging copy changed: '+str(target))


def _stage(plan, work):
    package = work/'Libs/population_dynamics'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    for name, row in plan['module_rows'].items():
        _copy_bound(row, package/(name+'.py'))
    for name, row in plan['source_rows'].items():
        _copy_bound(row, work/'source'/name)
    _copy_bound(plan['design_row'], work/'design.json')
    _copy_bound(plan['native_config_row'], work/'native_config.yaml')
    _copy_bound(plan['canonical_config_row'], work/'canonical_source.yaml')
    (work/'native_command.py').write_text(NATIVE_COMMAND_SOURCE)


def _runtime_from_text(text):
    for line in text.splitlines():
        if line.startswith(RUNTIME_PREFIX):
            return json.loads(line[len(RUNTIME_PREFIX):])
    raise ValueError('native child runtime observation missing')


def _validate_runtime(runtime, design, *, import_help=False):
    expected = {key.removesuffix('_version'):value for key,value in design['reference_runtime'].items()}
    _require(all(runtime.get(key) == value for key,value in expected.items())
             and isinstance(runtime.get('python'),str) and bool(runtime['python']),
             'native child runtime differs from preregistered stack')
    if import_help:
        _require(runtime.get('native_cli_import_help') is True
                 and runtime.get('source_entry_sha256') == design['source_entry_sha256'],
                 'staged native CLI/source import witness missing or changed')


def _command(inputs, work):
    # Preserve the absolute venv/bin/python path; resolving it can bypass the venv.
    return [str(inputs.native_python), '-B', str(work/'native_command.py'),
            '--source',str(work/'source'), '--config',str(work/'native_config.yaml'),
            '--output',str(work/'native')]


def preflight(inputs: ExecutionInputs) -> dict:
    for name in ('preparation','canonical_config','factorization_proof','pinned_source',
                 'native_python','work','output','native_matplotlib_cache','project_root'):
        _require(getattr(inputs,name).is_absolute(), 'explicit absolute '+name+' path required')
    _require(inputs.native_python.is_file() and os.access(inputs.native_python,os.X_OK),
             'absolute executable native Python required')
    _require(all(path.is_absolute() for path in inputs.source_runs),'explicit absolute source-run paths required')
    _require(not inputs.work.exists() and not inputs.output.exists()
             and not inputs.work.is_symlink() and not inputs.output.is_symlink(), 'fresh native work and output required')
    work, output = inputs.work.resolve(), inputs.output.resolve()
    _require(work.is_relative_to(Path('/tmp')) and work != Path('/tmp'), 'native work must be node-local under /tmp')
    _require(work != output and not output.is_relative_to(work) and not work.is_relative_to(output),
             'native work and durable output must be disjoint')
    cache = inputs.native_matplotlib_cache.resolve()
    _require(cache.is_relative_to(Path('/tmp')) and not cache.is_relative_to(work),
             'native Matplotlib cache must be node-local and outside fresh work')
    protected = [inputs.preparation, inputs.factorization_proof, inputs.pinned_source]
    protected += [path.parent if path.name == 'source_run' else path for path in inputs.source_runs]
    _require(all(not work.is_relative_to(path.resolve()) and not output.is_relative_to(path.resolve())
                 for path in protected), 'native destinations must preserve existing input archives')
    design_path = inputs.preparation/'design.json'
    native_path = inputs.preparation/'native_config.yaml'
    design = json.loads(design_path.read_text())
    _require(design.get('scope') == comparison.REFERENCE_CLOCK_SCOPE, 'explicit reference-clock diagnostic required')
    config, registered, _, dimensions = source._source_registration(inputs.canonical_config)
    config_sha = sha256_file(inputs.canonical_config)
    contract = comparison._streamed_design_contract(source,design,config,config_sha,inputs.factorization_proof)
    _require(sha256_file(native_path) == design['native_config_sha256']
             and yaml.safe_load(native_path.read_text()) == contract['native_config'],
             'frozen cases-only native projection changed')
    _require(len(inputs.source_runs) == design['cases'], 'complete preregistered source-run list required')
    guarded = [_row(design_path),_row(native_path),_row(inputs.canonical_config)]
    found = set()
    for local in inputs.source_runs:
        status, header = source._source_header(local,config,registered,config_sha)
        _require(header is not None and status['status'] == 'COMPLETE','complete existing source archive required')
        case_id = header['report']['case_id']
        _require(case_id in design['case_ids'] and case_id not in found,'duplicate or unregistered source case')
        expected = next(row for row in design['preexisting_source_bindings'] if row['case_id'] == case_id)
        _require(all(header['binding'][key] == expected[key] for key in ('report_sha256','delivery_sha256')),
                 'preregistered source report/delivery changed')
        _require(header['report'].get('backend') == 'numpy'
                 and header['report'].get('driver',{}).get('numpy_version') == design['candidate_numpy_version'],
                 'existing candidate runtime changed')
        source._source_execution_metadata(header,config,registered,dimensions,config_sha)
        guarded += [_row(header['source_run']/'report.json'),_row(header['archive']/'delivery.json')]
        found.add(case_id)
    _require(found == set(design['case_ids']), 'complete selected source denominator required')
    manifest_path = inputs.pinned_source/'source.json'
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get('revision') == design['source_revision']
             and manifest.get('repository') == 'https://github.com/EPFL-LCN/neuronaldynamics-exercises',
             'pinned native source identity changed')
    index = {row['path']:row for row in manifest['files']}
    source_rows = {'source.json':_row(manifest_path)}
    for name in SOURCE_FILES[1:]:
        row = _row(inputs.pinned_source/name)
        _require(name in index and row['sha256'] == index[name]['sha256'] and row['bytes'] == index[name]['bytes'],
                 'pinned native source helper changed: '+name)
        source_rows[name] = row
    _require(source_rows[SOURCE_FILES[-1]]['sha256'] == design['source_entry_sha256'],'native source entry changed')
    module_rows = {name:_row(inputs.project_root/'Libs/population_dynamics'/(name+'.py')) for name in NATIVE_MODULES}
    _require(module_rows['pfc_factorized_source']['sha256'] == design['reference_execution_implementation_sha256'],
             'current native factorization helper differs from prepared proof')
    for name in ('source_comparison.json','factorized/report.json','design.json',
                 'Libs/population_dynamics/pfc_factorized_source.py'):
        guarded.append(_row(inputs.factorization_proof/name))
    plan = dict(design=design, native_config=contract['native_config'], module_rows=module_rows,
        source_rows=source_rows, guarded_rows=guarded, design_row=_row(design_path),
        native_config_row=_row(native_path),canonical_config_row=_row(inputs.canonical_config))
    with tempfile.TemporaryDirectory(prefix='neuronsim-pfc-native-import-',dir='/tmp') as name:
        staged = Path(name)
        _stage(plan,staged)
        probe = subprocess.run([str(inputs.native_python),'-B',str(staged/'native_command.py'),'--preflight-help'],
            cwd=staged,env=_native_environment(inputs,staged),text=True,capture_output=True)
        _require(probe.returncode == 0, 'staged native import/help failed: '+str(probe.stderr))
        runtime = _runtime_from_text(probe.stdout)
        _validate_runtime(runtime,design,import_help=True)
    plan['preflight_runtime'] = runtime
    return plan


def _execution_allocation():
    _require(os.environ.get('SLURMD_NODENAME') == 'cpu03' and os.environ.get('SLURM_JOB_PARTITION') == 'gpu'
             and bool(os.environ.get('SLURM_JOB_ID')) and not os.environ.get('SLURM_JOB_GPUS')
             and not os.environ.get('CUDA_VISIBLE_DEVICES')
             and int(os.environ.get('SLURM_NTASKS','0')) == 1
             and int(os.environ.get('SLURM_CPUS_PER_TASK','0')) == 1,
             'native execution requires CPU03/gpu/no-GPU one-CPU allocation')
    memory = int(os.environ.get('SLURM_MEM_PER_NODE',os.environ.get('SLURM_MEM_PER_CPU','0')))
    _require(memory >= REQUIRED_MEMORY_GIB*1024, 'native execution requires the evidenced 8GiB allocation')


def _completed_outputs(plan, work, runtime):
    design = plan['design']
    report_path = work/'native/report.json'
    report = json.loads(report_path.read_text())
    _require(report.get('status') == 'NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS'
             and report.get('resolved_config') == plan['native_config']
             and report.get('source_revision') == design['source_revision']
             and report.get('source_entry_sha256') == design['source_entry_sha256']
             and report.get('synapse_execution') == design['reference_synapse_execution']
             and report.get('execution_implementation_sha256') == design['reference_execution_implementation_sha256']
             and report.get('full_scale_qualified') is False and report.get('scientific_fidelity_qualified') is False,
             'successful native report binding or diagnostic scope incomplete')
    _validate_runtime(runtime,design)
    _require(report.get('brian2_version') == runtime['brian2'] and report.get('numpy_version') == runtime['numpy'],
             'native report differs from same-child runtime')
    cases = report.get('cases')
    _require(isinstance(cases,list) and len(cases) == design['cases']
             and [case.get('case_id') for case in cases] == design['case_ids'],
             'successful native output has incomplete selected denominator')
    for case, expected in zip(cases,plan['native_config']['cases']):
        _require(all(case.get(key) == value for key,value in expected.items()) and case.get('core_units') == 100000,
                 'native case physical/RNG/unit metadata changed')
        name = case.get('raw_file')
        _require(isinstance(name,str) and Path(name).name == name and name not in ('','.','..')
                 and comparison._sha256_value(case.get('raw_sha256'))
                 and (work/'native'/name).is_file() and not (work/'native'/name).is_symlink()
                 and (work/'native'/name).stat().st_size > 0,
                 'successful native output lacks retained raw artifact')
        _require(type(case.get('native_events')) is int and case['native_events'] >= 0
                 and type(case.get('elapsed_seconds')) in (int,float)
                 and math.isfinite(case['elapsed_seconds']) and case['elapsed_seconds'] >= 0,
                 'native case event/cost metadata incomplete')
    catalog = json.loads((work/'native/dataset/dataset.json').read_text())
    _require(catalog.get('complete_episodes') == design['cases'],'successful native archive catalog incomplete')
    return report


def run(inputs: ExecutionInputs, *, preflight_only: bool) -> dict:
    plan = preflight(inputs)
    if preflight_only:
        return dict(status='PREFLIGHT_PASS_NATIVE_REFERENCE_CLOCK',command=_command(inputs,inputs.work),
            runtime_versions=plan['preflight_runtime'],staged_modules=list(NATIVE_MODULES),
            selected_cases=plan['design']['case_ids'],native_execution_performed=False,
            source_archive_created=False,required_memory_gib=REQUIRED_MEMORY_GIB,**QUALIFICATION_FLAGS)
    _execution_allocation()
    inputs.work.mkdir(parents=True,exist_ok=False)
    record = dict(status='NATIVE_REFERENCE_CLOCK_EXECUTION_FAILED',exit_code=1,native_exit_code=None,
        native_execution_performed=False,archive_status='NOT_STARTED',completed_cases=0,
        job_id=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),
        command=_command(inputs,inputs.work),started_utc=datetime.now(timezone.utc).isoformat(),
        design_sha256=plan['design_row']['sha256'],native_config_sha256=plan['native_config_row']['sha256'],
        source_config_sha256=plan['canonical_config_row']['sha256'],**QUALIFICATION_FLAGS)
    report = None
    try:
        for row in plan['guarded_rows']:
            _require(row['path'].stat().st_size == row['bytes'] and sha256_file(row['path']) == row['sha256'],
                     'registered native input changed before execution: '+str(row['path']))
        _stage(plan,inputs.work)
        with (inputs.work/'native.log').open('x') as log:
            record['native_execution_performed'] = True
            child = subprocess.run(record['command'],cwd=inputs.work,env=_native_environment(inputs,inputs.work),
                                   stdout=log,stderr=subprocess.STDOUT)
        record.update(native_exit_code=child.returncode,exit_code=child.returncode)
        if child.returncode == 0:
            with (inputs.work/'native.log').open() as log:
                runtime = _runtime_from_text(log.read(65536))
            record['runtime_versions'] = runtime
            report = _completed_outputs(plan,inputs.work,runtime)
            record.update(status='NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE',completed_cases=len(report['cases']),
                          native_report_sha256=sha256_file(inputs.work/'native/report.json'))
    except Exception as error:
        record.update(status='NATIVE_REFERENCE_CLOCK_EXECUTION_FAILED',exit_code=record['exit_code'] or 2,
                      error_type=type(error).__name__,error=str(error))
    record['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(inputs.work/'execution.json',record)
    try:
        files = archive_tree(inputs.work,inputs.output)
        index = {row['path']:row for row in files}
        if report is not None:
            _require(all(index['native/'+case['raw_file']]['sha256'] == case['raw_sha256']
                         for case in report['cases']), 'published native raw binding differs from producer report')
        delivery = dict(record,files=files,archive_status='COMPLETE')
        freeze_json(inputs.output/'delivery.json',delivery)
        record['archive_status'] = 'COMPLETE'
    except Exception as error:
        record.update(status='NATIVE_REFERENCE_ARCHIVE_FAILED',exit_code=2,archive_status='FAILED',
                      archive_error_type=type(error).__name__,archive_error=str(error))
        write_json(inputs.work/'execution.json',record)
        if inputs.output.is_dir():
            freeze_json(inputs.output/'delivery.json',dict(record,files=[]))
    if record['status'] == 'NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE' and record['archive_status'] == 'COMPLETE':
        shutil.rmtree(inputs.work)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation','canonical-config','factorization-proof','pinned-source','native-python',
                 'work','output','native-matplotlib-cache'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-run',type=Path,action='append',required=True)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    inputs = ExecutionInputs(args.preparation,args.canonical_config,args.source_run,args.factorization_proof,
        args.pinned_source,args.native_python,args.work,args.output,args.native_matplotlib_cache,args.project_root)
    report = run(inputs,preflight_only=args.preflight_only)
    print(json.dumps(report,default=str))
    return 0 if report['status'] in ('PREFLIGHT_PASS_NATIVE_REFERENCE_CLOCK','NATIVE_REFERENCE_CLOCK_EXECUTION_COMPLETE') else int(report['exit_code'] or 1)


if __name__ == '__main__':
    raise SystemExit(main())
