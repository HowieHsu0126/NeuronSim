"""Reuse a verified PFC readout archive to stage and verify a development dataset export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def configure_source_selection(env, work, origins):
    """Restage the exact archived sources; legacy stage scripts retain numeric IDs."""
    env.pop('NRNSIM_PFC_SOURCE_ARCHIVES_JSON',None)
    env.pop('NRNSIM_PFC_SOURCE_JOBS',None)
    if 'NRNSIM_PFC_SOURCE_ARCHIVES_JSON' in (work/'stage_sources.py').read_text():
        path=work/'source_archives.json'
        path.write_text(json.dumps([row['archive'] for row in origins])+'\n')
        env['NRNSIM_PFC_SOURCE_ARCHIVES_JSON']=str(path)
    else:
        env['NRNSIM_PFC_SOURCE_JOBS']=','.join(str(row['job_id']) for row in origins)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (os.environ.get('SLURMD_NODENAME') not in {'cpu03','cpu04','cpu05','cpu06'}
            or not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS')):
        raise ValueError('AUTHORIZED_CPU_ONLY_SLURM_ALLOCATION_REQUIRED')
    if args.output.exists():raise FileExistsError(args.output)
    receipt=json.loads((args.readout/'delivery.json').read_text())
    if receipt.get('status')!='ARCHIVED_PFC_SOURCE_READOUTS' or receipt.get('exit_code')!=0:
        raise ValueError('COMPLETED_SOURCE_READOUT_REQUIRED')
    is_distractor = receipt.get('readout_mode') == 'distractor'
    entries={row['path']:row for row in receipt['files']}
    if len(entries)!=len(receipt['files']):raise ValueError('DUPLICATE_READOUT_DELIVERY_PATH')
    work=Path(tempfile.mkdtemp(prefix='neuronsim-pfc-dataset-'+os.environ['SLURM_JOB_ID']+'.',dir='/tmp'))
    (work/'scratch').mkdir();(work/'phases').mkdir()
    record=dict(status='PREPARING',job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],
        runtime=str(work),readout_archive=str(args.readout),readout_delivery_sha256=digest(args.readout/'delivery.json'),
        started_utc=datetime.now(timezone.utc).isoformat(),commands=[],new_source_simulations=0,
        production_episodes=0,scientific_fidelity_qualified=False)
    code=1
    try:
        names=[name for name in entries if name.startswith('Libs/')]
        names+=['config.yaml','stage_sources.py','run_guarded.py','input_origins.json']
        if is_distractor:
            names+=['protocol.yaml']
            path=args.readout/'analysis/report.json'
            if digest(path)!=entries['analysis/report.json']['sha256']:
                raise ValueError('PAIRED_READOUT_REPORT_CHANGED')
            paired_report=json.loads(path.read_text())
            if paired_report['status']!='COMPLETE_PAIRED_DISTRACTOR_RESPONSE':
                raise ValueError('COMPLETE_PAIRED_DISTRACTOR_RESPONSE_REQUIRED')
            record['paired_readout_report_sha256']=digest(path)
        for name in names:
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('READOUT_PATH_INVALID')
            row=entries[name];target=work/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(args.readout/relative,target)
            if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:
                raise ValueError('VALIDATED_READOUT_INPUT_CHANGED:'+name)
        old_origins=json.loads((work/'input_origins.json').read_text())
        source_root=Path(__file__).resolve().parents[1]/'population_dynamics'
        record['implementation_sha256']={}
        for name in ['pfc_source_dataset.py','test_pfc_source_dataset.py','pfc_distractor_response.py',
                     'test_pfc_distractor_response.py','pfc_temporal_readouts.py','pfc_scale_analysis.py',
                     'benchmark_tasks.py','circuit_campaign.py','config.py','pfc_input_pairing.py','pfc_kernel_parameters.py']:
            target=work/'Libs/population_dynamics'/name;shutil.copy2(source_root/name,target)
            record['implementation_sha256'][name]=digest(target)
        config_target=work/'Libs/configs/neural_population_dynamics.yaml'
        config_target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_root.parent/'configs/neural_population_dynamics.yaml',config_target)
        shutil.copy2(Path(__file__),work/'launcher.py')
        env=dict(os.environ,PYTHONPATH=str(work/'Libs'),TMPDIR=str(work/'scratch'),
            PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',NRNSIM_PFC_READOUT_MODE='distractor' if is_distractor else 'phases')
        if is_distractor:
            by_role={row['role']:row for row in old_origins}
            if set(by_role)!={'baseline','candidate'}:raise ValueError('PAIRED_SOURCE_ORIGINS_REQUIRED')
            env['NRNSIM_PFC_DISTRACTOR_CANDIDATE']=by_role['candidate']['archive']
        else:
            configure_source_selection(env,work,old_origins)
        def run(phase, command):
            row=dict(phase=phase,command=command);record['commands'].append(row)
            result=subprocess.run([sys.executable,'-B',str(work/'run_guarded.py'),phase,*command],cwd=work,env=env)
            row['exit_code']=result.returncode
            if result.returncode:raise RuntimeError(phase+' failed:'+str(result.returncode))
        run('tests',[sys.executable,'-B','-m','unittest','-v',
                     'population_dynamics.test_pfc_source_dataset.PfcSourceDatasetTests'])
        run('stage',[sys.executable,'-B',str(work/'stage_sources.py')])
        origins=json.loads((work/'input_origins.json').read_text())
        signature=lambda rows:[(row.get('role',row.get('job_id')),row['case_id'],row['delivery_sha256']) for row in rows]
        if signature(origins)!=signature(old_origins):raise ValueError('SOURCE_CHANGED_AFTER_VERIFIED_READOUT')
        selected=origins
        origin_path=work/'input_origins.json'
        if is_distractor:
            selected=[row for row in origins if row['role']=='candidate']
            origin_path=work/'dataset_input_origins.json'
            origin_path.write_text(json.dumps(selected,indent=2)+'\n')
        command=[sys.executable,'-B','-m','population_dynamics.pfc_source_dataset',
                 '--config',str(work/('protocol.yaml' if is_distractor else 'config.yaml')),
                 '--output',str(work/'analysis'),'--input-origins',str(origin_path)]
        if is_distractor:command+=['--distractor-baseline',str(work/'distractor_inputs/baseline')]
        for row in selected:command+=['--source-run',row['staged']]
        run('export',command)
        report=json.loads((work/'analysis/report.json').read_text())
        if report['status']!='EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET':
            raise ValueError('DATASET_EXPORT_NOT_VERIFIED')
        record.update(status='EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET',export=report,source_origins=origins)
        code=0
    except Exception as error:
        record.update(status='FAILED',error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
    record.update(exit_code=code,completed_utc=datetime.now(timezone.utc).isoformat())
    (work/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    # Preserve the compute-local directory if archive transfer fails.
    args.output.mkdir(parents=True,exist_ok=False)
    files=[]
    for source in sorted(work.rglob('*')):
        relative=source.relative_to(work)
        if not source.is_file() or set(relative.parts)&{'inputs','distractor_inputs','scratch','__pycache__'}:continue
        target=args.output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        checksum=digest(source)
        if digest(target)!=checksum:raise ValueError('DATASET_ARCHIVE_COPY_MISMATCH:'+str(relative))
        files.append(dict(path=str(relative),bytes=source.stat().st_size,sha256=checksum))
    (args.output/'delivery.json').write_text(json.dumps(dict(record,files=files),indent=2)+'\n')
    shutil.rmtree(work)
    print(json.dumps({key:record.get(key) for key in ['status','job_id','exit_code','error','new_source_simulations','production_episodes']}))
    return code


if __name__=='__main__':
    raise SystemExit(main())
