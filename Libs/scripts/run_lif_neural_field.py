"""Project the complete graph on CPU03; select mean-field parameters on CPU04."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import yaml


def stage(repo, work, config, *, mode='diagnostic', projection_source=None):
    pending=['lif_neural_field','resource_probe'];seen=set();package=repo/'Libs/population_dynamics'
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);source=package/(name+'.py');target=work/'Libs/population_dynamics'/source.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node,ast.ImportFrom) and node.level==1:
                pending.extend([node.module] if node.module else [a.name for a in node.names])
    for name in ['Libs/__init__.py','Libs/population_dynamics/__init__.py']:
        shutil.copy2(repo/name,work/name)
    (work/'Libs/configs').mkdir();shutil.copy2(repo/'Libs/configs/neural_population_lif_neural_field.yaml',work/'Libs/configs/neural_population_lif_neural_field.yaml')
    shutil.copy2(Path(__file__),work/'launcher.py');shutil.copy2(Path(__file__).with_suffix('.sbatch'),work/'launcher.sbatch')
    spec=importlib.util.spec_from_file_location('field_resources',work/'Libs/population_dynamics/resource_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    health=module.probe(work/'resource.json',required_memory_gib=4)
    if health['status']!='PASS':raise RuntimeError('source-field resource gate:'+str(health['reasons']))
    if mode=='comparison':return
    helper_spec=importlib.util.spec_from_file_location('field_hash',work/'Libs/population_dynamics/provenance.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    base=repo/config['cohort'];cohort_rows=None
    if (base/'delivery.json').exists():
        proof=json.loads((base/'delivery.json').read_text())
        if (proof['status']!='VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE' or proof['completed_episodes']!=72
                or proof['cortical_units']!=100000):raise ValueError('accepted complete native development cohort required')
        cohort_rows={row['path']:row['sha256'] for row in proof['files']}
    def copy_input(relative):
        source=base/relative;target=work/'inputs'/relative
        if cohort_rows is not None and helper.sha256_file(source)!=cohort_rows[relative]:
            raise ValueError('accepted cohort input changed:'+relative)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    if mode=='diagnostic':
        graph=repo/config['source_graph'];manifest=json.loads((graph/'manifest.json').read_text())
        (work/'graph').mkdir();shutil.copy2(graph/'manifest.json',work/'graph/manifest.json');shutil.copy2(graph/'layout.npz',work/'graph/layout.npz')
        for row in manifest['parts']:
            if Path(row['path']).name!=row['path']:raise ValueError('source shard must have a local filename')
            shutil.copy2(graph/row['path'],work/'graph'/row['path'])
        cases=[config['diagnostic_case']]
    else:
        if projection_source is None:raise ValueError('completed diagnostic source required')
        delivery=json.loads((projection_source/'delivery.json').read_text())
        if delivery['status']!='COMPLETE_SOURCE_GRAPH_AND_NATIVE_PREFIX_MEAN_FIELD_REPLAY' or delivery['exit_code']!=0:
            raise ValueError('complete full-source diagnostic and fresh-process replay required')
        declared={row['path']:row['sha256'] for row in delivery['files']}
        (work/'projection').mkdir()
        for name in ['projection.json','projection.npz']:
            source=projection_source/'projection'/name
            if helper.sha256_file(source)!=declared['projection/'+name]:raise ValueError('completed projection changed')
            shutil.copy2(source,work/'projection'/name)
        design=json.loads((base/'train/design.json').read_text())
        cases=[row['case_id'] for row in design['cases'] if row['block'] in config['validation_blocks']]
        copy_input('train/design.json')
        scale=repo/config['fitting_scale'];scale_delivery=json.loads((scale.parents[1]/'delivery.json').read_text())
        scale_rows={row['path']:row['sha256'] for row in scale_delivery['files']}
        if (scale_delivery['status']!='PASS' or scale_delivery['training_seed']!=41
                or helper.sha256_file(scale)!=scale_rows['forecasts/fitting_field_scale.npz']):
            raise ValueError('accepted seed41 fit-only scale required')
        shutil.copy2(scale,work/'fitting_field_scale.npz')
    for case in cases:
        for folder,names in [('train/'+case,['report.json']),('train_fields/'+case,['summary.json','spatial_fields.npz'])]:
            for name in names:
                copy_input(folder+'/'+name)


def run(args):
    allowed=(('cpu03','gpu') if args.mode=='diagnostic' else ('cpu04','cpu'))
    if ((os.environ.get('SLURMD_NODENAME'),os.environ.get('SLURM_JOB_PARTITION'))!=allowed
            or os.environ.get('SLURM_JOB_GPUS') or not os.environ.get('SLURM_JOB_ID')):
        raise ValueError('registered non-GPU Slurm placement required')
    started=time.perf_counter();config=yaml.safe_load((args.repo/'Libs/configs/neural_population_lif_neural_field.yaml').read_text())
    output=args.repo/'Output/neural_population_dynamics/baselines/lif_neural_field'/('run_'+os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True,exist_ok=False);record=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],
        scientific_claim_qualified=False,training_performed=False,source_simulation_performed=False,started_utc=datetime.now(timezone.utc).isoformat())
    try:
        stage(args.repo,args.work,config,mode=args.mode,projection_source=args.projection_source)
        env=dict(os.environ,PYTHONPATH=str(args.work),PYTHONDONTWRITEBYTECODE='1')
        case=config['diagnostic_case']
        if args.mode=='diagnostic':
            command=[sys.executable,'-m','Libs.population_dynamics.lif_neural_field','--graph','graph',
                '--episode','inputs/train/'+case,'--fields','inputs/train_fields/'+case,'--output','projection']
            commands=[('projection',command)]
            common=[sys.executable,'-m','Libs.population_dynamics.lif_neural_field','--projection','projection',
                '--episode','inputs/train/'+case,'--fields','inputs/train_fields/'+case,'--output','forecast',
                '--relaxation-ms',str(config['diagnostic_relaxation_ms']),
                '--origin-ms',str(config['diagnostic_origin_ms']),'--horizon-ms',str(config['diagnostic_horizon_ms'])]
            commands.extend([('forecast',common),('replay',[*common,'--replay'])])
        elif args.mode=='campaign':
            commands=[('evaluation',[sys.executable,'-m','Libs.population_dynamics.lif_neural_field',
                '--projection','projection','--cohort','inputs','--fitting-scale','fitting_field_scale.npz',
                '--config','Libs/configs/neural_population_lif_neural_field.yaml','--output','evaluation'])]
        else:
            if args.selection_source is None or args.population_campaign is None:
                raise ValueError('delivered selection and population campaign required')
            helper_spec=importlib.util.spec_from_file_location('comparison_hash',args.work/'Libs/population_dynamics/provenance.py')
            helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
            for source,destination in [(args.selection_source,'selection'),
                *[(args.population_campaign/f'seed_{seed}',f'population/seed_{seed}') for seed in [41,42,43]]]:
                proof=json.loads((source/'delivery.json').read_text())
                if proof['exit_code']!=0:raise ValueError('successful delivered comparison inputs required')
                for row in proof['files']:
                    name=row['path']
                    if not (name.startswith(('forecasts/','evaluation/')) or name in ('fitting_field_scale.npz','replay.json')):
                        continue
                    if name=='forecasts/full_unit_samples.npz':continue
                    origin=source/name;target=args.work/destination/name
                    if helper.sha256_file(origin)!=row['sha256']:raise ValueError('comparison source changed:'+name)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,target)
                shutil.copy2(source/'delivery.json',args.work/destination/'delivery.json')
            commands=[('comparison',[sys.executable,'-m','Libs.population_dynamics.lif_neural_field',
                '--selection','selection','--population-campaign','population','--output','comparison'])]
        record['commands']=[]
        for name,command in commands:
            with (args.work/(name+'.log')).open('w') as stream:
                child=subprocess.run(command,cwd=args.work,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record['commands'].append(dict(name=name,command=command,exit_code=child.returncode))
            if child.returncode:raise RuntimeError(name+' failed; inspect '+name+'.log')
        statuses=dict(diagnostic='COMPLETE_SOURCE_GRAPH_AND_NATIVE_PREFIX_MEAN_FIELD_REPLAY',
            campaign='COMPLETE_DESCRIPTIVE_DEVELOPMENT_MEAN_FIELD_SELECTION',comparison='COMPLETE_DESCRIPTIVE_COMMON_MEAN_FIELD_COMPARISON')
        record.update(status=statuses[args.mode],exit_code=0)
    except Exception as error:
        record.update(status='FAILED',exit_code=2,error=str(error),traceback=traceback.format_exc())
    finally:
        # Archive declared small software/projection evidence; raw graph is already authoritative.
        spec=importlib.util.spec_from_file_location('field_provenance',args.repo/'Libs/population_dynamics/provenance.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);rows=[]
        for relative in ['Libs','projection','forecast','evaluation','comparison','fitting_field_scale.npz','projection.log','forecast.log','replay.log','evaluation.log','comparison.log','resource.json','launcher.py','launcher.sbatch']:
            path=args.work/relative
            paths=path.rglob('*') if path.is_dir() else [path]
            for source in paths:
                if not source.is_file() or '__pycache__' in source.parts:continue
                name=source.relative_to(args.work);target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);digest=module.sha256_file(target)
                if digest!=module.sha256_file(source):raise ValueError('archived source-field evidence differs')
                rows.append(dict(path=str(name),bytes=target.stat().st_size,sha256=digest))
        record.update(files=rows,elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        (output/'delivery.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    return record['exit_code']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--mode',choices=['diagnostic','campaign','comparison'],default='diagnostic');parser.add_argument('--projection-source',type=Path)
    parser.add_argument('--selection-source',type=Path);parser.add_argument('--population-campaign',type=Path)
    args=parser.parse_args();raise SystemExit(run(args))


if __name__=='__main__':main()
