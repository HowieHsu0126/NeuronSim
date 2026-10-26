"""Reuse delivered conditional LFADS forecasts in the canonical three-seed analyzer.

Both epoch budgets remain separate. No model or native simulation is executed.
Run in a CPU-only Slurm allocation with node-local scratch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

from run_population_trajectory_campaign import archive, copy_bound, digest, manifest_rows


def stage_receipt(root, work, destination, names):
    proof = json.loads((root/'delivery.json').read_text())
    rows = manifest_rows(proof)
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('declared relative artifact path required')
        copy_bound(root/path, work/destination/path, rows[name])
    source = root/'delivery.json'
    receipt_name = 'software_delivery.json' if destination == Path('.') else 'delivery.json'
    copy_bound(source, work/destination/receipt_name, dict(bytes=source.stat().st_size, sha256=digest(source)))
    return proof


def stage_cells(matrix, budget, work):
    """Follow authenticated completed-cell references, rather than directory scans."""
    proof = json.loads((matrix/'delivery.json').read_text())
    if (proof['status'] != 'DESCRIBED_REPLAYED_CONDITIONAL_PRIOR_MATRIX' or proof['exit_code'] != 0
            or proof['completed_cells'] != 6 or proof['new_forecast_origins'] != 0
            or proof['scientific_claim_qualified'] is not False):
        raise ValueError('complete development conditional LFADS matrix required')
    stage_receipt(matrix, work, Path('inputs/matrix'), ['completion_chain.json', 'analysis/analysis.json'])
    chain = json.loads((work/'inputs/matrix/completion_chain.json').read_text())
    matrix_analysis = json.loads((work/'inputs/matrix/analysis/analysis.json').read_text())
    if digest(work/'inputs/matrix/completion_chain.json') != proof['completion_chain_sha256']:
        raise ValueError('completed matrix chain changed')
    candidates = [Path(proof['reference_analysis'])] + [Path(c['analysis_archive']) for c in chain['cells']]
    inputs, bindings, fitting_costs = {}, {}, []
    for comparison in candidates:
        receipt = json.loads((comparison/'delivery.json').read_text())
        candidate = Path(receipt['candidate_delivery']['path']).parent
        candidate_proof = json.loads((candidate/'delivery.json').read_text())
        if candidate_proof['network_budget_epochs'] != budget:
            continue
        seed = candidate_proof['training_seed']
        if seed not in (41,42,43) or seed in inputs:
            raise ValueError('exact independent seed-cell identities required')
        matrix_cell = [cell for cell in matrix_analysis['inputs']
                       if cell['network_budget_epochs'] == budget and cell['training_seed'] == seed]
        if (len(matrix_cell) != 1 or matrix_cell[0]['comparison_delivery_sha256'] != digest(comparison/'delivery.json')
                or matrix_cell[0]['analysis_sha256'] != digest(comparison/'analysis/analysis.json')):
            raise ValueError('completed matrix cell identity changed')
        if (receipt['status'] != 'DESCRIBED_REPLAYED_CONDITIONAL_PRIOR_COMPARISON'
                or receipt['exit_code'] != 0 or receipt['paired_origins'] != 48
                or receipt['scientific_claim_qualified'] is not False
                or candidate_proof['status'] != 'REPLAYED_CONDITIONAL_MEAN_CANDIDATE'
                or candidate_proof['exit_code'] != 0 or candidate_proof['formal_test_accessed'] is not False
                or digest(candidate/'delivery.json') != receipt['candidate_delivery']['sha256']):
            raise ValueError('authenticated complete conditional forecast and comparison required')
        prefix = Path('inputs/cells')/str(seed)
        stage_receipt(comparison, work, prefix/'comparison', ['analysis/analysis.json'])
        evaluation = json.loads((candidate/'candidate/forecasts/evaluation.json').read_text())
        names = ['candidate/forecasts/'+name for name in evaluation['artifacts']]
        names += ['candidate/forecasts/evaluation.json','candidate/forecasts/scores.csv',
                  'candidate/replay.json','candidate/replay.csv']
        stage_receipt(candidate, work, prefix/'forecast', list(dict.fromkeys(names)))
        mean = Path(candidate_proof['mean_run'])
        if (digest(mean/'delivery.json') != candidate_proof['mean_delivery_sha256']
                or digest(mean/'delivery.json') != receipt['mean_delivery']['sha256']):
            raise ValueError('conditional mean fit delivery changed')
        mean_proof = stage_receipt(mean, work, prefix/'mean',
            ['mean/mean_prior.json','mean/controller_means.npz','phases/mean_fit/receipt.json'])
        if mean_proof['status'] != 'FITTED_TEMPLATE_TIME_CONTROLLER_MEAN' or mean_proof['exit_code'] != 0:
            raise ValueError('completed fitting-only controller mean required')
        from population_dynamics.lfads_conditional_prior import load_mean_prior
        _, mean_binding = load_mean_prior(work/prefix/'mean/mean', evaluation['prior_refit'],
            training_seed=seed, network_budget_epochs=budget, selected_model_artifacts=evaluation['selected_model_artifacts'])
        if ({k:v for k,v in mean_binding.items() if k!='implementation_sha256'} !=
                {k:v for k,v in evaluation['conditional_prior'].items() if k!='implementation_sha256'}):
            raise ValueError('forecast conditional mean binding changed')
        admitted = json.loads((work/prefix/'comparison/analysis/analysis.json').read_text())
        if (admitted['candidate_evaluation_sha256'] != digest(work/prefix/'forecast/candidate/forecasts/evaluation.json')
                or admitted['mean_binding'] != evaluation['conditional_prior']):
            raise ValueError('independently compared conditional forecast binding changed')
        phase = json.loads((work/prefix/'mean/phases/mean_fit/receipt.json').read_text())
        if phase['status'] != 'PASS' or phase['exit_code'] != 0:
            raise ValueError('completed measured mean-fit phase required')
        seconds = (datetime.fromisoformat(phase['finished_utc']) - datetime.fromisoformat(phase['started_utc'])).total_seconds()
        fitting_costs.append(dict(seed=seed, budget_epochs=budget, mean_fit_phase_seconds=seconds,
            cost_scope='measured mean-fit child phase only; inherited neural training and posterior cache costs excluded',
            source_delivery_sha256=digest(mean/'delivery.json')))
        inputs[seed] = (work/prefix/'forecast/candidate/forecasts',work/prefix/'forecast/candidate/replay.json')
        bindings[str(seed)] = dict(comparison=str(comparison), comparison_delivery_sha256=digest(comparison/'delivery.json'),
            candidate=str(candidate), candidate_delivery_sha256=digest(candidate/'delivery.json'),
            mean=str(mean), mean_delivery_sha256=digest(mean/'delivery.json'),
            forecast_continuation=evaluation.get('forecast_continuation'), graph_lifetimes=evaluation['graph_lifetimes'])
    if set(inputs) != {41,42,43}:
        raise ValueError('three complete conditional LFADS cells per budget required')
    return inputs, bindings, fitting_costs


def run(args):
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS')
            or os.environ.get('SLURMD_NODENAME') not in ('cpu03','cpu04')):
        raise ValueError('allowed CPU-only Slurm allocation required')
    if args.output.exists():
        raise FileExistsError(args.output)
    record = dict(status='RUNNING', job_id=os.environ['SLURM_JOB_ID'], command=sys.argv,
        node=os.environ['SLURMD_NODENAME'], training_seeds=[41,42,43], budget_epochs=args.budget,
        calibration_outcomes_accessed=False, test_outcomes_accessed=False, scientific_claim_qualified=False,
        new_forecasts=0, neural_training_updates=0, started_utc=datetime.now(timezone.utc).isoformat())
    with tempfile.TemporaryDirectory(prefix='neuronsim-lfads-saved-'+record['job_id']+'.',dir='/tmp') as name:
        work = Path(name)
        code = 1
        try:
            software = json.loads((args.software/'delivery.json').read_text())
            if software['status'] != 'PASS' or software['exit_code'] != 0:
                raise ValueError('accepted saved-analysis software required')
            names = [r['path'] for r in software['files'] if r['path'].startswith('Libs/')
                     and '/test_' not in r['path'] and r['path'].endswith(('.py','.yaml'))]
            stage_receipt(args.software, work, Path('.'), names)
            sys.path[:0] = [str(work/'Libs'), str(work)]
            from population_dynamics.resource_probe import probe
            if probe(work/'admission.json',required_memory_gib=4)['status'] != 'PASS':
                raise RuntimeError('saved-analysis resource admission failed')
            from population_dynamics.lfads_forecast_analysis import analyze
            from population_dynamics.trajectory_forecast_analysis import _write_rows
            from population_dynamics.population_forecast_analysis import _origin_cost_fidelity
            inputs, bindings, fitting_costs = stage_cells(args.matrix,args.budget,work)
            method = f'lfads_template_time_controller_mean_prior_budget{args.budget}'
            report = analyze(work/'cells',work/'analysis',forecast_inputs=inputs,method=method)
            import csv
            rows = list(csv.DictReader((work/'analysis/forecast_all_seeds.csv').open()))
            costs = list(csv.DictReader((work/'analysis/costs_by_origin.csv').open()))
            joined = _origin_cost_fidelity(rows,costs,report['configuration'],report['costs'])
            _write_rows(work/'analysis/cost_fidelity_by_origin.csv',joined)
            _write_rows(work/'analysis/fitting_costs_by_seed.csv',fitting_costs)
            report.update(saved_cell_bindings=bindings,
                conditioning='Known familiar template and elapsed native time; fitting-only controller means; unchanged network weights and AR scales',
                information_matched_to_population_model=False, network_budget_epochs=args.budget,
                end_to_end_cost_measured=False, formal_test_accessed=False,
                cost_fidelity_origins=len(joined))
            for filename in ('cost_fidelity_by_origin.csv','fitting_costs_by_seed.csv'):
                report['artifacts'][filename]=digest(work/'analysis'/filename)
            (work/'analysis/analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            record.update(status='ARCHIVED_VERIFIED_DEVELOPMENT',exit_code=0,
                matrix_delivery_sha256=digest(args.matrix/'delivery.json'),saved_cell_bindings=bindings,
                verification=dict(all_unit_metrics_independently_replayed=True,seeds=[41,42,43],score_rows=432))
            code=0
        except Exception as error:
            record.update(status='FAILED',exit_code=1,error=str(error),traceback=traceback.format_exc())
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
        archive(work,args.output,record)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','software','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--budget',type=int,choices=(20,80),required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__=='__main__':
    main()
