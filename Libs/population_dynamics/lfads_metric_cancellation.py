"""Replay the retained LFADS FP32 cancellation failure without changing evidence."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from .lfads_forecast_replay import (_exact_array, _loaded_model, _native_episode,
    _project_units, _sample_payload, _verified_training, LFADSRandomStreams, numpy_unit_metrics)
from .provenance import sha256_file
from .trajectory_sample_metrics import unit_count_metrics as original_metrics


def diagnose(acquisition, fields, training, forecasts, migration, candidate, output):
    paths = map(Path, (acquisition, fields, training, forecasts, migration, candidate, output))
    acquisition, fields, training, forecasts, migration, candidate, output = paths
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='RUNNING', scientific_claim_qualified=False,
                  scope='one previously failed development score; same checkpoint and sampled path',
                  candidate_metric_sha256=sha256_file(candidate),
                  implementation_sha256=sha256_file(Path(__file__)))
    model = None
    started = time.perf_counter()
    try:
        ev = json.loads((forecasts/'evaluation.json').read_text())
        failed_replay = forecasts.parent/'seed_42_replay.json'
        failed = json.loads(failed_replay.read_text())
        if (failed['status'] != 'MISMATCH' or failed['verified_origins'] != 48
                or failed['evaluation_sha256'] != sha256_file(forecasts/'evaluation.json')
                or failed['scores_sha256'] != sha256_file(forecasts/'scores.csv')
                or failed['csv_sha256'] != sha256_file(failed_replay.with_suffix('.csv'))):
            raise ValueError('completed bound original numerical failure required')
        with failed_replay.with_suffix('.csv').open() as stream:
            mismatches = [r for r in csv.DictReader(stream) if r['passed'] != 'True']
        if len(mismatches) != 1:
            raise ValueError('diagnostic scope requires the single recorded mismatch')
        mismatch = mismatches[0]
        spec = importlib.util.spec_from_file_location('candidate_count_metrics', candidate)
        revised = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(revised)
        cohort, run, directory, selected, loaded_migration = _verified_training(acquisition, fields, training, migration)
        binding = run['binding']
        if (binding != ev['training_binding'] or binding['seed'] != 42
                or selected['model_artifacts'] != ev['selected_model_artifacts']
                or loaded_migration != ev['inference_migration']):
            raise ValueError('unchanged failed-case training and inference binding required')
        case_index = next(i for i,c in enumerate(cohort.validation) if c['case_id'] == mismatch['case_id'])
        case = cohort.validation[case_index]
        cfg = ev['configuration'];origin = float(mismatch['origin_ms'])
        origin_index = cfg['origins_ms'].index(origin)
        boundary = int(origin/binding['config']['bin_ms'])
        horizons = [int(ms/binding['config']['bin_ms']) for ms in cfg['horizons_ms']]
        longest = max(horizons);draws = cfg['joint_samples']
        data = _native_episode(cohort, case)
        counts, stimulus = data.window(0, data.duration_ticks)
        model = _loaded_model(directory, cohort, binding, boundary, longest, 'cuda')
        state = model.infer_state(dict(counts=counts[None,:boundary],stimulus=stimulus[None,:boundary]),
                                 metadata=dict(unit_ids=list(map(int,cohort.unit_ids)),bin_seconds=.005))
        seed = cfg['sampling_seed_base'] + 42*100000 + case_index*100 + origin_index*3
        sample = model.simulate(state,stimulus[None,boundary:boundary+longest],
                                LFADSRandomStreams.from_seeds(seed,seed+1,seed+2),draws)
        payload = _sample_payload(sample,cohort.unit_ids,draws,boundary,longest,.005,binding['config']['model'])
        generated, observed = sample.counts[:,0], counts[boundary:boundary+longest]
        del sample
        projected, truth = _project_units(generated,observed,cohort.unit_field_index,len(cohort.field_ids))
        saved_path = forecasts/f"{case['case_id']}_origin_{origin:g}.npz"
        with np.load(saved_path,allow_pickle=False) as saved:
            for name,value in [('sampled_field_counts',projected),('observed_field_counts',truth),*payload.items()]:
                _exact_array(saved[name],value,'unchanged sampled field/path required:'+name)
        independent, exact, consumed = numpy_unit_metrics(generated,observed,data.minimum_interval_ticks,
                                      data.bin_ticks,horizons,cfg['refractory_support_windows_bins'])
        torch.set_num_threads(2)
        compared = []
        for ms,h in zip(cfg['horizons_ms'],horizons):
            args = (torch.from_numpy(generated[:,:h]),torch.from_numpy(observed[:h]).float(),
                    data.minimum_interval_ticks,data.bin_ticks,cfg['refractory_support_windows_bins'])
            old, new = original_metrics(*args), revised.unit_count_metrics(*args)
            for name,value in old.items():
                reference = independent[h][name]
                allowed = None if reference is None else 1e-7+5e-6*abs(new[name])
                passed = reference is None or abs(new[name]-reference) <= allowed
                compared.append(dict(horizon_ms=ms,metric=name,original=value,revised=new[name],
                                     independent_float64=reference,revised_passed=passed))
        h = int(float(mismatch['horizon_ms'])/binding['config']['bin_ms'])
        residuals = generated[:,:h].astype(np.int64)-observed[None,:h].astype(np.int64)
        largest = int(np.abs(residuals).max())
        if largest**2*h*len(cohort.unit_ids) > np.iinfo(np.int64).max:
            raise ValueError('independent ordered-pair oracle integer reduction would overflow')
        numerator = sum(2*int(np.sum(residuals[i]*residuals[j],dtype=np.int64))
                        for i in range(draws) for j in range(i+1,draws))
        denominator = draws*(draws-1)*h*len(cohort.unit_ids)
        value = numerator/denominator
        bad = next(r for r in compared if r['horizon_ms']==float(mismatch['horizon_ms'])
                   and r['metric']==mismatch['metric'])
        if (bad['original'] != float(mismatch['evaluator_float32'])
                or bad['revised'] != float(np.float32(value))
                or abs(value-float(mismatch['numpy_float64'])) > 1e-12
                or not all(r['revised_passed'] for r in compared)):
            raise ValueError('original failure or exact corrected estimator did not reproduce')
        np.savez_compressed(output/'failed_case_counts.npz',sampled_counts=generated[:,:h],
                            observed_counts=observed[:h],unit_ids=cohort.unit_ids,
                            minimum_interval_ticks=data.minimum_interval_ticks)
        result.update(status='REPRODUCED_FP32_CANCELLATION_EXACT_CORRECTION_PASS',
                      mismatch=mismatch, comparisons=compared, exact_numerator=numerator,
                      exact_denominator=denominator, exact_rational_value=value,
                      observed_integer_count_values=observed[:h].size,
                      sampled_integer_count_values=generated[:,:h].size,
                      all_saved_fields_and_latent_paths_exact=True,
                      full_unit_sample_scope='newly retained failed-case diagnostic; no preexisting full-unit array for this origin',
                      regenerated_count_values=consumed,
                      original_replay_sha256=sha256_file(failed_replay),
                      original_unit_metrics_sha256=ev['unit_metrics_implementation_sha256'],
                      counts_sha256=sha256_file(output/'failed_case_counts.npz'))
    except BaseException as error:
        result.update(status='FAILED',error=str(error),error_type=type(error).__name__)
        raise
    finally:
        if model is not None:
            model.close()
        result['elapsed_seconds'] = time.perf_counter()-started
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition','fields','training','forecasts','migration','candidate','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = diagnose(args.acquisition,args.fields,args.training,args.forecasts,args.migration,args.candidate,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','mismatch')},indent=2))


if __name__ == '__main__':
    main()
