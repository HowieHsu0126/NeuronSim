"""All-unit circular response readouts from replayed PFC cue/omission episodes.

Three joint initial/background source blocks give descriptive model results.
These data do not establish neuroscience, full-scale fidelity or control efficacy.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .circuit_data import CircuitDataset
from .provenance import sha256_file


def summarize_counts(counts, n_e, cue_center_deg, cue_width_deg, duration_ms):
    counts = np.asarray(counts)
    if (counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts < 0)
            or type(n_e) is not int or not 0 < n_e < len(counts)
            or not np.isfinite(duration_ms) or duration_ms <= 0):
        raise ValueError('finite complete E/I counts and positive phase exposure required')
    angles = 2 * np.pi * np.arange(n_e) / n_e
    total = int(counts[:n_e].sum())
    x = float(np.dot(counts[:n_e], np.cos(angles)))
    y = float(np.dot(counts[:n_e], np.sin(angles)))
    length = math.hypot(x, y)
    defined = total > 0 and length > 64 * np.finfo(np.float64).eps * total
    angle = math.degrees(math.atan2(y, x)) % 360 if defined else None
    center = int(round(n_e / 360. * cue_center_deg))
    half_width = int(round(n_e / 360. * cue_width_deg / 2))
    cue = np.zeros(n_e, dtype=bool)
    cue[np.arange(center - half_width, center + half_width + 1) % n_e] = True
    if not 0 < cue.sum() < n_e:
        raise ValueError('source cue must cover a proper nonempty E subset')
    seconds = duration_ms / 1000
    rate_in = float(counts[:n_e][cue].sum() / cue.sum() / seconds)
    rate_out = float(counts[:n_e][~cue].sum() / (~cue).sum() / seconds)
    return dict(excitatory_spikes=total, inhibitory_spikes=int(counts[n_e:].sum()),
        excitatory_rate_hz=total / n_e / seconds,
        inhibitory_rate_hz=float(counts[n_e:].sum() / (len(counts)-n_e) / seconds),
        cue_units=int(cue.sum()), cue_region_rate_hz=rate_in, noncue_region_rate_hz=rate_out,
        cue_selectivity_hz=rate_in-rate_out, vector_x_spikes=x, vector_y_spikes=y,
        resultant_length=length / total if total else None, decoded_angle_deg=angle,
        circular_cue_error_deg=(angle-cue_center_deg+180)%360-180 if defined else None,
        direction_defined=defined)


def analyze(original, replay_receipt, design_path, output):
    original, replay_receipt, design_path, output = map(Path, (original, replay_receipt, design_path, output))
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((original/'report.json').read_text())
    replay = json.loads(replay_receipt.read_text())
    design = json.loads(design_path.read_text())
    config = report['resolved_config']
    if (report['status'] != 'NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS' or replay['status'] != 'PASS'
            or replay.get('original_report_sha256') != sha256_file(original/'report.json')
            or replay.get('source_revision') != report['source_revision']
            or replay['complete_episodes'] != 6 or replay['refractory_violations'] != 0
            or replay['core_units'] != 1280 or config['duration_ms'] != 500.
            or report['independent_source_rng_blocks'] != 3
            or len(report['matched_prefix_checks']) != 3
            or report['source_revision'] != design['source_revision']
            or config['stimulus_start_ms'] != 100. or config['stimulus_duration_ms'] != 200.
            or config['stimulus_center_deg'] != 120. or config['stimulus_width_deg'] != 30.):
        raise ValueError('complete registered repeated-source diagnostic required')
    phases = design['phases_ms']
    if phases != {'baseline':[0.,100.], 'cue':[100.,300.], 'delay':[300.,500.]}:
        raise ValueError('registered baseline/cue/delay phases required')
    dataset = CircuitDataset(original/'dataset', splits=('development',))
    rows, unit_counts = [], {}
    n_e, n_i = config['excitatory_neurons'], config['inhibitory_neurons']
    dt, bin_ms = config['dt_ms'], config['readout_bin_ms']
    duration_ticks = int(round(config['duration_ms']/dt))
    expected = {(s,c) for s in (91001,91002,91003) for c in ('cue','omission')}
    seen = set()
    for case in report['cases']:
        seed = case['source_rng_seed']
        condition = 'omission' if case['stimulus_strength_nA'] == 0 else 'cue'
        if (seed,condition) in seen or (seed,condition) not in expected or case['case_id'] != f'pfc_{seed}_{condition}':
            raise ValueError('source block/condition identity changed')
        seen.add((seed,condition))
        if case['stimulus_strength_nA'] not in (0.,.06) or case['distractor_strength_nA'] != 0:
            raise ValueError('registered source control changed')
        raw_path = original/case['raw_file']
        if sha256_file(raw_path) != case['raw_sha256']:
            raise ValueError('native raw response changed')
        with np.load(raw_path, allow_pickle=False) as raw:
            ticks, ids = raw['spike_ticks'], raw['spike_unit_indices']
            native_counts = raw['native_spike_counts']
            window = dataset.window(case['case_id'], boundary_tick=duration_ticks//2,
                history_ticks=duration_ticks//2, horizon_ticks=duration_ticks//2,
                bin_ticks=int(round(bin_ms/dt)))
            dense = np.concatenate([window['history'],window['targets']])
            if dense.shape != (100,1280):
                raise ValueError('full-unit full-episode archive readout required')
            total = np.bincount(ids[ticks < duration_ticks],minlength=n_e+n_i)
            if not np.array_equal(total,dense.sum(axis=0)) or not np.array_equal(total,native_counts):
                raise ValueError('native count/recorded event/archive total differs')
            for phase,(start,stop) in phases.items():
                use = (ticks >= int(round(start/dt))) & (ticks < int(round(stop/dt)))
                count = np.bincount(ids[use],minlength=n_e+n_i)
                archive_count = dense[int(round(start/bin_ms)):int(round(stop/bin_ms))].sum(axis=0)
                if not np.array_equal(count,archive_count):
                    raise ValueError('native versus archive phase counts differ')
                key = f'{case["case_id"]}_{phase}'
                unit_counts[key] = count
                rows.append(dict(case_id=case['case_id'],source_rng_seed=seed,condition=condition,phase=phase,
                    start_ms=start,stop_ms=stop,**summarize_counts(count,n_e,120.,30.,stop-start)))
    if seen != expected:
        raise ValueError('all registered source blocks and conditions required')
    contrasts = []
    metrics = ('excitatory_rate_hz','inhibitory_rate_hz','cue_selectivity_hz','cue_region_rate_hz','noncue_region_rate_hz','vector_x_spikes','vector_y_spikes')
    for seed in (91001,91002,91003):
        for phase in phases:
            pair = {r['condition']:r for r in rows if r['source_rng_seed']==seed and r['phase']==phase}
            contrasts.append(dict(source_rng_seed=seed,phase=phase,**{m:pair['cue'][m]-pair['omission'][m] for m in metrics}))
    if any(any(r[m] != 0 for m in metrics) for r in contrasts if r['phase']=='baseline'):
        raise ValueError('matched pre-cue source readouts differ')
    output.mkdir(parents=True)
    for name,table in [('responses.csv',rows),('paired_cue_effects.csv',contrasts)]:
        with (output/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    np.savez_compressed(output/'phase_unit_counts.npz',**unit_counts)
    result=dict(status='DESCRIBED_REPLAYED_SOURCE_DIAGNOSTIC',source_rng_blocks=3,complete_episodes=6,units=1280,
        phase_count_values_verified=6*3*1280,original_report_sha256=sha256_file(original/'report.json'),
        replay_receipt_sha256=sha256_file(replay_receipt),design_sha256=sha256_file(design_path),
        configuration=config,source_revision=report['source_revision'],source_entry_sha256=report['source_entry_sha256'],
        scientific_fidelity_qualified=False,working_memory_qualified=False,neuroscientific_claim_qualified=False,
        population_coordinate='360*source E index/N_E degrees; I has no preferred angle',
        source_rng_scope='joint initial/background blocks, not separately controlled latent state/noise or training seeds',
        interpretation='descriptive source-model response and cue-omission contrast; no confidence intervals, inferential tests, full-scale or neuroscience claim',
        implementation_sha256=sha256_file(Path(__file__)),
        artifacts={n:sha256_file(output/n) for n in ('responses.csv','paired_cue_effects.csv','phase_unit_counts.npz')})
    (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','replay-receipt','design','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.original,args.replay_receipt,args.design,args.output)))

if __name__=='__main__':main()
