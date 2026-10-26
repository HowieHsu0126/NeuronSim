"""Describe one registered PFC distractor contrast using complete source archives.

A paired control intervention reuses the existing no-distractor episode. All
time bins remain visible; no working-memory success threshold is imposed here.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

from .pfc_delay_analysis import read, require, write_json, write_rows
from .pfc_scale_analysis import (
    _source_registration, _source_header, _source_values, _source_manifest,
    _source_bound_file, geometry,
)
from .pfc_temporal_readouts import temporal_rows, delay_summary
from .provenance import sha256_file


def registration(config_path, baseline):
    config_path, baseline = Path(config_path), Path(baseline)
    candidate = yaml.safe_load(config_path.read_text())
    protocol = candidate['distractor_response']
    require(protocol['status'] == 'REGISTERED_BEFORE_SOURCE_EXECUTION'
            and protocol['scientific_decision'] == 'descriptive_no_success_threshold'
            and protocol['independent_joint_rng_blocks'] == 1,
            'registered single-block descriptive protocol required')
    require(sha256_file(baseline/'delivery.json') == protocol['baseline_delivery_sha256'],
            'registered baseline delivery changed')
    manifest = _source_manifest(read(baseline/'delivery.json'))
    base_path = _source_bound_file(baseline, manifest, 'source_run/config.yaml')
    require(sha256_file(base_path) == protocol['baseline_config_sha256'],
            'registered baseline configuration changed')
    base, registered, _, dimensions = _source_registration(base_path)
    case_id = protocol['baseline_case_id']
    require(case_id in registered, 'registered baseline case missing')
    original = next(row for row in base['cases'] if row['case_id'] == case_id)
    require(original['distractor_strength_nA'] == 0.
            and original['stimulus_strength_nA'] == .5
            and protocol['expected_coupled_units'] == dimensions['coupled_core_units'],
            'strong-cue zero-distractor comparator and declared scale required')
    strength = protocol['distractor_strength_nA']
    require(type(strength) in (int, float) and math.isfinite(strength) and strength > 0,
            'positive finite registered distractor amplitude required')
    expected_case = dict(original, case_id=protocol['candidate_case_id'],
                         distractor_strength_nA=strength)
    require(candidate['cases'] == [expected_case]
            and expected_case['case_id'] not in registered
            and Path(expected_case['case_id']).name == expected_case['case_id'],
            'exact registered candidate case required')
    omitted = {'cases', 'purpose', 'distractor_response'}
    require({k:v for k,v in candidate.items() if k not in omitted}
            == {k:v for k,v in base.items() if k not in omitted},
            'only registered distractor amplitude and identifying metadata may change')
    require(base.get('random_stream_policy', 'source_global') == 'source_global'
            and base['stimulus_center_deg'] == 120. and base['stimulus_width_deg'] == 30.
            and base['distractor_center_deg'] == 90. and base['distractor_width_deg'] == 40.
            and base['distractor_start_ms'] == 400. and base['distractor_duration_ms'] == 50.
            and base['readout_bin_ms'] == 5.,
            'registered source geometry, clocks and joint RNG policy required')
    status, header = _source_header(baseline, base, registered, sha256_file(base_path))
    require(header is not None and status['status'] == 'COMPLETE'
            and status['case_id'] == case_id, 'completed registered baseline required')
    return dict(candidate=candidate, protocol=protocol, candidate_sha256=sha256_file(config_path),
        candidate_registry={expected_case['case_id']:(expected_case['source_rng_seed'], 'strong_cue')},
        baseline=base, baseline_registry=registered, baseline_sha256=sha256_file(base_path),
        dimensions=dimensions, baseline_header=header)


def observed_arrays(header):
    path = _source_bound_file(header['archive'], header['manifest'],
        'source_run/applied_current_observation.npz',
        header['report']['outputs']['applied_current_observation.npz'])
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def load_baseline(registered):
    raw, common, paired = _source_values(registered['baseline_header'], registered['baseline'],
        registered['baseline_registry'], registered['dimensions'], registered['baseline_sha256'])
    require(registered['baseline_header']['binding']['applied_current_observation']['status']
            == 'VERIFIED_ACTUAL_CURRENT', 'observed baseline currents required')
    return raw, common, paired, observed_arrays(registered['baseline_header'])


def verify_pair(base_raw, candidate_raw, base_common, candidate_common,
                base_paired, candidate_paired, base_current, candidate_current):
    require(base_common == candidate_common, 'paired execution program or backend changed')
    require(base_paired['inputs_used'] == candidate_paired['inputs_used'],
            'paired native input consumption differs')
    require(np.array_equal(base_paired['initial'], candidate_paired['initial']),
            'paired initial voltage differs')
    a, b = base_paired['final_rng'], candidate_paired['final_rng']
    require(set(a) == set(b) and all(a[k].dtype == b[k].dtype
            and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in a),
            'paired final RNG differs')
    for key in set(base_current)-{'current_A'}:
        require(np.array_equal(base_current[key], candidate_current[key]),
                'paired source current clocks or unit identities differ')
    changed = np.flatnonzero(np.any(base_current['current_A'] != candidate_current['current_A'], axis=1))
    require(len(changed) > 0 and np.array_equal(changed, np.arange(changed[0], changed[-1]+1))
            and candidate_current['distractor_active'][changed].all()
            and not candidate_current['cue_active'][changed].any(),
            'one nonoverlapping observed distractor interval required')
    start = int(candidate_current['segment_start_ticks'][changed[0]])
    stop = int(candidate_current['segment_stop_ticks'][changed[-1]])
    require(changed[-1]+1 < len(candidate_current['segment_start_ticks'])
            and not np.any(candidate_current['current_A'][changed[-1]+1:])
            and not np.any(base_current['current_A'][changed[-1]+1:]),
            'observed task current removal before recovery is required')
    masks = [raw['spike_ticks'] < start for raw in (base_raw, candidate_raw)]
    for key in ('spike_ticks', 'spike_unit_indices', 'spike_seconds'):
        require(np.array_equal(base_raw[key][masks[0]], candidate_raw[key][masks[1]]),
                'paired pre-distractor native events differ: '+key)
    return dict(status='PASS',first_different_current_tick=start,current_return_to_baseline_tick=stop,
        exact_pre_distractor_events=int(masks[0].sum()),exact_initial_voltage_values=len(base_paired['initial']),
        final_rng_arrays_compared=len(a),source_driver_and_consumption_identical=True,
        full_external_stream_arrays_compared=False,
        background_pairing_basis='same verified deterministic driver, NumPy version, seed, schedule, dimensions and consumption; full streams not archived')


def region_rows(raw, config, dimensions, observation, first_different_tick, removal_tick):
    n_e, n_i = dimensions['n_e'], dimensions['n_i']
    rows = temporal_rows(raw['spike_ticks'], raw['spike_unit_indices'], n_e=n_e,n_i=n_i,
        dt_ms=config['dt_ms'],bin_ms=config['readout_bin_ms'],duration_ms=config['duration_ms'])
    cue, cue_geometry = geometry(n_e,config['stimulus_center_deg'],config['stimulus_width_deg'])
    distractor, distractor_geometry = geometry(n_e,config['distractor_center_deg'],config['distractor_width_deg'])
    regions = dict(cue_only=cue-distractor,distractor_only=distractor-cue,
                   overlap=cue & distractor,neither=set(range(n_e))-(cue|distractor))
    require(all(regions.values()), 'nonempty disjoint source preference regions required')
    indices = {key:np.array(sorted(value),dtype=np.int64) for key,value in regions.items()}
    distractor_ids = np.array(sorted(distractor),dtype=np.int64)
    transitions = observation['segment_start_ticks'][1:][
        np.any(np.diff(observation['current_A'],axis=0) != 0,axis=1)]
    for row in rows:
        start,stop = np.searchsorted(raw['spike_ticks'],[row['start_tick'],row['stop_tick']])
        ids = raw['spike_unit_indices'][start:stop]
        counts = np.bincount(ids[ids<n_e],minlength=n_e)
        seconds = (row['stop_ms']-row['start_ms'])*.001
        for key,units in indices.items():
            row[key+'_spikes'] = int(counts[units].sum())
            row[key+'_rate_hz'] = row[key+'_spikes']/len(units)/seconds
        require(sum(row[key+'_spikes'] for key in regions) == row['excitatory_spikes'],
                'disjoint regions must preserve every E event')
        inside = int(counts[distractor_ids].sum())
        row['distractor_region_rate_hz'] = inside/len(distractor)/seconds
        row['nondistractor_region_rate_hz'] = (int(counts.sum())-inside)/(n_e-len(distractor))/seconds
        row['distractor_selectivity_hz'] = row['distractor_region_rate_hz']-row['nondistractor_region_rate_hz']
        angle = row['decoded_angle_deg']
        row['circular_distractor_error_deg'] = None if angle is None else (
            angle-config['distractor_center_deg']+180)%360-180
        row['different_task_current_exposure_ms'] = max(0,min(row['stop_tick'],removal_tick)
                                                       -max(row['start_tick'],first_different_tick))*config['dt_ms']
        row['source_current_transition_within_bin'] = bool(np.any(
            (transitions>row['start_tick']) & (transitions<row['stop_tick'])))
    geometry_info = dict(disjoint_E_unit_counts={key:len(value) for key,value in regions.items()},
        cue_region_units=len(cue),distractor_region_units=len(distractor),
        cue_geometry=cue_geometry,distractor_geometry=distractor_geometry,
        coordinates='source preference angles, not soma positions',
        equal_amplitudes_imply_equal_total_injected_current=False)
    return rows, geometry_info


def differences(base, candidate):
    require(len(base)==len(candidate), 'paired full temporal coverage required')
    excluded = {'bin_index','start_ms','stop_ms','start_tick','stop_tick','cue_units',
                'direction_defined','decoded_angle_deg','circular_cue_error_deg','circular_distractor_error_deg',
                'different_task_current_exposure_ms','source_current_transition_within_bin'}
    result = []
    for a,b in zip(base,candidate):
        require(all(a[k]==b[k] for k in ('bin_index','start_tick','stop_tick')), 'paired bin clocks differ')
        row = {k:b[k] for k in ('bin_index','start_ms','stop_ms','start_tick','stop_tick',
            'different_task_current_exposure_ms','source_current_transition_within_bin')}
        for key in (key for key in a if key not in excluded):
            row[key+'_difference'] = None if a[key] is None or b[key] is None else b[key]-a[key]
        for key in ('decoded_angle_deg','circular_cue_error_deg','circular_distractor_error_deg'):
            row[key+'_circular_change'] = None if a[key] is None or b[key] is None else (b[key]-a[key]+180)%360-180
        row.update(baseline_direction_defined=a['direction_defined'],candidate_direction_defined=b['direction_defined'])
        result.append(row)
    return result


def run(config_path, baseline, output, *, candidate=None, check_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(status='VALIDATING',scientific_fidelity_qualified=False,working_memory_qualified=False,
        production_episodes=0,independent_joint_rng_blocks=1,confidence_intervals=False,p_values=False,
        implementation_sha256=sha256_file(Path(__file__)))
    try:
        registered=registration(config_path,baseline)
        base_raw,base_common,base_paired,base_current=load_baseline(registered)
        report.update(config_sha256=registered['candidate_sha256'],protocol=registered['protocol'],
            coupled_core_units=registered['dimensions']['coupled_core_units'],
            baseline_binding=registered['baseline_header']['binding'])
        if check_only:
            require(candidate is None, 'preflight cannot consume a candidate outcome')
            report.update(status='READY_SOURCE_DISTRACTOR_PROTOCOL',new_source_simulations=0)
        else:
            require(candidate is not None, 'completed candidate source archive required')
            status,header=_source_header(candidate,registered['candidate'],registered['candidate_registry'],
                                         registered['candidate_sha256'])
            require(header is not None and status['status']=='COMPLETE','completed registered candidate required')
            raw,common,paired=_source_values(header,registered['candidate'],registered['candidate_registry'],
                                             registered['dimensions'],registered['candidate_sha256'])
            require(header['binding']['applied_current_observation']['status']=='VERIFIED_ACTUAL_CURRENT',
                    'observed candidate currents required')
            current=observed_arrays(header)
            pairing=verify_pair(base_raw,raw,base_common,common,base_paired,paired,base_current,current)
            first,removal=pairing['first_different_current_tick'],pairing['current_return_to_baseline_tick']
            tables={}
            for arm,values,observation in (('baseline',base_raw,base_current),('candidate',raw,current)):
                rows,geometry_info=region_rows(values,registered['candidate'],registered['dimensions'],
                                               observation,first,removal)
                tables[arm]=rows
            tables['paired_difference']=differences(tables['baseline'],tables['candidate'])
            report['artifacts']={}
            for name,rows in tables.items():
                path=output/(name+'.csv');write_rows(path,rows)
                report['artifacts'][path.name]=sha256_file(path)
            report.update(status='COMPLETE_PAIRED_DISTRACTOR_RESPONSE',candidate_binding=header['binding'],
                pairing=pairing,source_geometry=geometry_info,complete_source_episodes_reused=2,
                baseline_events=len(base_raw['spike_ticks']),candidate_events=len(raw['spike_ticks']),
                bins_per_arm=len(tables['baseline']),bin_ms=registered['candidate']['readout_bin_ms'],
                actual_distractor_interval_ms=[first*registered['candidate']['dt_ms'],removal*registered['candidate']['dt_ms']],
                post_removal_summary_scope='all complete5ms bins entirely after observed task-current removal; no response-based window selection',
                post_removal_summaries={})
            for arm in ('baseline','candidate'):
                rows=tables[arm];removal_ms=removal*registered['candidate']['dt_ms']
                report['post_removal_summaries'][arm]=dict(delay_summary(rows,removal_ms),
                    first_complete_bin_start_ms=next(row['start_ms'] for row in rows if row['start_ms']>=removal_ms))
    except Exception as error:
        report.update(status='FAILED',error_type=type(error).__name__,error=str(error));raise
    finally:
        write_json(output/'report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','baseline','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    run(args.config,args.baseline,args.output,candidate=args.candidate,check_only=args.check_only)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
