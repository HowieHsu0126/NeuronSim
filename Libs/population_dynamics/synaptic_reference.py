"""Check reference kernels against exponential PSC and refractory semantics.

This is a single-cell source qualification probe, not a network experiment or a
qualification of the generative model. It intentionally reports a nonzero exit
when necessary continuous-PSC conditions fail. The published MOD is preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

CLOCK_STEPS_MS = (0.1, 0.025, 0.00625)
SOLVER_STEP_MS = 0.00025
REFRACTORY_MS = 2.0
SYNAPTIC_TAU_MS = 0.5
MEMBRANE_TAU_MS = 10.0
CAPACITANCE_PF = 250.0
RESET_MV = -65.0
INITIAL_CURRENT_PA = 1000.0
PULSE_TIME_MS = 1.0
PULSE_WEIGHTS_PA = (0.0, 200.0)
READOUT_TIMES_MS = (2.001, 2.501)


def analytic_after_refractory(time_ms: float, pulse_pa: float) -> tuple[float, float]:
    """Continuous exponential PSC and subthreshold LIF after a spike at t=0.

Voltage is clamped at reset during [0, t_ref]. Synaptic current continues to
follow dI/dt=-I/tau_s, including an impulse at PULSE_TIME_MS. With E_L=V_reset,
C*dV/dt=-C*(V-V_reset)/tau_m+I after release. Units: ms, pA, mV, pF.
"""
    if not math.isfinite(time_ms) or time_ms < REFRACTORY_MS or not math.isfinite(pulse_pa):
        raise ValueError('finite post-refractory time and pulse weight required')
    release_current = (
        INITIAL_CURRENT_PA * math.exp(-REFRACTORY_MS / SYNAPTIC_TAU_MS)
        + pulse_pa * math.exp(-(REFRACTORY_MS - PULSE_TIME_MS) / SYNAPTIC_TAU_MS)
    )
    elapsed = time_ms - REFRACTORY_MS
    current = release_current * math.exp(-elapsed / SYNAPTIC_TAU_MS)
    voltage = RESET_MV + release_current / CAPACITANCE_PF * (
        math.exp(-elapsed / MEMBRANE_TAU_MS) - math.exp(-elapsed / SYNAPTIC_TAU_MS)
    ) / (1.0 / SYNAPTIC_TAU_MS - 1.0 / MEMBRANE_TAU_MS)
    return current, voltage


def probe(source_dir: Path, mechanisms_dir: Path) -> dict[str, Any]:
    """Run in a fresh Python process with the original, already built source."""
    original = (source_dir / 'IntFire_PD.mod').read_bytes()
    if original != (mechanisms_dir / 'IntFire_PD.mod').read_bytes():
        raise ValueError('compiled-input MOD does not match the recorded original source')
    # Loading simulator extensions is confined to this explicit validation CLI.
    from neuron import h, load_mechanisms
    if not load_mechanisms(str(mechanisms_dir)):
        raise RuntimeError('compiled IntFire_PD mechanism could not be loaded')
    h.load_file('stdrun.hoc')
    h.tausyn_IntFire_PD = SYNAPTIC_TAU_MS
    h.tauref_IntFire_PD = REFRACTORY_MS
    h.Cm_IntFire_PD = CAPACITANCE_PF
    h.taum_IntFire_PD = MEMBRANE_TAU_MS
    h.Vreset_IntFire_PD = RESET_MV
    h.Vteta_IntFire_PD = -50.0
    rows = []
    for clock_step in CLOCK_STEPS_MS:
        for pulse_weight in PULSE_WEIGHTS_PA:
            cell = h.IntFire_PD()
            cell.Iext = 0.0
            clock = h.NetCon(None, cell)
            clock.weight[0] = 0.0
            pulse = h.NetCon(None, cell)
            pulse.weight[0] = pulse_weight
            spikes = h.Vector()
            detector = h.NetCon(cell, None)
            detector.record(spikes)
            h.CVode().active(0)
            h.dt = SOLVER_STEP_MS
            h.finitialize(RESET_MV)
            # A known spike at zero isolates refractory handling from threshold
            # localization error and from stochastic network dynamics.
            cell.m = -49.0
            cell.i = INITIAL_CURRENT_PA
            for k in range(round(3.0 / clock_step) + 1):
                clock.event(k * clock_step)
            if pulse_weight:
                pulse.event(PULSE_TIME_MS)
            for requested_time in READOUT_TIMES_MS:
                h.continuerun(requested_time)
                actual_time = float(h.t)
                if abs(actual_time - requested_time) > SOLVER_STEP_MS / 2:
                    raise RuntimeError('recorded time does not match the requested probe time')
                if list(spikes) != [0.0]:
                    raise RuntimeError('probe requires exactly the imposed spike at zero')
                if requested_time == READOUT_TIMES_MS[0] and float(cell.m) != RESET_MV:
                    raise RuntimeError('first readout is not after refractory reset')
                reference_i, reference_v = analytic_after_refractory(actual_time, pulse_weight)
                rows.append({
                    'clock_step_ms': clock_step, 'pulse_weight_pa': pulse_weight,
                    'pulse_time_ms': PULSE_TIME_MS, 'requested_time_ms': requested_time,
                    'actual_time_ms': actual_time, 'spike_times_ms': list(spikes),
                    'port_current_pa': float(cell.i), 'port_voltage_mv': float(cell.m),
                    'continuous_current_pa': reference_i, 'continuous_voltage_mv': reference_v,
                    'current_error_pa': float(cell.i) - reference_i,
                    'voltage_error_mv': float(cell.m) - reference_v,
                })
    pairs = []
    for step in CLOCK_STEPS_MS:
        for readout in READOUT_TIMES_MS:
            selected = [r for r in rows if r['clock_step_ms'] == step and r['requested_time_ms'] == readout]
            baseline, stimulated = selected
            pairs.append({
                'clock_step_ms': step, 'requested_time_ms': readout,
                'port_pulse_current_effect_pa': stimulated['port_current_pa'] - baseline['port_current_pa'],
                'continuous_pulse_current_effect_pa': stimulated['continuous_current_pa'] - baseline['continuous_current_pa'],
                'port_pulse_voltage_effect_mv': stimulated['port_voltage_mv'] - baseline['port_voltage_mv'],
                'continuous_pulse_voltage_effect_mv': stimulated['continuous_voltage_mv'] - baseline['continuous_voltage_mv'],
            })
    # Strict necessary properties, not calibrated network-fidelity thresholds.
    # Their failure cannot be repaired by a smaller background-clock step.
    decays = all(0.0 < r['port_current_pa'] < INITIAL_CURRENT_PA for r in rows
                 if r['requested_time_ms'] == READOUT_TIMES_MS[0] and r['pulse_weight_pa'] == 0.0)
    retains_input = all(p['port_pulse_current_effect_pa'] > 0.0 for p in pairs)
    conditions = {'current_decays_during_refractory': decays,
                  'refractory_input_retained_after_release': retains_input}
    return {
        'execution_status': 'COMPLETED',
        'qualification_status': 'NECESSARY_CONDITIONS_PASSED' if all(conditions.values()) else 'SOURCE_SEMANTIC_MISMATCH',
        'scope': 'ModelDB 266872 IntFire_PD versus continuous exponential PSC LIF refractory semantics',
        'network_fidelity_qualified': False,
        'necessary_conditions': conditions,
        'source': json.loads((source_dir / 'source.json').read_text()),
        'original_mod_sha256': hashlib.sha256(original).hexdigest(),
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'neuron': h.nrnversion(0),
        'solver_step_ms': SOLVER_STEP_MS,
        'analytical_equations': 'dI/dt=-I/tau_s; I(1+)=I(1-)+w; V=V_reset while refractory; C*dV/dt=-C*(V-V_reset)/tau_m+I after release',
        'independent_source': 'https://github.com/nest/nest-simulator/blob/v3.8/models/iaf_psc_exp.cpp',
        'sampling_note': 'Actual h.t is used for analytical values. First readout follows reset; event-stored port states are unchanged between clocks.',
        'rows': rows, 'paired_pulse_effects': pairs,
    }


def probe_nest() -> dict[str, Any]:
    """Check the native kernel with delivered impulses during refractoriness.

All inputs land on the simulation grid. This analytically soluble unit test uses
absolute roundoff tolerances; it does not set network-fidelity thresholds.
"""
    from importlib.metadata import version
    import nest
    nest.verbosity = nest.VerbosityLevel.ERROR
    rows, cases = [], []
    current_tolerance_pa, voltage_tolerance_mv = 1e-9, 1e-10
    for step in (0.1, 0.05, 0.025):
        for pulse in (-200.0, 0.0, 200.0):
            nest.ResetKernel()
            nest.SetKernelStatus({'local_num_threads': 1, 'resolution': step, 'rng_seed': 1001})
            parameters = {
                'V_m': -49.0, 'E_L': RESET_MV, 'V_reset': RESET_MV, 'V_th': -50.0,
                'C_m': CAPACITANCE_PF, 'tau_m': MEMBRANE_TAU_MS,
                'tau_syn_ex': SYNAPTIC_TAU_MS, 'tau_syn_in': SYNAPTIC_TAU_MS,
                't_ref': REFRACTORY_MS, 'I_e': 0.0,
            }
            neuron = nest.Create('iaf_psc_exp', params=parameters)
            meter = nest.Create('multimeter', params={
                'record_from': ['V_m', 'I_syn_ex', 'I_syn_in'], 'interval': step})
            nest.Connect(meter, neuron)
            recorder = nest.Create('spike_recorder')
            nest.Connect(neuron, recorder)
            initial = nest.Create('spike_generator', params={'spike_times': [0.4]})
            nest.Connect(initial, neuron, syn_spec={'weight': INITIAL_CURRENT_PA, 'delay': 0.1})
            if pulse:
                extra = nest.Create('spike_generator', params={'spike_times': [0.9]})
                nest.Connect(extra, neuron, syn_spec={'weight': pulse, 'delay': 0.1})
            nest.Simulate(3.2)
            events = meter.get('events')
            spikes = recorder.get('events')['times'].tolist()
            if len(spikes) != 1 or abs(spikes[0] - step) > 1e-12:
                raise RuntimeError('NEST probe requires the imposed first-step spike only')
            release = spikes[0] + REFRACTORY_MS
            case_rows = []
            for t, i_ex, i_in, v in zip(events['times'], events['I_syn_ex'], events['I_syn_in'], events['V_m']):
                if t <= release + 1e-9:
                    continue
                # Generator events arrive at 0.5 and 1.0 ms after their 0.1-ms
                # synaptic delays. Both arrivals occur during the known refractory interval.
                reference_i = INITIAL_CURRENT_PA * math.exp(-(t - 0.5) / SYNAPTIC_TAU_MS) + pulse * math.exp(-(t - 1.0) / SYNAPTIC_TAU_MS)
                release_i = INITIAL_CURRENT_PA * math.exp(-(release - 0.5) / SYNAPTIC_TAU_MS) + pulse * math.exp(-(release - 1.0) / SYNAPTIC_TAU_MS)
                elapsed = t - release
                reference_v = RESET_MV + release_i / CAPACITANCE_PF * (
                    math.exp(-elapsed / MEMBRANE_TAU_MS) - math.exp(-elapsed / SYNAPTIC_TAU_MS)
                ) / (1.0 / SYNAPTIC_TAU_MS - 1.0 / MEMBRANE_TAU_MS)
                case_rows.append({
                    'resolution_ms': step, 'refractory_pulse_pa': pulse, 'time_ms': float(t),
                    'spike_time_ms': spikes[0], 'refractory_release_ms': release,
                    'native_excitatory_current_pa': float(i_ex), 'native_inhibitory_current_pa': float(i_in),
                    'native_current_pa': float(i_ex + i_in), 'reference_current_pa': reference_i,
                    'native_voltage_mv': float(v), 'reference_voltage_mv': reference_v,
                    'current_error_pa': float(i_ex + i_in) - reference_i,
                    'voltage_error_mv': float(v) - reference_v,
                })
            if not case_rows:
                raise RuntimeError('NEST post-refractory recording is empty')
            current_error = max(abs(row['current_error_pa']) for row in case_rows)
            voltage_error = max(abs(row['voltage_error_mv']) for row in case_rows)
            cases.append({'resolution_ms': step, 'refractory_pulse_pa': pulse,
                          'post_release_samples': len(case_rows),
                          'max_current_error_pa': current_error, 'max_voltage_error_mv': voltage_error,
                          'passed': current_error <= current_tolerance_pa and voltage_error <= voltage_tolerance_mv})
            rows.extend(case_rows)
    return {
        'execution_status': 'COMPLETED',
        'qualification_status': 'NATIVE_KERNEL_ANALYTIC_CHECK_PASSED' if all(c['passed'] for c in cases) else 'NATIVE_KERNEL_ANALYTIC_CHECK_FAILED',
        'scope': 'NEST iaf_psc_exp refractory PSC, including signed impulses and post-release voltage',
        'network_fidelity_qualified': False,
        'necessary_conditions': {'analytic_psc_and_voltage': all(c['passed'] for c in cases)},
        'absolute_tolerances': {'current_pa': current_tolerance_pa, 'voltage_mv': voltage_tolerance_mv},
        'tolerance_basis': 'Roundoff tolerance for exact exponential propagation with grid-aligned inputs; not an empirical network-fidelity threshold.',
        'parameters': parameters, 'native_model': 'iaf_psc_exp', 'nest': nest.__version__,
        'python': platform.python_version(),
        'dependencies': {name: version(name) for name in ('nest-simulator', 'numpy', 'cython', 'matplotlib')},
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stimulus_semantics': {'initial_impulse_pa': INITIAL_CURRENT_PA, 'initial_arrival_ms': 0.5,
                               'extra_arrival_ms': 1.0, 'connection_delay_ms': 0.1},
        'rows': rows, 'cases': cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('modeldb', 'nest'), default='modeldb')
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--mechanisms-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.backend == 'modeldb':
        if args.source_dir is None or args.mechanisms_dir is None:
            parser.error('--source-dir and --mechanisms-dir are required for modeldb')
        result = probe(args.source_dir.resolve(), args.mechanisms_dir.resolve())
    else:
        result = probe_nest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'qualification_status': result['qualification_status']}))
    if not all(result['necessary_conditions'].values()):
        sys.exit(2)


if __name__ == '__main__':
    main()
