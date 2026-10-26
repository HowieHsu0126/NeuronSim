"""Source-projected LIF conditional-mean field comparator.

The diffusion and fast-synapse approximations follow the Siegert/Fourcaud
transfer law (Fourcaud and Brunel, 2002, DOI:10.1162/089976602320264015).
This implementation uses SciPy quadrature; it does not copy an author solver.
It emits expected aggregate counts. It provides no identified unit samples or
joint fluctuation law. Source graph projection does not qualify these closures.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import yaml
from scipy.integrate import quad
from scipy.special import erfcx, roots_legendre, zetac

from .provenance import sha256_file
from .mesocircuit_continuation import prepare_pulse_protocol
from .spatial_connectivity import EDGE_DTYPE, SCHEMA, native_delay_ticks
from .spatial_field import PeriodicPopulationGrid

QUADRATURE_ORDER = 32
QUADRATURE_RELATIVE_TOLERANCE = 1e-7
QUADRATURE_ABSOLUTE_TOLERANCE_HZ = 1e-8
ALPHA_VOLTAGE_SHIFT = float(np.sqrt(2) * abs(zetac(.5) + 1))
LIMITS = [
    "homogeneous rates within each source population/spatial cell",
    "independent recurrent afferent spike trains in diffusion input moments",
    "Poisson input moments omit recurrent refractory and cross-source covariance",
    "different-delay copies of one source train omit colored cross-delay covariance",
    "diffusion intensity uses instantaneous delayed arrival rates; it is not filtered current variance",
    "fast exponential synapse threshold/reset shift is an approximation",
    "observed binned history approximates synaptic current and delay state",
    "conditional mean only; no unit samples or distribution scores",
]


@lru_cache(maxsize=4)
def _quadrature_nodes(order):
    return roots_legendre(order)


def lif_rate(mu_mv, sigma_mv, *, reset_mv, threshold_mv, tau_m_seconds,
             refractory_seconds, tau_syn_seconds, order=QUADRATURE_ORDER):
    """Stable Gauss-Legendre evaluation of the shifted Siegert integral.

    Voltages are relative to E_L. sigma² is the white diffusion parameter,
    whose stationary membrane variance is sigma²/2. The exponential rescaling
    avoids overflow for subthreshold inputs; zero diffusion uses the exact
    deterministic LIF stationary rate.
    """
    mu, sigma = np.broadcast_arrays(np.asarray(mu_mv, dtype=float), np.asarray(sigma_mv, dtype=float))
    if (not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma < 0)
            or not np.isfinite([reset_mv, threshold_mv, tau_m_seconds, refractory_seconds, tau_syn_seconds]).all()
            or reset_mv >= threshold_mv or min(tau_m_seconds, refractory_seconds) <= 0 or tau_syn_seconds < 0):
        raise ValueError("finite LIF inputs, ordered voltages and positive time constants required")
    result = np.zeros(mu.shape)
    zero = sigma == 0
    firing = zero & (mu > threshold_mv)
    result[firing] = 1 / (refractory_seconds + tau_m_seconds * np.log(
        (mu[firing] - reset_mv) / (mu[firing] - threshold_mv)))
    noisy = ~zero
    if noisy.any():
        shift = ALPHA_VOLTAGE_SHIFT / 2 * np.sqrt(tau_syn_seconds / tau_m_seconds)
        lower = (reset_mv - mu[noisy]) / sigma[noisy] + shift
        upper = (threshold_mv - mu[noisy]) / sigma[noisy] + shift
        # A lower bound on the positive-end integral bounds the firing rate.
        # Return zero only when that upper rate bound is below the declared
        # absolute tolerance. This avoids an underflowing 0/0 for tiny noise.
        span = upper - lower
        width = np.minimum(span, 1 / (2 * np.maximum(upper, 1)))
        positive_end = upper - width
        with np.errstate(over='ignore'):
            log_bound = -positive_end**2 - np.log(tau_m_seconds*np.sqrt(np.pi)*width)
        negligible = (positive_end > 0) & (log_bound < np.log(QUADRATURE_ABSOLUTE_TOLERANCE_HZ))
        values = np.zeros(lower.shape)
        broad = ~negligible & (span > 64)
        for index in np.flatnonzero(broad):
            a, b = lower[index], upper[index]
            # x=-sinh(v) resolves a long negative tail on logarithmic scale.
            integral = 0.
            if a < 0:
                lo = np.arcsinh(max(-b, 0)); hi = np.arcsinh(-a)
                integral += quad(lambda v: erfcx(np.sinh(v))*np.cosh(v), lo, hi,
                                 epsabs=1e-12, epsrel=1e-11)[0]
            if b > 0:
                integral += quad(lambda x: erfcx(-x), max(a, 0), b,
                                 epsabs=1e-12, epsrel=1e-11)[0]
            values[index] = 1 / (refractory_seconds + tau_m_seconds*np.sqrt(np.pi)*integral)
        regular = ~(negligible | broad)
        lower, upper = lower[regular], upper[regular]
        roots, weights = _quadrature_nodes(order)
        x = (lower[:, None] + upper[:, None]) / 2 + (upper-lower)[:, None] * roots / 2
        scale = np.square(np.maximum(upper, 0))
        # erfcx(-x) = 2 exp(x²) - erfcx(x) for x > 0.
        integrand = np.empty_like(x)
        positive = x > 0
        scaled = np.broadcast_to(scale[:, None], x.shape)
        integrand[positive] = 2*np.exp(x[positive]**2-scaled[positive]) - erfcx(x[positive])*np.exp(-scaled[positive])
        integrand[~positive] = erfcx(-x[~positive])*np.exp(-scaled[~positive])
        integral = (upper-lower)/2 * (integrand @ weights)
        values[regular] = np.exp(-scale) / (refractory_seconds*np.exp(-scale) + tau_m_seconds*np.sqrt(np.pi)*integral)
        result[noisy] = values
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1/refractory_seconds):
        raise FloatingPointError("invalid LIF transfer rate")
    return result


def projected_edge_moments(edges, ids, groups, group_sizes, tc_mask, dt_ms, *, diagnostics=None):
    """Aggregate native float32 weights and exact delay ticks from one shard.

    Sum weights for each target/source/delay before squaring. Different-delay
    copies retain every first moment, but their colored cross-delay covariance
    is omitted by the independent-arrival diffusion approximation.
    """
    if edges.dtype != EDGE_DTYPE or edges.ndim != 1:
        raise ValueError("source graph edge schema required")
    source = np.searchsorted(ids, edges['source']); target = np.searchsorted(ids, edges['target'])
    if (np.any(source >= len(ids)) or np.any(target >= len(ids))
            or not np.array_equal(ids[source], edges['source']) or not np.array_equal(ids[target], edges['target'])
            or np.any(groups[target] < 0) or not np.isfinite(edges['weight']).all()):
        raise ValueError("source/target identity or cortical target invalid")
    delay = native_delay_ticks(edges['raw_delay'], dt_ms)
    weight = edges['weight'].astype(np.float32).astype(np.float64)
    if not len(edges):
        return {}
    order = np.lexsort((delay, source, target))
    pair_change = (np.diff(source[order]) != 0) | (np.diff(target[order]) != 0)
    starts = np.r_[0, np.flatnonzero(pair_change | (np.diff(delay[order]) != 0))+1]
    pair_starts = np.r_[0, np.flatnonzero(pair_change)+1]
    pair_ends = np.r_[pair_starts[1:], len(edges)]
    multi = np.minimum.reduceat(delay[order], pair_starts) != np.maximum.reduceat(delay[order], pair_starts)
    if diagnostics is not None:
        for key, value in [('cross_delay_source_target_pairs', int(multi.sum())),
                           ('cross_delay_edges', int((pair_ends-pair_starts)[multi].sum()))]:
            diagnostics[key] = diagnostics.get(key, 0) + value
    if not np.isfinite(weight).all():raise ValueError('native weight overflows float32')
    summed = np.add.reduceat(weight[order], starts)
    selected_source, selected_target = source[order[starts]], target[order[starts]]
    selected_delay = delay[order[starts]]
    result = {}
    n = len(group_sizes)
    for d in np.unique(selected_delay):
        use = selected_delay == d
        s, t, w = selected_source[use], selected_target[use], summed[use]
        cortical = groups[s] >= 0
        key = groups[t[cortical]]*n + groups[s[cortical]]
        first = np.bincount(key, weights=w[cortical], minlength=n*n).reshape(n,n) / group_sizes[:,None]
        second = np.bincount(key, weights=w[cortical]**2, minlength=n*n).reshape(n,n) / group_sizes[:,None]
        active_tc = ~cortical & tc_mask[s]
        tc = np.bincount(groups[t[active_tc]], weights=w[active_tc], minlength=n) / group_sizes
        result[int(d)] = (first, second, tc)
    return result


def build_projection(graph: Path, field_view: Path, episode: Path, output: Path):
    """Read declared shards once; bind graph and the existing observation map."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = json.loads((graph/'manifest.json').read_text())
    if (manifest['schema'] != SCHEMA or manifest['status'] != 'GENERATED'
            or manifest['external_device_edges_included'] is not False):
        raise ValueError("complete source projection graph required")
    if sha256_file(graph/'layout.npz') != manifest['layout_sha256']:
        raise ValueError("source layout changed")
    with np.load(graph/'layout.npz', allow_pickle=False) as a:
        ids, labels, xy = (a[k].copy() for k in ('unit_ids','populations','positions_mm'))
    net = manifest['resolved_source_network']; dt = manifest['resolved_simulation']['sim_resolution']
    if (np.any(np.diff(ids) <= 0) or net['neuron_model'] != 'iaf_psc_exp'
            or net['dc_input'] or net['thalamic_input_type'] != 'pulses'):
        raise ValueError("source homogeneous LIF, pulse controls and increasing IDs required")
    field_report = json.loads((field_view/'summary.json').read_text())
    native = json.loads((episode/'report.json').read_text())
    if (native['status'] != 'EXECUTED' or native['graph_manifest_sha256'] != sha256_file(graph/'manifest.json')
            or field_report['episode_report_sha256'] != sha256_file(episode/'report.json')):
        raise ValueError("field and graph differ")
    if sha256_file(field_view/'spatial_fields.npz') != field_report['artifacts']['spatial_fields.npz']:
        raise ValueError("source field map changed")
    cortex = labels != 'TC'
    grid = PeriodicPopulationGrid.from_positions(ids[cortex], labels[cortex], xy[cortex], extent_mm=net['extent'],
        cells_per_axis=4, definition='source cortical soma positions; source population and periodic square top-hat cells')
    with np.load(field_view/'spatial_fields.npz',allow_pickle=False) as a:
        for key, value in [('source_unit_ids',ids[cortex]),('unit_field_index',grid.mapping.group_index),('field_ids',grid.mapping.field_ids)]:
            if not np.array_equal(a[key], value):raise ValueError('source observation map differs:'+key)
        sizes = a['total_unit_count'].copy()
    if (np.any(sizes <= 0) or not np.array_equal(sizes,np.bincount(grid.mapping.group_index,minlength=len(sizes)))):
        raise ValueError('exact nonempty source cell occupancy required')
    groups = np.full(len(ids), -1,dtype=np.int64);groups[cortex]=grid.mapping.group_index
    tc_mask = (~cortex) & (np.sum(xy**2,axis=1) <= net['th_radius']**2)
    moments, count, consumed, covariance_diagnostics = {}, 0, set(), {}
    last_target = {}
    for row in manifest['parts']:
        path = graph/row['path']
        if sha256_file(path) != row['sha256']:raise ValueError('source graph shard changed:'+row['path'])
        edges = np.load(path,mmap_mode='r',allow_pickle=False)
        if len(edges) != row['edges'] or row['path'] in consumed:raise ValueError('graph shard denominator differs')
        consumed.add(row['path']);count+=len(edges)
        if len(edges):
            projection=(row['source_population'],row['target_population'])
            if np.any(np.diff(edges['target'])<0) or int(edges['target'][0]) <= last_target.get(projection,-1):
                raise ValueError('a complete target must belong to one shard per projection')
            last_target[projection]=int(edges['target'][-1])
        for d, values in projected_edge_moments(edges,ids,groups,sizes,tc_mask,dt,diagnostics=covariance_diagnostics).items():
            if d not in moments:moments[d]=[np.zeros_like(v) for v in values]
            for dst, src in zip(moments[d],values):dst+=src
    if count != manifest['edges']:raise ValueError('complete graph edge count differs')
    delays=np.asarray(sorted(moments),dtype=np.int64)
    arrays=dict(delay_ticks=delays,mean_weights_pa=np.stack([moments[d][0] for d in delays]),
        squared_weights_pa2=np.stack([moments[d][1] for d in delays]),tc_impulses_pa=np.stack([moments[d][2] for d in delays]),
        field_ids=grid.mapping.field_ids,unit_ids=ids[cortex],unit_field_index=groups[cortex],unit_count=sizes,
        field_populations=grid.population_labels)
    np.savez(output/'projection.npz',**arrays)
    report=dict(status='PROJECTED_COMPLETE_SOURCE_GRAPH',graph_manifest_sha256=sha256_file(graph/'manifest.json'),
        field_summary_sha256=sha256_file(field_view/'summary.json'),source_edges=count,source_units=int(cortex.sum()),
        field_count=len(sizes),dt_ms=dt,network=net,parameters=net['neuron_params'],
        implementation_sha256=sha256_file(Path(__file__)),projection_sha256=sha256_file(output/'projection.npz'),
        seconds=time.perf_counter()-started,limits=LIMITS,covariance_diagnostics=covariance_diagnostics,scientific_claim_qualified=False)
    (output/'projection.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


class LIFNeuralField:
    """Advance a projected mean field with one source-scale relaxation time."""
    def __init__(self, projection: Path, relaxation_ms: float):
        self.report=json.loads((projection/'projection.json').read_text())
        if (self.report['status']!='PROJECTED_COMPLETE_SOURCE_GRAPH'
                or sha256_file(projection/'projection.npz')!=self.report['projection_sha256']):
            raise ValueError('bound completed graph projection required')
        with np.load(projection/'projection.npz',allow_pickle=False) as a:
            for key in a.files:setattr(self,key,a[key].copy())
        self.dt=self.report['dt_ms']/1000;self.relaxation=relaxation_ms/1000
        if not np.isfinite(self.relaxation) or self.relaxation<=0:raise ValueError('positive rate relaxation required')
        self.p=self.report['parameters'];self.net=self.report['network']
        if self.p['tau_syn_ex']!=self.p['tau_syn_in']:raise ValueError('equal source synaptic constants required by shifted transfer')
        self.ts=self.p['tau_syn_ex']/1000;self.tm=self.p['tau_m']/1000
        self.deadtime=(round(self.p['t_ref']/self.report['dt_ms'])+1)*self.dt
        names=self.net['populations'];lookup={name:i for i,name in enumerate(names)}
        self.bg_rate=np.asarray([self.net['bg_rate']*self.net['ext_indegrees'][lookup[str(name)]] for name in self.field_populations])
        self.dc=np.asarray([self.net['DC_amp'][lookup[str(name)]] for name in self.field_populations])
        self.bg_weight=float(np.float32(self.net['weight_ext']))
        # The existing native importer reports this two-tick generator offset.
        self.bg_delay=int(native_delay_ticks([self.net['delay_poisson']],self.report['dt_ms'])[0])+2
        self.mean_matrix=self.mean_weights_pa.transpose(1,0,2).reshape(len(self.unit_count),-1)
        self.square_matrix=self.squared_weights_pa2.transpose(1,0,2).reshape(len(self.unit_count),-1)

    def forecast(self, history_counts, pulse_ticks, *, boundary_tick, horizon_ticks, bin_ticks, check_quadrature=False):
        """Use only prefix counts and declared deterministic TC pulse times.

        History rates are piecewise constant within observed bins. The filter
        initialization integrates that complete prefix analytically. A rate at
        tick k belongs to [k,k+1); arrivals are applied before its transfer update.
        Known TC
        controls include their future arrivals; future response counts are absent.
        """
        h=np.asarray(history_counts);pulses=np.asarray(pulse_ticks)
        if (h.ndim!=2 or h.shape!=(boundary_tick//bin_ticks,len(self.unit_count))
                or boundary_tick%bin_ticks or horizon_ticks%bin_ticks or min(bin_ticks,horizon_ticks,boundary_tick)<1
                or not np.isfinite(h).all() or np.any(h<0) or np.any(h!=np.rint(h))
                or pulses.ndim!=1 or pulses.dtype.kind not in 'iu' or np.any(pulses<0) or np.any(np.diff(pulses)<=0)):
            raise ValueError('complete prefix integer counts and known ordered pulse ticks required')
        rates=h/(self.unit_count*(bin_ticks*self.dt))
        decay=np.exp(-self.dt/self.ts);relax=np.exp(-self.dt/self.relaxation)
        n=len(self.unit_count);current=np.zeros(n);tc_current=np.zeros(n)
        ends=np.arange(1,len(h)+1)*bin_ticks;starts=ends-bin_ticks
        for delay,w,tc in zip(self.delay_ticks,self.mean_weights_pa,self.tc_impulses_pa):
            begin=starts+delay;end=np.minimum(ends+delay,boundary_tick);valid=begin<end
            coefficients=self.ts*(np.exp(-(boundary_tick-end[valid])*self.dt/self.ts)-np.exp(-(boundary_tick-begin[valid])*self.dt/self.ts))
            current+=w@(coefficients@rates[valid])
            arrivals=pulses+delay;past=arrivals[arrivals<boundary_tick]
            tc_current+=tc*np.exp(-(boundary_tick-past)*self.dt/self.ts).sum()
        tc_updates=np.zeros((horizon_ticks,n))
        for delay,tc in zip(self.delay_ticks,self.tc_impulses_pa):
            offsets=pulses+delay-boundary_tick;offsets=offsets[(offsets>=0)&(offsets<horizon_ticks)]
            tc_updates[offsets]+=tc
        length=int(self.delay_ticks.max())+1
        # Buffer index equals absolute source emission tick modulo length.
        buffer=np.empty((length,n))
        for tick in range(boundary_tick-length,boundary_tick):
            buffer[tick%length]=0 if tick<0 else rates[tick//bin_ticks]
        rate=rates[-min(len(rates),max(1,round(.02/(bin_ticks*self.dt)))):].mean(0)
        predicted=np.zeros((horizon_ticks//bin_ticks,n));max_error=None;quadrature_checks=0
        for offset in range(horizon_ticks):
            tick=boundary_tick+offset
            delayed=buffer[(tick-self.delay_ticks)%length].reshape(-1)
            tc_current+=tc_updates[offset]
            elapsed=max(0,(tick-self.bg_delay)*self.dt)
            bg=self.bg_rate*self.bg_weight*self.ts*(-np.expm1(-elapsed/self.ts))
            mu=(current+tc_current+bg+self.dc)*self.p['tau_m']/self.p['C_m']
            intensity=self.square_matrix@delayed+self.bg_rate*self.bg_weight**2*(tick>=self.bg_delay)
            sigma=np.sqrt(intensity*self.tm)*(self.p['tau_syn_ex']/self.p['C_m'])
            args=dict(reset_mv=self.p['V_reset']-self.p['E_L'],threshold_mv=self.p['V_th']-self.p['E_L'],
                tau_m_seconds=self.tm,refractory_seconds=self.deadtime,tau_syn_seconds=self.ts)
            target=lif_rate(mu,sigma,**args)
            if check_quadrature and offset%max(1,horizon_ticks//32)==0:
                refined=lif_rate(mu,sigma,order=2*QUADRATURE_ORDER,**args)
                max_error=max(max_error or 0.,float(np.max(np.abs(refined-target))));quadrature_checks+=1
                if not np.allclose(target,refined,rtol=QUADRATURE_RELATIVE_TOLERANCE,atol=QUADRATURE_ABSOLUTE_TOLERANCE_HZ):
                    raise FloatingPointError('source-range transfer quadrature requires refinement')
            buffer[tick%length]=rate
            predicted[offset//bin_ticks]+=rate*self.unit_count*self.dt
            rate=rate*relax+target*(1-relax)
            current=current*decay+(self.mean_matrix@delayed)*self.ts*(1-decay)
            tc_current*=decay
        if not np.isfinite(predicted).all() or np.any(predicted<0):raise FloatingPointError('invalid expected field counts')
        return predicted,dict(transfer_quadrature_max_difference_hz=max_error,transfer_quadrature_checks=quadrature_checks,unit_outputs_available=False,
            distribution_scores_available=False,rate_bound_hz=1/self.deadtime)


def forecast_episode(projection, episode, fields, output, *, relaxation_ms, origin_ms=1000., horizon_ms=500., replay=False):
    """Execute one complete native-format prefix through the public comparator."""
    if output.exists() and not replay:raise FileExistsError(output)
    model=LIFNeuralField(projection,relaxation_ms)
    report=json.loads((episode/'report.json').read_text());view=json.loads((fields/'summary.json').read_text())
    if (report['status']!='EXECUTED' or report['graph_manifest_sha256']!=model.report['graph_manifest_sha256']
            or view['episode_report_sha256']!=sha256_file(episode/'report.json')
            or sha256_file(fields/'spatial_fields.npz')!=view['artifacts']['spatial_fields.npz']
            or report['dt_ms']!=model.report['dt_ms']):
        raise ValueError('native episode, source graph and observation binding required')
    with np.load(fields/'spatial_fields.npz',allow_pickle=False) as f:
        for key,expected in [('source_unit_ids',model.unit_ids),('unit_field_index',model.unit_field_index),('field_ids',model.field_ids),('total_unit_count',model.unit_count)]:
            if not np.array_equal(f[key],expected):raise ValueError('source field mapping changed:'+key)
        counts=f['count_sum'].copy();edges=f['bin_edges_ticks'].copy()
    if np.any(np.diff(edges)!=np.diff(edges)[0]) or edges[0]!=0:raise ValueError('complete aligned prefix field bins required')
    width=int(np.diff(edges)[0]);boundary=round(origin_ms/model.report['dt_ms']);horizon=round(horizon_ms/model.report['dt_ms'])
    if (boundary*model.report['dt_ms']!=origin_ms or horizon*model.report['dt_ms']!=horizon_ms
            or boundary+horizon>edges[-1] or len(counts)!=len(edges)-1
            or width*model.report['dt_ms']!=view.get('settings',{}).get('bin_ms',width*model.report['dt_ms'])):
        raise ValueError('native-clock forecast inside complete episode required')
    sim=dict(sim_resolution=report['dt_ms'],t_presim=0.,t_sim=report['duration_ms'])
    protocol=prepare_pulse_protocol(model.net,sim,report['config']['pulse_control'])
    pulse_ticks=np.asarray(protocol.effective_ticks,dtype=np.int64)+protocol.delay_ticks
    binding=dict(projection_sha256=model.report['projection_sha256'],episode_report_sha256=sha256_file(episode/'report.json'),
        field_summary_sha256=sha256_file(fields/'summary.json'),implementation_sha256=sha256_file(Path(__file__)),
        relaxation_ms=relaxation_ms,origin_ms=origin_ms,horizon_ms=horizon_ms,known_tc_ticks=pulse_ticks.tolist())
    started=time.perf_counter()
    predicted,diagnostic=model.forecast(counts[:boundary//width],pulse_ticks,boundary_tick=boundary,
        horizon_ticks=horizon,bin_ticks=width,check_quadrature=True)
    elapsed=time.perf_counter()-started
    truth=counts[boundary//width:(boundary+horizon)//width]
    hz=predicted/(model.unit_count*(width*model.dt));actual=truth/(model.unit_count*(width*model.dt))
    arrays=dict(expected_field_counts=predicted,expected_field_hz=hz,observed_field_counts=truth,
        field_ids=model.field_ids,unit_count=model.unit_count,known_tc_ticks=pulse_ticks)
    result=dict(status='EXECUTED_CONDITIONAL_MEAN_FIELD',binding=binding,forecast_seconds=elapsed,
        mean_field_rate_mse=float(np.mean((hz-actual)**2)),predicted_spikes_per_unit=float(predicted.sum()/model.unit_count.sum()),
        observed_spikes_per_unit=float(truth.sum()/model.unit_count.sum()),diagnostics=diagnostic,limits=LIMITS,
        core_neurons=len(model.unit_ids),field_count=len(model.field_ids),scientific_claim_qualified=False,
        independent_units='one native circuit/initial-noise realization',distribution_scores_available=False)
    if replay:
        original=json.loads((output/'forecast.json').read_text())
        if original['binding']!=binding or sha256_file(output/'forecast.npz')!=original['forecast_sha256']:
            raise ValueError('mean-field replay source or arrays changed')
        with np.load(output/'forecast.npz',allow_pickle=False) as f:
            if set(f.files)!=set(arrays):raise ValueError('mean-field output schema changed')
            for key,value in arrays.items():
                if f[key].dtype!=value.dtype or not np.array_equal(f[key],value):raise ValueError('mean-field replay differs:'+key)
        for key in ('mean_field_rate_mse','predicted_spikes_per_unit','observed_spikes_per_unit'):
            if result[key]!=original[key]:raise ValueError('mean-field score replay differs:'+key)
        result['status']='PASS_EXACT_MEAN_FIELD_FRESH_PROCESS_REPLAY'
        with (output/'replay.json').open('x') as stream:json.dump(result,stream,indent=2)
    else:
        output.mkdir(parents=True);np.savez(output/'forecast.npz',**arrays)
        result['forecast_sha256']=sha256_file(output/'forecast.npz')
        (output/'forecast.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate_campaign_config(config):
    """Keep the deterministic development comparator on the registered protocol."""
    if (config['schema']!='source_projected_lif_mean_field_development_v1'
            or config['fit_blocks']!=list(range(14)) or config['validation_blocks']!=list(range(14,18))
            or config['bin_ms']!=5. or config['origins_ms']!=[1000.,3000.,5000.]
            or config['horizons_ms']!=[20.,80.,500.]
            or config['relaxation_candidates_ms']!=[1.,5.,10.]
            or config['selection_split']!='validation'
            or config['selection_metric']!='fitting_standardized_field_rate_mse'
            or config['formal_evaluation'] is not False):
        raise ValueError('registered deterministic development protocol required')
    return config


def evaluate_campaign(projection, cohort, fitting_scale_path, config, output):
    """Select one relaxation using common development mean-field scores.

    Standardization reuses the accepted fit-only scale. Validation blocks are
    hyperparameter-selection data, so their scores are descriptive. Complete
    candidate/condition outputs remain available; no distribution is invented.
    """
    validate_campaign_config(config)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    design=json.loads((cohort/'train/design.json').read_text())
    if design['acquired_split']!='train':raise ValueError('development training cohort required')
    cases=sorted(design['cases'],key=lambda row:row['case_id'])
    if (len(cases)!=72 or {c['block'] for c in cases}!=set(range(18))
            or any(c['split']!='train' or c['family']!='trajectory' for c in cases)
            or len({c['case_id'] for c in cases})!=72
            or any(len([c for c in cases if c['block']==block])!=4 for block in range(18))):
        raise ValueError('complete registered72-episode development cohort required')
    validation=[c for c in cases if c['block'] in config['validation_blocks']]
    if len(validation)!=16:raise ValueError('complete16-episode validation denominator required')
    model=LIFNeuralField(projection,config['relaxation_candidates_ms'][0])
    with np.load(fitting_scale_path,allow_pickle=False) as scale:
        sd=scale['standard_deviation_hz'].copy()
        if (not np.array_equal(scale['field_ids'],model.field_ids)
                or not np.array_equal(scale['unit_count'],model.unit_count)
                or not np.isfinite(sd).all() or np.any(sd<=0)):
            raise ValueError('accepted fit-only scale and source observation map required')
    binding=dict(config=config,cohort_design_sha256=sha256_file(cohort/'train/design.json'),
        fitting_scale_sha256=sha256_file(fitting_scale_path),projection_sha256=model.report['projection_sha256'],
        implementation_sha256=sha256_file(Path(__file__)))
    record=dict(status='RUNNING_DEVELOPMENT_SELECTION',binding=binding,training_seeds=[],training_performed=False,
        source_simulation_performed=False,scientific_claim_qualified=False,formal_evaluation=False,
        selection_scores_are_independent_test=False,distribution_scores_available=False,unit_outputs_available=False,
        limits=LIMITS)
    (output/'evaluation.json').write_text(json.dumps(record,indent=2)+'\n')
    rows=[];started=time.perf_counter()
    for relaxation in config['relaxation_candidates_ms']:
        for case in validation:
            for origin in config['origins_ms']:
                name=f"relaxation_{relaxation:g}/{case['case_id']}_origin_{origin:g}"
                target=output/name
                result=forecast_episode(projection,cohort/'train'/case['case_id'],cohort/'train_fields'/case['case_id'],
                    target,relaxation_ms=relaxation,origin_ms=origin,horizon_ms=max(config['horizons_ms']))
                with np.load(target/'forecast.npz',allow_pickle=False) as arrays:
                    predicted=arrays['expected_field_hz'];truth=arrays['observed_field_counts']/(model.unit_count*config['bin_ms']/1000)
                for horizon in config['horizons_ms']:
                    bins=round(horizon/config['bin_ms']);error=predicted[:bins]-truth[:bins]
                    rows.append(dict(case_id=case['case_id'],block=case['block'],template_id=case['template_id'],
                        origin_ms=origin,horizon_ms=horizon,relaxation_ms=relaxation,
                        fitting_standardized_field_rate_mse=float(np.mean((error/sd)**2)),
                        mean_field_rate_mse=float(np.mean(error**2)),forecast_seconds=result['forecast_seconds'],
                        forecast_artifact=name+'/forecast.npz'))
        with (output/'candidate_scores.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    means={relaxation:float(np.mean([row[config['selection_metric']] for row in rows if row['relaxation_ms']==relaxation]))
        for relaxation in config['relaxation_candidates_ms']}
    selected=min(config['relaxation_candidates_ms'],key=lambda value:means[value])
    if len(rows)!=432 or not np.isfinite(list(means.values())).all():raise ValueError('complete finite432-row selection required')
    record.update(status='COMPLETE_DESCRIPTIVE_DEVELOPMENT_SELECTION',selected_relaxation_ms=selected,
        candidate_mean_scores=means,score_rows=len(rows),validation_episodes=len(validation),origins_per_episode=3,
        elapsed_seconds=time.perf_counter()-started,score_table_sha256=sha256_file(output/'candidate_scores.csv'))
    (output/'evaluation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def sampled_mean_field_mse(samples, truth, standard_deviation_hz, unit_count, bin_ms):
    """Estimate squared mean-field error, correcting finite-ensemble variance."""
    if (samples.ndim!=3 or samples.shape[1:]!=truth.shape or len(samples)<2
            or samples.dtype.kind not in 'iu' or truth.dtype.kind not in 'iu'
            or np.any(samples<0) or np.any(truth<0)):
        raise ValueError('complete nonnegative integer field ensemble required')
    scale=np.asarray(unit_count)*(bin_ms/1000)*np.asarray(standard_deviation_hz)
    if scale.shape!=(samples.shape[-1],) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('positive source field denominators and fit scale required')
    residual=(samples.astype(np.float64)-truth)/scale
    mean=residual.mean(0)
    # A negative estimate is valid and is preserved.
    return float(np.mean(mean**2-residual.var(0,ddof=1)/len(samples)))


def compare_population_campaign(selection, population_campaign, output):
    """Reuse three accepted population deliveries; generate no new samples."""
    if output.exists():raise FileExistsError(output)
    selected=json.loads((selection/'evaluation/evaluation.json').read_text())
    delivery=json.loads((selection/'delivery.json').read_text())
    if (delivery['status']!='COMPLETE_DESCRIPTIVE_DEVELOPMENT_MEAN_FIELD_SELECTION' or delivery['exit_code']!=0
            or selected['status']!='COMPLETE_DESCRIPTIVE_DEVELOPMENT_SELECTION' or selected['score_rows']!=432):
        raise ValueError('complete source-bound deterministic selection required')
    declared={row['path']:row['sha256'] for row in delivery['files']}
    for name in ['evaluation/evaluation.json','evaluation/candidate_scores.csv','fitting_field_scale.npz']:
        if sha256_file(selection/name)!=declared[name]:raise ValueError('delivered selection input changed:'+name)
    if sha256_file(selection/'evaluation/candidate_scores.csv')!=selected['score_table_sha256']:
        raise ValueError('selected score table changed')
    config=validate_campaign_config(selected['binding']['config'])
    with np.load(selection/'fitting_field_scale.npz',allow_pickle=False) as scale:
        sd,units,field_ids=(scale[key].copy() for key in ('standard_deviation_hz','unit_count','field_ids'))
    if sha256_file(selection/'fitting_field_scale.npz')!=selected['binding']['fitting_scale_sha256']:
        raise ValueError('selected fit scale changed')
    with (selection/'evaluation/candidate_scores.csv').open() as stream:
        baseline=[row for row in csv.DictReader(stream) if float(row['relaxation_ms'])==selected['selected_relaxation_ms']]
    if len(baseline)!=144:raise ValueError('selected complete144-cell mean-field denominator required')
    rows=[];bindings={}
    for seed in [41,42,43]:
        cell=population_campaign/f'seed_{seed}';proof=json.loads((cell/'delivery.json').read_text())
        replay=json.loads((cell/'replay.json').read_text());evaluation=json.loads((cell/'forecasts/evaluation.json').read_text())
        files={row['path']:row['sha256'] for row in proof['files']}
        for name in ['replay.json','forecasts/evaluation.json','forecasts/fitting_field_scale.npz']:
            if sha256_file(cell/name)!=files[name]:raise ValueError('accepted population metadata changed:'+name)
        if (proof['status']!='PASS' or proof['exit_code']!=0 or proof['training_seed']!=seed
                or replay['status']!='PASS' or replay['training_seed']!=seed or replay['verified_origins']!=48
                or replay['all_origin_fields_and_latents_exact'] is not True
                or replay['evaluation_sha256']!=sha256_file(cell/'forecasts/evaluation.json')
                or evaluation['status']!='EVALUATED_DEVELOPMENT' or evaluation['training_seed']!=seed
                or evaluation['acquisition_design_sha256']!=selected['binding']['cohort_design_sha256']
                or evaluation['configuration']['origins_ms']!=config['origins_ms']
                or evaluation['configuration']['horizons_ms']!=config['horizons_ms']
                or evaluation['configuration']['joint_samples']!=16 or evaluation['formal_test_accessed'] is not False):
            raise ValueError('three completed source-matched population campaign seeds required')
        with np.load(cell/'forecasts/fitting_field_scale.npz',allow_pickle=False) as fit:
            if not all(np.array_equal(fit[key],expected) for key,expected in
                       [('standard_deviation_hz',sd),('unit_count',units),('field_ids',field_ids)]):
                raise ValueError('population and neural-field fitting scales differ')
        cached={}
        for baseline_row in baseline:
            name=f"{baseline_row['case_id']}_origin_{float(baseline_row['origin_ms']):g}.npz"
            if name not in cached:
                path=cell/'forecasts'/name
                if sha256_file(path)!=files['forecasts/'+name]:raise ValueError('accepted population samples changed')
                with np.load(path,allow_pickle=False) as values:
                    samples=values['sampled_field_counts'].copy();truth=values['observed_field_counts'].copy()
                if samples.shape!=(16,100,len(units)) or truth.shape!=(100,len(units)):
                    raise ValueError('complete16-draw500ms common field outputs required')
                baseline_path='evaluation/'+baseline_row['forecast_artifact']
                if sha256_file(selection/baseline_path)!=declared[baseline_path]:raise ValueError('selected field prediction changed')
                with np.load(selection/baseline_path,allow_pickle=False) as values:
                    if (not np.array_equal(values['observed_field_counts'],truth)
                            or not np.array_equal(values['field_ids'],field_ids)
                            or not np.array_equal(values['unit_count'],units)):
                        raise ValueError('source truth/map differs between models')
                cached[name]=(samples,truth)
            samples,truth=cached[name];bins=round(float(baseline_row['horizon_ms'])/config['bin_ms'])
            value=sampled_mean_field_mse(samples[:,:bins],truth[:bins],sd,units,config['bin_ms'])
            baseline_value=float(baseline_row['fitting_standardized_field_rate_mse'])
            rows.append(dict(seed=seed,block=int(baseline_row['block']),case_id=baseline_row['case_id'],
                origin_ms=float(baseline_row['origin_ms']),horizon_ms=float(baseline_row['horizon_ms']),
                population_mean_field_mse_corrected=value,lif_mean_field_mse=baseline_value,
                population_minus_lif_mean_field_mse=value-baseline_value))
        bindings[str(seed)]=dict(delivery_sha256=sha256_file(cell/'delivery.json'),replay_sha256=sha256_file(cell/'replay.json'))
    output.mkdir(parents=True)
    with (output/'mean_field_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='COMPLETE_DESCRIPTIVE_COMMON_MEAN_FIELD_COMPARISON',training_seeds=[41,42,43],score_rows=len(rows),
        selected_relaxation_ms=selected['selected_relaxation_ms'],population_bindings=bindings,
        independent_blocks=4,scientific_claim_qualified=False,inferential_statistics_performed=False,
        samples_regenerated=False,source_simulation_repeated=False,distribution_comparison_performed=False,
        deterministic_baseline_unique_cells=144,selection_scores_are_independent_test=False,
        score_table_sha256=sha256_file(output/'mean_field_comparison.csv'))
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path);parser.add_argument('--fields',type=Path)
    parser.add_argument('--episode',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--projection',type=Path);parser.add_argument('--relaxation-ms',type=float,default=5.)
    parser.add_argument('--origin-ms',type=float,default=1000.);parser.add_argument('--horizon-ms',type=float,default=500.)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--config',type=Path);parser.add_argument('--cohort',type=Path);parser.add_argument('--fitting-scale',type=Path)
    parser.add_argument('--selection',type=Path);parser.add_argument('--population-campaign',type=Path)
    args=parser.parse_args()
    if args.selection or args.population_campaign:
        if not args.selection or not args.population_campaign:parser.error('comparison requires selection and population campaign')
        print(json.dumps(compare_population_campaign(args.selection,args.population_campaign,args.output)));return
    if args.config:
        if not args.projection or not args.cohort or not args.fitting_scale or args.graph or args.replay:
            parser.error('campaign requires --projection, --cohort, --fitting-scale and no graph/replay')
        result=evaluate_campaign(args.projection,args.cohort,args.fitting_scale,yaml.safe_load(args.config.read_text()),args.output)
        print(json.dumps(result));return
    if bool(args.graph)==bool(args.projection):parser.error('exactly one of --graph and --projection required')
    if not args.fields or not args.episode:parser.error('single episode requires --fields and --episode')
    if args.graph:
        if args.replay:parser.error('--replay requires a completed --projection')
        result=build_projection(args.graph,args.fields,args.episode,args.output)
    else:
        result=forecast_episode(args.projection,args.episode,args.fields,args.output,relaxation_ms=args.relaxation_ms,
            origin_ms=args.origin_ms,horizon_ms=args.horizon_ms,replay=args.replay)
    print(json.dumps(result))


if __name__=='__main__':main()
