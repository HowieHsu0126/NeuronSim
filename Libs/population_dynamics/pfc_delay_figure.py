"""Render all replayed PFC source blocks and currents in separate phase windows."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from PIL import Image
import yaml

from .pfc_delay_analysis import CONDITIONS, PHASES, SEEDS
from .provenance import sha256_file

COLORS = ('#0072B2', '#E69F00', '#7B3294')
CONDITION_STYLE = (
    ('omission', 'X', 'Omission (0 nA)'),
    ('cue', 'o', 'Weak cue (0.06 nA)'),
    ('strong_cue', 's', 'Strong cue (0.5 nA)'),
)
PANEL_METRICS = (
    ('excitatory_rate_hz', 'Excitatory population activity', 'E firing rate (Hz)'),
    ('cue_selectivity_hz', 'Cue-selective activity', 'Cue selectivity (Hz)'),
    ('circular_cue_error_deg', 'Direction relative to cue', 'Direction error (°)'),
    ('resultant_length', 'Concentration of the response', 'Circular resultant length'),
)
PHASE_LABELS = ('Baseline\n0–0.1', 'Cue\n0.1–0.3', 'Early\n0.3–0.5',
                'Middle\n1–2', 'Late\n4–5')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def checked_inputs(root):
    report=read(root/'analysis.json')
    require(report['status']=='DESCRIBED_REPLAYED_DELAY_DIAGNOSTIC'
            and report['complete_source_episodes']==9 and report['source_rng_blocks']==3
            and report['units']==1280 and report['duration_ms']==5000.
            and report['phases_ms']==PHASES and report['full_unit_500ms_prefixes_verified']==9
            and report['phase_unit_count_values_verified']==57600
            and all(report[k] is False for k in ('working_memory_qualified','full_scale_qualified',
                'scientific_fidelity_qualified','neuroscientific_claim_qualified',
                'confidence_intervals','p_values','q_values')) and report['production_episodes']==0,
            'complete unqualified nine-episode source delay analysis required')
    replay_path=root/'readout_replay.json'
    require(sha256_file(replay_path)==report['readout_replay_sha256'],'readout replay receipt changed')
    replay=read(replay_path)
    require(replay['status']=='PASS' and replay['complete_episodes']==9 and replay['source_rng_blocks']==3
            and replay['phase_unit_count_values_checked']==57600
            and replay['analysis_artifacts']==report['artifacts']
            and replay['original_report_sha256']==report['original_report_sha256']
            and replay['design_sha256']==report['design_sha256'],
            'independent raw-event readout replay must bind the plotted analysis')
    for name,digest in report['artifacts'].items():
        require(Path(name).name==name and sha256_file(root/name)==digest,'analysis artifact changed: '+name)
    with (root/'responses.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    expected={(seed,condition,phase) for seed in SEEDS for condition in CONDITIONS for phase in PHASES}
    require(len(rows)==45,'all 45 source block/current/phase rows required')
    parsed={}
    for row in rows:
        seed=int(row['source_rng_seed']);condition=row['condition'];phase=row['phase']
        key=(seed,condition,phase)
        require(key in expected and key not in parsed and row['case_id']==f'pfc_{seed}_{condition}'
                and [float(row['start_ms']),float(row['stop_ms'])]==PHASES[phase],
                'source row identity or phase exposure changed')
        values={name:None if row[name]=='' else float(row[name]) for name,_,_ in PANEL_METRICS}
        require(values['excitatory_rate_hz'] is not None and values['excitatory_rate_hz']>=0
                and values['cue_selectivity_hz'] is not None
                and all(v is None or np.isfinite(v) for v in values.values()),'invalid plotted response')
        require((values['circular_cue_error_deg'] is not None)==(row['direction_defined']=='True')
                and (values['circular_cue_error_deg'] is None or -180<=values['circular_cue_error_deg']<180)
                and (values['resultant_length'] is None or 0<=values['resultant_length']<=1+1e-12),
                'circular response domain or undefined direction changed')
        parsed[key]=values
    require(set(parsed)==expected,'complete plotted source denominator required')
    return report,parsed


def style_from(path):
    spec=importlib.util.spec_from_file_location('pfc_figure_style',path)
    style=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    return style


def geometry(fig):
    renderer=fig.canvas.get_renderer()
    texts=[(text,text.get_window_extent(renderer)) for text in fig.findobj(Text)
           if text.get_visible() and text.get_text().strip()]
    outside=[text.get_text() for text,box in texts if box.x0<0 or box.y0<0
             or box.x1>fig.bbox.x1 or box.y1>fig.bbox.y1]
    overlaps=[(a.get_text(),b.get_text()) for i,(a,box_a) in enumerate(texts)
              for b,box_b in texts[i+1:] if box_a.overlaps(box_b)]
    ticks={ax:set(ax.get_xticklabels()+ax.get_yticklabels()) for ax in fig.axes}
    spine_overlaps=[]
    for text,box in texts:
        for ax in fig.axes:
            for spine in ax.spines.values():
                if spine.get_visible() and text not in ticks[ax] and box.overlaps(spine.get_window_extent(renderer)):
                    spine_overlaps.append(text.get_text())
    return dict(outside=outside,text_overlaps=overlaps,spine_overlaps=spine_overlaps)


def render(analysis_root, style_kernel, output):
    analysis_root,style_kernel,output=map(Path,(analysis_root,style_kernel,output))
    report,data=checked_inputs(analysis_root)
    require(not (output/'pfc_long_delay.png').exists(),'existing figure will not be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    style=style_from(style_kernel)
    style.apply_figure_style(sizes=(9,8,7))
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.3))
    fig.subplots_adjust(left=.11,right=.98,bottom=.18,top=.79,hspace=.66,wspace=.34)
    missing=[]
    try:
        for letter,ax,(metric,title,ylabel) in zip('abcd',axes.flat,PANEL_METRICS):
            for phase_index,phase in enumerate(PHASES):
                for condition_index,(condition,marker,_) in enumerate(CONDITION_STYLE):
                    for seed_index,(seed,color) in enumerate(zip(SEEDS,COLORS)):
                        x=phase_index+(condition_index-1)*.29+(seed_index-1)*.075
                        value=data[seed,condition,phase][metric]
                        if value is None:
                            missing.append(dict(metric=metric,source_rng_seed=seed,condition=condition,phase=phase))
                            ax.text(x,.025,'×',color=color,ha='center',va='bottom',fontsize=8,
                                    transform=ax.get_xaxis_transform())
                        else:
                            ax.plot(x,value,linestyle='None',marker=marker,color=color,
                                    markersize=3.5,markeredgewidth=.65,zorder=3)
            ax.set_xticks(range(5),PHASE_LABELS)
            ax.set_xlim(-.55,4.55)
            ax.set_ylabel(ylabel)
            ax.set_title(title,loc='left',pad=9)
            ax.axhline(0,color='#888888',linestyle=':',linewidth=.65,zorder=0)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.margins(y=.15)
            style.panel_letter(ax,letter,dx=-.20,dy=1.12,fontsize=11)
            if metric=='excitatory_rate_hz':
                high=max(v[metric] for v in data.values())
                ax.set_ylim(-.04*max(high,1e-6),1.14*max(high,1e-6))
            elif metric=='cue_selectivity_hz':
                ax.set_yscale('symlog',linthresh=.25,linscale=1.)
                ax.set_yticks([-.25,0,.25,1,10,100])
                ax.set_yticklabels(['−0.25','0','0.25','1','10','100'])
                values=[v[metric] for v in data.values()]
                ax.set_ylim(min(-.4,min(values)*1.4),max(values)*1.5)
            elif metric=='circular_cue_error_deg':
                ax.set_ylim(-192,192);ax.set_yticks([-180,-90,0,90,180])
            elif metric=='resultant_length':
                ax.set_ylim(-.05,1.08);ax.set_yticks([0,.5,1])
        fig.suptitle('PFC source responses in five analysis windows',y=.995,fontsize=9)
        block_handles=[Line2D([],[],linestyle='None',marker='o',color=color,
                             markersize=4,label=f'Block {i+1}') for i,color in enumerate(COLORS)]
        condition_handles=[Line2D([],[],linestyle='None',marker=marker,color='#555555',
                                 markersize=4,label=label) for _,marker,label in CONDITION_STYLE]
        fig.legend(handles=block_handles,loc='upper center',bbox_to_anchor=(.55,.956),
                   ncol=3,frameon=False,handletextpad=.4,columnspacing=1.8)
        fig.legend(handles=condition_handles,loc='upper center',bbox_to_anchor=(.55,.915),
                   ncol=3,frameon=False,handletextpad=.4,columnspacing=1.5)
        fig.text(.5,.099,'Separate analysis windows (s); intervals between windows are not shown.',
                 ha='center',va='center',fontsize=8)
        fig.text(.5,.036,'1,024 E + 256 I units · 3 joint initial/background RNG blocks\n'
                 'Source-development diagnostic · individual episodes, no inferential intervals',
                 ha='center',va='center',fontsize=8)
        if missing:
            fig.text(.5,.139,'× at panel bottom: undefined circular quantity (listed in caption).',
                     ha='center',va='center',fontsize=8)
        fig.canvas.draw()
        checks=geometry(fig)
        for ext in ('png','pdf','svg'):
            fig.savefig(output/f'pfc_long_delay.{ext}',dpi=300)
        image=Image.open(output/'pfc_long_delay.png')
        scale=300/fig.dpi
        for letter,ax in zip('abcd',axes.flat):
            box=ax.get_tightbbox(fig.canvas.get_renderer())
            left=max(0,int((box.x0-35)*scale));right=min(image.width,int((box.x1+8)*scale))
            top=max(0,image.height-int((box.y1+15)*scale))
            bottom=min(image.height,image.height-int((box.y0-8)*scale))
            image.crop((left,top,right,bottom)).save(output/f'panel_{letter}.png')
        caption=('All 1,280 native source units are retained; firing and circular readouts use all 1,024 excitatory units. '
            'E angle denotes cue preference, not measured cortical position. '
            'Colors identify joint source RNG blocks 1/2/3 (91001/91002/91003); shapes identify omission (0 nA), '
            'weak cue (0.06 nA), and strong cue (0.5 nA). The 120-degree, 30-degree-wide cue is applied from 100 to 300 ms. '
            'Each point is one episode summary over the labeled half-open window; horizontal offsets only separate marks. '
            'Selectivity is cue-region firing minus noncue-region firing; panel b uses a symmetric-log axis, linear within ±0.25 Hz, to retain small signed effects. Signed circular error is relative to 120 degrees; '
            'resultant length measures spike-weighted directional concentration, from 0 to 1. '
            'No lines interpolate the gaps between early, middle, and late windows. These phase samples do not establish '
            'uninterrupted persistence over the full 4.7 s after cue offset. The three source RNG blocks are not training '
            'seeds or biological replicates. Extended episodes preserve all nine original 500 ms spike prefixes and add '
            'no independent blocks. Native arrays and phase readouts were independently replayed. '
            'This is an unqualified source-model development diagnostic, with no confidence intervals, p/q values, '
            'working-memory qualification, biological finding, or source-equivalence claim.')
        if missing:
            caption+=' Undefined plotted quantities: '+json.dumps(missing,sort_keys=True)+'.'
        (output/'caption.txt').write_text(caption+'\n')
        names=['pfc_long_delay.'+ext for ext in ('png','pdf','svg')]
        names.extend('panel_'+letter+'.png' for letter in 'abcd')
        names.append('caption.txt')
        status='RENDERED_PENDING_VISUAL_INSPECTION' if not any(checks.values()) else 'FAILED_GEOMETRY'
        receipt=dict(status=status,analysis_sha256=sha256_file(analysis_root/'analysis.json'),
            readout_replay_sha256=report['readout_replay_sha256'],responses_sha256=report['artifacts']['responses.csv'],
            implementation_sha256=sha256_file(Path(__file__)),style_kernel_sha256=sha256_file(style_kernel),
            geometry=checks,plotted_response_rows=45,source_rng_blocks=3,conditions=list(CONDITIONS),
            phases=list(PHASES),interpolation_between_phases=False,undefined_quantities=missing,
            scientific_fidelity_qualified=False,working_memory_qualified=False,neuroscientific_claim_qualified=False,
            artifacts={name:sha256_file(output/name) for name in names})
        (output/'render_checks.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        require(status!='FAILED_GEOMETRY','rendered figure failed geometric checks; inspect retained PNG')
        return receipt
    finally:
        plt.close(fig)


def checked_gain_points(analysis_root, protocol_path):
    """Pair the complete fixed bank within size and replay its saved summaries."""
    delivery = read(analysis_root.parent / 'delivery.json')
    require(delivery['status'] == 'COMPLETE_FIXED_REPLICATE_READOUTS'
            and delivery['exit_code'] == 0 and delivery['mode'] == 'analysis'
            and delivery['verified'] is True, 'complete fixed-bank delivery required')
    files = {row['path']: row for row in delivery['files']}
    bindings = {}

    def bound(name):
        key = 'analysis/' + name
        path = analysis_root / name
        require(key in files and path.stat().st_size == files[key]['bytes']
                and sha256_file(path) == files[key]['sha256'], 'figure input changed: ' + key)
        bindings[key] = files[key]['sha256']
        return path

    report = read(bound('report.json'))
    protocol = yaml.safe_load(protocol_path.read_text())
    require(sha256_file(protocol_path) == report['settings']['source_protocol_sha256'],
            'frozen replication protocol required')
    require(report['status'] == 'COMPLETE_DESCRIPTIVE_FIXED_GAIN_REPLICATION'
            and report['source_cases'] == protocol['full_source_cases'] == 384
            and report['source_simulations'] == report['production_episodes'] == 0
            and all(report[key] is False for key in (
                'scientific_parameter_domain_qualified', 'confirmation_execution_released',
                'confidence_intervals', 'p_values', 'activation_threshold_introduced')),
            'complete descriptive replication without scientific promotion required')
    require(protocol['Jpos_values'] == [1.6, 2.3]
            and protocol['fixed_source_clock_ms'] == .05 and protocol['duration_ms'] == 5000.
            and protocol['size_values'] == [2560, 100000]
            and protocol['full_blocks_per_size'] == 32
            and protocol['new_blocks_per_size'] == 29
            and protocol['exposed_pilot_blocks_per_size'] == 3
            and set(protocol['conditions']) == set(CONDITIONS), 'fixed figure design changed')
    phase = 'late_delay'
    require(PHASES[phase] == [4000., 5000.], 'declared late-delay window changed')
    cohorts = (('reference_2560', 2560, 'reference_full_rng_seeds'),
               ('expanded_100k', 100000, 'expanded_full_rng_seeds'))
    points, summaries = [], []
    for cohort, size, seed_field in cohorts:
        tables = {}
        for partition, blocks in (('new29', 29), ('full32', 32)):
            for parameter in ('default', 'candidate'):
                name = f'{partition}/{cohort}_{parameter}/responses.csv'
                with bound(name).open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                parsed = {}
                for row in rows:
                    seed = int(row['source_rng_seed'])
                    key = (seed, row['condition'], row['phase'])
                    value = float(row['excitatory_rate_hz'])
                    require(key not in parsed and key[1] in CONDITIONS and key[2] in PHASES
                            and row['case_id'] == f'pfc_{seed}_{key[1]}'
                            and np.isfinite(value) and value >= 0, 'invalid response identity or rate')
                    parsed[key] = value
                seeds = {key[0] for key in parsed}
                require(len(seeds) == blocks and set(parsed) == {
                    (seed, condition, window) for seed in seeds
                    for condition in CONDITIONS for window in PHASES},
                    'complete source block, condition and phase denominator required')
                tables[partition, parameter] = parsed
        for parameter in ('default', 'candidate'):
            full, new = tables['full32', parameter], tables['new29', parameter]
            require({key[0] for key in full} == set(protocol[seed_field])
                    and set(new) < set(full)
                    and all(new[key] == full[key] for key in new),
                    'new29 must be the unchanged subset of the full32 table')
        require(set(tables['full32', 'default']) == set(tables['full32', 'candidate'])
                and set(tables['new29', 'default']) == set(tables['new29', 'candidate']),
                'parameters must share the same within-size blocks')
        for seed in sorted(protocol[seed_field]):
            for condition in protocol['conditions']:
                key = (seed, condition, phase)
                default, candidate = (tables['full32', p][key] for p in ('default', 'candidate'))
                points.append(dict(cohort=cohort, coupled_units=size, source_rng_seed=seed,
                    case_id=f'pfc_{seed}_{condition}', condition=condition, phase=phase,
                    start_ms=PHASES[phase][0], stop_ms=PHASES[phase][1],
                    partition='new29' if key in tables['new29', 'default'] else 'pilot3',
                    default_rate_hz=default, candidate_rate_hz=candidate,
                    candidate_minus_default_hz=candidate-default))
    paired = report['input_pairing']
    require(len(paired) == len(points) == 192
            and all(row['paired_initial_background_input_law'] is True for row in paired)
            and {(row['size'], row['case_id'], row['source_rng_seed']) for row in paired}
            == {(row['coupled_units'], row['case_id'], row['source_rng_seed']) for row in points},
            'all plotted pairs must be bound to the accepted input-pairing report')
    for partition, blocks in (('new29', 29), ('full32', 32)):
        with bound(partition + '/paired_response_effects.csv').open(newline='') as stream:
            saved = [row for row in csv.DictReader(stream)
                     if row['phase'] == phase and row['metric'] == 'excitatory_rate_hz']
        require(len(saved) == 6 and {(row['cohort'], row['condition']) for row in saved}
                == {(cohort, condition) for cohort, _, _ in cohorts for condition in CONDITIONS},
                'all six saved gain-effect summaries required')
        for row in saved:
            values = np.asarray([point['candidate_minus_default_hz'] for point in points
                if point['cohort'] == row['cohort'] and point['condition'] == row['condition']
                and (partition == 'full32' or point['partition'] == 'new29')])
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            require(len(values) == int(row['blocks']) == int(row['defined_blocks']) == blocks
                    and row['units'] == 'Hz'
                    and np.allclose([mean, sd], [float(row['mean']), float(row['block_sd'])],
                                    rtol=1e-10, atol=1e-9), 'saved mean or block SD differs from plotted pairs')
            summaries.append(dict(partition=partition, cohort=row['cohort'], condition=row['condition'],
                                  blocks=blocks, mean_hz=mean, block_sd_hz=sd))
    return report, protocol, points, summaries, bindings


def render_gain(analysis_root, protocol_path, style_kernel, output):
    analysis_root, protocol_path, style_kernel, output = map(
        Path, (analysis_root, protocol_path, style_kernel, output))
    report, protocol, points, summaries, bindings = checked_gain_points(analysis_root, protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    style_from(style_kernel).apply_figure_style(sizes=(9, 8, 7))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(left=.12, right=.985, bottom=.23, top=.76)
    conditions = ('omission', 'cue', 'strong_cue')
    colors = ('#777777', '#0072B2', '#7B3294')
    ticks, labels = [], []
    try:
        for group, (cohort, size) in enumerate((('reference_2560', 2560), ('expanded_100k', 100000))):
            for ci, (condition, color) in enumerate(zip(conditions, colors)):
                x = 4 * group + ci
                ticks.append(x)
                labels.append(('Omission\n0 nA', 'Weak cue\n0.06 nA', 'Strong cue\n0.5 nA')[ci])
                for partition, left, right in (('pilot3', -.29, -.17), ('new29', -.09, .29)):
                    selected = [point for point in points if point['cohort'] == cohort
                                and point['condition'] == condition and point['partition'] == partition]
                    offsets = np.linspace(left, right, len(selected))
                    for point, offset in zip(selected, offsets):
                        point['plot_x'] = float(x + offset)
                    ax.scatter(x + offsets, [point['candidate_minus_default_hz'] for point in selected],
                               s=14 if partition == 'pilot3' else 9, linewidths=.7,
                               facecolors='none' if partition == 'pilot3' else color,
                               edgecolors=color, alpha=1 if partition == 'pilot3' else .7, zorder=3)
                mean = next(row['mean_hz'] for row in summaries if row['partition'] == 'new29'
                            and row['cohort'] == cohort and row['condition'] == condition)
                ax.plot([x-.10, x+.30], [mean, mean], color='#222222', linewidth=1.5, zorder=2)
            ax.text(4*group+1, 1.045, f'{size:,} coupled neurons', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=9)
        ax.axhline(0, color='#999999', linestyle=':', linewidth=.7, zorder=0)
        ax.axvline(3, color='#DDDDDD', linewidth=.7, zorder=0)
        ax.set_xticks(ticks, labels)
        ax.set_xlim(-.6, 6.6)
        values = [point['candidate_minus_default_hz'] for point in points]
        span = max(values)-min(values)
        ax.set_ylim(min(0, min(values))-.08*span, max(values)+.13*span)
        ax.set_ylabel('Excitatory firing-rate change (Hz)\nJpos 2.3 − Jpos 1.6')
        fig.suptitle('PFC response to stronger recurrence during the late delay', y=.98, fontsize=10)
        handles = [Line2D([], [], marker='o', linestyle='None', markerfacecolor='none',
                          color='#444444', markersize=4, label='3 exposed pilot blocks'),
                   Line2D([], [], marker='o', linestyle='None', color='#0072B2',
                          markersize=3.5, label='29 new blocks'),
                   Line2D([], [], color='#222222', linewidth=1.5, label='New-block mean')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .91), ncol=3,
                   handletextpad=.5, columnspacing=1.6)
        fig.text(.55, .093, 'Late window: 4–5 s · Source clock: 0.05 ms · Separate blocks at each size',
                 ha='center', va='center', fontsize=8)
        fig.text(.55, .047, 'Source-development diagnostic; no activation threshold or inferential intervals',
                 ha='center', va='center', fontsize=8)
        fig.canvas.draw()
        checks = geometry(fig)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'pfc_recurrent_gain.{ext}', dpi=300)
        with (output / 'paired_points.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
        caption = (
            'Late-delay excitatory firing-rate changes when the registered recurrence parameter Jpos '
            'increases from 1.6 to 2.3, at sigma=20 and the fixed 0.05 ms source clock. '
            'Each point is the candidate-minus-default difference for one joint initial/background '
            'RNG block and condition over 4–5 s after episode start. The complete bank contains '
            '384 episodes: 32 blocks × 3 conditions × 2 parameters at each of two network sizes. '
            'The figure retains all 192 within-size differences, including signed and near-zero values. '
            'Inputs are paired across parameters and conditions within a size/block; sizes are not paired. '
            'The 120-degree, 30-degree-wide cue is applied at 100–300 ms; all omission, weak and strong '
            'controls are shown. Hollow markers denote the three exposed pilot blocks, which motivated '
            'the replication; filled markers denote the 29 new blocks, identified from the actual '
            'new29 tables. Black ticks are the new29 arithmetic means; no CI is drawn. '
            'Horizontal offsets only separate marks and points may overlap. The corresponding full32 '
            'summaries include the same pilot and new blocks and are not an independent replication. '
            'This plot shows the gain effect separately for each condition, not the registered '
            'weak-minus-omission difference-in-differences; that contrast remains in the full analysis. '
            'Joint source RNG blocks are not training seeds or biological replicates. '
            'A late-window average does not establish uninterrupted persistence, an activation '
            'probability, bistability, working-memory fidelity, numerical-clock convergence or a '
            'qualified parameter domain. All controls and source failures remain in the evidence bank.')
        (output / 'caption.txt').write_text(caption+'\n')
        names = ['pfc_recurrent_gain.'+ext for ext in ('png', 'pdf', 'svg')]
        names += ['paired_points.csv', 'caption.txt']
        receipt = dict(status='FAILED_GEOMETRY' if any(checks.values()) else 'RENDERED_PENDING_VISUAL_INSPECTION',
            source_job=597232, source_delivery_sha256=sha256_file(analysis_root.parent/'delivery.json'),
            source_protocol_sha256=report['settings']['source_protocol_sha256'], input_sha256=bindings,
            implementation_sha256=sha256_file(Path(__file__)), style_kernel_sha256=sha256_file(style_kernel),
            geometry=checks, phase='late_delay', phase_ms=PHASES['late_delay'], plotted_pairs=len(points),
            plotted_pilot_pairs=sum(point['partition'] == 'pilot3' for point in points),
            plotted_new_pairs=sum(point['partition'] == 'new29' for point in points),
            full_source_cases=protocol['full_source_cases'], replayed_summary_rows=summaries,
            summary_float_tolerance=dict(rtol=1e-10, atol=1e-9), block_sd_ddof=1,
            sizes_paired=False, source_simulations=0, raw_arrays_read=0, confidence_intervals=False,
            activation_threshold_introduced=False, scientific_parameter_domain_qualified=False,
            confirmation_execution_released=False, production_episodes=0,
            artifacts={name:sha256_file(output/name) for name in names})
        (output/'render_checks.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        require(not any(checks.values()), 'rendered figure failed geometry; inspect retained PNG')
        return receipt
    finally:
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','style-kernel','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--kind', choices=('source-delay', 'recurrent-gain'), default='source-delay')
    parser.add_argument('--protocol', type=Path)
    args=parser.parse_args()
    if args.kind == 'recurrent-gain':
        if args.protocol is None:
            parser.error('--protocol is required for recurrent-gain')
        receipt = render_gain(args.analysis, args.protocol, args.style_kernel, args.output)
    else:
        receipt = render(args.analysis, args.style_kernel, args.output)
    print(json.dumps(receipt, indent=2))


if __name__=='__main__':
    main()
