"""Render one replay-bound, descriptive CA3 recruitment panel on a worker node."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import socket

from .provenance import sha256_file

CUE_WINDOWS = ((520,540),(750,770),(980,1000))
UNITS = {'target':137, 'other':274, 'eligible':138}
TOLERANCE = dict(rtol=1e-12, atol=1e-12)
ARTIFACTS = {'descriptive_readouts.csv','pyramidal_window_counts.npz'}


def _close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('CA3_FIGURE_INCONSISTENT_COUNT_OR_FRACTION')


def _integer(value):
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError('CA3_FIGURE_NONNEGATIVE_INTEGER_REQUIRED')
    return int(value)


def load_inputs(readouts, replay):
    report_path = readouts/'report.json'
    report = json.loads(report_path.read_text())
    proof = json.loads(replay.read_text())
    hashes = {'report.json':sha256_file(report_path), 'replay':sha256_file(replay)}
    if (report['status'] != 'DESCRIBED_SOURCE_SCHEDULED_CA3_RETRIEVAL_EVENTS'
            or report['cues_retained'] != 3 or report['duration_ms'] != 1001
            or report['full_record_recounted_units'] != 84053
            or report['window_counted_pyramidal_units'] != 75377
            or proof['status'] != 'PASS_INDEPENDENT_RAW_EVENT_READOUT_REPLAY'
            or proof['source_report_sha256'] != hashes['report.json']
            or proof['all_three_cues_retained'] is not True
            or proof['window_count_values_compared'] != 452262
            or proof['array_values_compared'] != 1357903
            or proof['scalar_values_compared'] != 141
            or proof['scalar_tolerance'] != TOLERANCE
            or proof['scientific_claim_qualified'] is not False):
        raise ValueError('CA3_FIGURE_COMPLETE_MATCHING_READOUT_REPLAY_REQUIRED')
    for name in ('scientific_fidelity_qualified','pattern_completion_qualified','causal_claim_qualified',
                 'neuroscientific_claim_qualified','full_scale_qualified','actual_current_trace_verified',
                 'terminal_native_clock_independently_verified','inferential_statistics_performed',
                 'p_values','confidence_intervals','success_thresholds_applied'):
        if report.get(name) is not False:
            raise ValueError('CA3_FIGURE_DESCRIPTIVE_SOURCE_SCOPE_REQUIRED:'+name)
    if (report['input_sha256']['native_report.json'] != report['native_report_sha256']
            or report['input_sha256']['observations/recording.json'] != report['recording_sha256']
            or set(report['artifacts']) != ARTIFACTS):
        raise ValueError('CA3_FIGURE_SOURCE_REPORT_IDENTITY_MISMATCH')
    for name in ARTIFACTS:
        hashes[name] = sha256_file(readouts/name)
        if hashes[name] != report['artifacts'][name]:
            raise ValueError('CA3_FIGURE_READOUT_ARTIFACT_CHANGED:'+name)
    events = [row for row in report['checked_event_files'] if row['group']=='CA3_Pyramidal']
    if (len(events)!=1 or events[0]['units']!=75377
            or events[0]['relative_path']!='native_run/results/spk_CA3_Pyramidal.dat'
            or events[0]['sha256']!=proof['raw_events_sha256']
            or events[0]['records']!=proof['pyramidal_events']
            or report['pyramidal_records']!=proof['pyramidal_events']):
        raise ValueError('CA3_FIGURE_RAW_EVENT_IDENTITY_MISMATCH')
    with (readouts/'descriptive_readouts.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        columns=reader.fieldnames
        rows=list(reader)
    identities={'cue_id','assembly_local_start','assembly_global_start','baseline_start_ms','baseline_stop_ms',
                'cue_start_ms','cue_stop_ms','following_source_background_ms'}
    measures={'units','baseline_event_count','cue_event_count','baseline_rate_hz','cue_rate_hz',
              'baseline_active_units','cue_active_units','baseline_spiking_fraction','cue_spiking_fraction',
              'delta_rate_hz','delta_spiking_fraction'}
    expected=identities | {p+'_'+m for p in UNITS for m in measures}
    expected |= {'target_minus_other_'+p+'_'+m for p in ('baseline','cue','delta') for m in ('rate_hz','spiking_fraction')}
    if columns is None or len(columns)!=len(set(columns)) or set(columns)!=expected or len(rows)!=3:
        raise ValueError('CA3_FIGURE_ALL_THREE_IDENTIFIED_CUE_ROWS_REQUIRED')
    numeric=[]
    for index,row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError('CA3_FIGURE_MALFORMED_CSV')
        values={name:float(value) for name,value in row.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('CA3_FIGURE_NONFINITE_VALUE')
        start,stop=CUE_WINDOWS[index]
        expected_ids=dict(cue_id=index+1,assembly_local_start=275*index,
            baseline_start_ms=start-20,baseline_stop_ms=start,cue_start_ms=start,cue_stop_ms=stop,
            following_source_background_ms=(210,210,1)[index])
        for name,value in expected_ids.items():
            if _integer(values[name])!=value:
                raise ValueError('CA3_FIGURE_SOURCE_SCHEDULE_OR_ASSEMBLY_CHANGED')
        global_start=_integer(values['assembly_global_start'])
        if index and global_start!=numeric[0]['assembly_global_start']+275*index:
            raise ValueError('CA3_FIGURE_GLOBAL_ASSEMBLY_IDENTITY_CHANGED')
        for prefix,n in UNITS.items():
            if _integer(values[prefix+'_units'])!=n:
                raise ValueError('CA3_FIGURE_SOURCE_SUBSET_SIZE_CHANGED')
            for phase in ('baseline','cue'):
                active=_integer(values[prefix+'_'+phase+'_active_units'])
                count=_integer(values[prefix+'_'+phase+'_event_count'])
                if active>n or count<active or (active==0)!=(count==0):
                    raise ValueError('CA3_FIGURE_ACTIVE_CELL_COUNT_INCONSISTENT')
                _close(values[prefix+'_'+phase+'_spiking_fraction'],active/n)
                _close(values[prefix+'_'+phase+'_rate_hz'],count/(n*.020))
            for measure in ('rate_hz','spiking_fraction'):
                _close(values[prefix+'_delta_'+measure],
                       values[prefix+'_cue_'+measure]-values[prefix+'_baseline_'+measure])
        for phase in ('baseline','cue','delta'):
            for measure in ('rate_hz','spiking_fraction'):
                key=phase+'_'+measure
                _close(values['target_minus_other_'+key],values['target_'+key]-values['other_'+key])
        numeric.append(values)
    starts=[[a-20,a] for a,b in CUE_WINDOWS];stops=[[a,b] for a,b in CUE_WINDOWS]
    if (report['source_scheduled_window_start_ms']!=starts or report['source_scheduled_window_stop_ms']!=stops
            or report['third_cue_post_source_support_ms']!=1):
        raise ValueError('CA3_FIGURE_REPORTED_WINDOW_IDENTITY_CHANGED')
    return numeric,report,proof,hashes


def _style(path):
    if path is None:
        raise ValueError('Supply --style-helper or NEURONSIM_FIGURE_STYLE_HELPER pointing to figure-style/kernel.py')
    path=Path(path)
    spec=importlib.util.spec_from_file_location('neuronsim_figure_style',path)
    if spec is None or spec.loader is None:
        raise ValueError('FIGURE_STYLE_HELPER_UNAVAILABLE')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,path


def _bbox_checks(fig, point_positions):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(Text)
           if t.get_visible() and t.get_text().strip()]
    frame=fig.bbox
    outside=[t.get_text() for t,b in texts
             if b.x0<frame.x0 or b.y0<frame.y0 or b.x1>frame.x1 or b.y1>frame.y1]
    overlaps=[[a.get_text(),b.get_text()] for i,(a,ba) in enumerate(texts)
              for b,bb in texts[i+1:] if ba.overlaps(bb)]
    crossings=[]
    for ax in fig.axes:
        ticks=set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for name,spine in ax.spines.items():
            if spine.get_visible():
                box=spine.get_window_extent(renderer)
                crossings.extend(dict(text=t.get_text(),spine=name) for t,b in texts
                                 if t not in ticks and b.overlaps(box))
    clipped=[]
    for ax,x,y,name in point_positions:
        px,py=ax.transData.transform((x,y))
        radius=3.5*fig.dpi/72
        if not (ax.bbox.x0+radius<px<ax.bbox.x1-radius and ax.bbox.y0+radius<py<ax.bbox.y1-radius):
            clipped.append(name)
    return dict(status='PASS' if not (outside or overlaps or crossings or clipped) else 'FAILED',
                text_count=len(texts),texts_outside_figure=outside,text_overlaps=overlaps,
                non_tick_text_spine_overlaps=crossings,clipped_markers=clipped,
                perceptual_inspection_performed=False)


CAPTION = (
    "Recruitment during three source-scheduled, 20-ms cues in one native CA3 source execution. "
    "Each cue addresses a different learned assembly at a different time and history; these are not "
    "same-stimulus repeats. Target denotes the 137 cells in the addressed assembly that are never directly "
    "boosted by the source program. Other (off-assembly) denotes the corresponding 274 cells in the other "
    "two learned assemblies, not the entire remaining network. Markers show the fraction with at least one "
    "recorded spike; labels give active/total target cells. Zero values retain visible square markers. "
    "The 138 source-eligible cells per assembly are not plotted; eligibility does not verify actual current "
    "delivery. Stems reference zero and do not connect cues. No averaging across cues, confidence intervals "
    "or inferential tests are shown. Cue-related background-current changes and different prior histories "
    "prevent state attribution. Current-delivery and internal-clock validation are reported separately. "
    "Only 1 ms of source-scheduled support follows the third cue. Pattern-completion and main-scale "
    "qualification are not established."
)


def render(readouts, replay, output, *, style_helper=None):
    if socket.gethostname().split('.')[0]=='mgmt2':
        raise RuntimeError('CA3_FIGURE_RENDER_REQUIRES_A_COMPUTE_NODE')
    readouts,replay,output=map(Path,(readouts,replay,output))
    if output.exists():
        raise FileExistsError(output)
    rows,report,proof,hashes=load_inputs(readouts,replay)
    # Keep backend/font initialization out of control-plane CLI imports.
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    helper,helper_path=_style(style_helper or os.environ.get('NEURONSIM_FIGURE_STYLE_HELPER'))
    helper.apply_figure_style(font='DejaVu Sans',sizes=(10,9,8),grid=False)
    helper_sha=sha256_file(helper_path)
    mpl.rcParams['savefig.bbox']=None
    fig,ax=plt.subplots(figsize=(5.4,3.4),dpi=300)
    fig.subplots_adjust(left=.27,right=.97,bottom=.31,top=.87)
    points=[];plotted=[];zero_count=0
    for prefix,label,color,marker,offset in (
        ('target','Target assembly: never-boosted cells','#0072B2','o',.12),
        ('other','Other two assemblies: never-boosted cells','#62676D','s',-.12)):
        for i,row in enumerate(rows):
            n=UNITS[prefix];active=_integer(row[prefix+'_cue_active_units']);fraction=active/n;y=2-i+offset
            ax.hlines(y,0,fraction,color='#BAC1C7',linewidth=.8,zorder=1)
            ax.scatter([fraction],[y],s=31,marker=marker,facecolors=color if prefix=='target' else 'white',
                       edgecolors=color,linewidths=1.,zorder=3,label=label if i==0 else None)
            if prefix=='target':
                ax.annotate(f'{active}/{n}',(fraction,y),xytext=(7,0),textcoords='offset points',
                            ha='left',va='center',fontsize=9,color=color)
            zero_count+=fraction==0
            points.append((ax,fraction,y,f'{prefix}_cue_{i+1}'))
            plotted.append(dict(cue_id=i+1,series=prefix,active_units=active,units=n,fraction=fraction))
    ax.axvline(0,color='#D5D8DA',linewidth=.6,zorder=0)
    ax.set_xlim(-.06,1.18);ax.set_ylim(-.42,2.42)
    ax.set_xticks([0,.25,.5,.75,1.]);ax.set_xticklabels(['0','0.25','0.5','0.75','1'])
    ax.set_yticks([2,1,0])
    ax.set_yticklabels([f'Assembly {i+1} / cue {i+1}\n{a}–{b} ms' for i,(a,b) in enumerate(CUE_WINDOWS)])
    ax.set_xlabel('Fraction of neurons spiking')
    ax.set_title('Recruitment during three source-scheduled cues',pad=12)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.60,.015),
               ncol=1,frameon=False,handletextpad=.6,labelspacing=.5)
    output.mkdir(parents=True,exist_ok=False)
    try:
        fig.savefig(output/'ca3_recruitment.png',dpi=300,bbox_inches=None)
        fig.savefig(output/'ca3_recruitment.pdf',bbox_inches=None,
                    metadata={'Creator':'NeuronSim ca3_readout_figure','CreationDate':None,'ModDate':None})
        qa=_bbox_checks(fig,points)
        (output/'caption.txt').write_text(CAPTION+'\n',encoding='utf-8')
        for name,digest in hashes.items():
            if sha256_file(replay if name=='replay' else readouts/name)!=digest:
                raise ValueError('CA3_FIGURE_INPUT_CHANGED_DURING_RENDER')
        result=dict(status='RENDERED_DESCRIPTIVE_CA3_FIGURE' if qa['status']=='PASS' else 'FAILED_RENDER_QA',
            created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=hashes,
            source_report_sha256=hashes['report.json'],raw_events_sha256=proof['raw_events_sha256'],
            verification_scope='bound readout artifacts and PASS independent raw-event replay; raw events not reopened',
            style_helper=str(helper_path.resolve()),style_helper_sha256=helper_sha,
            matplotlib_version=mpl.__version__,python_version=platform.python_version(),dpi=300,
            implementation_sha256=sha256_file(Path(__file__)),plotted_observations=plotted,
            visible_zero_markers=zero_count,source_eligible_cells_plotted=False,render_qa=qa,
            source_executions=1,same_stimulus_repeats=False,inferential_claim=False,
            scientific_claim_qualified=False,pattern_completion_qualified=False,state_attribution_qualified=False,
            artifacts={name:sha256_file(output/name) for name in
                       ('ca3_recruitment.png','ca3_recruitment.pdf','caption.txt')})
        with (output/'render_report.json').open('x') as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if qa['status']!='PASS':
            raise ValueError('CA3_FIGURE_GEOMETRIC_RENDER_QA_FAILED')
        return result
    finally:
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('readouts','replay','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--style-helper',type=Path,help='Staged figure-style/kernel.py; alternatively set NEURONSIM_FIGURE_STYLE_HELPER')
    args=parser.parse_args()
    result=render(args.readouts,args.replay,args.output,style_helper=args.style_helper)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
