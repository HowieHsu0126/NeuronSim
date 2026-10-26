"""Render source-backed developmental joint-forecast diagnostics without CIs."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from itertools import product
import json
from pathlib import Path

import numpy as np

from Libs.population_dynamics.provenance import sha256_file

METHODS = ('stochastic','latent_mean','independent_marginals')
LABELS = {'stochastic':'Stochastic LDS','latent_mean':'Latent mean + Poisson',
          'independent_marginals':'Independent marginals'}
COLORS = {'stochastic':'#0072B2','latent_mean':'#E69F00','independent_marginals':'#009E73'}
MARKERS = {'stochastic':'o','latent_mean':'^','independent_marginals':'s'}
LINES = {'stochastic':'-','latent_mean':'--','independent_marginals':':'}
HORIZONS = (20.,80.,500.)
METRICS = ('energy_score','unit_total_coverage_90','refractory_count_bound_violation')
SOURCE_NAMES = ('analysis.json','forecast_summary.csv','forecast_by_block.csv',
                'forecast_paired_block_differences.csv','covariance_diagnostic.csv')


def _profile(comparison):
    if comparison == 'latent_controls':
        return METHODS, METRICS, LABELS, COLORS, MARKERS, LINES
    if comparison != 'matched_observations':
        raise ValueError('unknown forecast comparison')
    return (('renewal', 'poisson'),
        ('energy_score', 'unit_total_coverage_90', 'refractory_count_violation_1bin'),
        {'renewal': 'Renewal + spike history', 'poisson': 'Poisson counts'},
        {'renewal': '#7B3294', 'poisson': '#0072B2'},
        {'renewal': 's', 'poisson': 'o'},
        {'renewal': '-', 'poisson': '--'})


def _csv(path):
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    if not rows:
        raise ValueError('empty source table: '+str(path))
    return rows


def _close(left,right):
    return bool(np.isclose(left,right,rtol=1e-10,atol=1e-14))


def _load(analysis, comparison='latent_controls'):
    methods, metrics, *_ = _profile(comparison)
    status=json.loads((analysis/'analysis.json').read_text())
    if (status['seeds']!=[41,42,43] or status['independent_blocks']!=4
            or status['scientific_claim_qualified'] is not False
            or status['formal_test_accessed'] is not False
            or status['status']!='REPLAYED_DEVELOPMENT'):
        raise ValueError('replayed three-seed/four-block developmental sources required')
    if comparison == 'matched_observations' and status.get('models') != ['poisson', 'renewal']:
        raise ValueError('matched Poisson/renewal analysis required')
    rows=_csv(analysis/'forecast_by_block.csv')
    blocks=sorted({int(row['block']) for row in rows})
    if len(blocks)!=4:
        raise ValueError('four complete realization blocks required')
    by_key={}
    for row in rows:
        key=(int(row['seed']),int(row['block']),row['method'],float(row['horizon_ms']))
        if key in by_key or int(row['episodes_origins'])!=12:
            raise ValueError('duplicate block row or changed setting denominator')
        values={metric:float(row[metric]) for metric in metrics}
        if not all(np.isfinite(list(values.values()))) or any(value<0 for value in values.values()):
            raise ValueError('invalid score in plotted source row')
        if values['unit_total_coverage_90']>1 or values[metrics[-1]]>1:
            raise ValueError('invalid fraction in plotted source row')
        by_key[key]=values
    if set(by_key)!=set(product(status['seeds'],blocks,methods,HORIZONS)):
        raise ValueError('incomplete seed/block/method/horizon grid')
    block_values={}
    for metric,method in product(metrics,methods):
        block_values[metric,method]=np.asarray([[np.mean([by_key[seed,block,method,h][metric]
            for seed in status['seeds']]) for h in HORIZONS] for block in blocks])
    summaries={(row['method'],float(row['horizon_ms']),row['metric']):row
               for row in _csv(analysis/'forecast_summary.csv')}
    for metric,method in product(metrics,methods):
        for index,h in enumerate(HORIZONS):
            row=summaries[method,h,metric]
            if (not _close(float(row['mean']),block_values[metric,method][:,index].mean())
                    or int(row['independent_blocks'])!=4 or int(row['training_seeds'])!=3):
                raise ValueError('plotted blocks disagree with canonical summary')
    paired=[row for row in _csv(analysis/'forecast_paired_block_differences.csv') if row['metric'] in metrics]
    if comparison == 'matched_observations':
        paired=[{**row, 'comparator': 'poisson',
                 'focal_minus_comparator': row['renewal_minus_poisson']} for row in paired]
    else:
        paired=[{**row, 'focal_minus_comparator': row['stochastic_minus_comparator']} for row in paired]
    seen=set()
    for row in paired:
        key=(int(row['block']),row['comparator'],float(row['horizon_ms']),row['metric'])
        if key in seen or key[1] not in methods[1:] or int(row['training_seeds'])!=3:
            raise ValueError('invalid paired block denominator')
        seen.add(key)
        b=blocks.index(key[0]);h=HORIZONS.index(key[2])
        difference=block_values[key[3],methods[0]][b,h]-block_values[key[3],key[1]][b,h]
        if not _close(difference,float(row['focal_minus_comparator'])):
            raise ValueError('paired block differences disagree with plotted source')
    if seen!=set(product(blocks,methods[1:],HORIZONS,metrics)):
        raise ValueError('missing paired block differences')
    covariance=[row for row in _csv(analysis/'covariance_diagnostic.csv') if row['statistic']=='cross_population']
    settings=sorted({(row['template_id'],float(row['origin_ms'])) for row in covariance})
    if len(settings)!=12:
        raise ValueError('12 matched template/origin covariance settings required')
    cov_by_key={}
    for row in covariance:
        key=(int(row['seed']),row['method'],(row['template_id'],float(row['origin_ms'])),float(row['horizon_ms']))
        if key in cov_by_key or int(row['independent_blocks'])!=4 or int(row['phases'])!=round(key[-1]/5):
            raise ValueError('changed covariance denominator or phase grid')
        values=tuple(float(row[k]) for k in ('observed_covariance_mean','predicted_covariance_mean'))
        if not np.isfinite(values).all():
            raise ValueError('nonfinite covariance source')
        cov_by_key[key]=values
    if set(cov_by_key)!=set(product(status['seeds'],methods,settings,HORIZONS)):
        raise ValueError('incomplete covariance source grid')
    observed=np.empty((12,3));predicted={method:np.empty((12,3)) for method in methods}
    for i,setting in enumerate(settings):
        for j,h in enumerate(HORIZONS):
            originals=[cov_by_key[seed,method,setting,h][0] for seed,method in product(status['seeds'],methods)]
            if not all(_close(value,originals[0]) for value in originals):
                raise ValueError('observed covariance changes across duplicate model/seed rows')
            observed[i,j]=originals[0]
            for method in methods:
                predicted[method][i,j]=np.mean([cov_by_key[seed,method,setting,h][1] for seed in status['seeds']])
    return status,blocks,block_values,observed,predicted,settings


def _geometry(fig,matplotlib):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[(text,text.get_window_extent(renderer)) for text in fig.findobj(matplotlib.text.Text)
           if text.get_visible() and text.get_text().strip()]
    outside=[text.get_text() for text,box in texts
             if not all(fig.bbox.contains(*point) for point in box.get_points())]
    overlaps=[(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(texts)
              for b,bb in texts[i+1:] if ba.overlaps(bb)]
    ticklabels={ax:set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both')) for ax in fig.axes}
    spine_overlaps=[(text.get_text(),side) for text,box in texts for ax in fig.axes
                    for side,spine in ax.spines.items() if spine.get_visible()
                    and text not in ticklabels[ax] and box.overlaps(spine.get_window_extent(renderer))]
    return dict(text_outside_canvas=outside,text_overlaps=overlaps,text_spine_overlaps=spine_overlaps)


def render(analysis,style_helper,output,unit_count,latent_dim,comparison='latent_controls'):
    methods, metrics, labels, colors, markers, lines = _profile(comparison)
    refractory_metric=metrics[-1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
    from PIL import Image
    spec=importlib.util.spec_from_file_location('forecast_figure_style',style_helper)
    style=importlib.util.module_from_spec(spec);spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(9,8,7))
    status,blocks,values,observed,predicted,settings=_load(analysis,comparison)
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.0))
    fig.subplots_adjust(left=.12,right=.985,bottom=.14,top=.815,wspace=.37,hspace=.52)
    titles=['Joint field prediction','Discrete central 90% intervals',
            'Cross-population covariance','Refractory count bounds']
    for ax,title,letter in zip(axes.flat,titles,'abcd'):
        ax.set_title(title,pad=7)
        style.panel_letter(ax,letter,dx=-.17,dy=1.015)
        ax.set_xscale('log');ax.set_xlim(15,670)
        ax.xaxis.set_major_locator(FixedLocator(HORIZONS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:str(int(x))))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis='y',color='#E7E7E7',linewidth=.4,zorder=0)
    for ax in axes[1]:ax.set_xlabel('Forecast horizon (ms; log scale)')
    for ax,metric,multiplier in ((axes[0,0],'energy_score',1.),
            (axes[0,1],'unit_total_coverage_90',1.),
            (axes[1,1],refractory_metric,1e6)):
        for method in methods[::-1]:
            y=values[metric,method]*multiplier
            for row in y:
                ax.plot(HORIZONS,row,color=colors[method],alpha=.25,lw=.65,zorder=1)
            ax.plot(HORIZONS,y.mean(axis=0),color=colors[method],linestyle=lines[method],
                marker=markers[method],ms=4.2 if method==methods[0] else 5.2,
                markerfacecolor=colors[method] if method==methods[0] else 'white',
                markeredgewidth=.9,lw=1.65 if method==methods[0] else 1.2,zorder=4)
        ax.margins(y=.10)
    axes[0,0].set_ylabel('Energy score')
    axes[0,0].text(.97,.05,'Lower is better',transform=axes[0,0].transAxes,
                      ha='right',va='bottom',fontsize=8,color='#666666')
    axes[0,1].set_ylabel('Observed coverage')
    axes[0,1].axhline(.9,color='#555555',ls=':',lw=.9,zorder=0)
    axes[0,1].text(.97,.905,'Nominal 0.90',transform=axes[0,1].get_yaxis_transform(),
                      ha='right',va='bottom',fontsize=8,color='#555555')
    coverage=np.concatenate([values['unit_total_coverage_90',m].ravel() for m in methods])
    axes[0,1].set_ylim(min(coverage.min()-.035,.85),1.005)
    lo,hi=axes[0,1].get_ylim()
    axes[0,1].yaxis.set_major_locator(FixedLocator([v for v in np.arange(.5,1.01,.1) if lo<=v<=hi]))
    axes[0,1].yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:.1f}'))
    axes[1,1].set_ylabel('Violations\n(per 10⁶ unit-bins)')
    violations=np.concatenate([values[refractory_metric,m].ravel()*1e6 for m in methods])
    axes[1,1].set_ylim(-.045*max(violations.max(),.01),1.12*max(violations.max(),.01))
    axes[1,1].axhline(0,color='#888888',lw=.5,zorder=0)
    covariance_ax=axes[1,0]
    jitter=np.exp(np.linspace(-.065,.065,len(settings)))
    for method in methods[::-1]:
        matrix=predicted[method]
        for j,h in enumerate(HORIZONS):
            covariance_ax.scatter(h*jitter,matrix[:,j],s=8,marker=markers[method],
                color=colors[method],alpha=.25,linewidths=0,zorder=1)
        covariance_ax.plot(HORIZONS,matrix.mean(0),color=colors[method],ls=lines[method],
            marker=markers[method],ms=4.2 if method==methods[0] else 5.2,
            markerfacecolor=colors[method] if method==methods[0] else 'white',lw=1.3,zorder=3)
    for j,h in enumerate(HORIZONS):
        covariance_ax.scatter(h*jitter,observed[:,j],s=9,color='#555555',alpha=.3,linewidths=0,zorder=1)
    covariance_ax.plot(HORIZONS,observed.mean(0),color='#303030',marker='D',ms=4,lw=1.4,
                       label='Native source',zorder=4)
    covariance_ax.legend(loc='upper left',fontsize=8,handlelength=1.5)
    covariance_ax.set_ylabel('Mean covariance\n(standardized field units²)')
    covariance_ax.set_yscale('symlog',linthresh=.001,linscale=.6)
    covariance_values=np.concatenate([observed.ravel(),*(predicted[m].ravel() for m in methods)])
    lower=min(-.0015,float(covariance_values.min())*1.5)
    upper=max(.005,float(covariance_values.max())*1.7)
    covariance_ax.set_ylim(lower,upper)
    major=[v for v in [-10,-1,-.1,-.01,-.001,0,.001,.01,.1,1,10] if lower<=v<=upper]
    covariance_ax.yaxis.set_major_locator(FixedLocator(major))
    covariance_ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    covariance_ax.yaxis.set_minor_locator(NullLocator())
    covariance_ax.axhline(0,color='#888888',lw=.6,zorder=0)
    fig.suptitle('Population-forecast diagnostics on epoch-selection blocks',y=.984,fontsize=9)
    fig.text(.5,.951,f'{unit_count:,} cortical units · latent dimension {latent_dim} · 3 training seeds · 4 realization blocks',
             ha='center',va='top',fontsize=8)
    handles=[Line2D([],[],color=colors[m],ls=lines[m],marker=markers[m],lw=1.4,ms=4,
                   markerfacecolor=colors[m] if m==methods[0] else 'white',label=labels[m]) for m in methods]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.921),ncol=3,
               handlelength=2.1,columnspacing=1.4,fontsize=8)
    fig.text(.5,.025,'Thin curves: four block means across three seeds. Covariance dots: matched settings, each using four blocks.\n'
        'Descriptive validation results; no inferential intervals or scientific qualification.',
        ha='center',va='bottom',fontsize=8,linespacing=1.25)
    geometry=_geometry(fig,matplotlib)
    stem=('matched_count_forecast_diagnostics' if comparison=='matched_observations'
          else 'trajectory_forecast_diagnostics')
    for suffix in ('png','pdf','svg'):
        fig.savefig(output/f'{stem}.{suffix}')
    crops=style.panel_crops(fig)
    with Image.open(output/f'{stem}.png') as picture:
        for letter,box in crops.items():picture.crop(box).save(output/f'panel_{letter}.png')
    caption=(
      'Developmental forecasts on the same 100,000-cortical-unit source graph, using a 16-dimensional controlled Poisson LDS. '
      'Training seeds 41, 42 and 43 all contribute. The four initial-state/background validation blocks (14–17) also selected epochs; '
      'they are not an independent final test. Panels a, b and d show four thin block curves, each averaged over all three seeds and '
      '12 template/origin settings; thick curves average the four blocks. No confidence interval or p-value is shown. '
      '(a) Reported energy score on joint standardized rate fields; lower is better. Field rates are counts divided by cortical '
      'units in the field and 0.005 seconds, standardized separately using fit blocks 0–13 only. The Euclidean norm is normalized '
      'by the square root of the number of time-by-field coordinates. (b) Coverage of discrete central 90% intervals for each unit’s '
      'total forecast count. The nominal 0.90 line is a reference, not evidence of calibration. '
      '(c) Cross-population covariance in squared standardized-field units. Each faint point averages the declared pairs and phases '
      'within one matched template/origin setting; predictive points additionally average all three seeds. Each covariance estimate '
      'uses four realization blocks. Native observations are deduplicated across model/seed rows. Thick curves average all 12 settings. '
      'This is total predictive covariance across contexts, not conditional covariance at one fixed inferred state. The symmetric-log '
      'axis is linear within ±0.001; zero and any negative reported values remain visible. Points are horizontally jittered only for visibility. '
      '(d) Reported refractory-count-bound violation fractions multiplied by one million; zero values are retained. '
      'The latent-mean ablation retains Poisson sampling and can change marginal means as well as dependence. '
      'All outputs are 5 ms count-space diagnostics; the source simulator and scientific claims remain unqualified.'
    )
    if comparison == 'matched_observations':
        caption=(
            'Matched count/field forecasts from Poisson and refractory-renewal observation models, each using the same '
            '16-dimensional stochastic latent dynamics and encoder capacity. Both use all 100,000 identified cortical units, '
            'the same 56 fitting episodes, 20-epoch maximum, initialization, and three training seeds (41,42,43). '
            'Renewal additionally uses native event-clock exposure and boundary availability; its training likelihood and '
            'history information differ from the unordered Poisson count model. They are compared only on the same count '
            'and projected-field forecasts. The four realization blocks 14–17 also selected epochs; these are descriptive '
            'development results, with no inferential intervals or significance tests. '
            '(a,b,d) Thin lines show four block means across three seeds and all twelve template/origin settings; thick '
            'lines average the blocks. (a) Joint field energy score, standardized from fitting blocks only; lower is better. '
            '(b) Coverage of discrete central 90% intervals for unit total counts; the dotted nominal level does not imply calibration. '
            '(c) Cross-population covariance at twelve matched settings, each estimated across four realization blocks. '
            'Native observations are deduplicated across seeds and models; predictive covariance includes within-state '
            'and between-context variation. This is not identified conditional covariance at one inferred state. '
            'The symmetric-log axis is linear within plus/minus 0.001; signed values are retained. '
            '(d) Count-support violations for the native refractory bound in one-bin windows, per million unit bins. '
            'Zero renewal values are shown explicitly. All horizons use sixteen joint samples at 5ms resolution. '
            'Budget termination is not demonstrated convergence; source and generative fidelity remain unqualified.')
    (output/f'{stem}.txt').write_text(caption+'\n')
    receipt=dict(status='RENDERED_PENDING_VISUAL_INSPECTION',
        source_hashes={name:sha256_file(analysis/name) for name in SOURCE_NAMES},
        renderer_sha256=sha256_file(Path(__file__)),style_helper_sha256=sha256_file(style_helper),
        seeds=status['seeds'],blocks=blocks,methods=list(methods),horizons_ms=HORIZONS,comparison=comparison,
        cortical_units=unit_count,latent_dimension=latent_dim,confidence_intervals=False,p_values=False,
        scientific_claim_qualified=False,formal_test_accessed=False,
        source_summary_and_paired_differences_verified=True,
        covariance_settings=len(settings),covariance_replicates_per_setting=4,
        covariance_axis=dict(scale='symlog',linear_threshold=.001),
        covariance_signed_values_preserved=True,
        block_series={f'{metric}:{method}':matrix.tolist() for (metric,method),matrix in values.items()},
        covariance_observed_means=observed.mean(0).tolist(),
        covariance_predicted_means={m:predicted[m].mean(0).tolist() for m in methods},
        geometry=geometry,panel_crop_boxes=crops,
        artifacts={f'{stem}.{ext}':sha256_file(output/f'{stem}.{ext}') for ext in ('png','pdf','svg','txt')})
    (output/'render_checks.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    plt.close(fig)
    print(json.dumps(dict(output=str(output),geometry=geometry,methods=list(methods),blocks=blocks)))
    if any(geometry.values()):raise ValueError('render geometry requires repair')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--style-helper',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--unit-count',type=int,required=True)
    parser.add_argument('--latent-dim',type=int,required=True)
    parser.add_argument('--comparison',choices=('latent_controls','matched_observations'),default='latent_controls')
    args=parser.parse_args()
    render(args.analysis,args.style_helper,args.output,args.unit_count,args.latent_dim,args.comparison)


if __name__=='__main__':main()
