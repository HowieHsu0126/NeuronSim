"""Render replay-backed LFADS conditioning diagnostics, without inferential claims."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from itertools import product
import json
from pathlib import Path

import numpy as np

from .lfads_boundary_analysis import BRANCHES, SEEDS
from .provenance import sha256_file
from .trajectory_forecast_figures import _geometry

BLOCKS = (14, 15, 16, 17)
HORIZONS = (20., 80., 500.)
TEMPLATES = tuple(f'familiar_{i}' for i in range(4))
METRICS = ('conditional_expected_spikes_per_unit', 'observed_spikes_per_unit', 'energy_score')
INNOVATION_METRIC = 'mean_square_per_channel'
UNCROSSED = (BRANCHES[0], BRANCHES[3])
LABELS = dict(zip(BRANCHES, ('Prefix boundary + prior input (forecast)', 'Response boundary + prior input',
                           'Prefix boundary + posterior input', 'Response boundary + posterior input')))
COLORS = dict(zip(BRANCHES, ('#0072B2', '#E69F00', '#7B3294', '#009E73')))
MARKERS = dict(zip(BRANCHES, ('o', 's', '^', 'v')))
LINES = dict(zip(BRANCHES, ('-', '--', ':', '-.')))
TABLES = ('diagnostic_all_seeds.csv', 'diagnostic_by_block.csv', 'diagnostic_summary.csv',
          'covariance_diagnostic.csv', 'covariance_summary.csv')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _close(actual, expected, message):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-14):
        raise ValueError(message)


def _index(rows, key, expected, label):
    indexed = {}
    for row in rows:
        item = key(row)
        if item in indexed or item not in expected:
            raise ValueError(f'duplicate or unexpected {label} row')
        indexed[item] = row
    if set(indexed) != set(expected):
        raise ValueError(f'incomplete {label} grid')
    return indexed


def _finite(row, names, *, nonnegative=False):
    values = np.asarray([float(row[name]) for name in names])
    if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
        raise ValueError('nonfinite or inadmissible plotted value')
    return values


def _csv(folder, status, name):
    path = folder / name
    if sha256_file(path) != status['artifacts'][name]:
        raise ValueError('plotted source hash changed: ' + name)
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _scope(status, controller):
    expected_config = dict(training_seeds=list(SEEDS), branches=list(BRANCHES),
        selection_blocks=list(BLOCKS), expected_cases_per_seed=16, expected_units=100000,
        joint_samples=16, horizons_ms=list(HORIZONS), bin_ms=5.,
        original_history_bins=200, original_horizon_bins=1000, forecast_horizon_bins=100,
        full_response_encoder_batch='joint_samples', formal_evaluation=False, update_weights=False)
    if (status['status'] != 'DESCRIBED_REPLAYED_DEVELOPMENT'
            or status['training_seeds'] != list(SEEDS) or status['independent_blocks'] != 4
            or status['seed_case_pairs'] != 48 or status['branches'] != 192 or status['score_rows'] != 576
            or status['connectivity_instances'] != 1
            or status['scientific_claim_qualified'] is not False or status['formal_test_accessed'] is not False
            or any(status['configuration'].get(k) != v for k, v in expected_config.items())):
        raise ValueError('complete registered three-seed selection diagnostics required')
    if (controller['status'] != 'DESCRIBED_REPLAYED_CONTROLLER_INNOVATIONS'
            or controller['training_seeds'] != list(SEEDS) or controller['branches'] != list(UNCROSSED)
            or controller['selection_cases_per_seed'] != 16 or controller['channel_rows'] != 1152
            or controller['case_horizon_rows'] != 288 or controller['controller_artifacts_verified'] != 96
            or controller['scientific_claim_qualified'] is not False or controller['formal_test_accessed'] is not False
            or controller['selection_only'] is not True
            or controller['channel_coordinates_pooled_across_seeds'] is not False):
        raise ValueError('complete unpooled controller diagnostics required')
    if (status['configuration'] != controller['configuration']
            or status['source_binding'] != controller['source_binding']):
        raise ValueError('boundary/controller source or protocol binding differs')
    if any(set(report['model_seed_bindings']) != {str(s) for s in SEEDS} for report in (status, controller)):
        raise ValueError('three model bindings required')
    for seed in SEEDS:
        left, right = (report['model_seed_bindings'][str(seed)] for report in (status, controller))
        for key in ('evaluation_sha256', 'replay_sha256', 'selected_model_artifacts'):
            if not left[key] or left[key] != right[key]:
                raise ValueError('boundary/controller evaluation, replay or checkpoint binding differs')


def _metric_values(tables):
    raw = _index(tables['diagnostic_all_seeds.csv'],
        lambda r: (int(r['seed']), int(r['block']), r['template_id'], r['branch'], float(r['horizon_ms'])),
        set(product(SEEDS, BLOCKS, TEMPLATES, BRANCHES, HORIZONS)), 'case')
    for (seed, block, template, branch, horizon), row in raw.items():
        role = 'forecast' if branch == BRANCHES[0] else 'response_conditioned_diagnostic'
        if (row['case_id'] != f'train_block_{block:03d}_{template}'
                or row['role'] != role or float(row['origin_ms']) != 1000.):
            raise ValueError('case identity, forecast role or origin differs')
        _finite(row, METRICS, nonnegative=True)
    for block, template, horizon in product(BLOCKS, TEMPLATES, HORIZONS):
        observed = [float(raw[s, block, template, branch, horizon]['observed_spikes_per_unit'])
                    for s, branch in product(SEEDS, BRANCHES)]
        _close(observed, observed[0], 'native count reference changes across duplicate branches/seeds')
    blocks = _index(tables['diagnostic_by_block.csv'],
        lambda r: (int(r['seed']), int(r['block']), r['branch'], float(r['horizon_ms'])),
        set(product(SEEDS, BLOCKS, BRANCHES, HORIZONS)), 'block')
    values = {metric: {branch: np.empty((3, 4, 3)) for branch in BRANCHES} for metric in METRICS}
    for i, seed in enumerate(SEEDS):
        for j, block in enumerate(BLOCKS):
            for branch in BRANCHES:
                for k, horizon in enumerate(HORIZONS):
                    member = blocks[seed, block, branch, horizon]
                    expected_values = np.mean([_finite(raw[seed, block, t, branch, horizon], METRICS)
                                               for t in TEMPLATES], axis=0)
                    _close(_finite(member, METRICS, nonnegative=True), expected_values,
                           'block values disagree with complete case table')
                    for metric, value in zip(METRICS, expected_values):
                        values[metric][branch][i, j, k] = value
    summaries = _index([r for r in tables['diagnostic_summary.csv'] if r['metric'] in METRICS],
        lambda r: (r['branch'], float(r['horizon_ms']), r['metric']),
        set(product(BRANCHES, HORIZONS, METRICS)), 'metric summary')
    for branch, metric in product(BRANCHES, METRICS):
        for k, horizon in enumerate(HORIZONS):
            row = summaries[branch, horizon, metric]
            means = values[metric][branch][:, :, k].mean(axis=1)
            if int(row['training_seeds']) != 3 or int(row['independent_blocks']) != 4:
                raise ValueError('metric summary denominator differs')
            _close(_finite(row, ('mean', 'minimum_seed_mean', 'maximum_seed_mean')),
                   [means.mean(), means.min(), means.max()], 'plotted values disagree with canonical summary')
    return values


def _covariance_values(tables):
    rows = [r for r in tables['covariance_diagnostic.csv'] if r['statistic'] == 'cross_population']
    indexed = _index(rows,
        lambda r: (int(r['seed']), r['branch'], r['template_id'], float(r['horizon_ms'])),
        set(product(SEEDS, BRANCHES, TEMPLATES, HORIZONS)), 'cross-population covariance')
    fields = ('observed_covariance_mean', 'predicted_covariance_mean', 'covariance_mse', 'covariance_mae')
    pair_counts = set()
    values = {branch: np.empty((3, 4, 3)) for branch in BRANCHES}
    observed = np.empty((4, 3))
    for row in rows:
        if int(row['independent_blocks']) != 4 or int(row['phases']) != int(float(row['horizon_ms']) / 5):
            raise ValueError('covariance replicate or phase denominator differs')
        pair_counts.add(int(row['pairs']))
        checked = _finite(row, fields)
        if (checked[2:] < 0).any():
            raise ValueError('negative covariance error')
    if len(pair_counts) != 1 or min(pair_counts) <= 0:
        raise ValueError('covariance pair set differs')
    for j, template in enumerate(TEMPLATES):
        for k, horizon in enumerate(HORIZONS):
            originals = [float(indexed[s, b, template, horizon][fields[0]]) for s, b in product(SEEDS, BRANCHES)]
            _close(originals, originals[0], 'native covariance changes across duplicate branches/seeds')
            observed[j, k] = originals[0]
            for i, seed in enumerate(SEEDS):
                for branch in BRANCHES:
                    values[branch][i, j, k] = float(indexed[seed, branch, template, horizon][fields[1]])
    summaries = _index([r for r in tables['covariance_summary.csv'] if r['statistic'] == 'cross_population'],
        lambda r: (r['branch'], float(r['horizon_ms'])), set(product(BRANCHES, HORIZONS)), 'covariance summary')
    for (branch, horizon), row in summaries.items():
        expected = np.mean([_finite(indexed[s, branch, t, horizon], fields) for s, t in product(SEEDS, TEMPLATES)], axis=0)
        if int(row['training_seeds']) != 3 or int(row['independent_blocks']) != 4:
            raise ValueError('covariance summary denominator differs')
        _close(_finite(row, fields), expected, 'plotted covariance disagrees with canonical summary')
    return values, observed


def _controller_values(controller):
    raw = _index(controller['case_scalar_moments'],
        lambda r: (int(r['seed']), int(r['block']), r['template_id'], r['branch'], float(r['horizon_ms'])),
        set(product(SEEDS, BLOCKS, TEMPLATES, UNCROSSED, HORIZONS)), 'controller case')
    values = {branch: np.empty((3, 4, 3)) for branch in UNCROSSED}
    for (seed, block, template, branch, horizon), row in raw.items():
        if (row['case_id'] != f'train_block_{block:03d}_{template}' or row['origin_ms'] != 1000.
                or row['role'] != ('forecast' if branch == BRANCHES[0] else 'response_conditioned_diagnostic')
                or row['vector_innovations'] != 16 * int(horizon / 5.)
                or row['scalar_innovations'] != 4 * row['vector_innovations']):
            raise ValueError('controller case identity or innovation denominator differs')
        norm, moment = _finite(row, ('mean_squared_norm', INNOVATION_METRIC), nonnegative=True)
        _close(norm, 4 * moment, 'scalar controller energy is inconsistent with four channels')
    for i, seed in enumerate(SEEDS):
        for j, block in enumerate(BLOCKS):
            for branch in UNCROSSED:
                for k, horizon in enumerate(HORIZONS):
                    values[branch][i, j, k] = np.mean([raw[seed, block, t, branch, horizon][INNOVATION_METRIC]
                                                       for t in TEMPLATES])
    by_seed = _index(controller['scalar_moments_by_seed'],
        lambda r: (r['seed'], r['branch'], r['horizon_ms']), set(product(SEEDS, UNCROSSED, HORIZONS)),
        'controller seed summary')
    for i, seed in enumerate(SEEDS):
        for branch in UNCROSSED:
            for k, horizon in enumerate(HORIZONS):
                row = by_seed[seed, branch, horizon]
                if (row['cases'] != 16 or row['vector_innovations'] != 16 * 16 * int(horizon / 5)
                        or row['scalar_innovations'] != 4 * row['vector_innovations']):
                    raise ValueError('controller seed denominator differs')
                mean = values[branch][i, :, k].mean()
                _close(_finite(row, ('mean_squared_norm', INNOVATION_METRIC)), [4 * mean, mean],
                       'controller seed summary disagrees with complete cases')
    summaries = _index([r for r in controller['scalar_summary_across_seeds'] if r['metric'] == INNOVATION_METRIC],
        lambda r: (r['branch'], r['horizon_ms']), set(product(UNCROSSED, HORIZONS)), 'controller scalar summary')
    for (branch, horizon), row in summaries.items():
        means = values[branch][:, :, HORIZONS.index(horizon)].mean(axis=1)
        if (row['training_seeds'] != 3 or row['selection_cases_per_seed'] != 16
                or row['channel_coordinates_pooled_across_seeds'] is not False):
            raise ValueError('controller scalar summary scope differs')
        _close(_finite(row, ('mean_of_seed_means', 'minimum_seed_mean', 'maximum_seed_mean')),
               [means.mean(), means.min(), means.max()], 'plotted controller moments disagree with canonical summary')
    return values


def load_sources(analysis, controller_analysis):
    """Validate small analysis artifacts instead of re-reading model/count arrays."""
    analysis, controller_analysis = Path(analysis), Path(controller_analysis)
    status = _read(analysis / 'analysis.json')
    controller = _read(controller_analysis / 'controller_diagnostics.json')
    _scope(status, controller)
    tables = {name: _csv(analysis, status, name) for name in TABLES}
    covariance, covariance_observed = _covariance_values(tables)
    return dict(status=status, controller=controller, metrics=_metric_values(tables),
                covariance=covariance, covariance_observed=covariance_observed,
                innovations=_controller_values(controller))


def _draw_series(ax, values, branches, *, all_points=False):
    for branch in reversed(branches):
        matrix = values[branch]
        for mean in matrix.mean(axis=1):
            ax.plot(HORIZONS, mean, color=COLORS[branch], lw=.6, alpha=.32, zorder=1)
        points = matrix.reshape(-1, 3) if all_points else matrix.mean(axis=0)
        jitter = np.exp(np.linspace(-.055, .055, len(points)))
        for k, horizon in enumerate(HORIZONS):
            ax.scatter(horizon * jitter, points[:, k], s=8, color=COLORS[branch],
                       marker=MARKERS[branch], linewidths=0, alpha=.3, zorder=2)
        ax.plot(HORIZONS, matrix.mean(axis=(0, 1)), color=COLORS[branch], ls=LINES[branch],
                marker=MARKERS[branch], ms=4.3, lw=1.6 if branch == BRANCHES[0] else 1.2,
                markerfacecolor=COLORS[branch] if branch == BRANCHES[0] else 'white',
                markeredgewidth=.9, zorder=4)


def _nonnegative_axis(ax, values):
    """Use logarithms only when every plotted value is strictly positive."""
    values = np.asarray(values, dtype=float)
    if values.min() > 0 and values.max() / values.min() > 40:
        ax.set_yscale('log')
        ax.margins(y=.16)
        return 'log'
    ax.margins(y=.12)
    return 'linear'


def render(analysis, controller_analysis, style_helper, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from PIL import Image

    analysis, controller_analysis, style_helper, output = map(Path, (analysis, controller_analysis, style_helper, output))
    data = load_sources(analysis, controller_analysis)
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location('lfads_boundary_figure_style', style_helper)
    if spec is None or spec.loader is None:
        raise ValueError('loadable explicit figure style helper required')
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.apply_figure_style(sizes=(9, 8, 7))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.775, wspace=.43, hspace=.53)
    titles = ('Conditional mean unit counts', 'Joint population-field prediction',
              'Signed cross-population covariance', 'Standardized controller innovations')
    for ax, title, letter in zip(axes.flat, titles, 'abcd'):
        ax.set_title(title, loc='left', pad=7)
        style.panel_letter(ax, letter, dx=-.19, dy=1.11)
        ax.set_xscale('log')
        ax.set_xlim(15, 670)
        ax.xaxis.set_major_locator(FixedLocator(HORIZONS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: str(int(value))))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis='y', color='#E7E7E7', linewidth=.4, zorder=0)
    for ax in axes[1]:
        ax.set_xlabel('Future horizon (ms; log scale)')
    counts = data['metrics'][METRICS[0]]
    energy = data['metrics']['energy_score']
    for ax, values in ((axes[0, 0], counts), (axes[0, 1], energy)):
        _draw_series(ax, values, BRANCHES)
    observed = data['metrics']['observed_spikes_per_unit'][BRANCHES[0]][0]
    axes[0, 0].plot(HORIZONS, observed.mean(axis=0), color='#303030', marker='D', ms=3.6, lw=1.2, zorder=5)
    for row in observed:
        axes[0, 0].plot(HORIZONS, row, color='#303030', lw=.55, alpha=.25, zorder=1)
    axis_scales = {}
    for name, ax, values, label in (
            ('counts', axes[0, 0], np.concatenate([observed.ravel(), *(v.ravel() for v in counts.values())]), 'Spikes per unit'),
            ('energy', axes[0, 1], np.concatenate([v.ravel() for v in energy.values()]), 'Energy score')):
        axis_scales[name] = _nonnegative_axis(ax, values)
        scale_note = 'log scale' if axis_scales[name] == 'log' else ''
        if name == 'energy':
            scale_note = 'lower is better' + ('; ' + scale_note if scale_note else '')
        ax.set_ylabel(label + ('\n(' + scale_note + ')' if scale_note else ''))
    _draw_series(axes[1, 0], data['covariance'], BRANCHES, all_points=True)
    observed_cov = data['covariance_observed']
    jitter = np.exp(np.linspace(-.055, .055, 4))
    for k, horizon in enumerate(HORIZONS):
        axes[1, 0].scatter(horizon*jitter, observed_cov[:, k], color='#303030', marker='D',
                          s=8, alpha=.4, linewidths=0, zorder=2)
    axes[1, 0].plot(HORIZONS, observed_cov.mean(axis=0), color='#303030', marker='D', ms=3.6, lw=1.2, zorder=5)
    axes[1, 0].set_yscale('symlog', linthresh=.001, linscale=.6)
    axes[1, 0].axhline(0, color='#AAAAAA', lw=.5, zorder=0)
    axes[1, 0].margins(y=.14)
    axes[1, 0].set_ylabel('Covariance (field units²)\n(symmetric log scale)')
    axes[1, 0].yaxis.set_minor_locator(NullLocator())
    _draw_series(axes[1, 1], data['innovations'], UNCROSSED)
    axes[1, 1].axhline(1, color='#555555', ls=':', lw=1, zorder=0)
    innovation_values = np.concatenate([[1.], *(v.ravel() for v in data['innovations'].values())])
    axis_scales['innovations'] = _nonnegative_axis(axes[1, 1], innovation_values)
    axes[1, 1].set_ylabel('Mean square per channel' + ('\n(log scale)' if axis_scales['innovations'] == 'log' else ''))
    # Matplotlib retains off-axis log Tick objects as visible; restrict ticks to
    # the displayed interval so both the figure and bbox checks use drawn labels.
    for ax in axes.flat:
        lo, hi = ax.get_ylim()
        ax.yaxis.set_major_locator(FixedLocator([v for v in ax.get_yticks() if lo <= v <= hi]))
        ax.yaxis.set_minor_locator(NullLocator())
    fig.suptitle('Response conditioning at the original LFADS boundary', y=.992, fontsize=9)
    fig.text(.5, .96, '100k cortical units · 3 training seeds · 4 epoch-selection blocks', ha='center', va='top', fontsize=8)
    handles = [Line2D([], [], color=COLORS[b], ls=LINES[b], marker=MARKERS[b], lw=1.3, ms=4,
                      markerfacecolor=COLORS[b] if b == BRANCHES[0] else 'white', label=LABELS[b]) for b in BRANCHES]
    handles += [Line2D([], [], color='#303030', marker='D', lw=1.2, ms=3.6, label='Native counts / covariance'),
                Line2D([], [], color='#555555', ls=':', lw=1, label='AR innovation reference: 1')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .927), ncol=2,
               handlelength=2.1, columnspacing=1.5, fontsize=8)
    fig.text(.5, .105, 'Only prefix boundary + prior input is a forecast. The other three branches use future responses.',
             ha='center', va='top', fontsize=8)
    fig.text(.5, .071, 'Thin curves: three seed means. Thick curves: overall means. Points: block or covariance-setting values.\n'
             'Selection-block diagnostics; no inferential intervals or biological causal claims.',
             ha='center', va='top', fontsize=8, linespacing=1.3)
    geometry = _geometry(fig, matplotlib)
    output.mkdir(parents=True, exist_ok=False)
    stem = 'lfads_boundary_diagnostics'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{stem}.{suffix}')
    crops = style.panel_crops(fig)
    with Image.open(output / f'{stem}.png') as picture:
        for letter, box in crops.items():
            picture.crop(box).save(output / f'panel_{letter}.png')
    caption = (
        'Replayed LFADS model-conditioning diagnostics on 100,000 identified cortical units. All three training seeds '
        '(41, 42, 43), four epoch-selection initial/background blocks (14–17), and four stimulus templates contribute. '
        'There is one connectivity instance. The origin is 1,000 ms, with 200 history bins and 100 future bins of 5 ms; '
        'each branch uses 16 joint draws. The four branches cross prefix versus full-response controller conditioning '
        'of the boundary state with prior versus posterior future controller inputs. Only the prefix-boundary/prior-input '
        'branch is a causal forecast. The other branches use future responses and are diagnostics, not microscopic '
        'oracles or mechanistic-simulator interventions. The original IC encoder is restricted to the first 200 bins '
        'in both training and inference; full-response conditioning concerns the controller encoder. '
        '(a) Conditional expected total spikes per unit, averaged over every unit and joint draw, with the native observed '
        'reference deduplicated across branches and seeds. (b) Joint field energy score in fitting-standardized field '
        'coordinates, lower is better. In (a,b,d), thin colored curves show each seed mean over all 16 cases; faint '
        'points show four block means over three seeds and four templates, and thick curves show the overall means. '
        'Native thin curves in (a) show four block means. '
        '(c) Cross-population covariance averaged over the declared field pairs and phases. Each faint predictive point '
        'is one seed/template setting and each native point one template; each covariance uses all four realization '
        'blocks. Thin colored curves average templates within a seed; thick curves average seeds and templates. '
        'This is total predictive covariance across the selected contexts, not conditional covariance at one inferred '
        'state. The symmetric-log axis is linear within plus/minus 0.001 and preserves zero and negative values. '
        '(d) For the two uncrossed branches only, standardized controller innovations are (u_t - phi*u_previous)/sigma, '
        'using the saved last historical input at the first step and checkpoint-bound AR parameters. The scalar plotted '
        'is the mean squared norm divided by four channels. No latent-coordinate means are pooled across seeds. '
        'One is the mathematical unit-second-moment reference, not an empirical calibration threshold. Agreement of '
        'this moment does not establish equal distributions or an adequate prior. Horizontal point jitter is visual only. '
        'Three seeds and recurrent selection blocks are not independent biological replications; no inferential '
        'confidence intervals, p-values or FDR claims are shown. These diagnostics establish neither source fidelity '
        'nor intervention success nor a neuroscientific finding.'
    )
    (output / f'{stem}.txt').write_text(caption + '\n', encoding='utf-8')
    source_hashes = {f'analysis/{name}': sha256_file(analysis / name) for name in ('analysis.json', *TABLES)}
    source_hashes['controller_analysis/controller_diagnostics.json'] = sha256_file(controller_analysis / 'controller_diagnostics.json')
    artifacts = [f'{stem}.{ext}' for ext in ('png', 'pdf', 'svg', 'txt')] + [f'panel_{letter}.png' for letter in crops]
    receipt = dict(status='RENDERED_PENDING_VISUAL_INSPECTION', source_hashes=source_hashes,
        renderer_sha256=sha256_file(Path(__file__)), style_helper_sha256=sha256_file(style_helper),
        training_seeds=list(SEEDS), epoch_selection_blocks=list(BLOCKS), branches=list(BRANCHES),
        causal_forecast_branch=BRANCHES[0], unit_count=100000, horizons_ms=list(HORIZONS),
        summary_consistency_verified=True, cross_analysis_bindings_verified=True,
        channel_coordinates_pooled_across_seeds=False, covariance_signed_values_preserved=True,
        covariance_axis=dict(scale='symlog', linear_threshold=.001), other_axis_scales=axis_scales,
        confidence_intervals=False, p_values=False, scientific_claim_qualified=False, formal_test_accessed=False,
        geometry=geometry, panel_crop_boxes=crops,
        artifacts={name: sha256_file(output / name) for name in artifacts})
    (output / 'render_checks.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    plt.close(fig)
    if any(geometry.values()):
        raise ValueError('render geometry requires repair: ' + json.dumps(geometry))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--controller-analysis', type=Path, required=True)
    parser.add_argument('--style-helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = render(args.analysis, args.controller_analysis, args.style_helper, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'training_seeds', 'geometry')}))


if __name__ == '__main__':
    main()

