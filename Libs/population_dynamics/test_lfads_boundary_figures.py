"""Meaningful source-scope, consistency, and export tests using temporary fixtures."""
from __future__ import annotations

import copy
import csv
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .lfads_boundary_figures import (BLOCKS, BRANCHES, HORIZONS, METRICS, SEEDS, TABLES,
                                     TEMPLATES, UNCROSSED, load_sources, render)
from .provenance import sha256_file


class BoundaryFigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.analysis = self.root / 'analysis'
        self.controllers = self.root / 'controller_analysis'
        self.analysis.mkdir(); self.controllers.mkdir()
        config = dict(training_seeds=list(SEEDS), branches=list(BRANCHES),
            selection_blocks=list(BLOCKS), expected_cases_per_seed=16, expected_units=100000,
            joint_samples=16, horizons_ms=list(HORIZONS), bin_ms=5.,
            original_history_bins=200, original_horizon_bins=1000, forecast_horizon_bins=100,
            full_response_encoder_batch='joint_samples', formal_evaluation=False, update_weights=False)
        bindings = {str(s): dict(evaluation_sha256=f'{s:064d}', replay_sha256=f'{s+10:064d}',
                    selected_model_artifacts={'checkpoint.npz': f'{s+20:064d}'}) for s in SEEDS}
        self.status = dict(status='DESCRIBED_REPLAYED_DEVELOPMENT', training_seeds=list(SEEDS),
            seed_case_pairs=48, branches=192, score_rows=576, independent_blocks=4, connectivity_instances=1,
            scientific_claim_qualified=False, formal_test_accessed=False, configuration=config,
            source_binding={'dataset_sha256': 'a'*64}, model_seed_bindings=bindings, artifacts={})
        raw = []
        for s, b, t, arm, h in product(SEEDS, BLOCKS, TEMPLATES, BRANCHES, HORIZONS):
            observed = h * .005 * (b-10 + int(t[-1])*.1)
            predicted = observed * (1 + .2*(s-40) + BRANCHES.index(arm))
            raw.append(dict(seed=s, block=b, template_id=t, branch=arm, horizon_ms=h,
                case_id=f'train_block_{b:03d}_{t}', origin_ms=1000.,
                role='forecast' if arm == BRANCHES[0] else 'response_conditioned_diagnostic',
                conditional_expected_spikes_per_unit=predicted, observed_spikes_per_unit=observed,
                energy_score=.1 + (s-40)*(BRANCHES.index(arm)+1)*h/500))
        blocks = []
        for s, b, arm, h in product(SEEDS, BLOCKS, BRANCHES, HORIZONS):
            members = [r for r in raw if (r['seed'],r['block'],r['branch'],r['horizon_ms']) == (s,b,arm,h)]
            blocks.append(dict(seed=s,block=b,branch=arm,horizon_ms=h,
                               **{m:float(np.mean([r[m] for r in members])) for m in METRICS}))
        summaries = []
        for arm,h,m in product(BRANCHES,HORIZONS,METRICS):
            means = [np.mean([r[m] for r in blocks if (r['seed'],r['branch'],r['horizon_ms']) == (s,arm,h)]) for s in SEEDS]
            summaries.append(dict(branch=arm,horizon_ms=h,metric=m,mean=float(np.mean(means)),
                minimum_seed_mean=float(min(means)),maximum_seed_mean=float(max(means)),training_seeds=3,independent_blocks=4))
        cov = []
        for s,arm,t,h in product(SEEDS,BRANCHES,TEMPLATES,HORIZONS):
            observed = (int(t[-1])-1.5)*.001*h/20
            predicted = observed + (BRANCHES.index(arm)-1.5)*.002 + (s-42)*.0001
            cov.append(dict(seed=s,branch=arm,template_id=t,horizon_ms=h,statistic='cross_population',
                independent_blocks=4,pairs=12,phases=int(h/5),observed_covariance_mean=observed,
                predicted_covariance_mean=predicted,covariance_mse=(predicted-observed)**2,covariance_mae=abs(predicted-observed)))
        cov_summary = []
        fields = ('observed_covariance_mean','predicted_covariance_mean','covariance_mse','covariance_mae')
        for arm,h in product(BRANCHES,HORIZONS):
            members = [r for r in cov if (r['branch'],r['horizon_ms']) == (arm,h)]
            cov_summary.append(dict(branch=arm,horizon_ms=h,statistic='cross_population',training_seeds=3,independent_blocks=4,
                **{m:float(np.mean([r[m] for r in members])) for m in fields}))
        self.tables = dict(zip(TABLES,(raw,blocks,summaries,cov,cov_summary)))
        cases=[]
        for s,b,t,arm,h in product(SEEDS,BLOCKS,TEMPLATES,UNCROSSED,HORIZONS):
            moment = 1 + (s-41)*.001 + (b-14)*.002 + int(t[-1])*.003
            if arm == UNCROSSED[1]:
                moment *= 1000*h/20
            vectors = 16*int(h/5)
            cases.append(dict(seed=s,block=b,template_id=t,branch=arm,horizon_ms=h,
                case_id=f'train_block_{b:03d}_{t}',origin_ms=1000.,
                role='forecast' if arm == UNCROSSED[0] else 'response_conditioned_diagnostic',
                mean_squared_norm=4*moment,mean_square_per_channel=moment,
                vector_innovations=vectors,scalar_innovations=4*vectors))
        seed_rows=[]
        for s,arm,h in product(SEEDS,UNCROSSED,HORIZONS):
            members=[r for r in cases if (r['seed'],r['branch'],r['horizon_ms']) == (s,arm,h)]
            mean=float(np.mean([r['mean_square_per_channel'] for r in members]))
            seed_rows.append(dict(seed=s,branch=arm,horizon_ms=h,cases=16,vector_innovations=16*16*int(h/5),
                                  scalar_innovations=4*16*16*int(h/5),mean_squared_norm=4*mean,mean_square_per_channel=mean))
        scalar=[]
        for arm,h in product(UNCROSSED,HORIZONS):
            means=[r['mean_square_per_channel'] for r in seed_rows if (r['branch'],r['horizon_ms']) == (arm,h)]
            scalar.append(dict(branch=arm,horizon_ms=h,metric='mean_square_per_channel',
                mean_of_seed_means=float(np.mean(means)),minimum_seed_mean=float(min(means)),maximum_seed_mean=float(max(means)),
                training_seeds=3,selection_cases_per_seed=16,channel_coordinates_pooled_across_seeds=False))
        self.controller=dict(status='DESCRIBED_REPLAYED_CONTROLLER_INNOVATIONS',training_seeds=list(SEEDS),
            branches=list(UNCROSSED),selection_cases_per_seed=16,channel_rows=1152,case_horizon_rows=288,
            controller_artifacts_verified=96,scientific_claim_qualified=False,formal_test_accessed=False,
            selection_only=True,channel_coordinates_pooled_across_seeds=False,
            configuration=copy.deepcopy(config),source_binding=copy.deepcopy(self.status['source_binding']),
            model_seed_bindings=copy.deepcopy(bindings),case_scalar_moments=cases,
            scalar_moments_by_seed=seed_rows,scalar_summary_across_seeds=scalar)
        self.write()

    def write(self):
        for name,rows in self.tables.items():
            with (self.analysis/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
            self.status['artifacts'][name]=sha256_file(self.analysis/name)
        (self.analysis/'analysis.json').write_text(json.dumps(self.status))
        (self.controllers/'controller_diagnostics.json').write_text(json.dumps(self.controller))

    def load(self):
        return load_sources(self.analysis,self.controllers)

    def test_complete_sources_preserve_seed_blocks_and_signed_covariance(self):
        data=self.load()
        self.assertEqual(data['metrics'][METRICS[0]][BRANCHES[0]].shape,(3,4,3))
        self.assertEqual(data['innovations'][UNCROSSED[1]].shape,(3,4,3))
        self.assertLess(data['covariance_observed'].min(),0)
        self.assertGreater(data['covariance_observed'].max(),0)
        np.testing.assert_allclose(data['covariance'][BRANCHES[0]][:,0,0],[-.0046,-.0045,-.0044])

    def test_missing_seed_case_is_rejected_even_with_updated_hash(self):
        self.tables['diagnostic_all_seeds.csv'].pop()
        self.write()
        with self.assertRaisesRegex(ValueError,'incomplete case'):
            self.load()

    def test_future_informed_branch_cannot_be_relabelled_forecast(self):
        self.tables['diagnostic_all_seeds.csv'][3]['role']='forecast'
        self.write()
        with self.assertRaisesRegex(ValueError,'forecast role'):
            self.load()

    def test_native_observation_cannot_differ_across_branches(self):
        self.tables['diagnostic_all_seeds.csv'][3]['observed_spikes_per_unit']+=1
        self.write()
        with self.assertRaisesRegex(ValueError,'native count reference'):
            self.load()

    def test_summary_and_block_disagreement_are_rejected(self):
        for name,field in [('diagnostic_by_block.csv','energy_score'),('diagnostic_summary.csv','mean'),
                           ('covariance_summary.csv','predicted_covariance_mean')]:
            with self.subTest(table=name):
                original=self.tables[name][0][field]
                self.tables[name][0][field]+=1
                self.write()
                with self.assertRaisesRegex(ValueError,'disagree'):
                    self.load()
                self.tables[name][0][field]=original

    def test_source_modification_without_rebinding_is_rejected(self):
        with (self.analysis/'diagnostic_summary.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            self.load()

    def test_cross_analysis_checkpoint_change_is_rejected(self):
        self.controller['model_seed_bindings']['42']['selected_model_artifacts']['checkpoint.npz']='b'*64
        self.write()
        with self.assertRaisesRegex(ValueError,'binding differs'):
            self.load()

    def test_controller_coordinate_pooling_is_rejected(self):
        self.controller['channel_coordinates_pooled_across_seeds']=True
        self.write()
        with self.assertRaisesRegex(ValueError,'unpooled'):
            self.load()

    def test_controller_moment_and_denominator_are_checked(self):
        for field in ('mean_squared_norm','vector_innovations'):
            with self.subTest(field=field):
                original=self.controller['case_scalar_moments'][0][field]
                self.controller['case_scalar_moments'][0][field]+=1
                self.write()
                with self.assertRaisesRegex(ValueError,'inconsistent|denominator'):
                    self.load()
                self.controller['case_scalar_moments'][0][field]=original

    def test_controller_summary_cannot_hide_case_variation(self):
        self.controller['scalar_summary_across_seeds'][0]['mean_of_seed_means']+=1
        self.write()
        with self.assertRaisesRegex(ValueError,'canonical summary'):
            self.load()

    def test_export_geometry_and_existing_output_protection(self):
        helper=self.root/'style.py'
        helper.write_text("""import matplotlib as mpl
def apply_figure_style(sizes):
    mpl.rcParams.update({'font.size':sizes[0], 'axes.titlesize':sizes[0], 'axes.labelsize':sizes[0],
        'xtick.labelsize':sizes[2], 'ytick.labelsize':sizes[2], 'legend.fontsize':sizes[1],
        'figure.dpi':100, 'savefig.dpi':100, 'axes.spines.top':False, 'axes.spines.right':False})
def panel_letter(ax,letter,dx,dy):
    ax.text(dx,dy,letter,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
def panel_crops(fig):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    height=fig.canvas.get_width_height()[1]
    return {letter:[int(b.x0),int(height-b.y1),int(b.x1),int(height-b.y0)]
        for letter,ax in zip('abcd',fig.axes) for b in [ax.get_tightbbox(renderer)]}
""")
        output=self.root/'rendered'
        receipt=render(self.analysis,self.controllers,helper,output)
        self.assertFalse(any(receipt['geometry'].values()))
        self.assertEqual(receipt['status'],'RENDERED_PENDING_VISUAL_INSPECTION')
        self.assertEqual(receipt['causal_forecast_branch'],BRANCHES[0])
        self.assertEqual(len(receipt['artifacts']),8)
        for name,digest in receipt['artifacts'].items():
            self.assertEqual(sha256_file(output/name),digest)
        self.assertFalse(receipt['scientific_claim_qualified'])
        with self.assertRaises(FileExistsError):
            render(self.analysis,self.controllers,helper,output)


if __name__ == '__main__':
    unittest.main()

