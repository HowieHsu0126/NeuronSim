import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .pulse_history_analysis import conditional_ridge
from .provenance import sha256_file
from .state_representation_probe import analyze, conditional_prediction, observed_features


CONFIG=dict(rate_windows_ms=[10.,50.,200.],temporal_window_ms=200.,temporal_bin_ms=5.,
    spatial_cells_per_axis=2,spatial_windows_ms=[10.,50.,200.],spike_age_windows_ms=[2.,10.,50.,200.])


class StateRepresentationTests(unittest.TestCase):
    def test_features_exclude_boundary_future_and_tc(self):
        ids=np.array([10,20,30]);pop=np.array(['E','I','TC']);xy=np.array([[-.2,.1],[.2,-.1],[0,0]])
        data=dict(unit_ids=ids,populations=pop,positions_mm=xy,spike_ticks=np.array([60,241,248,249,250,251,260]),
            spike_unit_ids=np.array([10,20,10,30,10,20,30]))
        ix=np.searchsorted(ids,data['spike_unit_ids'])
        a=observed_features(data,ix,250,1.,['E','I'],1.,CONFIG)
        changed={k:v.copy() for k,v in data.items()};changed['spike_ticks'][4:]=[250,900,1000];changed['spike_unit_ids'][3:]=[30,30,10,20]
        b=observed_features(changed,np.searchsorted(ids,changed['spike_unit_ids']),250,1.,['E','I'],1.,CONFIG)
        for left,right in zip(a,b):np.testing.assert_array_equal(left,right)
        values=dict(zip(a[1],a[0]));self.assertEqual(values['rate_hz_E_10ms'],100.);self.assertEqual(values['rate_hz_I_10ms'],100.)
        rich=dict(zip(a[3],a[2]));self.assertEqual(rich['fraction_spiked_E_2ms'],1.);self.assertEqual(rich['fraction_spiked_I_2ms'],0.)

    def test_linear_matches_original_ridge(self):
        rng=np.random.default_rng(30);cells=np.tile(np.arange(4),5);x=rng.normal(size=(20,27));x[:,-1]=cells
        y=rng.normal(size=20)+cells;tc=np.arange(4);tx=rng.normal(size=(4,27));tx[:,-1]=tc
        for alpha in [.01,1.,100.]:
            expected=conditional_ridge(x,y,cells,tx,tc,alpha);actual,_=conditional_prediction(x,y,cells,tx,tc,'linear',alpha)
            np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)

    def test_rbf_matches_joint_augmented_system(self):
        rng=np.random.default_rng(31);cells=np.tile(np.arange(4),5);tc=np.array([3,1,0,2]);x=rng.normal(size=(20,5));x[:,-1]=cells
        y=rng.normal(size=20);tx=rng.normal(size=(4,5));tx[:,-1]=tc
        means=np.array([x[cells==c].mean(0) for c in range(4)]);centered=x-means[cells];scale=centered.std(0);active=scale>0
        z=centered[:,active]/scale[active];zt=(tx-means[tc])[:,active]/scale[active]
        kernel=np.array([[np.exp(-np.linalg.norm(a-b)**2/(2*z.shape[1])) for b in z] for a in z])
        cross=np.array([[np.exp(-np.linalg.norm(a-b)**2/(2*z.shape[1])) for b in z] for a in zt]);c=np.eye(4)[cells];ct=np.eye(4)[tc]
        for alpha in [.0001,.01,1.]:
            matrix=np.block([[kernel+alpha*len(x)*np.eye(len(x)),c],[c.T,np.zeros((4,4))]])
            solution=np.linalg.solve(matrix,np.concatenate([y,np.zeros(4)]));expected=cross@solution[:len(x)]+ct@solution[len(x):]
            actual,_=conditional_prediction(x,y,cells,tx,tc,'rbf',alpha)
            np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-9)

    def test_cell_only_features_and_unseen_cell(self):
        cells=np.tile(np.arange(4),4);x=cells[:,None].astype(float);y=np.arange(16,dtype=float)
        expected=np.array([y[cells==c].mean() for c in range(4)])
        prediction,info=conditional_prediction(x,y,cells,np.arange(4)[:,None],np.arange(4),'rbf',.01)
        np.testing.assert_array_equal(prediction,expected);self.assertEqual(info['active_features'],0)
        with self.assertRaisesRegex(ValueError,'unseen'):conditional_prediction(x,y,cells,np.array([[4.]]),np.array([4]),'rbf',.01)

    def test_held_out_target_cannot_change_its_prediction_or_selection(self):
        rng=np.random.default_rng(32);blocks=np.repeat(np.arange(6),4);cells=np.tile(np.arange(4),6);x=rng.normal(size=(24,5));y=rng.normal(size=24)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'dataset';dataset.mkdir()
            def prepare(target):
                np.savez(dataset/'representations.npz',R=x[:,:2],S=x[:,:4],P=x,target=target,blocks=blocks,cells=cells)
                report=dict(status='BUILT_DEVELOPMENT',config={'cohort_role':'development','predictors':{'linear_regularization':[.01,1.,100.],'rbf_regularization':[.0001,.01,1.]}},
                    feature_dimensions={'R':2,'S':4,'P':5},arrays_sha256=sha256_file(dataset/'representations.npz'),
                    rows=[dict(block=int(b),cell=int(c),focal_arrival_ms=1000. if c<2 else 1500.,history=int(c%2)) for b,c in zip(blocks,cells)])
                (dataset/'features.json').write_text(json.dumps(report))
            prepare(y);analyze(dataset,root/'before');changed=y.copy();changed[blocks==0]+=100;prepare(changed);analyze(dataset,root/'after')
            with np.load(root/'before/predictions.npz') as a,np.load(root/'after/predictions.npz') as b:
                np.testing.assert_array_equal(a['predictions'][blocks==0],b['predictions'][blocks==0])
            import csv
            def selection(path):
                with path.open() as f:return {r['model']:r['selected_alpha'] for r in csv.DictReader(f) if r['block']=='0'}
            self.assertEqual(selection(root/'before/prediction_by_block.csv'),selection(root/'after/prediction_by_block.csv'))


if __name__=='__main__':unittest.main()
