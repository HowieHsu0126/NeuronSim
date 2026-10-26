"""Behavioral CUDA tests; use SPATIAL_CUDA_LIBRARY for the compiled project kernel."""
from dataclasses import replace
import math
import json
import os
from pathlib import Path
import unittest

import numpy as np

from .spatial_connectivity import CudaProjection,Projection,cell_order,create_layout,native_delay_ticks


LIBRARY = os.environ.get("SPATIAL_CUDA_LIBRARY")


def generate(ids,xy,target_ids=None,target_xy=None,p=None,batch=256,maximum=1000000):
    if p is None: p=Projection(1.,.5,1e300,1.,2.,0.,.5,.3,3,76001,0)
    if target_ids is None: target_ids,target_xy=ids,xy
    with CudaProjection(Path(LIBRARY),ids,xy,target_ids,target_xy,p) as generator:
        return np.concatenate(list(generator.chunks(batch,maximum)))


class ValidationTests(unittest.TestCase):
    def test_parameters_reject_wrong_domain(self):
        p=Projection(1.,.5,.3,.7,2.,.1,.5,.3,2,76001,0)
        for key,value in [("radius",.51),("p0",1.01),("seed",0),("repeats",2.5),
                          ("stream",72),("beta",float("nan")),("weight_sd",-1)]:
            with self.subTest(key=key),self.assertRaises(ValueError): replace(p,**{key:value}).validate()

    def test_native_delay_readbacks_including_half_step_neighbors(self):
        fixture=json.loads((Path(__file__).parent/"fixtures"/"nest_spatial_semantics.json").read_text())
        self.assertEqual(len(fixture["delays"]),178)
        for case in fixture["delays"]:
            with self.subTest(resolution=case["resolution_ms"],label=case["label"]):
                if case["accepted"]:
                    actual=native_delay_ticks([case["input_delay_ms"]],case["resolution_ms"])
                    self.assertEqual(int(actual[0]),case["native_ld_round_steps"])
                else:
                    with self.assertRaises(ValueError):
                        native_delay_ticks([case["input_delay_ms"]],case["resolution_ms"])

    def test_cell_order_is_canonical(self):
        xy=np.array([[.49,0],[-.49,0],[.2,.2],[0,0]])
        ids=np.array([30,20,10,40]); permutation=[2,0,3,1]
        a=cell_order(ids,xy,1.,.2);b=cell_order(ids[permutation],xy[permutation],1.,.2)
        for left,right in zip(a,b): np.testing.assert_array_equal(left,right)
        with self.assertRaises(ValueError): cell_order(np.array([1,1]),xy[:2],1.,.2)
        with self.assertRaises(ValueError): cell_order(ids,xy*10,1.,.2)


@unittest.skipUnless(LIBRARY,"SPATIAL_CUDA_LIBRARY is required; no CPU fallback")
class CudaTests(unittest.TestCase):
    def check_complete(self,ids,xy,p):
        result=generate(ids,xy,p=p)
        actual=sorted((int(e["source"]),int(e["target"]),int(e["repeat"]),int(e["anchor_index"])) for e in result)
        # Quadratic native-source oracle, independent of CUDA cell lookup/skipping.
        origin=(xy.max(axis=0)+xy.min(axis=0))/2-p.extent/2
        canonical=origin+np.fmod(xy-origin,p.extent)
        canonical=np.where(canonical<origin,canonical+p.extent,canonical)
        expected=[]
        for j,target in enumerate(ids):
            base=np.fmod(xy[j]-p.radius-origin,p.extent)
            base=np.where(base<0,base+p.extent,base)+p.radius+origin
            extra=base+p.radius-origin>p.extent
            for anchor in range(4):
                if (anchor&1 and not extra[0]) or (anchor&2 and not extra[1]): continue
                center=base-np.asarray([bool(anchor&1),bool(anchor&2)])*p.extent
                for i,source in enumerate(ids):
                    delta=np.abs(canonical[i]-center)
                    inside=(max(delta)<=p.radius and
                            (sum(delta)<=p.radius or math.sqrt(float(delta@delta))<=p.radius))
                    if source!=target and inside:
                        expected.extend((source,target,r,anchor) for r in range(p.repeats))
        self.assertEqual(actual,sorted(expected))
        self.assertEqual(len(actual),len(set(actual)))
        return result

    def test_matches_native_boundary_oracle_including_duplicate_images(self):
        fixture=json.loads((Path(__file__).parent/"fixtures"/"nest_spatial_semantics.json").read_text())
        for index,case in enumerate(fixture["spatial"]):
            if case["radius_mm"]>.5:
                with self.assertRaises(ValueError):
                    Projection(1.,case["radius_mm"],1.,1.,1.,0.,.5,1.,1,76001,0).validate()
                continue
            ids,xy=np.asarray(case["unit_ids"]),np.asarray(case["actual_positions_mm"])
            for repeats,key in [(1,"edges"),(3,"three_call_edges")]:
                if key not in case: continue
                p=Projection(1.,case["radius_mm"],1e300,1.,1.,0.,1e-300,1.,repeats,76001,0)
                actual=generate(ids,xy,p=p,batch=3)
                observed=sorted((int(e["source"]),int(e["target"]),float(e["raw_delay"])) for e in actual)
                expected=sorted((e["source"],e["target"],e["weight"]+1e-300) for e in case[key])
                with self.subTest(case=index,repeats=repeats): self.assertEqual(observed,expected)

    def test_wrapped_cell_aliases_and_repeat_multiplicity(self):
        for n in [1,4,9]:
            ids=np.arange(n)
            xy=np.random.default_rng(110+n).uniform(-.5,.5,(n,2))
            p=Projection(1.,.5,1e300,1.,2.,0.,.5,.3,3,76001,0)
            with self.subTest(n=n): self.check_complete(ids,xy,p)

    def test_cell_neighbors_match_exhaustive_periodic_pairs(self):
        xy=np.random.default_rng(120).uniform(-.5,.5,(100,2))
        xy[:6]=[[.499,0],[-.499,0],[0,0],[.08,0],[-.08,0],[0,.08]]
        p=Projection(1.,.08,1e300,1.,2.,0.,.5,.3,2,76001,0)
        result=self.check_complete(np.arange(100),xy,p)
        for edge in result:
            origin=(xy.max(axis=0)+xy.min(axis=0))/2-.5
            q=origin+np.fmod(xy[edge["source"]]-origin,1.)
            q=np.where(q<origin,q+1.,q)
            delta=q-xy[edge["target"]]
            whole=np.floor(np.abs(delta));rounded=np.copysign(whole+((np.abs(delta)-whole)>=.5),delta)
            delta=delta-rounded
            self.assertAlmostEqual(edge["raw_delay"],.5+math.sqrt(float(delta@delta))/.3,places=14)

    def test_source_order_and_batching_preserve_rng_realization(self):
        xy=np.random.default_rng(130).uniform(-.5,.5,(160,2));ids=np.arange(160)
        p=Projection(1.,.3,.2,.6,2.,.8,.5,.3,4,76001,5)
        left=generate(ids,xy,p=p,batch=160)
        right=generate(ids[::-1],xy[::-1],ids,xy,p=p,batch=7,maximum=800)
        np.testing.assert_array_equal(left,right)
        changed=generate(ids,xy,p=replace(p,seed=76002))
        self.assertFalse(np.array_equal(left,changed))
        weighted=generate(ids,xy,p=replace(p,weight_sd=0))
        for field in ["source","target","repeat","anchor_index","raw_delay"]:
            np.testing.assert_array_equal(left[field],weighted[field])

    def test_layout_separates_source_rng_roles_and_keeps_normal_tails(self):
        net={"num_neurons":[4096]*9,"populations":["L23E","L23I","L4E","L4I","L5E","L5I","L6E","L6I","TC"],
             "extent":1.,"V0_type":"original","neuron_params":{"V0_mean":{"original":0.},"V0_std":{"original":1.}}}
        cfg={"random_streams":{"spatial_layout":55001,"initial_state":55003}}
        a=create_layout(Path(LIBRARY),net,cfg)
        b=create_layout(Path(LIBRARY),net,cfg)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        changed=create_layout(Path(LIBRARY),net,{"random_streams":{"spatial_layout":55002,"initial_state":55003}})
        np.testing.assert_array_equal(a["cortical_initial_vm_mv"],changed["cortical_initial_vm_mv"])
        self.assertFalse(np.array_equal(a["positions_mm"],changed["positions_mm"]))
        xy,vm=a["positions_mm"],a["cortical_initial_vm_mv"]
        self.assertTrue(np.all((xy>=-.5)&(xy<.5)))
        self.assertLess(abs(float(xy.mean())),6/math.sqrt(12*xy.size))
        self.assertLess(abs(float(vm.mean())),6/math.sqrt(vm.size))
        self.assertLess(abs(float(np.mean(vm**2))-1),6*math.sqrt(2/vm.size))
        self.assertLess(vm.min(),-3.)
        self.assertGreater(vm.max(),3.)

    def test_zero_probability_and_output_budget(self):
        ids=np.arange(4);xy=np.zeros((4,2))
        p=Projection(1.,.5,1e300,0.,2.,0.,.5,.3,3,76001,0)
        self.assertEqual(len(generate(ids,xy,p=p)),0)
        with self.assertRaises(MemoryError): generate(ids,xy,p=replace(p,p0=1.),maximum=1)
        with CudaProjection(Path(LIBRARY),ids,xy,ids,xy,p) as generator: pass
        with self.assertRaises(RuntimeError): list(generator.chunks())

    def test_binomial_multiplicity_includes_zero_and_multiple_edges(self):
        n=8192;r=5;q=.35
        p=Projection(1.,.5,.2,q,2.,0.,.5,.3,r,76001,0)
        edges=generate(np.array([0]),np.zeros((1,2)),np.arange(1,n+1),np.zeros((n,2)),p)
        counts=np.bincount(edges["target"],minlength=n+1)[1:]
        self.assertLessEqual(counts.max(),r)
        observed=np.bincount(counts,minlength=r+1)
        for k in range(r+1):
            probability=math.comb(r,k)*q**k*(1-q)**(r-k)
            expectation=n*probability
            bound=6*math.sqrt(n*probability*(1-probability))
            self.assertLess(abs(observed[k]-expectation),bound)

    def test_boundary_images_receive_independent_bernoulli_draws(self):
        n=8192;r=5;q=.35
        p=Projection(1.,.5,1e300,q,2.,0.,.5,.3,r,76001,0)
        target_xy=np.tile([.5,0.],(n,1))
        edges=generate(np.array([0]),np.zeros((1,2)),np.arange(1,n+1),target_xy,p)
        counts=np.bincount(edges["target"],minlength=n+1)[1:]
        self.assertLessEqual(counts.max(),2*r)
        observed=np.bincount(counts,minlength=2*r+1)
        for k in range(2*r+1):
            probability=math.comb(2*r,k)*q**k*(1-q)**(2*r-k)
            self.assertLess(abs(observed[k]-n*probability),6*math.sqrt(n*probability*(1-probability)))

    def test_weight_redraw_has_truncated_normal_mean(self):
        n=8192
        for sign in [1,-1]:
            p=Projection(1.,.5,.2,1.,sign*.3,.5,.5,.3,8,76001,0)
            edges=generate(np.array([0]),np.zeros((1,2)),np.arange(1,n+1),np.zeros((n,2)),p)
            values=sign*edges["weight"]
            self.assertTrue(np.all(values>0))
            a=-.3/.5
            tail=.5*math.erfc(a/math.sqrt(2))
            ratio=math.exp(-a*a/2)/math.sqrt(2*math.pi)/tail
            expected=.3+.5*ratio
            variance=.5**2*(1+a*ratio-ratio**2)
            self.assertLess(abs(values.mean()-expected),6*math.sqrt(variance/len(values)))


if __name__ == "__main__": unittest.main()
