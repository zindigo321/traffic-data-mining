import tempfile
import unittest
from pathlib import Path

from v3_prepare import (aligned_lag,check_days,frozen_transitions,historical_transitions,inputs_only,
                        read_extension,recovery_target,temporal_extras,truth_only)
from v3_train import Study,build,epoch_ids,matched_indices,risk_metrics,set_seed
from evaluation import rank_score

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

CFG=dict(ratio=.4,cpu_threads=2,width=32,heads=4,layers=2,negative_budget=3,sampling_seed=741)


class V3Tests(unittest.TestCase):
    def test_past_transition_prior_does_not_see_today_or_eval(self):
        y=np.zeros((96,2),dtype=np.int8)
        y[24:48]=1
        mask=np.ones_like(y,dtype=bool)
        part=np.array([0,0,1,1])
        daytype=np.array([0,0,0,1])
        a,state=historical_transitions(y,mask,mask,part,daytype)
        change=y.copy()
        change[24:48]=0
        b,_=historical_transitions(change,mask,mask,part,daytype)
        np.testing.assert_array_equal(a[:48],b[:48])
        self.assertFalse(np.array_equal(a[48:72],b[48:72]))
        change=y.copy()
        change[48:]=1
        c,stat=historical_transitions(change,mask,mask,part,daytype)
        np.testing.assert_array_equal(a,c)
        for k in state:
            np.testing.assert_array_equal(state[k],stat[k])
        np.testing.assert_array_equal(a[48:],frozen_transitions(state,daytype[2:]))

    def test_mutating_target_and_future_changes_truth_only(self):
        times=pd.date_range("2025-08-01 07:00",periods=12,freq="10min")
        speed=np.ones((12,2))*3.
        grid=dict(times=times,speed=speed,bus=speed*0+2,message=speed*0+10,observed=np.ones_like(speed,dtype=bool))
        roads=pd.DataFrame(dict(direction=["NB","SB"],length=[.5,.4]))
        h=np.ones_like(speed)*10
        n=np.ones_like(h)*10
        phase=np.zeros(12,dtype=np.int8)
        t=7
        baseline=inputs_only(grid,h,n,phase,roads,CFG)
        extra=temporal_extras(grid,h,phase,.4)
        y,_,_=truth_only(grid,h,baseline["lag"],CFG)
        changed={k:v.copy() for k,v in grid.items()}
        for k in ("speed","bus","message"):
            changed[k][t:]=80
        other=inputs_only(changed,h,n,phase,roads,CFG)
        plus=temporal_extras(changed,h,phase,.4)
        y_new,_,_=truth_only(changed,h,other["lag"],CFG)
        self.assertEqual((y[t,0],y_new[t,0]),(1,0))
        np.testing.assert_array_equal(baseline["eligible"][:t+1],other["eligible"][:t+1])
        x=np.concatenate((baseline["x"],np.stack(extra,axis=-1)),axis=-1)
        x_new=np.concatenate((other["x"],np.stack(plus,axis=-1)),axis=-1)
        np.testing.assert_array_equal(x[:t+1],x_new[:t+1])
        for pool in ("mean","max","attention"):
            set_seed(11,CFG)
            model=build(x.shape[-1],CFG,pool,11).eval()
            adj=torch.tensor([[[0,1],[0,0]]],dtype=torch.bool)
            valid=torch.ones((1,2),dtype=torch.bool)
            center=torch.zeros(1,dtype=torch.long)
            def score(values):
                with torch.inference_mode():
                    return model(torch.from_numpy(np.nan_to_num(values[t:t+1]).astype(np.float32)),adj,valid,center)
            torch.testing.assert_close(score(x),score(x_new),atol=0,rtol=0)

    def test_excluded_ranges_raise(self):
        cfg=dict(exclusions=[["2025-02-17","2025-07-01"]])
        with self.assertRaises(ValueError):
            check_days(cfg,np.array(["2025-02-20"],dtype="datetime64[D]"),np.array(["2026-02-02"],dtype="datetime64[D]"))

    def test_lag_breaks_on_partition_and_gap(self):
        t=pd.DatetimeIndex(["2025-07-01 08:00","2025-07-01 08:10","2025-07-01 08:20","2025-07-01 17:00"])
        np.testing.assert_array_equal(aligned_lag(t,np.array([0,0,1,1]),10),[-1,0,-1,-1])

    def test_pair_budget_and_risk_sampling(self):
        y=np.array([0,0,1,0,0,-1,0,1,0]).reshape(9,1)
        flags=np.array([1,0,1,1,0,1,0,1,0],dtype=bool).reshape(9,1)
        grid=dict(y=y,aux=y,eligible=np.ones_like(flags),risk=flags,ratio_prior=np.where(flags,.3,.6),
                  split=np.array([0,0,0,0,0,0,0,1,1]),minute=np.arange(9)*10,road_ids=np.array([2]))
        a,na,_=matched_indices(Study(grid,"A"),CFG)
        b,nb,stat=matched_indices(Study(grid,"R1"),CFG)
        self.assertEqual((na,nb),(4,4))
        self.assertEqual(set(b),{0,2,3})
        self.assertNotIn(5,a)
        self.assertEqual(stat["sampled_hard"],2)
        draw=epoch_ids(b,nb,np.random.default_rng(1))
        self.assertEqual(len(draw),nb)
        self.assertEqual(set(draw),set(b))

    def test_streak_and_lag_counts_missing(self):
        t=pd.date_range("2025-08-03 07:00",periods=8,freq="10min")
        speed=np.array([10,2,np.nan,2,3,8,1,3.])[:,None]
        x=temporal_extras(dict(times=t,speed=speed),np.full_like(speed,10),np.zeros(8,dtype=int),.4)
        self.assertEqual((x[3][5,0],x[4][5,0],x[5][5,0]),(4,3,2))
        self.assertTrue(np.isnan(x[0][1,0]))

    def test_routed_AP_equals_risk_AP_on_full_support(self):
        y=np.array([0,1,0,1,0])
        risk=np.array([0,1,1,1,0],dtype=bool)
        p=rank_score(np.array([.99,.2,.3,.7,.9],dtype=np.float32),risk,True)
        result=risk_metrics(dict(y=y,score=p,risk=risk),.60)
        self.assertAlmostEqual(result["risk_AP"],average_precision_score(y,p))
        self.assertEqual(result["ordinary_FP"],0)

    def test_smoothl1_head_and_main_have_gradient(self):
        set_seed(33,CFG)
        for pool in ("mean","max","attention"):
            model=build(6,CFG,pool,33,True)
            x=torch.randn(2,3,6)
            z=model(x,torch.ones(2,3,3,dtype=torch.bool),torch.ones(2,3,dtype=torch.bool),torch.zeros(2,dtype=torch.long))
            self.assertEqual(z.shape,(2,2))
            (z[:,0].square().mean()+.4*z[:,1].square().mean()).backward()
            self.assertTrue(torch.isfinite(model.input.weight.grad).all())
            self.assertGreater(float(model.classifier.auxiliary.weight.grad.abs().sum()),0)

    def test_held_observations_require_unlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                read_extension(Path(tmp),"holdout_sealed")

    def test_recovery_target_has_no_unknown_truth(self):
        a=recovery_target(np.array([3.,8.,np.nan]),np.array([10.,10.,10.]),np.array([1,0,-1]))
        np.testing.assert_allclose(a[:2],np.log1p([.3,.8]),rtol=1e-6)
        self.assertTrue(np.isnan(a[-1]))

