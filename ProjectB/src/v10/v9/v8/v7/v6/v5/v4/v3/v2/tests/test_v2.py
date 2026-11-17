import tempfile
import unittest
from pathlib import Path

from v1 import runtime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from evaluation import event_metrics, rank_score, threshold
from experiment import Fold, proportional, sample, set_seed
from holdout import blocks
from model import build, primary
from prepare_chicago import (aligned_lag, causal_history, inputs_only, key_matrix, read_extension, truth_only)

CFG = dict(ratio=.4,minimum_history_days=3,cpu_threads=2,width=32,heads=4,layers=2,
           sampling_seed=741,aux_seed=1971,negative_budget=3)


class TimeAndSamplingTests(unittest.TestCase):
    def grid(self):
        times = pd.date_range("2024-07-01 07:00",periods=6,freq="10min")
        speed = np.full((len(times),2),16.)
        speed[0:3] = 3.
        bus = np.ones_like(speed)*3
        grid = dict(times=times,speed=speed,bus=bus,message=bus*4,observed=np.ones_like(speed,dtype=bool))
        roads = pd.DataFrame(dict(direction=["NB","SB"],length=[.5,.4]))
        return grid,roads

    def test_target_future_mutation_keeps_current_features_eligibility_and_scores(self):
        grid,roads = self.grid()
        t = 2
        hist = np.full_like(grid["speed"],16.)
        days = np.full_like(hist,4.)
        split = np.zeros(len(hist),dtype=int)
        prev = inputs_only(grid,hist,days,split,roads,CFG)
        y0,_,_ = truth_only(grid,hist,prev["lag"],CFG)
        changed = {k:v.copy() for k,v in grid.items()}
        for col in ("speed","bus","message"):
            changed[col][t:,:] = 90.
        new = inputs_only(changed,hist,days,split,roads,CFG)
        y1,_,_ = truth_only(changed,hist,new["lag"],CFG)
        np.testing.assert_array_equal(prev["x"][:t+1],new["x"][:t+1])
        np.testing.assert_array_equal(prev["eligible"][:t+1],new["eligible"][:t+1])
        self.assertTrue(prev["eligible"][t,0])
        self.assertEqual((y0[t,0],y1[t,0]),(1,0))
        set_seed(11,CFG)
        for pool in ("mean","max","attention"):
            net = build(prev["x"].shape[-1],CFG,pool,11,True).eval()
            edge = torch.tensor([[[0,1],[1,0]]],dtype=torch.bool)
            valid, center = torch.ones((1,2),dtype=torch.bool),torch.tensor([0])
            def run(row):
                x = torch.tensor(np.nan_to_num(row[t:t+1],nan=0.))
                with torch.inference_mode():
                    return primary(net(x,edge,valid,center)).sigmoid()
            torch.testing.assert_close(run(prev["x"]),run(new["x"]),atol=0,rtol=0)

    def test_lag_uses_its_own_history_and_strict_rule(self):
        grid,roads = self.grid()
        hist = np.full_like(grid["speed"],10.)
        hist[0] = 5.
        counts = np.full_like(hist,4.)
        x = inputs_only(grid,hist,counts,np.zeros(len(hist),dtype=int),roads,CFG)
        self.assertAlmostEqual(x["ratio_prior"][1,0],.6)
        self.assertFalse(x["risk"][1,0])
        self.assertTrue(x["risk"][2,0])
        y,_,aux = truth_only(grid,hist,x["lag"],CFG)
        self.assertEqual((y[1,0],aux[1,0]),(-1,1))
        self.assertEqual(y[2,0],1)
        grid["speed"][2] = 4.
        y,_,aux = truth_only(grid,hist,x["lag"],CFG)
        self.assertEqual((y[2,0],aux[2,0]),(0,0))

    def test_eval_updates_neither_history_nor_support(self):
        values = np.arange(1,8,dtype=float).reshape(7,1,1)
        weekend = np.zeros(7,dtype=bool)
        phases = np.array([0,0,0,0,0,1,1])
        hist,days = causal_history(values,weekend,phases,3)
        values[-2:]*=8000
        other = causal_history(values,weekend,phases,3)
        np.testing.assert_array_equal(hist,other[0])
        np.testing.assert_array_equal(days,other[1])
        self.assertEqual(days[-1,0],5)

    def test_no_lag_across_dates_gaps_or_partitions(self):
        times = pd.DatetimeIndex(["2024-08-24 08:00","2024-08-24 08:10","2024-08-24 08:20",
                                  "2024-08-24 17:00","2024-08-25 07:00"])
        np.testing.assert_array_equal(aligned_lag(times,np.array([0,0,1,1,1]),10),[-1,0,-1,-1,-1])

    def test_unknown_not_in_main_but_eligible_aux(self):
        y = np.array([0,0,0,0,0,1,-1,1,0]).reshape(9,1)
        data = dict(y=y,aux=np.maximum(y,0),eligible=np.ones_like(y,dtype=bool),risk=np.zeros_like(y,dtype=bool),
                    ratio_prior=np.full_like(y,.7,dtype=float),split=np.array([0,0,0,0,0,0,0,1,1]),
                    minute=np.arange(9)*10,road_ids=np.array([12]))
        data["risk"][[0,1,5,7]] = True
        data["aux"][6,0] = 1
        f = Fold(data,"V3")
        tr,aux,stat = sample(f,CFG)
        self.assertEqual(stat["sampled_hard"],2)
        self.assertEqual((len(tr),len(aux)),(4,7))
        self.assertTrue(set([0,1,5]) <= set(tr))
        self.assertTrue(6 in aux and 6 not in tr)
        self.assertTrue((f.phase[np.r_[tr,aux]]==0).all())
        np.testing.assert_array_equal(tr,sample(f,CFG)[0])

    def test_proportional_budget_when_hard_exceeds_budget(self):
        ids = np.arange(10)
        strat = np.array([0]*7+[1]*3)
        draw = proportional(ids,strat,4,np.random.default_rng(741))
        self.assertEqual(len(set(draw)),4)
        self.assertEqual((draw<7).sum(),3)

    def test_unopened_holdout_file_cannot_be_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                read_extension(Path(tmp),"holdout_sealed")

    def test_key_and_event_partition(self):
        # 3 true windows form 2 events. Long alert overlaps both but can only match 1.
        times = np.arange(5)*10+420
        keys = key_matrix(times,[1]).ravel()
        event = event_metrics(keys,np.array([1,1,0,1,0]),np.ones(5,dtype=bool))
        self.assertEqual((event["true_events"],event["alert_events"],event["matched_events"]),(2,1,1))

    def test_route_keeps_full_support_and_prioritizes_risk(self):
        p = np.array([1.,0.,.2,.3],dtype=np.float32)
        r = np.array([0,1,0,1],dtype=bool)
        x = rank_score(p,r,True)
        self.assertEqual(len(x),len(p))
        self.assertGreater(x[r].min(),x[~r].max())

    def test_date_bootstrap_with_weights_matches_independent_sklearn(self):
        y = np.array([1,0,1,0,1,0])
        p = np.array([.7,.8,.7,.3,.1,.1])
        day = np.array([0,0,1,1,2,2])
        weights = np.array([[1,1,1],[0,2,1],[1,0,2]])
        t = threshold(y,p)
        actual = blocks(y,p,p>=t,day,weights)
        expected = [average_precision_score(y,p,sample_weight=w[day]) for w in weights]
        np.testing.assert_allclose(actual[:,0],expected,atol=1e-14)

    def test_separate_auxiliary_backward_uses_unknown_only_for_aux(self):
        set_seed(11,CFG)
        x=torch.randn(4,2,6)
        adj=torch.ones(4,2,2,dtype=torch.bool)
        valid=torch.ones(4,2,dtype=torch.bool)
        center=torch.zeros(4,dtype=torch.long)
        for pool in ("mean","max","attention"):
            model=build(6,CFG,pool,11,True)
            logits=model(x,adj,valid,center)
            self.assertEqual(logits.shape,(4,2))
            # Rows 2 and 3 have synthetic unknown main states and valid auxiliary targets.
            loss=logits[:2,0].square().mean()+.25*logits[2:,1].square().mean()
            loss.backward()
            self.assertGreater(float(model.classifier.auxiliary.weight.grad.abs().sum()),0)
            self.assertTrue(torch.isfinite(model.input.weight.grad).all())

    def test_initial_backbone_and_aux_head_paired(self):
        models = []
        for pool in ("mean","max","attention"):
            set_seed(11,CFG)
            models.append(build(6,CFG,pool,11,True))
        for a in models[1:]:
            for name,param in models[0].state_dict().items():
                torch.testing.assert_close(param,a.state_dict()[name],atol=0,rtol=0)


if __name__=="__main__":
    unittest.main()



