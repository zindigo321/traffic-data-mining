import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v7_io import CONFIG,load
from v7_prepare import forecast_inputs,inputs_only,truth_only,historical_transitions
from v7_score import balanced_cutoff
from v7_train import build,dataset,sample_training,study_for

CFG=load(CONFIG)


class LeadTests(unittest.TestCase):
    def fixture(self):
        # Two adjacent roads. Same geometry is used in the target and its neighbor.
        days=pd.date_range("2026-07-01",periods=2)
        times=pd.DatetimeIndex([d+pd.Timedelta(minutes=m) for d in days
            for m in [*range(420,540,10),*range(1020,1140,10)]])
        mph=np.full((len(times),2),3.)
        raw=dict(times=times,speed=mph,half=np.stack([mph,mph],axis=-1),
            half_count=np.ones((*mph.shape,2)),bus=np.full_like(mph,np.nan),
            message=np.full_like(mph,np.nan),observed=np.ones_like(mph,dtype=bool))
        h=np.full_like(mph,10.)
        phase=np.repeat([0,1],24)
        roads=pd.DataFrame(dict(direction=["EB","EB"],length=[.5,.8]))
        return raw,h,np.full_like(h,5),phase,roads

    def pipeline(self,raw,h,n,phase,roads):
        current=inputs_only(raw,h,n,phase,roads,CFG)
        y=truth_only(raw,h,current["lag"],CFG)[0]
        prior,_=historical_transitions(y,current["risk"],current["eligible"],phase[::24],np.array([False,False]))
        x,rel=forecast_inputs(raw,h,n,phase,roads,prior,CFG)
        return x,rel,y,current["risk"]

    def test_rewrite_both_label_grids_and_neighbors(self):
        raw,h,n,phase,roads=self.fixture()
        t=32
        new={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in raw.items()}
        for key in ("speed","half","half_count"):
            new[key][t-1:]=42
        original=self.pipeline(raw,h,n,phase,roads)
        changed=self.pipeline(new,h,n,phase,roads)
        np.testing.assert_array_equal(original[0][:t+1],changed[0][:t+1])
        np.testing.assert_array_equal(original[1][:t+1],changed[1][:t+1])
        self.assertEqual((original[2][t,0],changed[2][t,0]),(1,0))
        self.assertNotEqual(bool(original[3][t,0]),bool(changed[3][t,0]))
        torch.set_num_threads(2)
        for pool in ("mean","max","attention"):
            torch.manual_seed(11)
            net=build(52,CFG,pool,11).eval()
            adj=torch.tensor([[[False,True],[False,False]]])
            valid=torch.ones((1,2),dtype=torch.bool)
            center=torch.zeros(1,dtype=torch.long)
            with torch.inference_mode():
                scores=[net(torch.from_numpy(np.nan_to_num(z[0][t:t+1])),adj,valid,center) for z in (original,changed)]
            torch.testing.assert_close(*scores,atol=0,rtol=0)

    def test_earliest_source_and_peak_boundary(self):
        raw,h,n,phase,roads=self.fixture()
        raw["speed"][4]=6.
        raw["speed"][5]=9.
        x,r,_,_=self.pipeline(raw,h,n,phase,roads)
        self.assertTrue(np.isnan(x[[0,1,12,13,24,25],:,0]).all())
        self.assertTrue((x[6,:,0]==6.).all())
        self.assertTrue((r[6]==.6).all())

    def test_frozen_prior_does_not_ingest_eval_truth(self):
        raw,h,n,phase,roads=self.fixture()
        x,_,_,_=self.pipeline(raw,h,n,phase,roads)
        new={**raw,"speed":raw["speed"].copy()}
        new["speed"][24:]=80.
        b,_,_,_=self.pipeline(new,h,n,phase,roads)
        np.testing.assert_array_equal(x[:,:,36:40],b[:,:,36:40])

    def test_calibration_tie_uses_larger_threshold(self):
        y=np.array([0,1,0,1],dtype=np.int8)
        score=np.array([.8,.8,.4,.4])
        self.assertGreater(balanced_cutoff(y,score),.8)
        self.assertEqual(balanced_cutoff(np.array([1,0]),np.array([.7,.5])),.7)

    def test_sampling_does_not_read_outcome_risk(self):
        d=dict(x=np.zeros((4,1,52)),y=np.array([0,1,0,1],np.int8)[:,None],
            aux=np.full((4,1),-1,np.int8),eligible=np.ones((4,1),bool),risk=np.ones((4,1),bool),
            ratio_prior=np.zeros((4,1)),split=np.array([0,0,0,1]),minute=np.arange(4)*10,road_ids=np.array([1]))
        one,stats=sample_training(study_for(d,"X1"),CFG)
        two,stats2=sample_training(study_for({**d,"risk":~d["risk"]},"X1"),CFG)
        np.testing.assert_array_equal(one,two)
        self.assertEqual(stats,stats2)

    def test_training_refuses_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(PermissionError):
                dataset(Path(temp),CFG,"holdout",True)
