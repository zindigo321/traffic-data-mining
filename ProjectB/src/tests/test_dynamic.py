"""Engineering fixtures only; no synthetic score is a real traffic result."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from full_io import CONFIG,load
from full_data import set_seed,TrainPreprocessor
from dynamic_model import build_model,weighted_attention
from v7_prepare import FEATURES,forecast_inputs,inputs_only,truth_only,historical_transitions

CFG=load(CONFIG)


class DynamicTests(unittest.TestCase):
    def setUp(self):
        set_seed(11,CFG)
        self.x=torch.randn(3,4,104)
        self.x[...,52:]=0
        self.adj=torch.zeros(3,4,4,dtype=torch.bool)
        self.adj[:,0,1]=True
        self.adj[:,1,2]=True
        self.valid=torch.tensor([[1,1,1,0]]*3,dtype=torch.bool)
        self.center=torch.tensor([0,1,2])

    def net(self,pool='mean',learned=False):
        net=build_model(104,CFG,'dynamic_'+pool,11).eval()
        if learned:
            with torch.no_grad():
                for gate in net.gates:
                    gate.net[-1].weight.fill_(.15)
                    gate.net[-1].bias.fill_(.7)
        return net

    def test_zero_gate_exact_static_all_pools(self):
        for pool in ('mean','max','attention'):
            set_seed(11,CFG)
            static=build_model(104,CFG,'graph_'+pool,11).eval()
            set_seed(11,CFG)
            dynamic=self.net(pool)
            torch.testing.assert_close(static(self.x,self.adj,self.valid,self.center),
                dynamic(self.x,self.adj,self.valid,self.center),atol=0,rtol=0)

    def test_direction_self_loop_padding_and_bound(self):
        gate=self.net(learned=True).gates[0]
        b=gate(self.x,self.adj,self.valid)
        self.assertEqual(tuple(b.shape),(3,4,4,4))
        self.assertTrue((b[:,:,1,0]>0).all())
        self.assertTrue((b[:,:,0,1]==0).all())
        self.assertTrue((b.diagonal(dim1=-2,dim2=-1)==0).all())
        self.assertTrue((b[:,:,:,3]==0).all() and (b[:,:,3,:]==0).all())
        self.assertTrue((b.exp()>=.5).all() and (b.exp()<=2).all())

    def test_missing_endpoint_has_neutral_prior(self):
        net=self.net(learned=True)
        self.x[:,0,52+FEATURES.index('lag20_relative_speed')]=1
        b=net.gates[0](self.x,self.adj,self.valid)
        self.assertTrue((b[:,:,1,0]==0).all())
        self.assertTrue((b[:,:,2,1]!=0).any())

    def test_valid_past_edge_change_changes_output(self):
        net=self.net(learned=True)
        args=(self.x,self.adj,self.valid,self.center)
        with torch.no_grad():
            on=net(*args)
            net.edge_enabled=False
            off=net(*args)
        self.assertGreater(float((on-off).abs().max()),1e-6)

    def test_static_topology_mask_still_changes_output(self):
        net=self.net(learned=True)
        with torch.no_grad():
            a=net(self.x,self.adj,self.valid,self.center)
            b=net(self.x,torch.zeros_like(self.adj),self.valid,self.center)
        self.assertGreater(float((a-b).abs().max()),1e-6)

    def test_singleton_gate_cannot_change_output(self):
        net=self.net(learned=True)
        x=self.x[:,:1]
        adj=self.adj[:,:1,:1]
        valid=self.valid[:,:1]
        center=torch.zeros(3,dtype=torch.long)
        with torch.no_grad():
            on=net(x,adj,valid,center)
            net.edge_enabled=False
            off=net(x,adj,valid,center)
        torch.testing.assert_close(on,off,atol=0,rtol=0)

    def test_batch_isolation_all_pools(self):
        changed=self.x.clone()
        changed[1:,:3,:52]+=3
        for pool in ('mean','max','attention'):
            net=self.net(pool,True)
            with torch.no_grad():
                a=net(self.x,self.adj,self.valid,self.center)[0]
                b=net(changed,self.adj,self.valid,self.center)[0]
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_target_mapping_node_permutation_all_pools(self):
        perm=torch.tensor([2,0,3,1])
        center=torch.argsort(perm)[self.center]
        for pool in ('mean','max','attention'):
            net=self.net(pool,True)
            with torch.no_grad():
                a=net(self.x,self.adj,self.valid,self.center)
                b=net(self.x[:,perm],self.adj[:,perm][:,:,perm],self.valid[:,perm],center)
            torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-6)

    def test_reject_empty_invalid_center_nonfinite(self):
        net=self.net()
        for x,valid,center in ((self.x,torch.zeros_like(self.valid),self.center),
            (self.x,self.valid,torch.full_like(self.center,3)),
            (torch.full_like(self.x,float('nan')),self.valid,self.center)):
            with self.assertRaises(ValueError):net(x,self.adj,valid,center)

    def test_backward_and_safe_weight_roundtrip(self):
        for pool in ('mean','max','attention'):
            net=self.net(pool).train()
            z=net(self.x,self.adj,self.valid,self.center)
            z.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
            self.assertGreater(sum(float(g.net[-1].weight.grad.abs().sum()) for g in net.gates),0)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'weight.pt'
                torch.save(net.state_dict(),path)
                restored=self.net(pool)
                restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
                torch.testing.assert_close(z,restored(self.x,self.adj,self.valid,self.center),atol=0,rtol=0)

    def test_single_batch_overfit(self):
        net=self.net().train()
        x=self.x.repeat(8,1,1)
        y=(torch.arange(len(x))%2).float()
        x[:,0,0]=(y*2-1)*4
        adj=self.adj.repeat(8,1,1)
        valid=self.valid.repeat(8,1)
        center=torch.zeros(len(x),dtype=torch.long)
        opt=torch.optim.Adam(net.parameters(),lr=.005)
        for _ in range(100):
            opt.zero_grad()
            z=net(x,adj,valid,center)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
            loss.backward()
            opt.step()
        self.assertLess(float(loss.detach()),.02)
        self.assertTrue(((z>=0)==(y==1)).all())

    def test_rewrite_label_and_future_cannot_change_edge_or_score(self):
        dates=pd.date_range('2026-07-01',periods=2)
        times=pd.DatetimeIndex([d+pd.Timedelta(minutes=m) for d in dates
              for m in [*range(420,540,10),*range(1020,1140,10)]])
        mph=np.full((len(times),2),3.)
        raw=dict(times=times,speed=mph,half=np.stack([mph,mph],-1),half_count=np.ones((*mph.shape,2)),
                 bus=np.full_like(mph,np.nan),message=np.full_like(mph,np.nan),observed=np.ones_like(mph,dtype=bool))
        h=np.full_like(mph,10.)
        support=np.full_like(h,5)
        phase=np.repeat([0,1],24)
        roads=pd.DataFrame(dict(direction=['EB','EB'],length=[.5,.8]))
        def pipeline(r):
            current=inputs_only(r,h,support,phase,roads,CFG)
            y=truth_only(r,h,current['lag'],CFG)[0]
            prior,_=historical_transitions(y,current['risk'],current['eligible'],phase[::24],np.array([False,False]))
            x,rel=forecast_inputs(r,h,support,phase,roads,prior,CFG)
            return x,rel,y
        t=32
        changed={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in raw.items()}
        for k in ('speed','half','half_count','bus','message'):
            changed[k][t-1:]=42
        a,b=pipeline(raw),pipeline(changed)
        np.testing.assert_array_equal(a[0][:t+1],b[0][:t+1])
        self.assertEqual((a[2][t,0],b[2][t,0]),(1,0))
        prep=TrainPreprocessor().fit(a[0][:24],np.ones((24,2),bool))
        xx=[torch.from_numpy(prep.transform(v[0][t:t+1],np.ones((1,2),bool))) for v in (a,b)]
        adj=torch.tensor([[[0,1],[0,0]]],dtype=torch.bool)
        valid=torch.ones((1,2),dtype=torch.bool)
        center=torch.zeros(1,dtype=torch.long)
        for pool in ('mean','max','attention'):
            net=self.net(pool,True)
            with torch.no_grad():
                torch.testing.assert_close(net(xx[0],adj,valid,center),net(xx[1],adj,valid,center),atol=0,rtol=0)
                torch.testing.assert_close(net.gates[0](xx[0],adj,valid),net.gates[0](xx[1],adj,valid),atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
