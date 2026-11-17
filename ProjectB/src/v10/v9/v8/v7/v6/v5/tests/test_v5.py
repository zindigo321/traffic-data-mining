import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v5_download import source_files
from v5_graph import graph_arrays, parse_polyline
from v5_io import save, sha
from v5_prepare import grid, half_features, inputs_only, truth_only
from v5_train import build, pinned_development, rank_penalty

CFG=dict(ratio=.4,source=dict(hours=[7,8,17,18],max_speed_mph=85),width=32,heads=4,layers=2,
         endpoint_near_m=25,endpoint_same_corridor_m=80,turn_angle_max=100,max_nodes=12)


class NYCProtocolTests(unittest.TestCase):
    def test_sealed_file_and_training_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(PermissionError):
                list(source_files(root,"holdout_sealed"))
            with self.assertRaises(PermissionError):
                pinned_development(root,{"folds":{"f1":None}},"holdout")

    def test_two_halves_and_legacy_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            file=root/"2026-05-11_2026-05-12_am.csv"
            file.write_text("data_as_of,link_id,speed,travel_time,status\n"
                "2026-05-11T07:01:00,1,4,900,-101\n"
                "2026-05-11T07:01:00,1,8,450,0\n"
                "2026-05-11T07:07:00,1,10,360,-101\n"
                "2026-05-11T07:13:00,1,15,240,0\n"
                "2026-05-11T07:18:00,1,0,360,0\n",encoding="utf-8")
            save(root/"evidence/download_development.json",dict(status="complete",directory=str(root),
                 files=[dict(file=file.name,rows=5,sha256=sha(file))]))
            result,receipt=grid(root,CFG,"development",pd.DataFrame(dict(link_id=["1"],length=[1.])),
                                ["2026-05-11","2026-05-12"])
            np.testing.assert_array_equal(result["half"][0,0],[6.,10.])
            self.assertEqual(result["speed"][0,0],8.)
            self.assertTrue(np.isnan(result["speed"][1,0]))
            self.assertEqual(receipt["duplicate_valid_timestamps"],1)
            self.assertEqual(receipt["valid_rows_by_status"]["-101"],2)
            self.assertEqual(receipt["first_half_only"],1)

    def test_directed_connectivity(self):
        t=pd.DataFrame(dict(corridor=["A","A","A"]),index=[1,2,3])
        p=[np.array([[40.7,-74.02],[40.7,-74.01]]),np.array([[40.7,-74.01],[40.7,-74.00]]),
           np.array([[40.7,-74.00],[40.7,-74.01]])]
        g,e=graph_arrays(p,t,CFG)
        self.assertEqual([(edge["src"],edge["dst"]) for edge in e],[(1,2)])
        self.assertTrue(g["adj"][0,0,1])
        self.assertFalse(g["adj"][0,1,0])
        self.assertEqual(int(g["valid"][2].sum()),1)
        np.testing.assert_allclose(parse_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq\x60@"),
                                   [[38.5,-120.2],[40.7,-120.95],[43.252,-126.453]])
        with self.assertRaises(ValueError):
            parse_polyline("?_")

    def test_rewrite_current_and_future_changes_truth_not_model_inputs(self):
        times=pd.date_range("2026-06-02 07:00",periods=12,freq="10min")
        a=np.full((12,2),3.)
        raw=dict(times=times,speed=a,half=np.stack([a,a],axis=-1),half_count=np.ones((12,2,2)),
                 bus=np.full_like(a,np.nan),message=np.full_like(a,np.nan),observed=np.ones_like(a,dtype=bool))
        history=np.full_like(a,10.)
        phase=np.zeros(12,dtype=np.int8)
        roads=pd.DataFrame(dict(direction=["NB","NB"],length=[.5,.7]))
        t=7
        changed={**raw,"speed":a.copy(),"half":raw["half"].copy(),"half_count":raw["half_count"].copy()}
        changed["speed"][t:]=40.
        changed["half"][t:]=40.
        changed["half_count"][t:]=7
        data=[]
        for grid_raw in (raw,changed):
            x=inputs_only(grid_raw,history,np.full_like(history,5),phase,roads,CFG)
            ext=half_features(grid_raw,history,phase,.4)
            y=truth_only(grid_raw,history,x["lag"],CFG)[0]
            data.append((x["eligible"],np.concatenate([x["x"],ext],axis=-1),y))
        np.testing.assert_array_equal(data[0][0][:t+1],data[1][0][:t+1])
        np.testing.assert_array_equal(data[0][1][:t+1],data[1][1][:t+1])
        self.assertEqual((data[0][2][t,0],data[1][2][t,0]),(1,0))
        torch.set_num_threads(2)
        for pool in ("mean","max","attention"):
            torch.manual_seed(11)
            model=build(data[0][1].shape[-1],CFG,pool,11,auxiliary=True).eval()
            adj=torch.tensor([[[False,True],[False,False]]])
            valid=torch.ones((1,2),dtype=torch.bool)
            center=torch.zeros(1,dtype=torch.long)
            with torch.inference_mode():
                outputs=[model(torch.from_numpy(np.nan_to_num(item[1][t:t+1])),adj,valid,center) for item in data]
            torch.testing.assert_close(*outputs,atol=0,rtol=0)

    def test_aux_loss_has_finite_gradient_and_safe_checkpoint(self):
        torch.manual_seed(33)
        torch.set_num_threads(2)
        net=build(6,CFG,"attention",33,auxiliary=True)
        data=torch.randn(4,2,6)
        mask=torch.ones(4,2,dtype=torch.bool)
        adj=torch.zeros(4,2,2,dtype=torch.bool)
        center=torch.zeros(4,dtype=torch.long)
        y=torch.tensor([0.,1.,0.,1.])
        z=net(data,adj,mask,center)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(z[:,0],y)+
              .35*rank_penalty(z[:,0],y,.25)+.30*torch.nn.functional.smooth_l1_loss(z[:,1],y-1.))
        loss.backward()
        self.assertTrue(torch.isfinite(net.classifier.auxiliary.weight.grad).all())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"weight.pt"
            torch.save(net.state_dict(),path)
            other=build(6,CFG,"attention",33,auxiliary=True)
            other.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
            torch.testing.assert_close(other(data,adj,mask,center),z,atol=0,rtol=0)
