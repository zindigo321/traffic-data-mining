from datetime import date
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from v8_thresholds import cutoffs,transfer_masks
from v8_train import arrays,rank_penalty


class V8Tests(unittest.TestCase):
    def inputs(self):
        return np.array([1,0,1,0,0],dtype=np.int8),np.array([.9,.8,.7,.6,.5]),np.array([1,1,1,0,0],bool)

    def test_max_F1_and_exact_BA(self):
        y,s,r=self.inputs()
        z=cutoffs(y,s,r,1,dict(FP_budget_day=12,recovery_budget_day=6))
        self.assertEqual(z["max_f1"]["threshold"],.7)
        self.assertAlmostEqual(z["max_f1"]["fit_F1"],.8)
        self.assertEqual(z["max_ba"]["threshold"],.7)

    def test_larger_threshold_for_same_BA(self):
        y=np.array([0,1,0,1],dtype=np.int8)
        s=np.array([.8,.8,.4,.4])
        z=cutoffs(y,s,y==1,1,dict(FP_budget_day=12,recovery_budget_day=6))
        self.assertGreater(z["max_ba"]["threshold"],.8)
        self.assertEqual(z["max_ba"]["fit_FP"],0)

    def test_both_error_caps_apply_in_integer_counts(self):
        y,s,r=self.inputs()
        for caps in (dict(FP_budget_day=2,recovery_budget_day=0),dict(FP_budget_day=0,recovery_budget_day=2)):
            out=cutoffs(y,s,r,1,caps)["constrained"]
            self.assertEqual(out["threshold"],.9)
            self.assertEqual((out["fit_FP"],out["fit_recovery_FP"]),(0,0))
            self.assertTrue(out["fit_FP_budget_ok"])

    def test_threshold_transfer_uses_disjoint_dates(self):
        start=(date(2026,6,22)-date(1970,1,1)).days
        keys=(np.arange(start,start+14,dtype=np.int64)*144+42)*1000+1
        early,late,total=transfer_masks(keys,["2026-06-22","2026-07-06"],7)
        self.assertEqual((early.sum(),late.sum(),total),(7,7,14))
        self.assertFalse(np.any(early&late))
        self.assertLess(keys[early].max(),keys[late].min())

    def test_fit_loader_rejects_holdout(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(PermissionError):
                arrays(Path(root),dict(folds={}),"holdout")

    def test_rank_loss_uses_main_label_and_top_negative(self):
        z=torch.tensor([.6,.5,.1,-.2,-.8],requires_grad=True)
        y=torch.tensor([1.,0.,0.,0.,0.])
        loss=rank_penalty(z,y,.25)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertLess(float(z.grad[0]),0)
        self.assertGreater(float(z.grad[1]),0)
        self.assertTrue(torch.equal(z.grad[2:],torch.zeros(3)))
