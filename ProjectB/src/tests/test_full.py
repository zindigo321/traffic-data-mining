"""Engineering regressions for the user's full-data and open-admission protocol."""
from types import SimpleNamespace
import unittest

import numpy as np

from full_data import MODELS,training_ids,epoch_order
from full_score import choose_thresholds,phase_metrics
from v3_prepare import historical_transitions


class FullDataTest(unittest.TestCase):
    def make_study(self):
        y=np.r_[np.ones(4,dtype=np.int8),np.zeros(16,dtype=np.int8),np.array([-1],dtype=np.int8)]
        return SimpleNamespace(train=np.arange(20),y=y,nroad=1,d={"minute":np.arange(21)*10})

    def test_full_training_ignores_legacy_negative_cap(self):
        study=self.make_study()
        cfg={"negative_budget":2,"sampling_seed":741}
        for model in MODELS:
            ids,n=training_ids(study,cfg,model)
            self.assertEqual(n,20)
            if model=="center_sampled_matched":
                self.assertEqual(len(ids),6)
            else:
                np.testing.assert_array_equal(ids,study.train)

    def test_full_epoch_uses_each_known_row_once(self):
        full=np.arange(500)
        order=epoch_order(full,len(full),np.random.default_rng(11))
        np.testing.assert_array_equal(np.sort(order),full)

    def test_sampled_control_matches_updates_with_repetitions(self):
        selected=np.arange(12)
        order=epoch_order(selected,500,np.random.default_rng(11))
        self.assertEqual(len(order),500)
        np.testing.assert_array_equal(np.unique(order),selected)

    def test_unknown_is_scored_but_excluded_from_metrics(self):
        raw=dict(sample_id=np.array([1,2,3]),phase=np.ones(3,dtype=int),y=np.array([-1,0,1]),
                 score=np.array([.99,.1,.9]),risk=np.array([True,False,True]))
        row=phase_metrics(raw,1,.5,{"FP_budget_day":12,"recovery_budget_day":6},"test",11,"constrained")
        self.assertEqual((row["scheduled"],row["known"],row["unknown"]),(3,2,1))
        self.assertEqual((row["tp"],row["fp"],row["fn"],row["tn"]),(1,0,0,1))

    def test_single_class_does_not_block_threshold_stage(self):
        options=choose_thresholds(np.zeros(10,dtype=int),np.linspace(.1,.9,10),np.zeros(10,dtype=bool),1,
                                 {"FP_budget_day":12,"recovery_budget_day":6})
        self.assertEqual(options["constrained"]["threshold"],.5)
        self.assertIn("single-class",options["constrained"]["reason"])

    def test_no_minimum_coverage_needed_to_report_na_metrics(self):
        raw=dict(sample_id=np.array([1,2,3]),phase=np.ones(3,dtype=int),y=np.array([-1,-1,0]),
                 score=np.array([.2,.6,.1]),risk=np.zeros(3,dtype=bool))
        row=phase_metrics(raw,1,.5,{"FP_budget_day":12,"recovery_budget_day":6},"test",11,"constrained")
        self.assertAlmostEqual(row["coverage"],1/3)
        self.assertIsNone(row["AP"])
        self.assertIsNone(row["f1"])

    def test_calibration_and_test_labels_do_not_update_prior_features(self):
        y=np.zeros((3*24,2),dtype=np.int8)
        risk=np.ones_like(y,dtype=bool)
        eligible=np.ones_like(y,dtype=bool)
        phases=np.array([0,1,2])
        weekends=np.array([False,False,False])
        first,state=historical_transitions(y,risk,eligible,phases,weekends)
        y[24:]=1
        later,changed=historical_transitions(y,risk,eligible,phases,weekends)
        np.testing.assert_array_equal(first,later)
        for k in state:
            np.testing.assert_array_equal(state[k],changed[k])


if __name__=="__main__":
    unittest.main()
