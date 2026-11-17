"""Synthetic checks for V9 policy math and date isolation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from v9_analysis import trace
from v9_check import MODELS
from v9_policy import adjust, authorized, group_mask, selection
from v9_verify import independent_cut
from v8_thresholds import cutoffs


class V9PolicyTest(unittest.TestCase):
    def test_group_and_identity(self):
        ratio = np.array([.39, .4, np.nan, np.inf, .5, .1])
        p = np.array([.7, .7, .7, .4, .6, 1.0], dtype=np.float64)
        low = group_mask(ratio, .4)
        np.testing.assert_array_equal(low, [True, False, False, False, False, True])
        self.assertIs(adjust(p, low, 0), p)
        self.assertEqual(adjust(p, low, -1)[-1], 1.0)
        self.assertLess(adjust(p, low, -.25)[0], p[0])
        self.assertGreater(adjust(p, low, .25)[0], p[0])
        np.testing.assert_array_equal(adjust(p, low, -.25)[~low], p[~low])

    def test_policy_reads_only_truncated_inputs(self):
        features = np.array([.2, .8, np.nan])
        score = np.array([.7, .4, .2])
        observation_at_t_minus_1 = np.array([.35, .9, .2])
        observation_at_t = np.array([.7, .9, .35])
        actual_recovery = (observation_at_t_minus_1 < .4) & (observation_at_t >= .4)
        truth = (observation_at_t_minus_1 < .4) & (observation_at_t < .4)
        new_recovery = (observation_at_t_minus_1[::-1] < .4) & (observation_at_t[::-1] >= .4)
        new_truth = (observation_at_t_minus_1[::-1] < .4) & (observation_at_t[::-1] < .4)
        self.assertFalse(np.array_equal(actual_recovery, new_recovery))
        self.assertFalse(np.array_equal(truth, new_truth))
        first = adjust(score, group_mask(features, .4), -.5)
        replay = adjust(score.copy(), group_mask(features.copy(), .4), -.5)
        np.testing.assert_array_equal(first, replay)
        np.testing.assert_array_equal(first >= .6, replay >= .6)

    def test_budgets_are_inclusive_and_all_negative_valid(self):
        y = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        p = np.array([.98, .92, .9, .88, .8, .7])
        risk = np.array([True, True, True, False, True, False])
        for total, recovered, expected in [(0, 0, .98), (2, 1, .8), (1, 1, .9), (4, 0, .98)]:
            cfg = {"FP_budget_day": total, "recovery_budget_day": recovered}
            result = cutoffs(y, p, risk, 1, cfg)["constrained"]
            self.assertEqual(result["threshold"], expected)
            self.assertEqual(result["threshold"], independent_cut(y, p, risk, total, recovered))
        y[0] = 0
        cfg = {"FP_budget_day": 1, "recovery_budget_day": 0}
        self.assertGreater(cutoffs(y, p, risk, 1, cfg)["constrained"]["threshold"], p.max())

    def test_score_ties_are_indivisible_for_thresholds(self):
        y = np.array([1, 0, 1, 0], dtype=np.int8)
        p = np.array([.8, .8, .7, .6])
        risk = np.array([True, True, False, False])
        cfg = {"FP_budget_day": 2, "recovery_budget_day": 0}
        self.assertGreater(cutoffs(y, p, risk, 1, cfg)["constrained"]["threshold"], p.max())

    def test_calibration_and_holdout_gate(self):
        cfg = {"folds": {"f1": ["a", "b"]}, "calibration": ["b", "c"]}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            authorized(root, cfg, "f1")
            with self.assertRaises(PermissionError):
                authorized(root, cfg, "final")
            (root / "evidence").mkdir()
            (root / "evidence/development_selection.json").touch()
            authorized(root, cfg, "final")
            for fold in ("holdout", "old_test", "NYC"):
                with self.assertRaises(PermissionError):
                    authorized(root, cfg, fold)

    def test_selection_requires_two_seeds_and_both_caps(self):
        cfg = {"seeds": [11, 22, 33], "folds": {"f1": []}, "v9_offsets": [-1, 0, 1],
               "FP_budget_day": 12, "recovery_budget_day": 6, "v9_min_seeds_improved": 2}
        rows = []
        for model in MODELS:
            for seed in cfg["seeds"]:
                for d in cfg["v9_offsets"]:
                    gain = .05 if d == 1 or (d == -1 and seed != 33 and model == "attention") else 0
                    rows.append(dict(model=model, fold="f1", seed=seed, delta=d, f1=.6+gain, recall=.55+gain,
                                     FP_day=12, recovery_FP_day=7 if d == 1 else 6))
        result = selection(rows, cfg)
        self.assertEqual(result["center"]["delta"], 0)
        self.assertEqual(result["rank"]["delta"], 0)
        self.assertEqual(result["attention"]["delta"], -1)

    def test_date_weights_recompute_equal_recall_with_ties(self):
        keys = np.array([1, 2, 3, 144001, 144002, 144003])
        y = np.array([1, 0, 1, 0, 1, 0])
        p = np.array([.8, .9, .7, .8, .8, .4])
        recovery = np.array([True, True, False, False, True, False])
        pred = p >= .8
        weights = np.array([[1, 1], [2, 0], [0, 2]])
        actual = trace(keys, y, p, recovery, pred, weights, 0, .6)
        expected = []
        for w in weights:
            idx = np.repeat(np.arange(len(keys)), w[keys // 144000])
            yy, rr, pp, pred2 = y[idx], recovery[idx], p[idx], pred[idx]
            tp = int(((yy == 1) & pred2).sum())
            fp = int(((yy == 0) & pred2).sum())
            recfp = int(((yy == 0) & pred2 & rr).sum())
            order = np.lexsort((keys[idx], -pp))
            hit = np.flatnonzero(yy[order] == 1)[int(np.ceil(yy.sum() * .6)) - 1]
            window = order[:hit + 1]
            expected.append([2 * tp / (yy.sum() + tp + fp), tp / yy.sum(), fp / 2, recfp / 2,
                             (yy[window] == 0).sum() / 2, ((yy[window] == 0) & rr[window]).sum() / 2])
        np.testing.assert_allclose(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
