"""Targeted regression tests for causal group sampling and training objectives."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from v10_analysis import summarize
from v10_data import balanced_group_weights, data_for, groups, training_sample
from v10_io import python, record
from v10_score import measure, select_recipe
from v10_train import low_group_rank


class V10Test(unittest.TestCase):
    def test_group_contract_and_boundary(self):
        ratio = np.array([.399, .4, .8, np.nan, np.inf])
        np.testing.assert_array_equal(groups(ratio), [1, 0, 0, 2, 2])
        np.testing.assert_array_equal(groups(ratio.copy()), groups(ratio))

    def test_group_rank_gradient_excludes_other_groups(self):
        z = torch.tensor([.2, .4, .8, -.4, .9], requires_grad=True)
        y = torch.tensor([1., 0., 0., 1., 1.])
        g = torch.tensor([1, 1, 0, 2, 0])
        loss = low_group_rank(z, y, g, .25)
        loss.backward()
        self.assertLess(z.grad[0].item(), 0)
        self.assertGreater(z.grad[1].item(), 0)
        np.testing.assert_array_equal(z.grad[2:].numpy(), np.zeros(3))
        higher_negative = z.detach().clone()
        higher_negative[1] += 1.
        self.assertGreater(low_group_rank(higher_negative, y, g, .25).item(), loss.item())

    def test_group_rank_missing_class_safe_backward(self):
        z = torch.tensor([.2, .8], requires_grad=True)
        loss = low_group_rank(z, torch.tensor([0., 1.]), torch.tensor([0, 2]), .25)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        np.testing.assert_array_equal(z.grad.numpy(), [0, 0])

    def test_macro_weights_are_train_class_balanced(self):
        group = np.array([0, 0, 0, 0, 1, 1, 2], dtype=np.int8)
        y = np.array([0, 0, 0, 1, 0, 1, 1], dtype=np.int8)
        weights, counts = balanced_group_weights(group, y)
        np.testing.assert_array_equal(counts, [[3, 1], [1, 1], [0, 1]])
        self.assertAlmostEqual(float(weights[group, y].mean()), 1., places=6)
        masses = (weights * counts)[counts > 0]
        np.testing.assert_allclose(masses, np.repeat(len(y)/5, 5), rtol=1e-7)
        self.assertEqual(weights[2, 0], 0.)

    def test_legal_low_reservation_ignores_realized_risk(self):
        y = np.r_[np.ones(4, dtype=np.int8), np.zeros(16, dtype=np.int8)]
        study = SimpleNamespace(train=np.arange(20), y=y, ratio=np.r_[np.repeat(.2, 10), np.repeat(.8, 10)],
                                nroad=1, d={"minute": np.arange(20) * 10}, risk=np.zeros(20, dtype=bool))
        cfg = {"negative_budget": 6, "sampling_seed": 741, "ratio": .4}
        for recipe in ("S1", "R1", "R2"):
            ids, stats = training_sample(study, cfg, recipe)
            low = ids[(study.y[ids] == 0) & (study.ratio[ids] < .4)]
            np.testing.assert_array_equal(low, np.arange(4, 10))
            self.assertEqual(len(ids), 10)
            self.assertFalse(stats["uses_realized_t_minus_1_risk"])
            study.risk = ~study.risk
            np.testing.assert_array_equal(ids, training_sample(study, cfg, recipe)[0])

    def test_holdout_cannot_be_opened_by_fit_loader(self):
        cfg = {"folds": {"f1": []}}
        with TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                data_for(Path(tmp), cfg, "holdout")
            with self.assertRaises(PermissionError):
                data_for(Path(tmp), cfg, "final")

    def test_na_precision_and_aggregation(self):
        y, p, risk = np.array([0, 1]), np.array([.1, .2]), np.array([False, True])
        result = measure(y, p, risk, .9, 1)
        self.assertIsNone(result["precision"])
        stats = summarize([{"seed": 11, "precision": None}, {"seed": 22, "precision": np.nan}], ["precision"])
        self.assertIsNone(stats["precision"])
        self.assertIsNone(stats["precision_sd"])
        self.assertEqual(stats["precision_valid_seeds"], 0)

    def test_selection_fallback_and_two_seed_guard(self):
        cfg = {"v10_recipes": ["B0", "S1", "R1", "R2"], "seeds": [11, 22, 33], "main_recall": .6,
               "v10_max_AP_drop": .01, "v10_max_group_recall_drop": .02,
               "v10_min_seeds_recovery_improved": 2, "FP_budget_day": 12, "recovery_budget_day": 6}
        op, matched, gr = [], [], []
        for recipe in cfg["v10_recipes"]:
            for seed in cfg["seeds"]:
                gain = .01 if recipe == "R1" else 0
                op.append(dict(recipe=recipe, seed=seed, AP=.72+gain, f1=.66+gain, recall=.58+gain, FP_day=11, recovery_FP_day=5.8))
                matched.append(dict(recipe=recipe, seed=seed, scope="all", target_recall=.6,
                                    recovery_FP_per_day=5.9 if recipe == "R1" else 6.3))
                for group in ("legal_low", "legal_not_low", "legal_missing"):
                    gr.append(dict(recipe=recipe, seed=seed, scope=group, AP=.85+gain, recall=.6))
        self.assertEqual(select_recipe(op, matched, gr, cfg)["primary"], "R1")
        for row in op:
            if row["recipe"] == "R1":
                row["f1"] = .65
        self.assertEqual(select_recipe(op, matched, gr, cfg)["primary"], "B0")

    def test_record_normalizes_integer_argument(self):
        with TemporaryDirectory() as tmp:
            command = python("-c", "import sys; assert sys.argv[1] == '8'") + [8]
            receipt = record(Path(tmp), "typed_command", command)
            self.assertEqual(receipt["exit_code"], 0)
            self.assertEqual(receipt["command"][-1], "8")


if __name__ == "__main__":
    unittest.main()
