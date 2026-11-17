import unittest

import numpy as np

from v4_data import interval_ok, pinned_array
from v4_metrics import matched_points
from v4_train import neighbor_features, rank_penalty

import torch
import pandas as pd
from prepare_chicago import inputs_only, truth_only
from model import build


class V4Tests(unittest.TestCase):
    def test_same_recall_exact_with_ties_and_full_denominator(self):
        ids = np.arange(6)
        y = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([.8, .8, .9, .8, .7, .6])
        risk = np.array([1, 1, 1, 1, 0, 1], dtype=bool)
        values, chosen = matched_points(ids, y, score, risk, [.33, .66], 2)
        np.testing.assert_array_equal(chosen[.33], [2, 0, 1])
        self.assertEqual((values[0]["TP"], values[0]["FP"], values[0]["FN"], values[0]["TN"]), (1, 2, 2, 1))
        self.assertEqual((values[0]["score_only_TP"], values[0]["score_only_FP"]), (2, 2))
        self.assertEqual(values[1]["TP"], 2)
        self.assertEqual(values[0]["FP_per_day"], 1)

    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):
            matched_points(np.arange(3), np.array([0, 1, -1]), np.array([.1, .8, .5]), np.ones(3, bool), [.6], 1)

    def test_embargo_and_no_holdout_loader(self):
        banned = [["2026-02-02", "2026-05-01"]]
        self.assertFalse(interval_ok(["2026-02-01", "2026-02-03"], banned))
        self.assertTrue(interval_ok(["2025-11-10", "2026-02-02"], banned))
        with self.assertRaises(PermissionError):
            pinned_array({}, "holdout")

    @staticmethod
    def topology():
        lookup = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 2]])
        valid = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=bool)
        adj = np.ones((3, 3, 3), dtype=bool)
        adj[0, 0, 1] = False
        adj[1, 1, 0] = False
        return lookup, valid, adj

    def test_neighbor_direction_and_missing(self):
        data = np.array([[.25, .32, .14], [.42, np.nan, .05]], dtype=np.float32)
        x = neighbor_features(data, *self.topology(), .4)
        np.testing.assert_allclose(x[0, 0], [1, 1, .32, 0, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(x[1, 1], [0, np.nan, np.nan, 1, 0, .42], equal_nan=True)
        np.testing.assert_allclose(x[:, 2], [[0, np.nan, np.nan, 0, np.nan, np.nan]] * 2, equal_nan=True)

    def test_target_future_mutation_does_not_change_neighbor_feature_or_score(self):
        times = pd.date_range("2025-08-01 07:00", periods=12, freq="10min")
        speed = np.full((12, 3), 3.)
        roads = pd.DataFrame(dict(direction=["NB", "SB", "EB"], length=[.5, .4, .4]))
        grid = dict(times=times, speed=speed, bus=speed * 0 + 2, message=speed * 0 + 10,
                    observed=np.ones_like(speed, dtype=bool))
        baseline = np.full_like(speed, 10.)
        support = np.full_like(speed, 5.)
        phase = np.zeros(len(times), dtype=np.int8)
        cfg = dict(ratio=.4, width=32, heads=4, layers=2)
        t = 7
        modified = {**grid, "speed": grid["speed"].copy(), "bus": grid["bus"].copy()}
        modified["speed"][t:] = 18.
        modified["bus"][t:] = 32.
        lookup, valid, adjacency = self.topology()
        outputs = [inputs_only(data, baseline, support, phase, roads, cfg) for data in (grid, modified)]
        truths = [truth_only(data, baseline, x["lag"], cfg)[0] for data, x in zip((grid, modified), outputs, strict=True)]
        features = [neighbor_features(x["ratio_prior"], lookup, valid, adjacency, .4) for x in outputs]
        np.testing.assert_array_equal(features[0][:t+1], features[1][:t+1])
        np.testing.assert_array_equal(outputs[0]["eligible"][:t+1], outputs[1]["eligible"][:t+1])
        self.assertEqual((truths[0][t, 0], truths[1][t, 0]), (1, 0))
        torch.set_num_threads(2)
        for pool in ("mean", "max", "attention"):
            torch.manual_seed(11)
            model = build(6, cfg, pool, 11).eval()
            graphs = [torch.from_numpy(np.nan_to_num(f[t:t+1, lookup[0]])) for f in features]
            with torch.inference_mode():
                result = [model(f, torch.from_numpy(adjacency[0:1]), torch.from_numpy(valid[0:1]), torch.zeros(1, dtype=torch.long))
                          for f in graphs]
            torch.testing.assert_close(*result, atol=0, rtol=0)

    def test_tail_gradient_only_top_recovery_and_positives(self):
        p = torch.tensor([-2., 1., 2., 3., 4., .5], requires_grad=True)
        y = torch.tensor([0, 0, 0, 0, 0, 1])
        loss = rank_penalty(p, y, .25)
        loss.backward()
        torch.testing.assert_close(p.grad[:3], torch.zeros(3))
        self.assertTrue(bool((p.grad[3:5] > 0).all()))
        self.assertLess(float(p.grad[5]), 0)
        empty = rank_penalty(p, torch.zeros_like(y), .25)
        self.assertEqual(float(empty.detach()), 0.)

    def test_pool_batches_do_not_mix_and_graph_changes_outputs(self):
        torch.set_num_threads(2)
        cfg = dict(width=32, heads=4, layers=2)
        a = torch.randn((1, 3, 12), generator=torch.Generator().manual_seed(123))
        b = torch.full_like(a, 100.)
        adj = torch.zeros((1, 3, 3), dtype=torch.bool)
        changed = adj.clone()
        changed[0, 1, 0] = True
        valid = torch.tensor([[True, True, False]])
        for pool in ("mean", "max", "attention"):
            torch.manual_seed(11)
            model = build(12, cfg, pool, 11).eval()
            with torch.inference_mode():
                solo = model(a, adj, valid, torch.zeros(1, dtype=torch.long))
                joint = model(torch.cat([a, b]), adj.repeat(2, 1, 1), valid.repeat(2, 1),
                              torch.zeros(2, dtype=torch.long))
                with_edge = model(a, changed, valid, torch.zeros(1, dtype=torch.long))
            torch.testing.assert_close(solo, joint[:1], atol=1e-6, rtol=0)
            self.assertGreater(float(torch.abs(solo - with_edge)), 1e-6)

    def test_date_bootstrap_equals_explicit_duplicate_rows(self):
        from v4_analysis import bootstrap_scores
        from sklearn.metrics import average_precision_score

        z = dict(sample_id=np.array([1000, 2000, 3000, 144001, 145001, 146001]),
                 y=np.array([1, 0, 0, 1, 0, 0]), score=np.array([.9, .8, .3, .8, .85, .2]),
                 risk=np.array([1, 1, 0, 1, 1, 0], dtype=bool),
                 pred=np.array([1, 1, 0, 1, 1, 0], dtype=bool))
        counts = np.array([[1, 2], [1, 1], [2, 0]])
        current = bootstrap_scores(z, counts, 0, .6)
        for values, actual in zip(counts, current, strict=True):
            weight = values[z["sample_id"] // 144000]
            repeated = np.repeat(np.arange(6), weight)
            keys = z["sample_id"][repeated] * 10 + np.arange(len(repeated))
            y = z["y"][repeated]
            risk = z["risk"][repeated]
            score = z["score"][repeated]
            ref, _ = matched_points(keys, y, score, risk, [.6], 2)
            self.assertAlmostEqual(actual[0], average_precision_score(y, score))
            self.assertAlmostEqual(actual[2], ref[0]["FP_per_day"])
