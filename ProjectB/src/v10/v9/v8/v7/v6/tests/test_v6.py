from pathlib import Path
import tempfile
import unittest

from v6_io import load,CONFIG,save,sha
from v6_lock import check_transport
from v6_stats import stability
from v6_train import configurations,training_budget
from v5_download import source_files


class V6Tests(unittest.TestCase):
    def test_unique_fair_pool_and_control_jobs(self):
        cfg=load(CONFIG)
        models=configurations(cfg)
        self.assertEqual(len(models),len(set(models)))
        self.assertEqual(len(models),12)
        self.assertEqual(cfg["primary_pool"],"attention")
        self.assertEqual(training_budget(),12)
        self.assertIsInstance(cfg["epochs"],dict)
        self.assertEqual(set(s for r,p,s in models if (r,p)==("X1","mean")),{11,22,33})

    def test_v5_test_is_outside_both_training_and_new_test(self):
        cfg=load(CONFIG)
        self.assertLessEqual(cfg["calibration"][1],cfg["prior_holdout"][0])
        self.assertGreaterEqual(cfg["holdout"][0],cfg["prior_holdout"][1])
        self.assertEqual(cfg["epochs"],{"X1_mean":4,"X1_max":2,"X1_attention":3,"B_mean":2})

    def test_stability_requires_all_three_conditions(self):
        self.assertFalse(stability([-3.,0.,0.],-.2))
        self.assertFalse(stability([-3.,-2.,0.],.001))
        self.assertFalse(stability([-1.,-1.,3.],-.01))
        self.assertTrue(stability([-2.,-1.,1.],-.001))

    def test_transport_refuses_speed_reads_until_unlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                list(source_files(Path(tmp),"holdout_sealed"))

    def test_transport_rejects_time_before_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            file=root/"sealed.csv"
            file.write_text("time,link_id,speed\n",encoding="utf-8")
            period=["2026-10-12","2026-11-09"]
            save(root/"evidence/download_holdout_sealed.json",dict(status="complete",interval=period,
                observation_values_parsed=False,transport_only=True,rows=1,directory=str(root),
                files=[dict(file=file.name,sha256=sha(file),bytes=file.stat().st_size,first_requested_utc="2026-11-10T00:00:00+00:00")]))
            with self.assertRaisesRegex(ValueError,"before exact"):
                check_transport(root,dict(holdout=period,formal_eligible=True),dict(at_utc="2026-11-10T01:00:00+00:00"))
