"""Runtime relocation regressions separate from historical model engineering tests."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('portable_launcher_for_tests',ROOT/'run.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PortableTests(unittest.TestCase):
    def test_original_numerical_sources_and_configuration_unchanged(self):
        frozen=json.loads((ROOT/'reference/evidence/dynamic_threshold_freeze.json').read_text(encoding='utf-8'))
        self.assertEqual(launcher.sha(ROOT/'src/config.json'),frozen['config_sha256'])
        for name,digest in frozen['code'].items():
            self.assertEqual(launcher.sha(ROOT/'src'/name),digest,name)

    def test_missing_local_arrays_never_fall_back_to_old_drive(self):
        from full_data import arrays
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ,{'PROJECTB_PREPARED_DIR':str(Path(directory)/'missing')}):
                with self.assertRaises(FileNotFoundError):arrays(ROOT/'reference','final')

    def test_manifest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=root/'manifest.csv'
            with manifest.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(('path','size_bytes','sha256','purpose'))
                writer.writerow(('../outside.py',1,'a'*64,'invalid'))
            (root/'SHA256SUMS').write_text('a'*64+'  ../outside.py\n'+launcher.sha(manifest)+'  manifest.csv\n',encoding='utf-8')
            with mock.patch.object(launcher,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'Unsafe manifest path'):launcher.verify_integrity()

    def test_public_raw_manifest_is_relative_and_preserves_bytes(self):
        snapshot=json.loads((ROOT/'data/source_snapshot.json').read_text(encoding='utf-8'))
        self.assertEqual(len(snapshot['files']),54)
        for row in snapshot['files']:
            rel=Path(row['path'])
            self.assertFalse(rel.is_absolute())
            self.assertNotIn('..',rel.parts)
            self.assertTrue((ROOT/rel).resolve().is_relative_to(ROOT/'data'))
            self.assertEqual(launcher.sha(ROOT/rel),row['sha256'])

    def test_reject_wrong_python_environment(self):
        with mock.patch.object(launcher.sys,'prefix',str(ROOT/'not_niulaoshi')):
            with self.assertRaisesRegex(RuntimeError,'不是 niulaoshi'):launcher.environment()

    def test_legacy_external_file_guard_is_active(self):
        with self.assertRaises(PermissionError):
            with open('G:/course-jiaotongshujuwajue/ProjectB/portable_guard_probe.txt',encoding='utf-8'):
                pass


if __name__=='__main__':unittest.main()
