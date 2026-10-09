import unittest
from pathlib import Path
import tempfile
import reproduce_stage1_evidence as r
class VerificationTests(unittest.TestCase):
 def test_equal_files(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'a').mkdir();(p/'b').mkdir()
   for d in ['a','b']:(p/d/'x.csv').write_text('n\n3\n')
   self.assertEqual(r.compare(p/'a',p/'b',['x.csv']),[])
 def test_changed_count_fails(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'a').mkdir();(p/'b').mkdir()
   (p/'a'/'x.csv').write_text('n\n3\n');(p/'b'/'x.csv').write_text('n\n4\n')
   self.assertEqual(r.compare(p/'a',p/'b',['x.csv']),['x.csv'])
 def test_missing_fails(self):
  with tempfile.TemporaryDirectory() as t:self.assertEqual(r.compare(Path(t),Path(t),['missing']),['missing'])
 def test_preflight_missing_manifest(self):
  with tempfile.TemporaryDirectory() as t:self.assertTrue(r.preflight(Path(t)))
 def test_preflight_changed_file(self):
  import json,hashlib
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'x').write_text('changed');(p/'RELEASE_CHECKSUMS.json').write_text(json.dumps({'x':hashlib.sha256(b'original').hexdigest()}))
   self.assertTrue(r.preflight(p))
 def test_preflight_missing_expected_export(self):
  import json
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'RELEASE_CHECKSUMS.json').write_text(json.dumps({}));self.assertTrue(r.preflight(p))
