#!/usr/bin/env python3
"""Replay existing analysis on frozen numerical records; never call a model."""
from pathlib import Path
import sys,tempfile,json
ROOT=Path(__file__).resolve().parents[1]

def compare(actual,expected,names):
 failures=[]
 for name in names:
  a,b=actual/name,expected/name
  if not a.is_file() or not b.is_file():failures.append(name);continue
  if name.endswith('.json'):
   equal=json.loads(a.read_text())==json.loads(b.read_text())
  else:equal=a.read_bytes()==b.read_bytes()
  if not equal:failures.append(name)
 return failures

def main():
 sys.path.insert(0,str(ROOT/'src'))
 import sim_common as common
 import analyze_variance as analysis
 import generate_identity_exports as identity
 common.EMBEDDINGS_ROOT=ROOT/'evidence/embeddings'
 with tempfile.TemporaryDirectory(prefix='stage1-analysis-') as t:
  out=Path(t)
  sys.argv=['analyze_variance','--coupling-summary','--simulation-results-dir',str(ROOT/'evidence/cells'),'--analysis-export-dir',str(out),'--coupling-summary-output',str(out/'coupling_summary.json')]
  analysis.main()
  identity.RESULTS=ROOT/'evidence/cells'
  identity.EXPORTS=out
  identity.main()
  names=sorted(p.name for p in (ROOT/'analysis/exports').iterdir() if p.suffix in ('.csv','.json'))
  failures=compare(out,ROOT/'analysis/exports',names)
  if failures:
   print('FAIL: mismatched or missing exports: '+', '.join(failures),file=sys.stderr)
   return 1
  print(f'PASS: {len(names)} frozen exports reproduced; no model calls; reference files unchanged.')
 return 0
if __name__=='__main__':sys.exit(main())
