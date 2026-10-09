#!/usr/bin/env python3
"""Replay existing analysis on frozen numerical records; never call a model."""
from pathlib import Path
import sys,tempfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_EXPORTS = ('coupling_summary.json', 'experiment_status.json', 'fig1_throughput.csv', 'fig1_throughput.json', 'fig2_similarity_distributions.csv', 'fig2_similarity_distributions.json', 'fig3_collision_by_tick.csv', 'fig3_collision_by_tick.json', 'fig4_top_attractors.csv', 'fig4_top_attractors.json', 'fig5_modal_concentration.csv', 'fig5_modal_concentration.json', 'l1_l2_identity_null_jaccard.json', 'l3_collision_rates_from_summary.json', 'l3_cross_model_comparison_seeds_11_22_33.json', 'l3_model_a_replication_seeds_11_22_33.json', 'l3_model_b_replication_seeds_11_22_33.json', 'standardized_tables.json')

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


def preflight(root):
    manifest_path = root / 'RELEASE_CHECKSUMS.json'
    if not manifest_path.is_file():
        return ['missing release checksum manifest']
    try:
        manifest = json.loads(manifest_path.read_text())
    except (ValueError, OSError):
        return ['unreadable release checksum manifest']
    failures = []
    for name in EXPECTED_EXPORTS:
        if 'analysis/exports/' + name not in manifest:
            failures.append('export missing from manifest: ' + name)
    for name, digest in manifest.items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            failures.append('unsafe manifest path'); continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append('missing or changed: ' + name)
    actual = {p.name for p in (root / 'analysis/exports').glob('*') if p.suffix in ('.csv','.json')}
    if actual != set(EXPECTED_EXPORTS):
        failures.append('export inventory differs from frozen 18-file inventory')
    return failures

def main():
 failures=preflight(ROOT)
 if failures:
  print("FAIL: " + "; ".join(failures),file=sys.stderr)
  return 1
 sys.path.insert(0,str(ROOT/'src'))
 import sim_common as common
 import analyze_variance as analysis
 import generate_identity_exports as identity
 from evidence_invariants import audit_pack
 reports=audit_pack(ROOT)
 bad={k:v['failures'] for k,v in reports.items() if v['failures']}
 if bad:
  print('FAIL: evidence invariants: '+json.dumps(bad),file=sys.stderr)
  return 1
 anomalies=sum(len(r['response_anomalies']) for r in reports.values())
 print(f'PASS: {len(reports)} cells checked (33 primary + 3 historical); {anomalies} ineligible individual repeat(s) reported, not committed.')
 common.EMBEDDINGS_ROOT=ROOT/'evidence/embeddings'
 with tempfile.TemporaryDirectory(prefix='stage1-analysis-') as t:
  out=Path(t)
  sys.argv=['analyze_variance','--coupling-summary','--simulation-results-dir',str(ROOT/'evidence/cells'),'--analysis-export-dir',str(out),'--coupling-summary-output',str(out/'coupling_summary.json')]
  analysis.main()
  identity.RESULTS=ROOT/'evidence/cells'
  identity.EXPORTS=out
  identity.main()
  names=EXPECTED_EXPORTS
  failures=compare(out,ROOT/'analysis/exports',names)
  if failures:
   print('FAIL: mismatched or missing exports: '+', '.join(failures),file=sys.stderr)
   return 1
  print(f'PASS: {len(names)} frozen exports reproduced; no model calls; reference files unchanged.')
 return 0
if __name__=='__main__':sys.exit(main())
