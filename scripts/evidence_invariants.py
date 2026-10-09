"""Independent evidence checks; no mutation, inference calls, or result repair."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def cosine(a, b):
    denominator = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/denominator if denominator else 0.


def validate_cell(data, items, events=None):
    failures=[]
    anomalies=[]
    mapping=data['solution_decoupling']
    arrivals=data['arrival_schedule']
    horizon=data['run_manifest']['n_ticks']
    gated=data.get('condition') not in ('autonomous','C_autonomous')
    couplings=data['couplings']
    if len(set(mapping.values())) != len(mapping):failures.append('nonbijective mapping')
    for key in ('problem_id','solution_id'):
        if len({c[key] for c in couplings})!=len(couplings):failures.append('noninjective '+key)

    def inspect(record, similarity_key):
        p,s,t=record['problem_id'],record['solution_id'],record['tick']
        if p not in items or s not in mapping or mapping[s] not in items:
            failures.append('unknown ID');return None
        if mapping[s]=='S'+p[1:]:failures.append('origin reunion')
        if not 0<=t<=horizon or arrivals[p]>t or arrivals[s]>t:failures.append('arrival/horizon violation')
        value=cosine(items[p]['vector'],items[mapping[s]]['vector'])
        if abs(value-record[similarity_key])>0.00000051:failures.append('similarity mismatch')
        return value

    for record in couplings:
        value=inspect(record,'cosine_similarity')
        if gated and value is not None and value<.60:failures.append('gate violation')

    replay=[];collisions=0;trace=[]
    if gated:
        energy={};used_p=set();used_s=set();per_tick=Counter();last_tick=-1
        for ev in data.get('evaluations') or []:
            p,s,t=ev['problem_id'],ev['solution_id'],ev['tick']
            value=inspect(ev,'true_similarity')
            if t<last_tick:failures.append('evaluation order')
            last_tick=t
            if p in used_p or s in used_s:failures.append('evaluation after commitment')
            per_tick[t,p]+=1
            if per_tick[t,p]>2:failures.append('breadth violation')
            before=energy.get(p,1.)
            if before<.15:failures.append('exhausted energy')
            if abs(before-ev['energy_before'])>0.00000051:failures.append('energy mismatch')
            energy[p]=before-.15
            if ev.get('fidelity_degraded') or abs(ev['perceived_similarity']-ev['true_similarity'])>0.00000051:failures.append('unexpected noise')
            if value is not None and bool(ev['accepted'])!=(value>=.60):failures.append('acceptance mismatch')
            if ev['accepted']:
                used_p.add(p);used_s.add(s);replay.append((t,p,s))
    else:
        if events is None:failures.append('missing events');events=[]
        grouped=defaultdict(dict)
        expected_repeats=data['run_manifest'].get('n_repeats_per_tick',3)
        for event in events:
            key=event['tick'],event['problem_id'];repeat=event['repeat']
            if repeat in grouped[key]:failures.append('duplicate repeat')
            grouped[key][repeat]=(event.get('parsed') or {}).get(event['problem_id'])
        used_p=set();used_s=set()
        for tick in range(horizon+1):
            start_s=set(used_s)
            for (t,p),choices in sorted(grouped.items()):
                if t!=tick:continue
                if p in used_p or arrivals.get(p,horizon+1)>tick:failures.append('event for inactive problem')
                if set(choices)!=set(range(expected_repeats)):failures.append('missing/invalid repeat')
                for s in choices.values():
                    if s is not None and (s not in mapping or arrivals.get(s,horizon+1)>tick or s in start_s or mapping.get(s)=='S'+p[1:]):anomalies.append({'tick':tick,'problem_id':p,'solution_id':s,'kind':'ineligible individual repeat'})
                serialized=[json.dumps(choices[r],sort_keys=True) for r in sorted(choices)]
                s=json.loads(Counter(serialized).most_common(1)[0][0]) if serialized else None
                if s is None:continue
                if s not in mapping or arrivals.get(s,horizon+1)>tick or s in start_s or mapping.get(s)=='S'+p[1:]:
                    failures.append('ineligible modal selection');continue
                collision=s in used_s
                trace.append({'tick':tick,'problem_id':p,'solution_id':s,'collision':collision})
                if collision:collisions+=1;continue
                if s not in mapping:continue
                used_p.add(p);used_s.add(s);replay.append((tick,p,s))
        if any(t<0 or t>horizon for t,p in grouped):failures.append('event outside horizon')
    observed=[(c['tick'],c['problem_id'],c['solution_id']) for c in couplings]
    if replay!=observed:failures.append('commitment replay mismatch')
    return {'failures':sorted(set(failures)),'response_anomalies':anomalies,'couplings':len(couplings),'collisions':collisions if not gated else None,'trace':trace}


def audit_pack(root):
    items=json.loads((root/'evidence/embeddings/sbir_v2_nomic-embed-text.json').read_text())['items']
    expected=json.loads((root/'analysis/exports/l3_collision_rates_from_summary.json').read_text())['rows']
    tiers={'tier_1_lightly_aligned':'A','tier_2_heavily_aligned':'B','tier_3_nemotron_nano_4b':'C','tier_4_qwen25_7b':'D','tier_5_gemma2_9b':'E'}
    collision_counts={(r['model'],r['seed']):r['collisions'] for r in expected}
    reports={}
    for path in sorted((root/'evidence/cells').glob('*.json')):
        data=json.loads(path.read_text());ep=root/'evidence/events'/path.name
        events=json.loads(ep.read_text()) if ep.exists() else None
        result=validate_cell(data,items,events)
        if events is not None:
            key=tiers[data['alignment_tier']],data['run_manifest']['seed']
            if result['collisions']!=collision_counts[key]:result['failures'].append('collision summary mismatch')
        result.pop('trace');reports[path.name]=result
    if len(reports)!=36:raise ValueError('Expected 33 primary and 3 historical cells')
    return reports

if __name__=='__main__':
    import sys
    root=Path(__file__).resolve().parents[1]
    reports=audit_pack(root)
    print(json.dumps(reports,indent=2))
    sys.exit(int(any(r['failures'] for r in reports.values())))
