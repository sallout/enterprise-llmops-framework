"""Without-replacement allocation and deterministic, component-disjoint holdout splits."""
from collections import Counter
import json
import math
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import digest

PIPELINES = ('pipeline1','pipeline2','pipeline3')
SPLITS = ('train','validation','test')


def ratios(values, allowed):
    if set(values) != set(allowed) or any(not math.isfinite(v) or v < 0 for v in values.values()) or not sum(values.values()):
        raise DissertationError('Ratios require nonnegative finite weights for all expected names.')
    total = sum(values.values())
    return {k:values[k]/total for k in allowed}


def mix_counts(available, weights, maximum=None):
    active = {p:w for p,w in weights.items() if w > 0 and available.get(p,0) > 0}
    if not active:
        return {},0
    total_weight = sum(active.values())
    active = {p:w/total_weight for p,w in active.items()}
    # Compute the largest continuous-bottleneck cohort, then apply Hamilton rounding.
    # The upper bound is based on the continuous bottleneck; round down if rounding exceeds supply.
    total = min(sum(available.get(p,0) for p in active), math.floor(min(available[p]/w for p,w in active.items())))
    if maximum is not None:
        total = min(total,maximum)
    def allocate(n):
        counts = {p:math.floor(n*w) for p,w in active.items()}
        for p in sorted(active,key=lambda p:(-(n*active[p]-counts[p]),p))[:n-sum(counts.values())]:
            counts[p]+=1
        return counts
    while total > 0:
        counts = allocate(total)
        if all(counts[p] <= available[p] for p in counts):
            return counts,total
        total-=1
    return {},0


def mix(store, weights, maximum, seed):
    available = store.counts('pipeline','payload IS NOT NULL AND reason IS NULL')
    counts,total = mix_counts(available,weights,maximum)
    with store.db:
        store.db.execute('UPDATE records SET selected=0,split=NULL')
        domain_selection={}
        for p,n in counts.items():
            domains={row[0]:row[1] for row in store.db.execute('SELECT domain,count(*) FROM records WHERE pipeline=? AND payload IS NOT NULL AND reason IS NULL GROUP BY domain',(p,))}
            chosen={d:1 if n>=len(domains) else 0 for d in domains}
            left=n-sum(chosen.values())
            supply={d:domains[d]-chosen[d] for d in domains}
            if left:
                supply_total=sum(supply.values())
                fractions={d:left*supply[d]/supply_total for d in domains}
                for d in domains:chosen[d]+=math.floor(fractions[d])
                for d in sorted(domains,key=lambda d:(-(fractions[d]-math.floor(fractions[d])),d))[:n-sum(chosen.values())]:chosen[d]+=1
            domain_selection[p]=chosen
            for domain,number in chosen.items():
                for row in store.db.execute('SELECT rid FROM records WHERE pipeline=? AND domain=? AND payload IS NOT NULL AND reason IS NULL ORDER BY priority DESC,rank,rid LIMIT ?',(p,domain,number)):
                    store.db.execute('UPDATE records SET selected=1 WHERE rid=?',(row[0],))
    return {'strategy':'continuous_bottleneck_without_replacement; Hamilton rounding; domain coverage; verified coding prioritized within source',
            'available':available,'requested_weights':weights,'selected':counts,'total':total,
            'not_selected':sum(available.values())-total,'domain_selection':domain_selection}


def split(store, weights, seed):
    total = store.count('selected=1')
    targets = {s:total*weights[s] for s in SPLITS}
    db=store.db
    db.execute('DROP TABLE IF EXISTS split_groups')
    db.execute('CREATE TEMP TABLE split_groups (root TEXT PRIMARY KEY,n INTEGER,counts TEXT,forced TEXT,conflict INTEGER,rank TEXT)')
    for g in db.execute('SELECT root,count(*) AS n FROM records WHERE selected=1 GROUP BY root'):
        counts=Counter();forced=set()
        for row in store.rows('root=?',(g['root'],)):
            if row['payload']:
                record=json.loads(row['payload'])
                if record['metadata'].get('official_holdout'):forced.add(record['metadata']['official_holdout'])
                if row['selected']:
                    counts[row['pipeline']]+=1
                    counts['domain:'+row['domain']]+=1
                    counts['verification:'+row['status']]+=1
        official='test' if 'test' in forced else 'validation' if 'validation' in forced else None
        db.execute('INSERT INTO split_groups VALUES (?,?,?,?,?,?)',
            (g['root'],g['n'],json.dumps(counts),official,int(len(forced)>1),digest([seed,g['root']])))
    actual=Counter();strata={name:Counter() for name in SPLITS};all_strata=Counter()
    for g in db.execute('SELECT counts FROM split_groups'):all_strata.update(json.loads(g[0]))
    def assign(g,name):
        db.execute('UPDATE records SET split=? WHERE root=? AND selected=1',(name,g['root']))
        actual[name]+=g['n'];strata[name].update(json.loads(g['counts']))
    for g in db.execute('SELECT * FROM split_groups WHERE forced IS NOT NULL ORDER BY root'):assign(g,g['forced'])
    free_count=db.execute('SELECT count(*) FROM split_groups WHERE forced IS NULL').fetchone()[0]
    for i,g in enumerate(db.execute('SELECT * FROM split_groups WHERE forced IS NULL ORDER BY n DESC,rank,root')):
        counts=json.loads(g['counts'])
        allowed=[name for name in SPLITS if weights[name]>0]
        empty=[name for name in allowed if actual[name]==0]
        if free_count-i<=len(empty):allowed=empty
        def cost(name):
            score=0.0
            for s in SPLITS:
                value=actual[s]+(g['n'] if name==s else 0)
                score+=(value-targets[s])**2/max(targets[s],1)
                for key,n in all_strata.items():
                    target=n*weights[s]
                    value=strata[s][key]+(counts.get(key,0) if name==s else 0)
                    score+=.25*(value-target)**2/max(target,1)
            return score,digest([seed,g['root'],name])
        assign(g,min(allowed,key=cost))
    db.commit()
    errors=[]
    if not actual['train']:errors.append('empty_training_split')
    if any(weights[name]>0 and not actual[name] for name in SPLITS):errors.append('insufficient_independent_groups_for_requested_splits')
    cross=db.execute('SELECT count(*) FROM (SELECT root FROM records WHERE selected=1 GROUP BY root HAVING count(DISTINCT split)>1)').fetchone()[0]
    near=db.execute('SELECT count(*) FROM near_pairs p JOIN records a ON a.rid=p.a JOIN records b ON b.rid=p.b WHERE a.selected=1 AND b.selected=1 AND a.split!=b.split').fetchone()[0]
    exact=db.execute('SELECT count(*) FROM (SELECT exact_key FROM records WHERE selected=1 GROUP BY exact_key HAVING count(DISTINCT split)>1)').fetchone()[0]
    if cross or near or exact:errors.append('cross_split_leakage_detected')
    group_count=db.execute('SELECT count(*) FROM split_groups').fetchone()[0]
    official_count=db.execute('SELECT count(*) FROM split_groups WHERE forced IS NOT NULL').fetchone()[0]
    conflicts=db.execute('SELECT sum(conflict) FROM split_groups').fetchone()[0] or 0
    return {'method':'seeded disk-backed group-wise greedy allocation; source/domain/verification stratification; no group splitting',
            'requested_weights':weights,'target_counts':targets,'actual_counts':dict(actual),'groups':group_count,
            'forced_official_groups':official_count,'official_holdout_conflicts':conflicts,
            'component_leakage':cross,'near_pair_leakage':near,'exact_leakage':exact,'errors':errors}
