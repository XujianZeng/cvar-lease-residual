#!/usr/bin/env python3
"""Exploratory post-hoc sensitivity analyses; see methodology.md.

Each proceeds scenario refits the complete LR/LP pipeline on the matching
training scenario, then evaluates fixed rules on the matching test scenario.
Observed nonpositive proceeds are a stress scenario, not known final prices.
"""
from pathlib import Path
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from . import buffer_study as bs
from . import revision_study as rs
from . import cycle_study as cs
from .analyze import cvar

HERE = Path(__file__).parent.parent
BOOT=1000
PAIRS=((rs.CA,'uniform'),(rs.BA,'scaled_base'))
SCENARIOS={
    'primary_3m_positive': {},
    '3m_recorded_nonpositive_included': {'include_nonpositive':True},
    'mature_3m_positive': {'require_followup_months':6},
    'mature_6m_positive': {'proceeds_months':6,'require_followup_months':6},
}

def json_safe(value):
    """Use standard JSON null for mathematically undefined percentages."""
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [json_safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def compare(values, kind, d):
    benchmark=rs.benchmark(kind,values,d)
    loss=bs.shortfall(values,d); ref=bs.shortfall(benchmark,d)
    a,b=cvar(loss,.99),cvar(ref,.99)
    return {'pct':float('nan') if b==0 else 100*(a/b-1),
            'cvar_rule':a,'cvar_benchmark':b,'gap_usd':a-b,
            'positive_loss_rule':int((loss>0).sum()),'positive_loss_benchmark':int((ref>0).sum())}

def bootstrap(values,d,rng,cluster=False):
    out={f'{rule}|{kind}':{'pct':[],'gap_usd':[]} for rule,kind in PAIRS}
    groups=[np.flatnonzero(d['event_month']==m) for m in np.unique(d['event_month'])]
    for _ in range(BOOT):
        idx=(np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
             if cluster else rng.integers(0,d['n'],d['n']))
        sub=bs.resample(d,idx);sub['n']=len(idx)
        for rule,kind in PAIRS:
            result=compare(values[rule][idx],kind,sub)
            for metric in ('pct','gap_usd'):out[f'{rule}|{kind}'][metric].append(result[metric])
    return {k:{metric:np.asarray(x) for metric,x in v.items()} for k,v in out.items()}

def evaluate(train,test,seed):
    brands,score,rules,target=cs.fit(dict(train)); test=bs.add_features(dict(test),brands)
    values={r:rules[r].value(test) for r,k in PAIRS}
    draws=bootstrap(values,test,np.random.default_rng(seed))
    entry={'n':test['n'],'returned':int(test['r'].sum()),'training_target_cvar':target,
           'auc':bs.auc(score.prob(test),test['r']),'comparisons':{},
           'base_fit_diagnostics':rules[rs.BA].fit_diagnostics}
    for r,k in PAIRS:
        key=f'{r}|{k}';point=compare(values[r],k,test)
        entry['comparisons'][key]={**point,**rs.summarize(point['pct'],draws[key]['pct']),
                                  'gap_ci95_usd':np.percentile(draws[key]['gap_usd'],[2.5,97.5]).tolist()}
    return entry,draws

def aggregate(entries,draws):
    out={}
    for r,k in PAIRS:
        key=f'{r}|{k}'
        point=float(np.mean([e['comparisons'][key]['pct'] for e in entries]))
        combined=np.mean([d[key]['pct'] for d in draws],axis=0)
        gaps=np.mean([d[key]['gap_usd'] for d in draws],axis=0)
        out[key]={**rs.summarize(point,combined),
                  'mean_gap_usd':float(np.mean([e['comparisons'][key]['gap_usd'] for e in entries])),
                  'mean_gap_ci95_usd':np.percentile(gaps,[2.5,97.5]).tolist()}
    return out

def paired_gb_change(gb,lr,d):
    # Scale both strategies down to the smaller original mean, preserving caps.
    common=min(gb.mean(),lr.mean());gb=gb*common/gb.mean();lr=lr*common/lr.mean()
    a,b=cvar(bs.shortfall(gb,d),.99),cvar(bs.shortfall(lr,d),.99)
    return 100*(a/b-1) if b>0 else float('nan')

def align_primary_intervals(result):
    """Reuse already reported primary ratio draws; new dollar and LOO CIs remain diagnostic."""
    rev=json.loads((HERE/'revision_results.json').read_text())
    cyc=json.loads((HERE/'cycle_results.json').read_text())
    primary=result['scenarios']['primary_3m_positive']
    fields=('pct','ci95','p_boot','undefined_draws')
    for entry,original in zip(primary['main'],rev['holdouts']):
        assert (entry['issuer'],entry['cohort'])==(original['issuer'],original['cohort'])
        for key in entry['comparisons']:
            entry['comparisons'][key].update({k:original['comparisons'][key][k] for k in fields})
    for entry,original in zip(primary['cycle'],cyc['tests']):
        assert entry['test_year']==original['test_year']
        for key in entry['comparisons']:
            entry['comparisons'][key].update({k:original['comparisons'][key][k] for k in fields})
    for tier in primary['main_pooled']:
        for key in primary['main_pooled'][tier]:
            primary['main_pooled'][tier][key].update({k:rev['pooled'][tier][key][k] for k in fields})
    for key in primary['cycle_pooled']:
        primary['cycle_pooled'][key].update({k:cyc['pooled'][key][k] for k in fields})
    result['primary_interval_source']='Primary percentage intervals reuse revision_results and cycle_results. Dollar-gap and leave-one-year-out intervals use the separately seeded robustness draws.'

def main():
    result={'analysis_status':'post_hoc_sensitivity_analysis','bootstrap':BOOT,
            'scenario_definitions':SCENARIOS,'scenarios':{},'month_cluster':[], 'gb_vs_lr':[]}
    primary={}
    for sid,(name,settings) in enumerate(SCENARIOS.items()):
        print('SCENARIO',name,flush=True)
        main_entries=[];main_draws=[];flow=[]
        for ii,(issuer,specs) in enumerate(bs.ISSUERS.items()):
            data={c:bs.load(*spec,**settings) for c,spec in specs.items()}
            if sid==0:primary[issuer]=data
            for c,d in data.items():flow.append({'dataset':issuer+'/'+c,'n':d['n'],'counts':d['counts'],'construction':d['construction']})
            for jj,cohort in enumerate(('validation','later')):
                e,draws=evaluate(data['train'],data[cohort],20260929+100*sid+10*ii+jj)
                e.update(issuer=issuer,cohort=cohort,tier=rs.TIER[issuer,cohort]);main_entries.append(e);main_draws.append(draws)
            print(name,issuer,'done',flush=True)
        main_pooled={}
        for tier in ('all','confirmatory'):
            idx=[j for j,e in enumerate(main_entries) if tier=='all' or e['tier']==tier]
            main_pooled[tier]=aggregate([main_entries[j] for j in idx],[main_draws[j] for j in idx])
        annual={y:bs.load(p,f'{y}-01',f'{y}-09',**settings) for y,p in cs.COHORTS.items()}
        for y,d in annual.items():flow.append({'dataset':str(y),'n':d['n'],'counts':d['counts'],'construction':d['construction']})
        cycles=[];cycle_draws=[]
        for y in range(2019,2024):
            # All five pairs are retained for diagnostics, including sparse-tail years.
            e,draws=evaluate(annual[y],annual[y+1],20261929+100*sid+y)
            e.update(train_year=y,test_year=y+1);cycles.append(e);cycle_draws.append(draws)
        leave_one_out={str(y):aggregate([e for e in cycles if e['test_year']!=y],
                            [d for e,d in zip(cycles,cycle_draws) if e['test_year']!=y]) for y in range(2020,2025)}
        result['scenarios'][name]={'flow':flow,'main':main_entries,'main_pooled':main_pooled,
                                   'cycle':cycles,'cycle_pooled':aggregate(cycles,cycle_draws),
                                   'cycle_leave_one_out':leave_one_out}
        (HERE/'robustness_results.partial.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False))
    # Conditional month-cluster checks and directly paired GB/LR decision comparisons.
    for ii,issuer in enumerate(bs.ISSUERS):
        data,rules,aucs=rs.fit_issuer(issuer)
        for jj,cohort in enumerate(('validation','later')):
            d=data[cohort];vals={name:r.value(d) for name,r in rules.items()}
            draws=bootstrap(vals,d,np.random.default_rng(20262929+10*ii+jj),cluster=True)
            cluster={'issuer':issuer,'cohort':cohort,'months':len(np.unique(d['event_month'])),'comparisons':{}}
            for r,k in PAIRS:
                key=f'{r}|{k}';cluster['comparisons'][key]=rs.summarize(compare(vals[r],k,d)['pct'],draws[key]['pct'])
            result['month_cluster'].append(cluster)
            paired={'issuer':issuer,'cohort':cohort,'comparisons':{}}
            rng=np.random.default_rng(20263929+10*ii+jj)
            pairs=((rs.GBM_CA,rs.CA),(rs.GBM_BA,rs.BA));samples={a:[] for a,b in pairs}
            for _ in range(BOOT):
                idx=rng.integers(0,d['n'],d['n']);sub=bs.resample(d,idx)
                for gb,lr in pairs:samples[gb].append(paired_gb_change(vals[gb][idx],vals[lr][idx],sub))
            for gb,lr in pairs:paired['comparisons'][gb]=rs.summarize(paired_gb_change(vals[gb],vals[lr],d),np.array(samples[gb]))
            result['gb_vs_lr'].append(paired)
        print('CLUSTER/PAIRED',issuer,'done',flush=True)
    # Same statistic on the original eight and expanded thirteen holdouts.
    rev=json.loads((HERE/'revision_results.json').read_text());cycle=json.loads((HERE/'cycle_results.json').read_text())
    points=[(h['auc']['logit'],h['comparisons'][rs.BA+'|scaled_base']['pct']) for h in rev['holdouts']]
    extra=[(t['auc_test'],t['comparisons'][rs.BA+'|scaled_base']['pct']) for t in cycle['tests']]
    result['correlations']={}
    for name,p in [('main_8',points),('all_13',points+extra)]:
        x,y=zip(*p); result['correlations'][name]={}
        for label,fn in [('pearson',pearsonr),('spearman',spearmanr)]:
            stat,pval=fn(x,y);result['correlations'][name][label]={'statistic':float(stat),'p':float(pval)}
    align_primary_intervals(result)
    (HERE/'robustness_results.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False))
    print('Saved robustness_results.json',flush=True)

if __name__=='__main__':main()
