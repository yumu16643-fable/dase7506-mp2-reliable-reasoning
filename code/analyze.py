"""Post-freeze validation analysis. Does not load test labels or change the model."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
import csv,json,math
from pathlib import Path
from collections import Counter
import numpy as np
from common import load_inputs,parse_answer,read_jsonl,score_choices
from features import choose,majority
from submission import score

def wilson(k,n):
    z=1.959963984540054; p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def main():
    reports=Path('../reports'); reports.mkdir(exist_ok=True)
    bank,arrays=load_inputs('validation'); refs=read_jsonl('data/gsm8k_validation.jsonl')
    ex=json.loads(Path('runs/search/experiments.json').read_text())
    saved=np.load('runs/search/validation_analysis.npz'); y=saved['labels']; n=len(bank)
    s=score(bank,arrays,Path('runs/model_bundle')); picked=np.array(choose(s,bank)); correct=y[np.arange(n),picked]
    base=np.array(choose(majority(bank),bank)); bc=y[np.arange(n),base]
    assert np.array_equal(picked,np.array(choose(saved[ex['selected']],bank)))
    # Cross-check every validation result against the unchanged independent scorer.
    for exp in ex['experiments']:
        check=score_choices('validation',[r['id'] for r in bank],choose(saved[exp['name']],bank))
        assert abs(check['accuracy']/100-exp['validation_accuracy'])<1e-12
    conf=1/(1+np.exp(-np.clip(s[np.arange(n),picked],-30,30)))
    bins=[]; ece=0
    for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(conf>=low)&(conf<high if high<1 else conf<=high)
        if mask.any():
            acc=float(correct[mask].mean()); avg=float(conf[mask].mean()); count=int(mask.sum()); ece+=count/n*abs(acc-avg)
            bins.append({'low':float(low),'high':float(high),'n':count,'accuracy':acc,'confidence':avg})
    coverage=[]; order=np.argsort(-conf,kind='stable')
    for fraction in [.25,.5,.75,1.]:
        k=math.ceil(n*fraction); coverage.append({'coverage':k/n,'n':k,'accuracy':float(correct[order[:k]].mean()),'minimum_confidence':float(conf[order[k-1]])})
    distinct=np.array([len(set(a for a in [parse_answer(c['text']) for c in r['candidates']] if a is not None)) for r in bank])
    groups=[]
    for k in sorted(set(distinct)):
        mask=distinct==k; groups.append({'distinct_answers':int(k),'n':int(mask.sum()),'majority_accuracy':float(bc[mask].mean()),'model_accuracy':float(correct[mask].mean())})
    delta=correct-bc; rng=np.random.default_rng(2026)
    diffs=np.array([delta[rng.integers(n,size=n)].mean() for _ in range(5000)])
    both_disagree=base!=picked
    # Different indices can have the same final answer; use numeric answer disagreement.
    answer_disagree=np.array([parse_answer(r['candidates'][i]['text'])!=parse_answer(r['candidates'][j]['text']) for r,i,j in zip(bank,base,picked)])
    stats={'n':n,'correct':int(correct.sum()),'accuracy':float(correct.mean()),'wilson95':wilson(int(correct.sum()),n),'majority_correct':int(bc.sum()),'net_gain_questions':int(delta.sum()),'paired_bootstrap95_gain':np.quantile(diffs,[.025,.975]).tolist(),'model_fixes_majority':int(((correct==1)&(bc==0)).sum()),'model_breaks_majority':int(((correct==0)&(bc==1)).sum()),'different_answer_n':int(answer_disagree.sum()),'accuracy_when_different':float(correct[answer_disagree].mean()),'confidence_brier':float(np.mean((conf-correct)**2)),'confidence_ece_5_equal_width_bins':ece,'confidence_bins':bins,'coverage':coverage,'distinct_answer_groups':groups,'random_expected_accuracy':float(y.mean()),'raw_candidate_zero_accuracy':float(y[:,0].mean())}
    hard=distinct>=2; hard_order=np.flatnonzero(hard)[np.argsort(-conf[hard],kind='stable')]
    stats['informative_subset']={'n':int(hard.sum()),'model_accuracy':float(correct[hard].mean()),'majority_accuracy':float(bc[hard].mean()),'top_half_n':int(math.ceil(hard.sum()/2)),'top_half_accuracy':float(correct[hard_order[:math.ceil(hard.sum()/2)]].mean())}
    (reports/'analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
    errors=[]
    with (reports/'validation_choices.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['id','selected','majority','correct','confidence','distinct_answers'])
        for i,row in enumerate(bank):
            writer.writerow([row['id'],int(picked[i]),int(base[i]),int(correct[i]),float(conf[i]),int(distinct[i])])
            if not correct[i]: errors.append({'position':i,'question':row['question'],'reference':refs[i]['answer'],'selected':int(picked[i]),'majority':int(base[i]),'confidence':float(conf[i]),'candidates':row['candidates']})
    (reports/'validation_errors.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(stats,indent=2))
    print('FIRST ERRORS')
    for r in errors[:4]:
        print(json.dumps({'position':r['position'],'question':r['question'],'reference':r['reference'],'selected':r['selected'],'candidate':r['candidates'][r['selected']]['text']},ensure_ascii=False))

if __name__=='__main__': main()
