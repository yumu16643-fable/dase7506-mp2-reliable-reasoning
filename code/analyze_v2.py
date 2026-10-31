"""Validation-only comparison of the frozen old and new methods, after selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
import json,math
from pathlib import Path
import numpy as np
from common import load_inputs,read_jsonl,parse_answer,DATA
from features import choose
from train import labels
from submission import score as oldscore
from submission_v2 import score

if __name__=='__main__':
    bank,arrays=load_inputs('validation'); refs=read_jsonl(DATA/'gsm8k_validation.jsonl'); y=labels(bank,refs); n=len(bank)
    s=score(bank,arrays,Path('runs/model_bundle_v2')); selected=np.array(choose(s,bank)); new=y[np.arange(n),selected]
    old=y[np.arange(n),choose(oldscore(bank,arrays,Path('runs/model_bundle')),bank)]
    distinct=[]; conf=[]
    for i,row in enumerate(bank):
        answers=[parse_answer(c['text']) for c in row['candidates']]; groups=list(dict.fromkeys(a for a in answers if a is not None)); distinct.append(len(groups))
        group_scores=np.array([s[i,answers.index(a)] for a in groups]); probs=np.exp(group_scores-group_scores.max()); probs/=probs.sum(); conf.append(float(probs.max()))
    distinct=np.asarray(distinct); conf=np.asarray(conf); bins=[]; ece=0.
    for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(conf>=low)&(conf<high if high<1 else conf<=high)
        if mask.any():
            row={'low':float(low),'high':float(high),'n':int(mask.sum()),'accuracy':float(new[mask].mean()),'confidence':float(conf[mask].mean())}; bins.append(row); ece+=mask.mean()*abs(row['accuracy']-row['confidence'])
    groups=[]
    for d in sorted(set(distinct)):
        m=distinct==d; groups.append({'distinct':int(d),'n':int(m.sum()),'old_accuracy':float(old[m].mean()),'v2_accuracy':float(new[m].mean())})
    hard=distinct>=2; order=np.argsort(-conf,kind='stable'); hard_order=np.flatnonzero(hard)[np.argsort(-conf[hard],kind='stable')]
    delta=new-old; rng=np.random.default_rng(2026); boot=np.array([delta[rng.integers(n,size=n)].mean() for _ in range(5000)])
    stats={'n':n,'v2_correct':int(new.sum()),'old_correct':int(old.sum()),'fixes':int(((new==1)&(old==0)).sum()),'breaks':int(((new==0)&(old==1)).sum()),'paired_bootstrap_gain95':np.quantile(boot,[.025,.975]).tolist(),'informative_n':int(hard.sum()),'informative_v2_accuracy':float(new[hard].mean()),'informative_old_accuracy':float(old[hard].mean()),'top_half_accuracy':float(new[order[:math.ceil(n/2)]].mean()),'informative_top_half_accuracy':float(new[hard_order[:math.ceil(hard.sum()/2)]].mean()),'groups':groups,'confidence_kind':'softmax over unique-answer pooled scores; no held-out calibration','brier':float(np.mean((conf-new)**2)),'ece':float(ece),'confidence_bins':bins}
    Path('../reports/analysis-v2.json').write_text(json.dumps(stats,indent=2)+'\n')
    errors=[{'position':int(i),'question':bank[i]['question'],'reference':refs[i]['answer'],'selected':int(selected[i]),'confidence':conf[i],'candidate':bank[i]['candidates'][selected[i]]['text']} for i in np.flatnonzero(new==0)]
    Path('../reports/validation-errors-v2.json').write_text(json.dumps(errors,indent=2)+'\n')
    print(json.dumps(stats,indent=2)); print('FIRST FAILURES'); print(json.dumps(errors[:3],indent=2))
