"""Matched training-only ablation: same selected loss/regularization, omit arithmetic."""
from pathlib import Path
import json,time
import numpy as np
from sklearn.model_selection import KFold
from common import DATA,load_inputs,read_jsonl,parse_answer
from features import make_features,choose
from train import labels
from train_arithmetic import fit_model,scores,acc

if __name__=='__main__':
    start=time.perf_counter();root=Path('runs/arithmetic-search');e=json.loads((root/'experiments-arithmetic.json').read_text());selected=e['selected'];bank,arrays=load_inputs('train');y=labels(bank,read_jsonl(DATA/'gsm8k_train.jsonl'));valid=np.array([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank]);n=len(bank);x=make_features(bank,arrays,selected['mode']).reshape(n,4,-1);oof=np.zeros((n,4));records=[]
    for fold,(fit,hold) in enumerate(KFold(3,shuffle=True,random_state=2026).split(bank)):
        model=fit_model(x[fit],y[fit],valid[fit],selected['kind'],selected['parameter']);oof[hold]=scores(x[hold],model);records.append({'fold':fold,'without_arithmetic_accuracy':acc(oof[hold],[bank[i] for i in hold],y[hold])})
    original=np.load(root/'oof-scores.npz');with_arith=original[selected['name']]
    result={'selected':selected,'without_arithmetic_accuracy':acc(oof,bank,y),'with_arithmetic_accuracy':acc(with_arith,bank,y),'without_arithmetic_correct':int(y[np.arange(n),choose(oof,bank)].sum()),'with_arithmetic_correct':int(y[np.arange(n),choose(with_arith,bank)].sum()),'n':n,'folds':records,'seconds':time.perf_counter()-start,'test_accessed':False,'purpose':'Matched feature ablation after configuration selection; no new selection or tuning.'}
    (root/'ablation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
