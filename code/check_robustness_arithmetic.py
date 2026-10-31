"""Reshuffled training-only CV of the already selected configuration; no reselection."""
from pathlib import Path
import json,time
import numpy as np
from sklearn.model_selection import KFold
from common import DATA,load_inputs,read_jsonl,parse_answer
from features import make_features
from arithmetic_features import arithmetic_features
from train import labels
from train_arithmetic import fit_model,scores,acc

if __name__=='__main__':
    start=time.perf_counter();root=Path('runs/arithmetic-search');e=json.loads((root/'experiments-arithmetic.json').read_text());selected=e['selected'];bank,arrays=load_inputs('train');y=labels(bank,read_jsonl(DATA/'gsm8k_train.jsonl'));valid=np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank]);n=len(bank)
    base=make_features(bank,arrays,'context').reshape(n,4,-1);raw=make_features(bank,arrays,selected['mode']).reshape(n,4,-1)
    if selected['arithmetic']:raw=np.concatenate([raw,arithmetic_features(bank).reshape(n,4,-1)],axis=2)
    original_oof=np.zeros((n,4));selected_oof=np.zeros((n,4));rows=[]
    for fold,(fit,hold) in enumerate(KFold(3,shuffle=True,random_state=2037).split(bank)):
        old=fit_model(base[fit],y[fit],valid[fit],'bce',.01);new=fit_model(raw[fit],y[fit],valid[fit],selected['kind'],selected['parameter'])
        original_oof[hold]=scores(base[hold],old);selected_oof[hold]=scores(raw[hold],new)
        row={'fold':fold,'original_accuracy':acc(original_oof[hold],[bank[i] for i in hold],y[hold]),'selected_accuracy':acc(selected_oof[hold],[bank[i] for i in hold],y[hold])};rows.append(row);print(row,flush=True)
    result={'seed':2037,'selected_fixed_before_diagnostic':selected,'original_accuracy':acc(original_oof,bank,y),'selected_accuracy':acc(selected_oof,bank,y),'n':n,'folds':rows,'seconds':time.perf_counter()-start,'test_accessed':False,'note':'Same training bank reshuffled; diagnostic only, not a new independent dataset or configuration-selection step.'}
    (root/'robustness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
