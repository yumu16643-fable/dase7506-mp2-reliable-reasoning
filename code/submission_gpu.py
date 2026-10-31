"""Portable CPU/NumPy evaluation of GPU-trained weights; never imports torch."""
import json
from pathlib import Path
import numpy as np
from common import parse_answer
from features import make_features,choose

def equality(bank):
    answer=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    same=np.zeros((len(bank),4,4),dtype=np.float32)
    for i,a in enumerate(answer):
        for j in range(4):
            for k in range(4): same[i,j,k]=float(j==k or (a[j] is not None and a[j]==a[k]))
    return same

def member_scores(bank,arrays,path,architecture,base_path):
    raw=make_features(bank,arrays,'combined'); same=equality(bank)
    with np.load(path,allow_pickle=False) as w:
        x=np.clip((raw-w['mean'])/w['scale'],-5,5).reshape(len(bank),4,-1)
        h=np.maximum(x@w['first_w']+w['first_b'],0)
        if 'context' in architecture:
            mean=np.broadcast_to(h.mean(axis=1,keepdims=True),h.shape)
            same_mean=(same@h)/np.maximum(same.sum(axis=2,keepdims=True),1)
            h=np.concatenate([h,mean,same_mean,h*mean],axis=2)
        h=np.maximum(h@w['second_w']+w['second_b'],0)
        logits=(h@w['output_w']+w['output_b']).reshape(len(bank),4)
    if architecture.startswith('residual'):
        context=make_features(bank,arrays,'context')
        with np.load(base_path,allow_pickle=False) as b:
            baseline=((context-b['mean'])/b['scale'])@b['w0']+b['b0']
        logits+=baseline.reshape(len(bank),4)
    return logits

def score(bank,arrays,checkpoint):
    folder=Path(checkpoint); config=json.loads((folder/'config.json').read_text())
    outputs=[member_scores(bank,arrays,folder/m['file'],m['architecture'],folder/'baseline.npz') for m in config['members']]
    return np.mean(outputs,axis=0)

def predict(bank,arrays,checkpoint):
    if not bank: return []
    return choose(score(bank,arrays,checkpoint),bank)
