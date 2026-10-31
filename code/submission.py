"""Portable NumPy inference; no reference IO, sklearn, training or cached choices."""
import json
from pathlib import Path
import numpy as np
from features import make_features, choose

def score(bank,arrays,checkpoint):
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'config.json').read_text())
    cache={}; outputs=[]
    for member in config['members']:
        mode=member['mode']
        if mode not in cache: cache[mode]=make_features(bank,arrays,mode)
        with np.load(checkpoint/member['file'],allow_pickle=False) as model:
            x=(cache[mode]-model['mean'])/model['scale']
            for layer in range(member['layers']):
                x=x@model[f'w{layer}']+model[f'b{layer}']
                if layer+1<member['layers']: x=np.maximum(x,0)
            outputs.append(x.reshape(len(bank),4))
    return np.mean(outputs,axis=0)

def predict(bank,arrays,checkpoint):
    if not bank: return []
    if checkpoint is None: raise ValueError('Supply --checkpoint runs/model_bundle')
    return choose(score(bank,arrays,checkpoint),bank)
