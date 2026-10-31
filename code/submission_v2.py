"""Frozen ranking inference with NumPy only; no reference files or training."""
import json
from pathlib import Path
import numpy as np
from features import make_features,choose
from ranking import aggregate

def score(bank,arrays,checkpoint):
    folder=Path(checkpoint); config=json.loads((folder/'config.json').read_text())
    cache={}; results=[]
    for member in config['members']:
        mode=member['mode']
        if mode not in cache: cache[mode]=make_features(bank,arrays,mode)
        with np.load(folder/member['file'],allow_pickle=False) as f:
            x=(cache[mode]-f['mean'])/f['scale']
            for layer in range(member.get('layers',1)):
                x=x@f['w'+str(layer)]+f['b'+str(layer)]
                if layer+1<member.get('layers',1): x=np.maximum(x,0)
            results.append(member.get('weight',1.)*x.reshape(len(bank),4))
    raw=np.sum(results,axis=0)/sum(m.get('weight',1.) for m in config['members'])
    return aggregate(raw,bank,**config.get('aggregation',{'kind':'max'}))

def predict(bank,arrays,checkpoint):
    if not bank: return []
    return choose(score(bank,arrays,checkpoint),bank)
