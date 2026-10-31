"""NumPy inference with optional bounded arithmetic features; no reference IO."""
import json
from pathlib import Path
import numpy as np
from features import make_features,choose
from arithmetic_features import arithmetic_features

def score(bank,arrays,checkpoint):
    path=Path(checkpoint); config=json.loads((path/'config.json').read_text()); m=config['member']
    x=make_features(bank,arrays,m['mode'])
    if m['arithmetic']:x=np.concatenate([x,arithmetic_features(bank)],axis=1)
    with np.load(path/m['file'],allow_pickle=False) as w:z=((x-w['mean'])/w['scale'])@w['w0']+w['b0']
    return z.reshape(len(bank),4)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    return choose(score(bank,arrays,checkpoint),bank)
