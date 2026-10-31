"""Reference-free score aggregation over candidates sharing a numeric answer."""
import numpy as np
from common import parse_answer
from features import choose

def aggregate(scores,bank,kind='max',temperature=1.,count_penalty=0.):
    scores=np.asarray(scores,dtype=np.float64).reshape(len(bank),4)
    if kind=='max': return scores.copy()
    result=np.full_like(scores,-1e12)
    for i,row in enumerate(bank):
        answers=[parse_answer(c['text']) for c in row['candidates']]
        for a in set(answers)-{None}:
            members=[j for j,b in enumerate(answers) if b==a]
            z=scores[i,members]/temperature; m=z.max()
            group=temperature*(m+np.log(np.exp(z-m).sum())-count_penalty*np.log(len(members)))
            # Same group score for every representative; numeric output is invariant.
            result[i,members]=group
        if not any(a is not None for a in answers): result[i]=scores[i]
    return result
