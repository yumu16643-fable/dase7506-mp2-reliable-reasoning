"""Check inference on a differently sized, renamed and permuted validation input."""
from pathlib import Path
import copy,json
import numpy as np
from common import load_inputs,parse_answer
from submission import predict,score

if __name__=='__main__':
    bank,arrays=load_inputs('validation'); bank=copy.deepcopy(bank[:7]); arrays={'embeddings':arrays['embeddings'][:7].copy()}; checkpoint=Path('runs/model_bundle')
    before=predict(bank,arrays,checkpoint); scores=score(bank,arrays,checkpoint)
    for i,row in enumerate(bank): row['id']='unseen-id-'+str(i)
    assert predict(bank,arrays,checkpoint)==before
    perm=np.array([2,0,3,1])
    changed=copy.deepcopy(bank)
    for row in changed:
        row['candidates']=[row['candidates'][j] for j in perm]
        for j,c in enumerate(row['candidates']): c['index']=j
    after_score=score(changed,{'embeddings':arrays['embeddings'][:,perm]},checkpoint)
    assert np.allclose(after_score,scores[:,perm],atol=3e-5,rtol=1e-5)
    after=predict(changed,{'embeddings':arrays['embeddings'][:,perm]},checkpoint)
    for row,old,new in zip(bank,before,after):
        assert parse_answer(row['candidates'][old]['text'])==parse_answer(row['candidates'][int(perm[new])]['text'])
    assert all(type(x) is int and 0<=x<4 for x in before)
    assert predict([],{'embeddings':np.zeros((0,4,1024),dtype=np.float32)},checkpoint)==[]
    result={'status':'passed','checks':['seven-question unseen-size bank','renamed IDs do not change predictions','permuted candidate scores and selected answers','Python integer outputs','empty bank']}
    Path('runs/inference-checks.json').write_text(json.dumps(result,indent=2)+'\n'); print(result)
