"""Freeze the training-CV-selected arithmetic checkpoint without test access."""
from pathlib import Path
import json,hashlib,datetime,shutil,copy
import numpy as np
from common import load_inputs
from submission_arithmetic import score,predict

if __name__=='__main__':
    src=Path('runs/arithmetic-search/selected_bundle');dest=Path('runs/model_bundle_arithmetic');dest.mkdir(exist_ok=True)
    for f in src.iterdir():shutil.copyfile(f,dest/f.name)
    e=json.loads(Path('runs/arithmetic-search/experiments-arithmetic.json').read_text());bank,arrays=load_inputs('validation'); small=copy.deepcopy(bank[:7]); data={'embeddings':arrays['embeddings'][:7].copy()}
    z=score(small,data,dest);old=predict(small,data,dest)
    for i,row in enumerate(small):row['id']='new-bank-'+str(i)
    assert predict(small,data,dest)==old
    perm=[2,0,3,1]; changed=copy.deepcopy(small)
    for row in changed:
        row['candidates']=[row['candidates'][j] for j in perm]
        for j,c in enumerate(row['candidates']):c['index']=j
    pz=score(changed,{'embeddings':data['embeddings'][:,perm]},dest);assert np.allclose(pz,z[:,perm],atol=1e-4,rtol=1e-5)
    sources=['submission_arithmetic.py','arithmetic_features.py','features.py','train_arithmetic.py','cross_validate_v2.py','arithmetic_plan.json']
    files=[Path(x) for x in sources]+list(dest.iterdir())
    result={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected':e['selected'],'validation_correct':e['validation_correct'],'validation_n':e['validation_n'],'inference_asset_bytes':sum(f.stat().st_size for f in dest.iterdir()),'test_accessed':False,'sha256':{str(f).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'checks':['numeric expression behavioral tests','NumPy export validation choices match','renamed IDs','seven-question bank','candidate score permutation equivariance']}
    Path('runs/freeze-arithmetic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
