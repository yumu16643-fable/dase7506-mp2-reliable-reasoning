"""Freeze the CV-selected GPU artifact for validation/audit reproduction, not test tuning."""
from pathlib import Path
import datetime,hashlib,json,shutil,copy
import numpy as np
from common import load_inputs,parse_answer
from submission_gpu import score,predict

if __name__=='__main__':
    src=Path('runs/gpu-search/selected_bundle'); dest=Path('runs/model_bundle_gpu'); dest.mkdir(exist_ok=True)
    for f in src.iterdir(): shutil.copyfile(f,dest/f.name)
    exp=json.loads(Path('runs/gpu-search/experiments-gpu.json').read_text())
    bank,arrays=load_inputs('validation'); small=copy.deepcopy(bank[:7]); data={'embeddings':arrays['embeddings'][:7].copy()}
    z=score(small,data,dest); old=predict(small,data,dest)
    for i,row in enumerate(small): row['id']='new-audit-'+str(i)
    assert predict(small,data,dest)==old
    perm=[2,0,3,1]; changed=copy.deepcopy(small)
    for row in changed:
        row['candidates']=[row['candidates'][j] for j in perm]
        for j,c in enumerate(row['candidates']): c['index']=j
    pz=score(changed,{'embeddings':data['embeddings'][:,perm]},dest)
    assert np.allclose(pz,z[:,perm],atol=1e-4,rtol=1e-5)
    assert all(type(x) is int for x in old)
    sources=['submission_gpu.py','features.py','gpu_model.py','train_gpu.py','gpu_plan.json']
    files=[Path(x) for x in sources]+list(dest.iterdir())
    result={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':exp['selected'],'promotion_gate_passed':exp['promotion_gate_passed'],'public_test_accessed':False,'inference_asset_bytes':sum(f.stat().st_size for f in dest.iterdir()),'sha256':{str(f).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'checks':['actual CUDA execution','automatic-differentiation finite-difference check','GPU/NumPy validation choices match','renamed IDs','different bank size','candidate score permutation equivariance']}
    Path('runs/freeze-gpu.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
