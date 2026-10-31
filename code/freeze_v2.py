"""Freeze the validation-selected v2 configuration before any new test scoring."""
from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
from common import load_inputs
from features import choose
from ranking import aggregate
from submission_v2 import score,predict

if __name__=='__main__':
    source=Path('runs/v2-search'); target=Path('runs/model_bundle_v2'); target.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'config.json').read_text()); summary=json.loads((source/'experiments-v2.json').read_text())
    assert summary['selected']['validation_accuracy']>summary['baseline_validation_accuracy']
    for name in ['config.json']+[m['file'] for m in config['members']]: shutil.copyfile(source/name,target/name)
    bank,arrays=load_inputs('validation'); outputs=score(bank,arrays,target)
    saved=np.load(source/'validation-v2.npz'); expected=aggregate(saved[config['name']],bank,**config['aggregation'])
    assert choose(outputs,bank)==choose(expected,bank),'FP32 export changed selected answers'
    import copy
    small=copy.deepcopy(bank[:7]); smallarrays={'embeddings':arrays['embeddings'][:7].copy()}; before=predict(small,smallarrays,target)
    for i,row in enumerate(small): row['id']='new-'+str(i)
    assert predict(small,smallarrays,target)==before
    perm=[2,0,3,1]; changed=copy.deepcopy(small)
    for row in changed:
        row['candidates']=[row['candidates'][j] for j in perm]
        for j,c in enumerate(row['candidates']): c['index']=j
    after=score(changed,{'embeddings':smallarrays['embeddings'][:,perm]},target)
    assert np.allclose(after,score(small,smallarrays,target)[:,perm],atol=1e-5,rtol=1e-5)
    files=[Path(f) for f in ['submission_v2.py','features.py','ranking.py','train_v2.py']]+list(target.iterdir())
    frozen={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':config['name'],'aggregation':config['aggregation'],'validation_correct':round(summary['selected']['validation_accuracy']*len(bank)),'validation_n':len(bank),'inference_asset_bytes':sum(p.stat().st_size for p in target.iterdir()),'sha256':{str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'checks':['FP32 export preserves all validation choices','seven-question bank','renamed IDs','permuted candidate scores'],'test_accessed_by_v2_development':False}
    Path('runs/freeze-v2.json').write_text(json.dumps(frozen,indent=2)+'\n'); print(json.dumps(frozen,indent=2))
