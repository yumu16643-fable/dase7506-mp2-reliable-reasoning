"""Copy the validation-selected inference assets and record hashes before test scoring."""
from pathlib import Path
import datetime,hashlib,json,shutil

if __name__=='__main__':
    source=Path('runs/search'); target=Path('runs/model_bundle'); target.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'config.json').read_text())
    for name in ['config.json']+[m['file'] for m in config['members']]: shutil.copyfile(source/name,target/name)
    files=[Path('features.py'),Path('submission.py'),Path('train.py')]+list(target.iterdir())
    hashes={str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    record={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection':'Highest validation accuracy; ties follow fixed search order. No test inputs/labels used.', 'model':config['name'],'upstream_commit':'bb801f4f2e7629d270df765ab12c333b2dc616bb','inference_asset_bytes':sum(p.stat().st_size for p in target.iterdir()),'inference_asset_limit_bytes':16*2**20,'sha256':hashes}
    Path('runs/freeze.json').write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record,indent=2))
