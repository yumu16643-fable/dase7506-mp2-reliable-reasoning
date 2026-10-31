"""Explicit final test scoring of the previously frozen arithmetic checkpoint."""
import os,json,subprocess,sys,time,platform,hashlib
from pathlib import Path

if __name__=='__main__':
    frozen=json.loads(Path('runs/freeze-arithmetic.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['sha256'].items())
    assert Path('runs/arithmetic-final-evaluation.json').exists(),'Record final-evaluation scope first'
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
    commands=[[sys.executable,'predict.py','--method','submission_arithmetic.py','--checkpoint','runs/model_bundle_arithmetic','--split','test','--output','runs/arithmetic-test-predictions.json'],[sys.executable,'score_predictions.py','--split','test','--predictions','runs/arithmetic-test-predictions.json','--output','runs/arithmetic-test-score.json']]
    start=time.perf_counter();durations=[]
    for command in commands:
        tick=time.perf_counter();subprocess.run(command,env=env,check=True);durations.append(time.perf_counter()-tick)
    record={'split':'test','prediction_process_seconds':durations[0],'scoring_process_seconds':durations[1],'total_wall_seconds':time.perf_counter()-start,'threads':4,'python':platform.python_version(),'processor':'Intel Core i7-13650HX','gpu_used':False,'frozen_source_and_assets_verified':True,'measurement':'Two fresh official processes; startup/imports/hashes/data/asset loading/preprocessing/scoring included. OS disk cache not flushed; reference Xeon not measured.'}
    Path('runs/arithmetic-test-resources.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
