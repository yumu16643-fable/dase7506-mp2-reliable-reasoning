"""Fresh-process validation scoring; this experimental round does not open test."""
import os,json,subprocess,sys,time,platform
from pathlib import Path

if __name__=='__main__':
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='4'
    commands=[[sys.executable,'predict.py','--method','submission_arithmetic.py','--checkpoint','runs/model_bundle_arithmetic','--split','validation','--output','runs/arithmetic-validation-predictions.json'],[sys.executable,'score_predictions.py','--split','validation','--predictions','runs/arithmetic-validation-predictions.json','--output','runs/arithmetic-validation-score.json']]
    start=time.perf_counter();seconds=[]
    for command in commands:
        tick=time.perf_counter();subprocess.run(command,env=env,check=True);seconds.append(time.perf_counter()-tick)
    record={'split':'validation','prediction_process_seconds':seconds[0],'scoring_process_seconds':seconds[1],'total_wall_seconds':time.perf_counter()-start,'threads':4,'python':platform.python_version(),'processor':'Intel Core i7-13650HX','gpu_used':False,'measurement':'Fresh official prediction/scoring processes with startup, imports, data hashes/loading, assets and preprocessing; OS disk cache not flushed; reference Xeon not measured.'}
    Path('runs/arithmetic-validation-resources.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
