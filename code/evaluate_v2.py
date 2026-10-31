"""Fresh official prediction/scoring processes for the frozen v2 checkpoint."""
import argparse,json,os,platform,subprocess,sys,time
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['validation','test'],required=True); a=p.parse_args()
    prefix='runs/v2-'+a.split; env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[k]='4'
    commands=[[sys.executable,'predict.py','--method','submission_v2.py','--checkpoint','runs/model_bundle_v2','--split',a.split,'--output',prefix+'-predictions.json'],[sys.executable,'score_predictions.py','--split',a.split,'--predictions',prefix+'-predictions.json','--output',prefix+'-score.json']]
    elapsed=[]; start=time.perf_counter()
    for command in commands:
        tick=time.perf_counter(); subprocess.run(command,check=True,env=env); elapsed.append(time.perf_counter()-tick)
    result={'split':a.split,'prediction_process_seconds':elapsed[0],'scoring_process_seconds':elapsed[1],'total_wall_seconds':time.perf_counter()-start,'threads':4,'processor':'Intel Core i7-13650HX','platform':platform.platform(),'python':platform.python_version(),'measurement':'Two fresh processes; startup/imports/hashes/data/assets/preprocessing/scoring included. OS disk cache not flushed.'}
    Path(prefix+'-resources.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
