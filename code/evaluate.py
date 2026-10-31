"""Externally time fresh official prediction + independent scoring processes."""
import argparse,json,os,platform,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--checkpoint',default='runs/model_bundle'); p.add_argument('--prefix')
    a=p.parse_args(); prefix=a.prefix or 'runs/'+a.split
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[k]='4'
    commands=[ [sys.executable,'predict.py','--method','submission.py','--checkpoint',a.checkpoint,'--split',a.split,'--output',prefix+'-predictions.json'], [sys.executable,'score_predictions.py','--split',a.split,'--predictions',prefix+'-predictions.json','--output',prefix+'-score.json'] ]
    start=time.perf_counter(); timings=[]
    for command in commands:
        tick=time.perf_counter(); subprocess.run(command,env=env,check=True); timings.append(time.perf_counter()-tick)
    result={'split':a.split,'prediction_process_seconds':timings[0],'scoring_process_seconds':timings[1], 'total_wall_seconds':time.perf_counter()-start,'threads':4,'processor':'Intel Core i7-13650HX','platform':platform.platform(),'python':platform.python_version(),'measurement':'Two fresh processes; includes startup, imports, data hashes/loading, assets, preprocessing and scoring. OS disk cache not flushed.'}
    Path(prefix+'-resources.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
