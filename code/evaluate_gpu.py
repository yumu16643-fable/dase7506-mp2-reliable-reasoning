"""Fresh CPU-only official evaluation of the GPU-trained artifact (validation only)."""
import json,os,platform,subprocess,sys,time
from pathlib import Path

if __name__=='__main__':
    env=os.environ.copy()
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[k]='4'
    commands=[[sys.executable,'predict.py','--method','submission_gpu.py','--checkpoint','runs/model_bundle_gpu','--split','validation','--output','runs/gpu-validation-predictions.json'],[sys.executable,'score_predictions.py','--split','validation','--predictions','runs/gpu-validation-predictions.json','--output','runs/gpu-validation-score.json']]
    durations=[]; start=time.perf_counter()
    for command in commands:
        tick=time.perf_counter(); subprocess.run(command,check=True,env=env); durations.append(time.perf_counter()-tick)
    result={'split':'validation','questions':798,'prediction_process_seconds':durations[0],'scoring_process_seconds':durations[1],'total_wall_seconds':time.perf_counter()-start,'threads':4,'processor':'Intel Core i7-13650HX','python':platform.python_version(),'numpy_only_inference':True,'cuda_used_during_inference':False,'measurement':'Fresh official prediction/scoring; startup/imports/hashes/loading/preprocessing included; OS disk cache not flushed. Not a reference Xeon measurement.'}
    Path('runs/gpu-validation-resources.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
