"""Confirm actual CUDA execution and portable export on synthetic candidate groups."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from gpu_model import Verifier,loss_value
from submission_gpu import member_scores,equality
from features import make_features,choose
from train_gpu import export

if __name__=='__main__':
    assert torch.cuda.is_available(),'No CUDA device'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    bank=[{'id':'synthetic-'+str(i),'question':'There are 4 apples and 2 are eaten. How many remain?','candidates':[{'index':j,'text':t} for j,t in enumerate(['4 - 2 = 2. #### 2','The answer is 4. #### 4','Two remain. \\boxed{2}','Unparseable answer'])]} for i in range(3)]
    rng=np.random.default_rng(2026); arrays={'embeddings':rng.normal(size=(3,4,1024)).astype(np.float32)}
    x=make_features(bank,arrays,'combined').reshape(3,4,-1); sc=SimpleNamespace(mean_=np.zeros(x.shape[-1]),scale_=np.ones(x.shape[-1])); x=np.clip(x,-5,5); same=equality(bank)
    folder=Path('runs/gpu-check'); folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/'baseline.npz',mean=np.zeros(2094,dtype=np.float32),scale=np.ones(2094,dtype=np.float32),w0=np.zeros((2094,1),dtype=np.float32),b0=np.array([.25],dtype=np.float32))
    results=[]
    for architecture in ['shared_bce','shared_marginal','context_marginal','residual_context_marginal']:
        torch.manual_seed(2026); model=Verifier(x.shape[-1],architecture).cuda().eval(); path=folder/(architecture+'.npz'); export(model,sc,path)
        baseline=np.ones((3,4),dtype=np.float32)*.25
        with torch.no_grad(): z=model(torch.tensor(x,device='cuda'),torch.tensor(same,device='cuda'),torch.tensor(baseline,device='cuda')).cpu().numpy()
        nz=member_scores(bank,arrays,path,architecture,folder/'baseline.npz'); assert np.allclose(z,nz,atol=1e-4,rtol=1e-4),architecture
        perm=[2,0,3,1]; changed=[{**row,'candidates':[row['candidates'][j] for j in perm]} for row in bank]
        pz=member_scores(changed,{'embeddings':arrays['embeddings'][:,perm]},path,architecture,folder/'baseline.npz'); assert np.allclose(pz,nz[:,perm],atol=1e-4,rtol=1e-4),architecture
        results.append({'architecture':architecture,'max_cuda_numpy_difference':float(np.max(np.abs(z-nz))),'permutation_scores_passed':True})
    logits=torch.randn(3,4,dtype=torch.double,device='cuda',requires_grad=True)
    y=torch.tensor([[1,0,1,0],[0,1,0,0],[0,0,1,1]],dtype=torch.double,device='cuda'); valid=torch.ones_like(y); valid[0,3]=0
    assert torch.autograd.gradcheck(lambda z:loss_value(z,y,valid,'shared_marginal'),(logits,),eps=1e-6,atol=1e-4)
    result={'cuda_available':True,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'synthetic_export_tests':results,'marginal_loss_gradient_check':'passed','test_bank_accessed':False}
    Path('runs/gpu-check/checks.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
