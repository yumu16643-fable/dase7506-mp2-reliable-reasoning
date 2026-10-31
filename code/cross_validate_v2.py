"""Question-disjoint three-fold diagnostics entirely within the supplied train split."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[key]='4'
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,check_grad
from scipy.special import softmax,logsumexp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import DATA,load_inputs,read_jsonl,parse_answer
from train import labels
from features import make_features,choose
from ranking import aggregate

def objective(w,x,y,valid,reg):
    z=np.where(valid,x@w,-1e12); positive=np.where(y,z,-1e12)
    loss=(logsumexp(z,axis=1)-logsumexp(positive,axis=1)).mean()+reg*np.dot(w,w)/2
    grad=np.einsum('nkd,nk->d',x,softmax(z,axis=1)-softmax(positive,axis=1),optimize=True)/len(x)+reg*w
    return loss,grad

if __name__=='__main__':
    started=time.perf_counter(); bank,arrays=load_inputs('train'); y=labels(bank,read_jsonl(DATA/'gsm8k_train.jsonl'))
    e=make_features(bank,arrays,'combined').reshape(len(bank),4,-1)
    context=make_features(bank,arrays,'context').reshape(len(bank),4,-1)
    valid=np.array([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank])
    informative=((y==0)&valid).any(axis=1)
    # Finite-difference check on an independently generated masked, multi-positive problem.
    rng=np.random.default_rng(2026); tx=rng.normal(size=(5,4,7)); ty=np.array([[1,0,0,0],[1,1,0,0],[0,1,0,0],[0,0,1,1],[1,0,1,0]],dtype=bool); tv=np.ones((5,4),dtype=bool); tv[0,3]=False; w=rng.normal(size=7)*.1
    err=check_grad(lambda w:objective(w,tx,ty,tv,.1)[0],lambda w:objective(w,tx,ty,tv,.1)[1],w)
    assert err<1e-5,err
    baseline_correct=[]; improved_correct=[]; folds=[]
    for fold,(fit,hold) in enumerate(KFold(3,shuffle=True,random_state=2026).split(bank)):
        foldbank=[bank[i] for i in hold]
        sc=StandardScaler().fit(context[fit].reshape(-1,context.shape[-1])); xt=sc.transform(context[fit].reshape(-1,context.shape[-1])); xv=sc.transform(context[hold].reshape(-1,context.shape[-1]))
        model=LogisticRegression(C=.01,max_iter=400,random_state=2026).fit(xt,y[fit].ravel())
        old=np.array(choose(model.decision_function(xv).reshape(-1,4),foldbank)); bc=y[hold,old]
        sc=StandardScaler().fit(e[fit].reshape(-1,e.shape[-1])); xfit=sc.transform(e[fit].reshape(-1,e.shape[-1])).reshape(len(fit),4,-1)
        mask=informative[fit]; xfit=xfit[mask].astype(np.float64); yf=y[fit][mask].astype(bool); vf=valid[fit][mask]
        trained=minimize(lambda w:objective(w,xfit,yf,vf,.1),np.zeros(e.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-9})
        xv=sc.transform(e[hold].reshape(-1,e.shape[-1])); raw=(xv@trained.x).reshape(-1,4)
        new=np.array(choose(aggregate(raw,foldbank,'logsumexp',.5,1.),foldbank)); nc=y[hold,new]
        baseline_correct.extend(bc); improved_correct.extend(nc)
        row={'fold':fold,'heldout_questions':len(hold),'baseline_accuracy':float(bc.mean()),'v2_accuracy':float(nc.mean()),'v2_optimizer_converged':bool(trained.success)}; folds.append(row); print(row,flush=True)
    result={'folds':folds,'baseline_accuracy':float(np.mean(baseline_correct)),'v2_accuracy':float(np.mean(improved_correct)),'baseline_correct':int(np.sum(baseline_correct)),'v2_correct':int(np.sum(improved_correct)),'n':len(bank),'gradient_finite_difference_error':float(err),'seconds':time.perf_counter()-started,'scope':'training split only; selected configuration diagnostic, not independent confirmation after validation model selection'}
    Path('runs/v2-search/cross-validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
