"""CPU ranking development. Reads only train/validation, never test inputs/references."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='4'
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import DATA,load_inputs,read_jsonl,parse_answer
from features import make_features,choose
from train import labels
from ranking import aggregate

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path('runs/v2-search')); args=parser.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    tr,ta=load_inputs('train'); va,vb=load_inputs('validation')
    yt=labels(tr,read_jsonl(DATA/'gsm8k_train.jsonl')); yv=labels(va,read_jsonl(DATA/'gsm8k_validation.jsonl'))
    valid=np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in tr])
    informative=((yt==0)&valid).any(axis=1)
    records=[]; models={}; scores={}
    def add(name,mode,w,b,scaler,sv,seconds,details):
        np.savez(out/(name+'.npz'),w0=np.asarray(w,dtype=np.float32).reshape(-1,1),b0=np.asarray([b],dtype=np.float32),mean=scaler.mean_.astype(np.float32),scale=scaler.scale_.astype(np.float32))
        models[name]={'file':name+'.npz','mode':mode,'layers':1}; scores[name]=sv
        row={'name':name,'validation_accuracy':float(np.mean(yv[np.arange(len(va)),choose(sv,va)])),'seconds':seconds,**details}
        records.append(row); print(json.dumps(row),flush=True)
    # Pairwise logistic: only correct-versus-valid-wrong pairs can affect final decisions.
    for mode in ['text','combined']:
        scaler=StandardScaler().fit(make_features(tr,ta,mode))
        xt=scaler.transform(make_features(tr,ta,mode)); xv=scaler.transform(make_features(va,vb,mode))
        xg=xt.reshape(len(tr),4,-1)
        pairs=[]; pair_weights=[]
        for i in np.flatnonzero(informative):
            pos=np.flatnonzero(yt[i]); neg=np.flatnonzero((yt[i]==0)&valid[i]); weight=1/(len(pos)*len(neg))
            for a in pos:
                for b in neg: pairs.append(xg[i,a]-xg[i,b]); pair_weights.append(weight)
        diffs=np.asarray(pairs); sw=np.asarray(pair_weights); sw*=len(sw)/sw.sum()
        paired=np.concatenate([diffs,-diffs]); target=np.r_[np.ones(len(diffs)),np.zeros(len(diffs))]; sw=np.r_[sw,sw]
        for c in [.001,.01,.1,1.]:
            tick=time.perf_counter(); model=LogisticRegression(C=c,fit_intercept=False,max_iter=300,random_state=2026).fit(paired,target,sample_weight=sw)
            name=f'pair_{mode}_{c}'
            add(name,mode,model.coef_[0],0.,scaler,model.decision_function(xv).reshape(-1,4),time.perf_counter()-tick,{'objective':'pairwise logistic','C':c,'training_informative_questions':int(informative.sum()),'positive_negative_pairs':len(diffs)})
        # Linear listwise losses, with only training-only normalization/statistics.
        x=np.asarray(xg[informative],dtype=np.float64); y=yt[informative].astype(bool); ok=valid[informative]; m=len(x); d=x.shape[-1]
        for loss in ['uniform','marginal']:
            for reg in [.001,.01,.1]:
                tick=time.perf_counter()
                def objective(w):
                    z=x@w; z=np.where(ok,z,-1e12)
                    lse=logsumexp(z,axis=1); p=softmax(z,axis=1)
                    if loss=='uniform':
                        q=y/y.sum(axis=1,keepdims=True); value=(lse-(z*q).sum(axis=1)).mean()
                    else:
                        positive=np.where(y,z,-1e12); q=softmax(positive,axis=1); value=(lse-logsumexp(positive,axis=1)).mean()
                    grad=np.einsum('nkd,nk->d',x,p-q,optimize=True)/m+reg*w
                    return value+reg*np.dot(w,w)/2,grad
                fitted=minimize(objective,np.zeros(d),jac=True,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-9})
                name=f'list_{loss}_{mode}_{reg}'
                add(name,mode,fitted.x,0.,scaler,(xv@fitted.x).reshape(-1,4),time.perf_counter()-tick,{'objective':loss,'l2':reg,'iterations':int(fitted.nit),'converged':bool(fitted.success)})
    # Baseline included so a failed optimization cannot silently replace it.
    from submission import score
    baseline=score(va,vb,Path('runs/model_bundle'))
    baseline_config=json.loads(Path('runs/model_bundle/config.json').read_text())
    import shutil
    for member in baseline_config['members']: shutil.copyfile(Path('runs/model_bundle')/member['file'],out/member['file'])
    models['baseline']=baseline_config['members'][0]; scores['baseline']=baseline
    records.append({'name':'baseline','validation_accuracy':float(np.mean(yv[np.arange(len(va)),choose(baseline,va)])),'seconds':0})
    # Pre-specified bounded answer pooling comparison, no test feedback.
    options=[{'kind':'max'}]+[{'kind':'logsumexp','temperature':t,'count_penalty':a} for t in [.5,1.,2.] for a in [0.,.5,1.]]
    trials=[]; best=None
    for name,raw in scores.items():
        for aggregation in options:
            z=aggregate(raw,va,**aggregation); selected=choose(z,va); acc=float(yv[np.arange(len(va)),selected].mean())
            item={'model':name,'aggregation':aggregation,'validation_accuracy':acc}; trials.append(item)
            if best is None or acc>best['validation_accuracy']: best=item
    config={'name':best['model'],'members':[models[best['model']]],'aggregation':best['aggregation'],'selection_split':'validation','training_split':'train','development':'CPU ranking v2; no test IO'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    summary={'baseline_validation_accuracy':records[-1]['validation_accuracy'],'selected':best,'fits':records,'pooling_trials':trials,'total_seconds':time.perf_counter()-started,'seed':2026,'test_accessed':False}
    (out/'experiments-v2.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez(out/'validation-v2.npz',labels=yv,**scores)
    print('SELECTED',json.dumps(best),flush=True)

if __name__=='__main__': main()
