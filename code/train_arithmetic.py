"""Training-only question CV selects arithmetic models; validation is inspected once."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from common import DATA,load_inputs,read_jsonl,parse_answer
from features import make_features,choose
from arithmetic_features import arithmetic_features,candidate_stats
from train import labels
from cross_validate_v2 import objective

def fit_model(x,y,valid,kind,parameter):
    scaler=StandardScaler().fit(x.reshape(-1,x.shape[-1])); z=scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape)
    if kind=='bce':
        model=LogisticRegression(C=parameter,max_iter=400,random_state=2026).fit(z.reshape(-1,z.shape[-1]),y.ravel())
        return scaler,model.coef_[0],float(model.intercept_[0]),{'iterations':int(model.n_iter_[0])}
    informative=((y==0)&valid).any(axis=1)
    if kind=='pair':
        differences=[]; weights=[]
        for i in np.flatnonzero(informative):
            pos=np.flatnonzero(y[i]); neg=np.flatnonzero((y[i]==0)&valid[i]); weight=1/(len(pos)*len(neg))
            for a in pos:
                for b in neg:differences.append(z[i,a]-z[i,b]);weights.append(weight)
        d=np.asarray(differences); sw=np.asarray(weights);sw*=len(sw)/sw.sum()
        model=LogisticRegression(C=parameter,max_iter=400,fit_intercept=False,random_state=2026).fit(np.concatenate([d,-d]),np.r_[np.ones(len(d)),np.zeros(len(d))],sample_weight=np.r_[sw,sw])
        return scaler,model.coef_[0],0.,{'iterations':int(model.n_iter_[0])}
    xx=z[informative].astype(np.float64); yy=y[informative].astype(bool); vv=valid[informative]
    result=minimize(lambda w:objective(w,xx,yy,vv,parameter),np.zeros(z.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-9})
    return scaler,result.x,0.,{'iterations':int(result.nit),'converged':bool(result.success)}

def scores(x,model):
    scaler,w,b,_=model
    return (scaler.transform(x.reshape(-1,x.shape[-1]))@w+b).reshape(len(x),4)

def acc(z,bank,y):return float(np.mean(y[np.arange(len(bank)),choose(z,bank)]))

if __name__=='__main__':
    started=time.perf_counter(); out=Path('runs/arithmetic-search');out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(Path('arithmetic_plan.json').read_text());bank,arrays=load_inputs('train');refs=read_jsonl(DATA/'gsm8k_train.jsonl');y=labels(bank,refs)
    valid=np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank]); n=len(bank)
    arithmetic=arithmetic_features(bank).reshape(n,4,-1)
    original={mode:make_features(bank,arrays,mode).reshape(n,4,-1) for mode in ['combined','context']}
    extended={mode:np.concatenate([x,arithmetic],axis=2) for mode,x in original.items()}
    configs=[{'name':'original','kind':'bce','mode':'context','arithmetic':False,'parameter':.01}]
    for kind,mode,parameters in [('bce','context',plan['model_families']['bce_context_C']),('pair','combined',plan['model_families']['pair_combined_C']),('marginal','combined',plan['model_families']['marginal_combined_l2'])]:
        for parameter in parameters:configs.append({'name':f'{kind}_{mode}_{parameter}','kind':kind,'mode':mode,'arithmetic':True,'parameter':parameter})
    oof={c['name']:np.zeros((n,4),dtype=np.float32) for c in configs}; records=[]
    for fold,(fit,hold) in enumerate(KFold(3,shuffle=True,random_state=2026).split(bank)):
        holdbank=[bank[i] for i in hold]
        for c in configs:
            tick=time.perf_counter(); x=(extended if c['arithmetic'] else original)[c['mode']]
            model=fit_model(x[fit],y[fit],valid[fit],c['kind'],c['parameter']); z=scores(x[hold],model);oof[c['name']][hold]=z
            record={'fold':fold,**c,'accuracy':acc(z,holdbank,y[hold]),'seconds':time.perf_counter()-tick,**model[-1]};records.append(record);print(json.dumps(record),flush=True)
    candidates=[{**c,'oof_accuracy':acc(oof[c['name']],bank,y)} for c in configs];selected=max(candidates,key=lambda c:c['oof_accuracy'])
    print('SELECTED',json.dumps(selected),flush=True)
    x=(extended if selected['arithmetic'] else original)[selected['mode']];model=fit_model(x,y,valid,selected['kind'],selected['parameter']);sc,w,b,_=model
    bundle=out/'selected_bundle';bundle.mkdir(exist_ok=True)
    np.savez(bundle/'weights.npz',mean=sc.mean_.astype(np.float32),scale=sc.scale_.astype(np.float32),w0=np.asarray(w,dtype=np.float32).reshape(-1,1),b0=np.asarray([b],dtype=np.float32))
    config={'name':selected['name'],'member':{'file':'weights.npz','mode':selected['mode'],'arithmetic':selected['arithmetic']},'selection':'training-only 3-fold question OOF accuracy','plan_sha256':hashlib.sha256(Path('arithmetic_plan.json').read_bytes()).hexdigest()}
    (bundle/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    # Validation is not consulted until all configurations and the winner are fixed.
    vb,va=load_inputs('validation');vy=labels(vb,read_jsonl(DATA/'gsm8k_validation.jsonl'))
    vx=make_features(vb,va,selected['mode']).reshape(len(vb),4,-1)
    if selected['arithmetic']:vx=np.concatenate([vx,arithmetic_features(vb).reshape(len(vb),4,-1)],axis=2)
    predicted=scores(vx,model)
    from submission_arithmetic import score
    exported=score(vb,va,bundle);assert choose(predicted,vb)==choose(exported,vb)
    baseline=oof['original'];correct=y[np.arange(n),choose(oof[selected['name']],bank)]; bc=y[np.arange(n),choose(baseline,bank)]
    rng=np.random.default_rng(2026);delta=correct-bc;bootstrap=np.array([delta[rng.integers(n,size=n)].mean() for _ in range(5000)])
    coverage=(arithmetic[:,:,0]>0); bad=(arithmetic[:,:,4]>0)
    summary={'plan':plan,'candidates':candidates,'selected':selected,'fold_records':records,'oof_correct':int(correct.sum()),'baseline_oof_correct':int(bc.sum()),'oof_n':n,'oof_fixes':int(((correct==1)&(bc==0)).sum()),'oof_breaks':int(((correct==0)&(bc==1)).sum()),'oof_bootstrap95_gain':np.quantile(bootstrap,[.025,.975]).tolist(),'validation_correct':round(acc(exported,vb,vy)*len(vb)),'validation_n':len(vb),'validation_accuracy':acc(exported,vb,vy),'training_candidate_check_coverage':float(coverage.mean()),'bad_rate_correct_training_candidate':float(bad[y.astype(bool)].mean()),'bad_rate_wrong_training_candidate':float(bad[~y.astype(bool)].mean()),'total_seconds':time.perf_counter()-started,'inference_asset_bytes':sum(p.stat().st_size for p in bundle.iterdir()),'test_accessed':False,'numpy_export_choices_match':True}
    (out/'experiments-arithmetic.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez(out/'oof-scores.npz',labels=y,**oof);np.savez(out/'validation-scores.npz',labels=vy,scores=exported)
    audit=[]
    for i,j in np.argwhere(bad)[:20]:
        r=bank[int(i)];text=r['candidates'][int(j)]['text'];audit.append({'train_position':int(i),'candidate':int(j),'correct_final_answer':bool(y[i,j]),'question':r['question'],'equations':candidate_stats(text)[1],'text':text})
    (out/'parser-audit-train.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('RESULT',json.dumps({k:summary[k] for k in ['selected','oof_correct','baseline_oof_correct','validation_correct','validation_n','validation_accuracy','training_candidate_check_coverage','bad_rate_correct_training_candidate','bad_rate_wrong_training_candidate','total_seconds']}),flush=True)
