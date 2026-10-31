"""Train only on train; select using validation. Never loads test inputs or labels."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='4'
import argparse,json,platform,time,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from common import DATA,load_inputs,read_jsonl,parse_answer
from features import make_features,choose,majority

def labels(bank,refs):
    assert len(bank)==len(refs)
    assert all(b['id']==r['id'] and b['question']==r['question'] for b,r in zip(bank,refs))
    return np.array([[int(parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer'])) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=np.int64)

def accuracy(scores,bank,y):
    return float(np.mean(y[np.arange(len(y)),choose(scores,bank)]))

def export_model(model,scaler,mode,path,kind):
    weights={'mean':scaler.mean_.astype('float32'),'scale':scaler.scale_.astype('float32')}
    if kind=='logistic': weights.update(w0=model.coef_.T.astype('float32'),b0=model.intercept_.astype('float32'))
    else:
        for i,(w,b) in enumerate(zip(model.coefs_,model.intercepts_)):
            weights[f'w{i}']=w.astype('float32'); weights[f'b{i}']=b.astype('float32')
    np.savez(path,**weights)
    return {'mode':mode,'file':path.name,'kind':kind,'layers':1 if kind=='logistic' else len(model.coefs_)}

def run(args):
    start=time.perf_counter(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    tr,ta=load_inputs('train',args.data_dir); va,vb=load_inputs('validation',args.data_dir)
    yt=labels(tr,read_jsonl(args.data_dir/'gsm8k_train.jsonl'))
    yv=labels(va,read_jsonl(args.data_dir/'gsm8k_validation.jsonl'))
    records=[]; candidates=[]; preds={}; specs={}
    def record(name,mode,kind,sec,sv,st=None,config=None):
        item={'name':name,'mode':mode,'kind':kind,'validation_accuracy':accuracy(sv,va,yv),'training_seconds':sec,'config':config or {}}
        if st is not None: item['train_accuracy']=accuracy(st,tr,yt)
        records.append(item); preds[name]=sv; print(json.dumps(item),flush=True)
    record('first_parseable','none','baseline',0,np.zeros((len(va),4)))
    record('majority','none','baseline',0,majority(va))
    for mode in ['agreement','style','text','embedding','combined','context']:
        xt=make_features(tr,ta,mode); xv=make_features(va,vb,mode)
        scaler=StandardScaler().fit(xt); xt=scaler.transform(xt); xv=scaler.transform(xv)
        for c in ([0.01,0.1,1.] if mode in ['embedding','combined','context'] else [0.1,1.]):
            tick=time.perf_counter(); name=f'lr_{mode}_{c}'
            model=LogisticRegression(C=c,max_iter=400,solver='lbfgs',random_state=2026).fit(xt,yt.ravel())
            sv=model.decision_function(xv).reshape(-1,4); st=model.decision_function(xt).reshape(-1,4)
            specs[name]=export_model(model,scaler,mode,out/(name+'.npz'),'logistic')
            record(name,mode,'logistic',time.perf_counter()-tick,sv,st,{'C':c}); candidates.append(name)
        if mode in ['text','combined','context']:
            for seed in [2026,2027]:
                tick=time.perf_counter(); name=f'mlp_{mode}_{seed}'
                model=MLPClassifier(hidden_layer_sizes=(64,),alpha=1.,learning_rate_init=0.001,batch_size=256,max_iter=100,early_stopping=True,validation_fraction=.12,n_iter_no_change=10,random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning); model.fit(xt,yt.ravel())
                pv=np.clip(model.predict_proba(xv)[:,1],1e-6,1-1e-6); pt=np.clip(model.predict_proba(xt)[:,1],1e-6,1-1e-6)
                sv=np.log(pv/(1-pv)).reshape(-1,4); st=np.log(pt/(1-pt)).reshape(-1,4)
                specs[name]=export_model(model,scaler,mode,out/(name+'.npz'),'mlp')
                record(name,mode,'mlp',time.perf_counter()-tick,sv,st,{'seed':seed,'alpha':1.,'iterations':model.n_iter_}); candidates.append(name)
        if mode=='text':
            for leaves in [7,15]:
                tick=time.perf_counter(); name=f'hgb_text_{leaves}'
                model=HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=150,l2_regularization=10,random_state=2026).fit(xt,yt.ravel())
                record(name,mode,'hgb_analysis_only',time.perf_counter()-tick,model.decision_function(xv).reshape(-1,4),config={'leaves':leaves})
    for mode in ['text','combined','context']:
        names=[f'mlp_{mode}_{seed}' for seed in [2026,2027]]
        name=f'ensemble_{mode}'; sv=np.mean([preds[x] for x in names],axis=0)
        record(name,mode,'ensemble',0,sv,config={'members':names}); candidates.append(name)
    best=max(candidates,key=lambda k:accuracy(preds[k],va,yv))
    members=next((r['config']['members'] for r in records if r['name']==best and r['kind']=='ensemble'),[best])
    config={'name':best,'members':[specs[k] for k in members],'seed':2026,'selection_split':'validation','training_split':'train','mask_invalid':True}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    np.savez(out/'validation_analysis.npz',labels=yv,**preds)
    summary={'selected':best,'experiments':records,'training_total_seconds':time.perf_counter()-start,'platform':platform.platform(),'python':platform.python_version(),'train_questions':len(tr),'validation_questions':len(va),'train_correct_candidate_histogram':np.bincount(yt.sum(axis=1),minlength=5).tolist(),'validation_correct_candidate_histogram':np.bincount(yv.sum(axis=1),minlength=5).tolist()}
    (out/'experiments.json').write_text(json.dumps(summary,indent=2)+'\n'); print('SELECTED',best,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=DATA); p.add_argument('--output',type=Path,default=Path('runs/search')); run(p.parse_args())
