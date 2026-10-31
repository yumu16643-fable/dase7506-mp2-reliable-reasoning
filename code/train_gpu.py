"""Fixed-plan CUDA development. Only train/validation are ever opened."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[key]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,copy,hashlib,json,platform,random,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
from gpu_model import Verifier,loss_value
from submission_gpu import equality,score as numpy_score
from common import DATA,load_inputs,read_jsonl,parse_answer
from features import make_features,choose
from train import labels

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cpu_accuracy(scores,bank,y):
    return float(np.mean(y[np.arange(len(bank)),choose(scores,bank)]))

def baseline_fit(context,y,fit):
    sc=StandardScaler().fit(context[fit].reshape(-1,context.shape[-1]))
    x=sc.transform(context[fit].reshape(-1,context.shape[-1]))
    model=LogisticRegression(C=.01,max_iter=400,random_state=2026).fit(x,y[fit].ravel())
    return sc,model

def prep(raw,context,y,valid,same,fit):
    scaler=StandardScaler().fit(raw[fit].reshape(-1,raw.shape[-1]))
    x=np.clip(scaler.transform(raw.reshape(-1,raw.shape[-1])),-5,5).reshape(raw.shape)
    bsc,baseline=baseline_fit(context,y,fit)
    b=baseline.decision_function(bsc.transform(context.reshape(-1,context.shape[-1]))).reshape(len(raw),4)
    tensors=tuple(torch.as_tensor(v,dtype=torch.float32,device='cuda') for v in [x,y,valid,same,b])
    return scaler,bsc,baseline,tensors

def train_one(architecture,seed,tensors,fit,stop,plan,epochs=None):
    set_seed(seed); model=Verifier(tensors[0].shape[-1],architecture,plan['width'],plan['second_width'],plan['dropout']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    fit=np.asarray(fit)
    if not architecture.endswith('bce'):
        yy=tensors[1].cpu().numpy(); vv=tensors[2].cpu().numpy().astype(bool)
        fit=fit[((yy[fit]==0)&vv[fit]).any(axis=1)]
    fit=torch.as_tensor(fit,device='cuda',dtype=torch.long)
    best_loss=float('inf'); best_state=None; best_epoch=0; stalled=0
    max_epochs=plan['max_epochs'] if epochs is None else epochs
    for epoch in range(1,max_epochs+1):
        model.train(); order=fit[torch.randperm(len(fit),device='cuda')]
        for indexes in order.split(plan['batch_questions']):
            x,y,valid,same,base=[t[indexes] for t in tensors]
            optimizer.zero_grad(set_to_none=True)
            loss=loss_value(model(x,same,base),y,valid,architecture)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        if stop is not None:
            model.eval()
            with torch.no_grad():
                x,y,valid,same,base=[t[stop] for t in tensors]
                current=float(loss_value(model(x,same,base),y,valid,architecture).item())
            if current<best_loss-1e-5:
                best_loss=current; best_epoch=epoch; stalled=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: stalled+=1
            if stalled>=plan['early_stopping_patience']: break
    if stop is not None:
        model.load_state_dict(best_state)
    else: best_epoch=max_epochs
    return model,best_epoch

def infer(model,tensors,indexes):
    model.eval()
    with torch.no_grad():
        x,_,_,same,base=[t[indexes] for t in tensors]
        return model(x,same,base).cpu().numpy()

def export(model,scaler,path):
    w={'mean':scaler.mean_.astype(np.float32),'scale':scaler.scale_.astype(np.float32)}
    for name in ['first','second','output']:
        layer=getattr(model,name); w[name+'_w']=layer.weight.detach().cpu().numpy().T.copy(); w[name+'_b']=layer.bias.detach().cpu().numpy().copy()
    np.savez(path,**w)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('runs/gpu-search')); args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(Path('gpu_plan.json').read_text()); started=time.perf_counter()
    assert torch.cuda.is_available(),'CUDA required: refusing silent CPU fallback'
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False
    bank,arrays=load_inputs('train'); y=labels(bank,read_jsonl(DATA/'gsm8k_train.jsonl'))
    raw=make_features(bank,arrays,'combined').reshape(len(bank),4,-1); context=make_features(bank,arrays,'context').reshape(len(bank),4,-1)
    valid=np.array([[parse_answer(c['text']) is not None for c in row['candidates']] for row in bank],dtype=np.float32); same=equality(bank)
    gpu={'name':torch.cuda.get_device_name(0),'memory_total_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'platform':platform.platform(),'plan_sha256':hashlib.sha256(Path('gpu_plan.json').read_bytes()).hexdigest(),'test_accessed':False}
    print('GPU',json.dumps(gpu),flush=True)
    oof={f'{a}_{s}':np.zeros((len(bank),4),dtype=np.float32) for a in plan['architectures'] for s in plan['seeds']}; old=np.zeros((len(bank),4),dtype=np.float32); records=[]
    for fold,(fit,hold) in enumerate(KFold(3,shuffle=True,random_state=plan['outer_cv_seed']).split(bank)):
        inner,stop=train_test_split(fit,test_size=plan['inner_holdout_fraction'],random_state=2026+fold)
        isc,_,_,it=prep(raw,context,y,valid,same,inner); sc,bsc,base,full=prep(raw,context,y,valid,same,fit)
        old[hold]=full[-1][hold].cpu().numpy()
        for architecture in plan['architectures']:
            for seed in plan['seeds']:
                tick=time.perf_counter(); name=f'{architecture}_{seed}'
                interim,best=train_one(architecture,seed,it,inner,stop,plan)
                del interim
                model,_=train_one(architecture,seed,full,fit,None,plan,epochs=best)
                pred=infer(model,full,hold); oof[name][hold]=pred
                acc=cpu_accuracy(pred,[bank[i] for i in hold],y[hold]); row={'fold':fold,'name':name,'architecture':architecture,'seed':seed,'selected_epochs':best,'outer_accuracy':acc,'seconds':time.perf_counter()-tick}
                records.append(row); print(json.dumps(row),flush=True); del model
        del it,full; torch.cuda.empty_cache()
    baseline_acc=cpu_accuracy(old,bank,y); candidates=[]
    for architecture in plan['architectures']:
        for seed in plan['seeds']:
            name=f'{architecture}_{seed}'; candidates.append({'name':name,'architecture':architecture,'seeds':[seed],'oof_accuracy':cpu_accuracy(oof[name],bank,y)})
        ensemble=np.mean([oof[f'{architecture}_{s}'] for s in plan['seeds']],axis=0)
        candidates.append({'name':architecture+'_ensemble','architecture':architecture,'seeds':plan['seeds'],'oof_accuracy':cpu_accuracy(ensemble,bank,y)})
    selected=max(candidates,key=lambda r:(r['oof_accuracy'],-len(r['seeds'])))
    print('CV_SELECTED',json.dumps({'baseline_oof':baseline_acc,'selected':selected}),flush=True)
    bundle=out/'selected_bundle'; bundle.mkdir(exist_ok=True)
    sc,bsc,base,tensors=prep(raw,context,y,valid,same,np.arange(len(bank))); members=[]; final_models=[]; final_cost=0
    for seed in selected['seeds']:
        epochs=round(float(np.median([r['selected_epochs'] for r in records if r['architecture']==selected['architecture'] and r['seed']==seed])))
        tick=time.perf_counter(); model,_=train_one(selected['architecture'],seed,tensors,np.arange(len(bank)),None,plan,epochs=epochs); final_cost+=time.perf_counter()-tick
        name=f'{selected["architecture"]}_{seed}.npz'; export(model,sc,bundle/name); members.append({'file':name,'architecture':selected['architecture'],'seed':seed,'epochs':epochs}); final_models.append(model)
    if selected['architecture'].startswith('residual'):
        np.savez(bundle/'baseline.npz',mean=bsc.mean_.astype(np.float32),scale=bsc.scale_.astype(np.float32),w0=base.coef_.T.astype(np.float32),b0=base.intercept_.astype(np.float32))
    config={'name':selected['name'],'members':members,'selection':'training-only question-disjoint three-fold OOF accuracy','feature_clip':[-5,5],'plan':plan['protocol']}
    (bundle/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    # Only now inspect validation labels; they did not choose architecture, epoch or seed.
    vb,va=load_inputs('validation'); vy=labels(vb,read_jsonl(DATA/'gsm8k_validation.jsonl'))
    ns=numpy_score(vb,va,bundle); validation_accuracy=cpu_accuracy(ns,vb,vy)
    xv=np.clip(sc.transform(make_features(vb,va,'combined')),-5,5).reshape(len(vb),4,-1); sv=equality(vb)
    cv=make_features(vb,va,'context'); bz=base.decision_function(bsc.transform(cv)).reshape(len(vb),4)
    with torch.no_grad():
        inputs=[torch.as_tensor(a,dtype=torch.float32,device='cuda') for a in [xv,sv,bz]]
        tp=np.mean([m.eval()(*inputs).cpu().numpy() for m in final_models],axis=0)
    assert choose(ns,vb)==choose(tp,vb),'NumPy export changed selected validation choices'
    gate=selected['oof_accuracy']>=baseline_acc+.005 and validation_accuracy>=736/798
    summary={**gpu,'plan':plan,'baseline_oof_accuracy':baseline_acc,'candidates':candidates,'selected':selected,'validation_correct':round(validation_accuracy*len(vb)),'validation_n':len(vb),'validation_accuracy':validation_accuracy,'promotion_gate_passed':gate,'fold_records':records,'final_refit_seconds':final_cost,'total_seconds':time.perf_counter()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(0),'inference_asset_bytes':sum(f.stat().st_size for f in bundle.iterdir()),'numpy_export_choices_match_cuda':True}
    (out/'experiments-gpu.json').write_text(json.dumps(summary,indent=2)+'\n'); np.savez(out/'oof-scores.npz',baseline=old,labels=y,**oof); np.savez(out/'validation-scores.npz',scores=ns,labels=vy)
    print('RESULT',json.dumps({k:summary[k] for k in ['selected','validation_accuracy','promotion_gate_passed','total_seconds','peak_gpu_memory_bytes','inference_asset_bytes']}),flush=True)

if __name__=='__main__': main()
