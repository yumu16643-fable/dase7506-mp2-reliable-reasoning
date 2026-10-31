"""Post-selection diagnostics using train OOF and validation; no test IO."""
from pathlib import Path
import json
import numpy as np
from common import load_inputs
from features import choose
from submission import score as baseline_score

if __name__=='__main__':
    root=Path('runs/gpu-search'); e=json.loads((root/'experiments-gpu.json').read_text()); oof=np.load(root/'oof-scores.npz'); bank,arrays=load_inputs('train'); labels=oof['labels']; selected=e['selected']
    scores=np.mean([oof[selected['architecture']+'_'+str(seed)] for seed in selected['seeds']],axis=0)
    old=labels[np.arange(len(bank)),choose(oof['baseline'],bank)]; new=labels[np.arange(len(bank)),choose(scores,bank)]
    delta=new-old; rng=np.random.default_rng(2026); boot=np.array([delta[rng.integers(len(bank),size=len(bank))].mean() for _ in range(5000)])
    vb,va=load_inputs('validation'); saved=np.load(root/'validation-scores.npz'); vy=saved['labels']; vs=saved['scores']; vc=vy[np.arange(len(vb)),choose(vs,vb)]; bc=vy[np.arange(len(vb)),choose(baseline_score(vb,va,Path('runs/model_bundle')),vb)]
    result={'train_oof_n':len(bank),'original_oof_correct':int(old.sum()),'gpu_oof_correct':int(new.sum()),'oof_fixes':int(((new==1)&(old==0)).sum()),'oof_breaks':int(((new==0)&(old==1)).sum()),'oof_paired_bootstrap95_gain':np.quantile(boot,[.025,.975]).tolist(),'validation_n':len(vb),'original_validation_correct':int(bc.sum()),'gpu_validation_correct':int(vc.sum()),'validation_fixes':int(((vc==1)&(bc==0)).sum()),'validation_breaks':int(((vc==0)&(bc==1)).sum()),'public_test_evaluated_this_round':False,'uncertainty_note':'Descriptive after configuration selection; folds overlap in fitting data; not corrected for model search.'}
    Path('../reports/analysis-gpu.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
