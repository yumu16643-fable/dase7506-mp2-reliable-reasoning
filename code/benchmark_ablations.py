"""Measure validation-only ablations in fresh official processes after freezing."""
import json,shutil,subprocess,sys
from pathlib import Path

if __name__=='__main__':
    choices=[('lr_agreement_0.1','agreement'),('lr_style_0.1','style'),('lr_text_0.1','text'),('lr_embedding_0.1','embedding'),('lr_combined_1.0','combined'),('lr_context_0.01','context'),('mlp_combined_2026','combined'),('mlp_context_2027','context'),('ensemble_combined','combined')]
    result=[]
    for name,mode in choices:
        folder=Path('runs/ablations')/name; bundle=folder/'bundle'; bundle.mkdir(parents=True,exist_ok=True)
        members=[name] if not name.startswith('ensemble') else ['mlp_combined_2026','mlp_combined_2027']
        specs=[]
        for m in members:
            shutil.copyfile(Path('runs/search')/(m+'.npz'),bundle/(m+'.npz'))
            specs.append({'file':m+'.npz','mode':mode,'layers':2 if m.startswith('mlp') else 1})
        (bundle/'config.json').write_text(json.dumps({'members':specs},indent=2)+'\n')
        prefix=str(folder/'validation')
        r=subprocess.run([sys.executable,'evaluate.py','--split','validation','--checkpoint',str(bundle),'--prefix',prefix],capture_output=True,text=True,check=True)
        (folder/'evaluation.log').write_text(r.stdout)
        score=json.loads(Path(prefix+'-score.json').read_text()); resources=json.loads(Path(prefix+'-resources.json').read_text())
        row={'name':name,'accuracy':score['accuracy'],'total_wall_seconds':resources['total_wall_seconds'],'asset_bytes':sum(p.stat().st_size for p in bundle.iterdir())}
        result.append(row); print(row,flush=True)
    Path('../reports/ablation_resources.json').write_text(json.dumps(result,indent=2)+'\n')
