"""Download only official data from the recorded upstream commit; verify every hash."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,hashlib,json,time,urllib.request

COMMIT='bb801f4f2e7629d270df765ab12c333b2dc616bb'
BASE='https://raw.githubusercontent.com/xudongwu-0/DASE7506-MP2/'+COMMIT+'/code/data/'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data'); a=p.parse_args(); a.data_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=a.data_dir/'manifest.json'
    if not manifest_path.exists(): urllib.request.urlretrieve(BASE+'manifest.json',manifest_path)
    manifest=json.loads(manifest_path.read_text())
    assert manifest['protocol']=='dase7506-mp2-open-hard-verifier-v5'
    def fetch(item):
        name,digest=item; target=a.data_dir/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest: return
        target.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(3):
            temp=target.with_name(target.name+'.part')
            try:
                with urllib.request.urlopen(BASE+name,timeout=180) as r,temp.open('wb') as f:
                    while True:
                        chunk=r.read(1024*1024)
                        if not chunk: break
                        f.write(chunk)
                if hashlib.sha256(temp.read_bytes()).hexdigest()!=digest: raise ValueError('Hash mismatch: '+name)
                temp.replace(target); print('Verified '+name,flush=True); return
            except Exception:
                if attempt==2: raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch,manifest['sha256'].items()))
    print('All official data verified. Prediction/scoring can now run offline.')

if __name__=='__main__': main()
