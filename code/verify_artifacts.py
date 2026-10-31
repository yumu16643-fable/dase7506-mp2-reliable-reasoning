"""Verify the frozen inference source/assets and optional whole-package hashes."""
from pathlib import Path
import hashlib,json

if __name__=='__main__':
    code=Path(__file__).resolve().parent; root=code.parent
    frozen=json.loads((code/'runs/freeze.json').read_text())
    for name,digest in frozen['sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest, name
    count=0
    manifest=root/'SHA256SUMS.txt'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest,name=line.split('  ',1)
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
            count+=1
    print(json.dumps({'status':'passed','frozen_files':len(frozen['sha256']),'package_files':count}))
