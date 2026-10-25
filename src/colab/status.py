from pathlib import Path
import json,hashlib
root=Path('/content/seqcredit')
print('SETUP_RETURN',job.poll() if 'job' in globals() else 'not in kernel globals')
log=root/'artifacts/colab_setup.log'
if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-18:]))
p=root/'data/yara_telecel_dump.csv.gz'
print('DATA_UPLOAD',json.dumps({'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None}))
if p.exists() and p.stat().st_size==92503821:print('DATA_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
(root/'artifacts/upload-parts').mkdir(parents=True,exist_ok=True)
