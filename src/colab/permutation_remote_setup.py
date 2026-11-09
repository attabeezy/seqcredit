"""Verify uploaded inputs and start the isolated repeated-control batch."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

root = Path('/content/seqcredit')
transport = Path('/content/permutation_transport')
root.mkdir(exist_ok=True)
with tarfile.open(transport / 'source.tar.gz') as bundle:
    bundle.extractall(root, filter='data')
manifest = json.loads((transport / 'source_manifest.json').read_text())
for name, digest in manifest.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise ValueError('Source transfer mismatch: ' + name)
data = json.loads((transport / 'data_manifest.json').read_text())
(root / 'data').mkdir(exist_ok=True)
target = root / 'data/yara_telecel_dump.csv.gz'
whole = hashlib.sha256()
with target.open('wb') as output:
    for part in data['parts']:
        payload = (transport / part['name']).read_bytes()
        if len(payload) != part['bytes'] or hashlib.sha256(payload).hexdigest() != part['sha256']:
            raise ValueError('Data chunk mismatch')
        output.write(payload)
        whole.update(payload)
if target.stat().st_size != data['bytes'] or whole.hexdigest() != data['sha256']:
    raise ValueError('Accepted data transfer mismatch')
subprocess.run(['git', 'init', '-b', 'colab-snapshot', str(root)], check=True)
subprocess.run(['git', 'add', '.'], cwd=root, check=True)
subprocess.run(['git', '-c', 'user.name=seqcredit execution snapshot', '-c',
                'user.email=execution@localhost', 'commit', '-m', 'Isolated execution snapshot'], cwd=root, check=True)
logdir = root / 'artifacts'
logdir.mkdir(exist_ok=True)
only_clause = ''
batch_file = transport / 'batch.json'
if batch_file.exists():
    batch = json.loads(batch_file.read_text())
    names = batch.get('conditions')
    if not isinstance(names, list) or not names:
        raise ValueError('Batch file must list conditions')
    only_clause = " --only '" + ','.join(names) + "'"
script = f'''set -eu
cd /content/seqcredit
trap 'code=$?; printf "%s\\n" "$code" > /content/seqcredit/artifacts/permutation_exit_code' EXIT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
uv venv --python 3.13.16 .venv
uv pip install --python .venv/bin/python numpy==2.4.6 scipy==1.18.0 scikit-learn==1.8.0 lightgbm==4.6.0 'tensorflow[and-cuda]==2.21.0' keras==3.15.0 joblib==1.5.3
.venv/bin/python -c "import sys;sys.path.insert(0,'src/scripts');from input_freeze import verify_accepted_dataset;from dataset_io import DATA_FILE;verify_accepted_dataset(DATA_FILE)"
.venv/bin/python src/scripts/repeated_permutation.py{only_clause}
'''
(logdir / 'permutation_launch.sh').write_text(script)
with (logdir / 'permutation_run.log').open('w') as log:
    proc = subprocess.Popen(['bash', str(logdir / 'permutation_launch.sh')], cwd=root,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(logdir / 'permutation_pid.json').write_text(json.dumps({'pid': proc.pid}))
print('Verified inputs; batch PID:', proc.pid)
