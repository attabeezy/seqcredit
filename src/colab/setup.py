import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path('/content/seqcredit')
with tarfile.open('/content/seqcredit-source.tar.gz','r:gz') as archive:
    archive.extractall(root,filter='data')
manifest=json.loads((root/'artifacts/colab_source_manifest.json').read_text())
for relative,expected in manifest['files'].items():
    actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    if actual!=expected:raise ValueError('Source transport mismatch: '+relative)
print('Source bundle verified',len(manifest['files']),'files')
subprocess.run(['git','init','-b','colab-snapshot',str(root)],check=True)
subprocess.run(['git','add','dataset.py','train.py','pyproject.toml','uv.lock','.gitignore','README.md','src','docs'],cwd=root,check=True)
subprocess.run(['git','-c','user.name=seqcredit execution snapshot','-c','user.email=execution@localhost','commit','-m','Accepted source snapshot for isolated Colab execution'],cwd=root,check=True)
setup='''set -eu
cd /content/seqcredit
uv sync --frozen --no-dev
uv pip install --python .venv/bin/python 'tensorflow[and-cuda]==2.21.0'
.venv/bin/python -c "import json,sys,tensorflow as tf; from importlib.metadata import version; devices=tf.config.list_physical_devices('GPU'); print('GPU_DEVICES',devices); assert devices, 'No TensorFlow GPU detected'; print('VERSIONS',json.dumps({x:version(x) for x in ['numpy','scipy','scikit-learn','lightgbm','tensorflow','keras','joblib']}))"
'''
(root/'src/colab').mkdir(parents=True,exist_ok=True)
(root/'src/colab/setup.sh').write_text(setup)
log=(root/'artifacts/colab_setup.log').open('w')
job=subprocess.Popen(['bash',str(root/'src/colab/setup.sh')],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'artifacts/colab_setup_pid.json').write_text(json.dumps({'pid':job.pid}))
print('SETUP_PID',job.pid)
