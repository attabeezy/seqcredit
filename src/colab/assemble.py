from pathlib import Path
import hashlib,json,os,subprocess
root=Path('/content/seqcredit');manifest=json.loads((root/'artifacts/colab_upload_manifest.json').read_text());parts_root=root/'artifacts/upload-parts';target=root/'data/yara_telecel_dump.csv.gz';temporary=target.with_suffix('.transfer');whole=hashlib.sha256();size=0
with temporary.open('wb') as out:
    for item in manifest['parts']:
        part=(parts_root/item['name']).resolve()
        assert part.is_relative_to(parts_root.resolve())
        payload=part.read_bytes();assert len(payload)==item['bytes'];assert hashlib.sha256(payload).hexdigest()==item['sha256'];out.write(payload);whole.update(payload);size+=len(payload)
assert size==manifest['bytes'];assert whole.hexdigest()==manifest['dataset_sha256'];os.replace(temporary,target)
print('ACCEPTED_DATASET_TRANSFER_VERIFIED',json.dumps({'bytes':size,'sha256':whole.hexdigest()}))
for item in manifest['parts']:(parts_root/item['name']).unlink()
parts_root.rmdir()
result=subprocess.run([str(root/'.venv/bin/python'),'-c',"import sys;sys.path.insert(0,'src/scripts');from input_freeze import verify_accepted_dataset;from dataset_io import DATA_FILE;print(verify_accepted_dataset(DATA_FILE)['status'])"],cwd=root,capture_output=True,text=True);print('FREEZE_CHECK',result.returncode,result.stdout,result.stderr);assert result.returncode==0
print('GPU_PREFLIGHT_RETURN',preflight_job.poll() if 'preflight_job' in globals() else 'unknown')
p=root/'artifacts/colab_gpu_preflight.json'
if p.exists():print('GPU_PREFLIGHT',p.read_text())
else:print('\n'.join((root/'artifacts/colab_gpu_preflight.log').read_text(errors='replace').splitlines()[-20:]))
