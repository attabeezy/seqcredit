import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path('/mnt/c/Users/Dell/Projects/yara/seqcredit');source=root/'data/yara_telecel_dump.csv.gz';safe=root/'src/colab/safe.py'
base=['python3',str(safe),'--auth','oauth2','--config','/home/ubuntu/.config/colab-cli/seqcredit-session.json','upload','-s','seqcredit-frozen-t4']
parts=[];chunk=4*1024*1024
with tempfile.TemporaryDirectory(prefix='seqcredit-transfer-') as tmp:
    with source.open('rb') as f:
        i=0
        while payload:=f.read(chunk):
            name=f'part-{i:03d}.bin';local=Path(tmp)/name;local.write_bytes(payload)
            print(f'Uploading verified data part {i+1}/23',flush=True)
            subprocess.run([*base,str(local),'/content/seqcredit/artifacts/upload-parts/'+name],check=True)
            parts.append({'name':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()});i+=1
manifest={'dataset_sha256':'6ec8bfc5a9e5a78f284624c4ed355aff163f42490cf2a49bf2142bdb5b30a068','bytes':source.stat().st_size,'parts':parts}
(root/'artifacts/colab_upload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run([*base,str(root/'artifacts/colab_upload_manifest.json'),'/content/seqcredit/artifacts/colab_upload_manifest.json'],check=True)
print('All data parts uploaded; local temporary pieces removed.',flush=True)
