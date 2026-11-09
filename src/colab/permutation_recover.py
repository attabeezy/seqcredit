"""Recover the task-owned batch only after verifying each transferred byte."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile

root = Path(__file__).resolve().parents[2]
art = root / 'artifacts'
base = ['python3', str(root / 'src/colab/safe.py'), '--auth', 'oauth2', '--config',
        '/home/ubuntu/.config/colab-cli/seqcredit-session.json', 'download',
        '-s', 'seqcredit-permutation-20261107']
destination = art / 'repeated_permutation_20261107'
destination_exists = destination.exists()
with tempfile.TemporaryDirectory(prefix='seqcredit-recovery-') as tmp:
    staging = Path(tmp)
    manifest_file = staging / 'manifest.json'
    subprocess.run([*base, '/content/permutation_export/manifest.json', str(manifest_file)], check=True)
    manifest = json.loads(manifest_file.read_text())
    names = [part['name'] for part in manifest['parts']]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError('Invalid or repeated part names')

    def get(part: dict) -> None:
        path = staging / part['name']
        subprocess.run([*base, '/content/permutation_export/' + part['name'], str(path)], check=True,
                       stdout=subprocess.DEVNULL)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != part['bytes'] or digest != part['sha256']:
            raise ValueError('Corrupt part: ' + part['name'])
        print('Verified', part['name'], flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(get, manifest['parts']))
    archive = staging / 'results.tar'
    with archive.open('wb') as out:
        for name in names:
            with (staging / name).open('rb') as stream:
                shutil.copyfileobj(stream, out)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if archive.stat().st_size != manifest['bytes'] or digest != manifest['sha256']:
        raise ValueError('Combined archive failed verification')
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if not (staging / member.name).resolve().is_relative_to(staging.resolve()) or member.issym() or member.islnk():
                raise ValueError('Unsafe archive path: ' + member.name)
            if member.name != 'permutation_run.log' and not member.name.startswith('repeated_permutation_20261107/') and member.name != 'repeated_permutation_20261107':
                raise ValueError('Unexpected archive member: ' + member.name)
        bundle.extractall(staging, filter='data')
    staged = staging / 'repeated_permutation_20261107'
    if not destination_exists:
        shutil.move(str(staged), str(destination))
        shutil.copy2(staging / 'permutation_run.log', destination / 'permutation_run.log')
    else:
        count = len(list(destination.glob('execution_batch*.json'))) + 1
        for item in staged.iterdir():
            if item.is_dir():
                if (destination / item.name).exists():
                    raise FileExistsError(f'Refusing to replace {item.name}')
                shutil.move(str(item), str(destination / item.name))
            elif item.name == 'execution.json':
                shutil.copy2(item, destination / f'execution_batch{count}.json')
            elif item.name == 'summary.json':
                shutil.copy2(item, destination / 'summary.json')
        shutil.copy2(staging / 'permutation_run.log', destination / f'permutation_run_batch{count}.log')
    (destination / 'transport_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Recovered verified batch at', destination)
