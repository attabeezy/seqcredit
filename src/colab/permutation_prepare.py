"""Prepare a minimal, checksummed bundle for the repeated-control run."""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parents[2]
out = root / 'artifacts/permutation_transport'
out.mkdir(parents=True, exist_ok=True)
files = ['.gitattributes', '.gitignore', 'pyproject.toml', 'uv.lock', 'docs/EXPERIMENT.md',
         'docs/calibration/ledger.json', 'docs/calibration/category_mapping.json']
files += [p.relative_to(root).as_posix() for p in (root / 'src/configs').glob('*.json')]
files += [p.relative_to(root).as_posix() for p in (root / 'src/scripts').glob('*.py')]
hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
with tarfile.open(out / 'source.tar.gz', 'w:gz') as archive:
    for name in files:
        archive.add(root / name, arcname=name)
(out / 'source_manifest.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
source = root / 'data/yara_telecel_dump.csv.gz'
accepted = json.loads((root / 'src/configs/accepted_dataset.json').read_text())
digest = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
if digest != accepted['dataset']['sha256']:
    raise ValueError('Accepted data hash mismatch')
parts = []
with source.open('rb') as stream:
    while data := stream.read(4 * 1024 * 1024):
        name = f'part-{len(parts):03}.bin'
        (out / name).write_bytes(data)
        parts.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
(out / 'data_manifest.json').write_text(json.dumps({'sha256': digest, 'bytes': source.stat().st_size,
                                                    'parts': parts}, indent=2), encoding='utf-8')
print(f'Prepared {len(files)} source files and {len(parts)} accepted-data chunks')
