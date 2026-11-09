"""Package any completed batch subset into hash-verified download chunks."""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path('/content/seqcredit')
batch = root / 'artifacts/repeated_permutation_20261107'
execution = json.loads((batch / 'execution.json').read_text())
if execution['status'] != 'passed' or not (1 <= len(execution['conditions']) <= 22):
    raise ValueError('Cannot export an incomplete successful batch')
out = Path('/content/permutation_export')
out.mkdir(exist_ok=True)
archive = out / 'results.tar'
with tarfile.open(archive, 'w') as bundle:
    bundle.add(batch, arcname='repeated_permutation_20261107')
    bundle.add(root / 'artifacts/permutation_run.log', arcname='permutation_run.log')
parts = []
with archive.open('rb') as stream:
    while payload := stream.read(4 * 1024 * 1024):
        name = f'part-{len(parts):04}.bin'
        (out / name).write_bytes(payload)
        parts.append({'name': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'manifest.json').write_text(json.dumps({'sha256': digest, 'bytes': archive.stat().st_size,
                                              'parts': parts}, indent=2))
print('Export ready:', len(parts), 'parts', archive.stat().st_size, 'bytes', digest)
