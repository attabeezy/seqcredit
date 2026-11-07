"""Upload only the prepared synthetic input bundle to the task-owned session."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
transport = root / 'artifacts/permutation_transport'
base = ['python3', str(root / 'src/colab/safe.py'), '--auth', 'oauth2', '--config',
        '/home/ubuntu/.config/colab-cli/seqcredit-session.json']
files = sorted(transport.glob('part-*.bin')) + [transport / name for name in
        ('source.tar.gz', 'source_manifest.json', 'data_manifest.json')]

def upload(path: Path) -> None:
    subprocess.run([*base, 'upload', '-s', 'seqcredit-permutation-20261107', str(path),
                    '/content/permutation_transport/' + path.name], check=True)
    print('Uploaded', path.name, flush=True)

with ThreadPoolExecutor(max_workers=3) as executor:
    list(executor.map(upload, files))
