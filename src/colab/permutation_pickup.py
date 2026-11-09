"""Download and verify one completed condition's artifacts immediately.

Remote helpers stay untouched; this local script pulls each file through the
same safe Colab CLI wrapper and verifies size + SHA-256 against the per-run
manifest, retrying transient failures. Completed conditions survive a VM
death; only the in-flight download is lost.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/content/seqcredit/artifacts/repeated_permutation_20261107'
DEST = ROOT / 'artifacts/repeated_permutation_20261107'
BASE = ['python3', str(ROOT / 'src/colab/safe.py'), '--auth', 'oauth2', '--config',
        '/home/ubuntu/.config/colab-cli/seqcredit-session.json', 'download',
        '-s', 'seqcredit-permutation-20261107']


def sha(p: Path) -> str:
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def missing(manifest: dict, dest: Path) -> list[str]:
    """Return manifest files absent or hash-mismatched under dest."""
    return [name for name, digest in manifest['files'].items()
            if not (dest / name).is_file() or sha(dest / name) != digest]


def fetch(name: str, dest: Path, manifest: dict, attempts: int = 3) -> None:
    path = dest / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run([*BASE, f'{REMOTE}/{name}', str(path)], check=True, timeout=900)
            if sha(path) == manifest['files'][name]:
                print(f'Picked {name} ({path.stat().st_size} bytes)', flush=True)
                return
            path.unlink(missing_ok=True)
            print(f'Hash mismatch {name}, attempt {attempt}', flush=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            print(f'Download failed {name}, attempt {attempt}: {error.__class__.__name__}', flush=True)
            path.unlink(missing_ok=True)
    raise IOError(f'Could not fetch verified {name} after {attempts} attempts')


def pickup(condition: str) -> Path:
    name = condition
    dest = DEST / name
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as handle:
        manifest_file = Path(handle.name)
    try:
        subprocess.run([*BASE, f'{REMOTE}/{name}/manifest.json', str(manifest_file)],
                       check=True, timeout=300)
        manifest = json.loads(manifest_file.read_text())
        if manifest['status'] != 'passed':
            raise ValueError(f'{name} has not passed validation')
        todo = missing(manifest, dest)
        if todo:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda item: fetch(item, dest, manifest), todo))
        leftover = missing(manifest, dest)
        if leftover:
            raise IOError(f'Unverified after pickup: {leftover}')
        (dest / '.picked').write_text(json.dumps(
            {'condition': name, 'manifest_sha256': sha(dest / 'manifest.json')}, indent=2) + '\n')
        return dest
    finally:
        manifest_file.unlink(missing_ok=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: permutation_pickup.py <condition>')
    print('Picked condition at', pickup(sys.argv[1]))
