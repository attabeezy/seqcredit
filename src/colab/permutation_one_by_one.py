"""Run the 22-condition permutation plan one condition per fresh Colab VM.

Each iteration creates a task-owned T4 session, uploads the frozen bundle
plus a one-condition batch.json, starts the isolated batch, and lets the
watcher pick up the condition's artifacts immediately after it passes
(permutation_pickup), then exports, merges, and stops the session. A VM
preemption retries that single condition on a new VM (up to MAX_ATTEMPTS),
so loss is capped at one in-flight condition.

Already-picked conditions (local .picked marker) are skipped, making the
script safe to re-run after an interruption.

Usage (from WSL, repo root):
    python3 src/colab/permutation_one_by_one.py           # all remaining conditions
    python3 src/colab/permutation_one_by_one.py --only headline_original,headline_one_epoch
    python3 src/colab/permutation_one_by_one.py --help
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SAFE = str(ROOT / 'src/colab/safe.py')
CONFIG = '/home/ubuntu/.config/colab-cli/seqcredit-session.json'
SESSION = 'seqcredit-permutation-20261107'
TRANSPORT = ROOT / 'artifacts/permutation_transport'
BATCH = TRANSPORT / 'batch.json'
DEST = ROOT / 'artifacts/repeated_permutation_20261107'
STATUS = TRANSPORT / 'orchestrate_status.json'
BASE = ['python3', SAFE, '--auth', 'oauth2', '--config', CONFIG]
MAX_ATTEMPTS = 3

PLAN = [
    'headline_original', 'headline_one_epoch',
    'permutation_1051_original', 'permutation_1051_one_epoch',
    'permutation_1052_original', 'permutation_1052_one_epoch',
    'permutation_1053_original', 'permutation_1053_one_epoch',
    'permutation_1054_original', 'permutation_1054_one_epoch',
    'permutation_1055_original', 'permutation_1055_one_epoch',
    'permutation_1056_original', 'permutation_1056_one_epoch',
    'permutation_1057_original', 'permutation_1057_one_epoch',
    'permutation_1058_original', 'permutation_1058_one_epoch',
    'permutation_1059_original', 'permutation_1059_one_epoch',
    'permutation_1060_original', 'permutation_1060_one_epoch',
]


def sh(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    print('RUN', ' '.join(args), flush=True)
    return subprocess.run(args, **kwargs)


def sessions() -> str:
    result = sh([*BASE, 'sessions'], capture_output=True, text=True, timeout=120)
    return (result.stdout or '') + (result.stderr or '')


def prepare() -> None:
    sh(['python3', str(ROOT / 'src/colab/permutation_prepare.py')], check=True)
    from hashlib import sha256
    manifest = json.loads((TRANSPORT / 'source_manifest.json').read_text())
    for name, digest in manifest.items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Bundle stale for {name}; re-run prepare')


def create_session() -> None:
    if SESSION in sessions():
        print(f'Session {SESSION} already active', flush=True)
        return
    sh([*BASE, 'new', '-s', SESSION, '--gpu', 'T4'], check=True, timeout=600)


def stop_session() -> None:
    try:
        sh([*BASE, 'stop', '-s', SESSION], check=True, timeout=120)
    except subprocess.CalledProcessError:
        print('Stop best-effort: session may already be gone', flush=True)


def write_status(data: dict) -> None:
    STATUS.write_text(json.dumps(data, indent=2) + '\n')


def run_condition(name: str) -> str:
    """Run one condition on its own VM; return 'skipped' or 'recovered'."""
    marker = DEST / name / '.picked'
    if marker.exists():
        print(f'SKIP {name} (already picked)', flush=True)
        return 'skipped'
    BATCH.write_text(json.dumps({'conditions': [name]}, indent=2) + '\n')
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            create_session()
            sh([*BASE, 'exec', '-s', SESSION, '-f',
                str(ROOT / 'src/colab/permutation_mkdir.py')],
               check=True, timeout=300)
            sh(['python3', str(ROOT / 'src/colab/permutation_upload.py')], check=True, timeout=1200)
            sh([*BASE, 'exec', '-s', SESSION, '-f',
                str(ROOT / 'src/colab/permutation_remote_setup.py'), '--timeout', '300'],
               check=True, timeout=600)
            sh(['python3', str(ROOT / 'src/colab/permutation_watch.py')], check=True, timeout=None)
            if marker.exists():
                stop_session()  # watch already stops on success; harmless
                return 'recovered'
            raise RuntimeError(f'Watch succeeded but {name} has no .picked marker')
        except subprocess.CalledProcessError as error:
            print(f'ATTEMPT {attempt}/{MAX_ATTEMPTS} FAILED for {name}: {error}', flush=True)
            stop_session()
            if marker.exists():
                return 'recovered'
        time.sleep(10)
    raise RuntimeError(f'Gave up on {name} after {MAX_ATTEMPTS} attempts')


def main(argv: list[str]) -> None:
    names = list(PLAN)
    if '--help' in argv:
        print(__doc__)
        return
    if '--only' in argv:
        value = argv[argv.index('--only') + 1]
        names = [item.strip() for item in value.split(',') if item.strip()]
        unknown = [item for item in names if item not in PLAN]
        if unknown:
            raise SystemExit(f'Unknown conditions: {unknown}')
    write_status({'status': 'preparing', 'plan': names})
    prepare()
    results: dict[str, str] = {}
    for index, name in enumerate(names, start=1):
        print(f'\n===== [{index}/{len(names)}] {name} =====', flush=True)
        write_status({'status': 'running', 'current': name, 'results': results})
        results[name] = run_condition(name)
        write_status({'status': 'running', 'current': None, 'results': results})
    write_status({'status': 'done', 'results': results})
    print('\nDONE', results, flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
