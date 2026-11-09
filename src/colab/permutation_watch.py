"""Monitor, recover a completed batch, and stop only its Colab session."""
from pathlib import Path
import json
import os
import subprocess
import time
import traceback

root = Path(__file__).resolve().parents[2]
os.environ['PATH'] = '/home/ubuntu/.local/bin:' + os.environ.get('PATH', '')
out = root / 'artifacts/permutation_transport'
base = ['python3', str(root / 'src/colab/safe.py'), '--auth', 'oauth2', '--config',
        '/home/ubuntu/.config/colab-cli/seqcredit-session.json']
session = 'seqcredit-permutation-20261107'
watch_file = out / 'watch_status.json'

def update(value: dict) -> None:
    watch_file.write_text(json.dumps(value, indent=2) + '\n')


def download_diagnostics() -> None:
    """Best-effort download of remote failure evidence while the VM is alive."""
    for remote in ('/content/seqcredit/artifacts/permutation_run.log',
                   '/content/seqcredit/artifacts/repeated_permutation_20261107/execution.json'):
        try:
            subprocess.run([*base, 'download', '-s', session, remote, str(out / Path(remote).name)],
                           check=True, timeout=600)
        except Exception:
            pass


last_state = None
try:
    failures = 0
    while True:
        try:
            process = subprocess.run([*base, 'exec', '-s', session, '-f',
                                      str(root / 'src/colab/permutation_watch_probe.py')],
                                     check=False, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            process = None
        if process is None or process.returncode:
            failures += 1
            if failures >= 5:
                raise RuntimeError('Remote progress query failed repeatedly: ' +
                                   (process.stdout[-2000:] if process else 'timed out'))
            update({'state': 'retrying_status', 'attempt': failures})
            time.sleep(20)
            continue
        failures = 0
        lines = [line.split('PERMUTATION_WATCH=', 1)[1] for line in process.stdout.splitlines()
                 if 'PERMUTATION_WATCH=' in line]
        if len(lines) != 1:
            raise RuntimeError('Could not read the remote completion signal')
        state = json.loads(lines[0])
        last_state = state
        update(state)
        if state['exit_code'] is not None:
            if state['exit_code'] != 0 or state['state'] != 'passed' or state['completed'] != 22:
                raise RuntimeError('Remote batch failed or did not complete; session retained for diagnosis')
            break
        time.sleep(45)
    subprocess.run([*base, 'exec', '-s', session, '-f', str(root / 'src/colab/permutation_export.py'),
                    '--timeout', '900'], check=True, timeout=960)
    subprocess.run(['python3', str(root / 'src/colab/permutation_recover.py')], check=True)
    subprocess.run([*base, 'stop', '-s', session], check=True)
    update({'state': 'recovered', 'completed': 22, 'session': 'stopped'})
except Exception:
    download_diagnostics()
    update({'state': 'watch_failed', 'traceback': traceback.format_exc(),
            'last_probe': last_state,
            'note': 'Remote diagnostics downloaded to artifacts/permutation_transport/ where available; '
                    'session retained for diagnosis.'})
    raise
