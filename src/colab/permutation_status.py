"""Print bounded progress and the batch exit status from the remote runtime."""
from pathlib import Path
import json
import subprocess
root = Path('/content/seqcredit/artifacts')
log = root / 'permutation_run.log'
if log.exists():
    print('\n'.join(log.read_text(errors='replace').splitlines()[-18:]))
status = root / 'repeated_permutation_20261107/execution.json'
if status.exists():
    value = json.loads(status.read_text())
    print('BATCH', value['status'], 'CONDITIONS', len(value['conditions']))
    if value['conditions']:
        print('LAST', list(value['conditions'].items())[-1])
exit_code = root / 'permutation_exit_code'
if exit_code.exists():
    print('EXIT_CODE', exit_code.read_text().strip())
else:
    lines = subprocess.run(['ps', '-eo', 'pid,etime,rss,args'], capture_output=True, text=True).stdout.splitlines()
    print('PROCESSES', [line for line in lines if 'src/scripts/repeated_permutation.py' in line])
