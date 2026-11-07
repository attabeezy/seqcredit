"""Return a machine-readable completion signal without exposing session details."""
from pathlib import Path
import json
root = Path('/content/seqcredit/artifacts')
exit_file = root / 'permutation_exit_code'
state = root / 'repeated_permutation_20261107/execution.json'
print('PERMUTATION_WATCH=' + json.dumps({
    'exit_code': int(exit_file.read_text()) if exit_file.exists() else None,
    'state': json.loads(state.read_text()).get('status') if state.exists() else 'starting',
    'completed': len(json.loads(state.read_text()).get('conditions', {})) if state.exists() else 0,
}))
