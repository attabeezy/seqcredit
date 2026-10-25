import json,platform,shutil,subprocess,sys
from pathlib import Path
print('PYTHON',sys.version)
print('PLATFORM',platform.platform())
print('TOOLS',json.dumps({name:shutil.which(name) for name in ['python3','uv','git','nvidia-smi']}))
print(subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout)
Path('/content/seqcredit/data').mkdir(parents=True,exist_ok=True)
