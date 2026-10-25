from pathlib import Path
import subprocess
root=Path('/content/seqcredit')
preflight_log=(root/'artifacts/colab_gpu_preflight.log').open('w')
preflight_job=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'src/colab/gpu_preflight.py')],cwd=root,stdout=preflight_log,stderr=subprocess.STDOUT,start_new_session=True)
print('GPU_PREFLIGHT_PID',preflight_job.pid)
