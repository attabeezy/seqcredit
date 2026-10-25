from pathlib import Path
import json,subprocess
root=Path('/content/seqcredit');log=(root/'artifacts/colab_driver.log').open('w')
driver_job=subprocess.Popen([str(root/'.venv/bin/python'),str(root/'src/colab/driver.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('COLAB_COMPARISON_DRIVER_PID',driver_job.pid)
