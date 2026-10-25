from pathlib import Path
import json
root=Path('/content/seqcredit');art=root/'artifacts'
print('DRIVER_RETURN',driver_job.poll() if 'driver_job' in globals() else 'unknown')
job=art/'colab_job.json'
if job.exists():print(job.read_text())
for name in ['colab_driver.log','full_comparison.log','evaluation.log']:
 p=art/name
 if p.exists():print('\nLOG',name,'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-8:]))
plan=json.loads((art/'comparison_plan.json').read_text());head=plan.get('headline_run')
if head:
 for control in ['label-permutation','order-shuffle']:
  p=art/(head+'-'+control+'.log')
  if p.exists():print('\nCONTROL',control,'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-6:]))
results=art/'comparison_results.json'
if results.exists():
 report=json.loads(results.read_text())
 print('COMPARISON_SUMMARY',json.dumps({'status':report['status'],'test_metrics':report['test_metrics'],'differences':report['uncertainty']['differences'],'epochs':report['epochs_completed']},indent=2))
