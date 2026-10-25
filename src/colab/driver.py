"""Run one matched Colab comparison and preserve all outputs for download."""
import hashlib,json,os,platform,subprocess,sys,tarfile,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts';sys.path.insert(0,str(ROOT/'src/scripts'))
from input_freeze import file_sha256,verify_accepted_dataset
from dataset_io import DATA_FILE

def status(stage,**fields):
    value={'status':stage,'updated_at':datetime.now(timezone.utc).isoformat(),**fields}
    (ART/'colab_job.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)

def child(args,log_name,stage):
    with (ART/log_name).open('w') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        status(stage,pid=process.pid,log=log_name)
        return process

def archive_outputs():
    archive=Path('/content/seqcredit-artifacts.tar')
    def clean(member):
        if '__pycache__' in Path(member.name).parts or member.name.endswith('.pyc'):return None
        return member
    with tarfile.open(archive,'w') as out:out.add(ART,arcname='artifacts',filter=clean)
    export=Path('/content/seqcredit-export');export.mkdir(exist_ok=True);parts=[]
    with archive.open('rb') as stream:
        i=0
        while payload:=stream.read(4*1024*1024):
            name=f'part-{i:03d}.bin';(export/name).write_bytes(payload);parts.append({'name':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()});i+=1
    manifest={'archive':'seqcredit-artifacts.tar','bytes':archive.stat().st_size,'sha256':file_sha256(archive),'parts':parts}
    (export/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

env=os.environ.copy();env.update({'TF_FORCE_GPU_ALLOW_GROWTH':'true','CUBLAS_WORKSPACE_CONFIG':':4096:8','TF_CPP_MIN_LOG_LEVEL':'2'})
try:
    accepted=verify_accepted_dataset(DATA_FILE)
    probe=json.loads((ART/'colab_gpu_preflight.json').read_text());assert probe['status']=='passed'
    environment={'backend':'Google Colab T4 via WSL google-colab-cli','python':sys.version,'platform':platform.platform(),'versions':{name:version(name) for name in ['numpy','scipy','scikit-learn','lightgbm','tensorflow','keras','joblib']},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'deterministic_gpu_preflight':probe,'environment_flags':{key:env[key] for key in ['TF_FORCE_GPU_ALLOW_GROWTH','CUBLAS_WORKSPACE_CONFIG','TF_CPP_MIN_LOG_LEVEL']},'git_snapshot_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'git_snapshot_scope':'Ephemeral source-only Git snapshot of the uploaded accepted source; not the original local repository history.','dataset_sha256':accepted['dataset']['sha256'],'accepted_snapshot_sha256':file_sha256(ROOT/'src/configs/accepted_dataset.json'),'cpu_results_policy':'Preserved separately; not pooled or selected against the Colab results.'}
    (ART/'colab_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    main_process=child([sys.executable,'train.py'],'full_comparison.log','headline_running')
    headline=None
    while main_process.poll() is None and headline is None:
        lines=(ART/'full_comparison.log').read_text(errors='replace').splitlines()
        found=[line[len('Results: '):] for line in lines if line.startswith('Results: ')]
        if found:headline=Path(found[0]).name;break
        time.sleep(.5)
    if headline is None:raise RuntimeError('Headline process did not report a run path')
    plan=json.loads((ART/'comparison_plan.json').read_text());plan['headline_run']=headline;plan['backend_environment']='colab_environment.json';(ART/'comparison_plan.json').write_text(json.dumps(plan,indent=2)+'\n');status('headline_running',headline=headline,pid=main_process.pid,log='full_comparison.log')
    code=main_process.wait()
    if code:raise RuntimeError(f'Headline training exited {code}')
    evaluation=child([sys.executable,'artifacts/evaluate_comparison.py'],'evaluation.log','controls_and_uncertainty_running')
    code=evaluation.wait()
    if code:raise RuntimeError(f'Controls/evaluation exited {code}')
    verify_accepted_dataset(DATA_FILE)
    status('comparison_completed',headline=headline,report='comparison_report.md',results='comparison_results.json')
    export=archive_outputs();status('ready_for_download',headline=headline,export_manifest='/content/seqcredit-export/manifest.json',archive_sha256=export['sha256'],parts=len(export['parts']))
except Exception as error:
    status('failed',error=str(error),traceback=traceback.format_exc())
    raise
