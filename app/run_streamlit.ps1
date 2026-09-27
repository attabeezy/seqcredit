$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:LOKY_MAX_CPU_COUNT = "1"
Set-Location -LiteralPath $projectRoot
streamlit run src\seqcredit_mvp\streamlit_app.py
