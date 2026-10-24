param(
    [ValidateSet('smoke', 'pilot', 'main')]
    [string]$Profile = 'smoke',
    [ValidateSet('no_signal', 'aggregate_signal', 'order_signal')]
    [string]$Scenario = 'no_signal',
    [int]$Seed = 42,
    [ValidateSet('primary', 'secondary')]
    [string]$Target = 'primary',
    [ValidateSet('none', 'label_permutation', 'order_shuffle')]
    [string]$Control = 'none'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Push-Location -LiteralPath $projectRoot
try {
    & uv sync --locked --group dev
    if ($LASTEXITCODE -ne 0) { throw 'Environment synchronization failed' }
    $pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Expected .venv Python executable is missing' }
    foreach ($script in @('test_extract_calibration.py', 'test_population_spec.py', 'test_contract_checks.py', 'test_experiment_v2.py', 'test_validate_pilots.py')) {
        & $pythonExe (Join-Path $PSScriptRoot $script)
        if ($LASTEXITCODE -ne 0) { throw "Checks failed: $script" }
    }
    & $pythonExe (Join-Path $PSScriptRoot 'run_smoke.py') --profile $Profile --scenario $Scenario --seed $Seed --target $Target --control $Control
    if ($LASTEXITCODE -ne 0) { throw "End-to-end $Profile experiment failed; inspect its failure.json" }
} finally {
    Pop-Location
}
