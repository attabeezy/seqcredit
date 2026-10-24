param(
    [ValidateSet('no_signal', 'aggregate_signal', 'order_signal')]
    [string]$Scenario = 'no_signal',
    [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'test-smoke.ps1') -Profile pilot -Scenario $Scenario -Seed $Seed
