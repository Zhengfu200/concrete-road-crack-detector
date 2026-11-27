$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$reviewPython = Join-Path $env:USERPROFILE 'miniconda3/envs/concrete-yolov10/python.exe'
& $reviewPython connected_review.py
if ($LASTEXITCODE -ne 0) { throw "Review exited with code $LASTEXITCODE" }
