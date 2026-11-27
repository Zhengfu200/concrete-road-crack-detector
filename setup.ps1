$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:YOLO_CONFIG_DIR = Join-Path $env:USERPROFILE '.cache/concrete-yolov10'
function Check-Exit {
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE). See output above." }
}
$taskEnvs = conda env list --json | ConvertFrom-Json
$taskExists = $taskEnvs.envs | Where-Object { (Split-Path $_ -Leaf) -eq 'concrete-yolov10' }
if (-not $taskExists) {
    conda create -n concrete-yolov10 python=3.12 pip tk -y --override-channels -c conda-forge
    Check-Exit
}
conda run -n concrete-yolov10 --no-capture-output python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
Check-Exit
conda run -n concrete-yolov10 --no-capture-output python -m pip install -r requirements.txt -c constraints-gpu.txt
Check-Exit
conda run -n concrete-yolov10 --no-capture-output python fetch_weights.py
Check-Exit
conda run -n concrete-yolov10 --no-capture-output python -c "from train import gpu_check; print(gpu_check())"
Check-Exit
conda run -n concrete-yolov10 --no-capture-output python -m pip freeze | Set-Content -LiteralPath requirements.lock.txt -Encoding utf8
Write-Host 'Environment ready: conda activate concrete-yolov10'
