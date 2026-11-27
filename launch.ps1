param([ValidateSet('annotate','detect','train','demo')][string]$Action = 'detect')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:YOLO_CONFIG_DIR = Join-Path $env:USERPROFILE '.cache/concrete-yolov10'
$taskPrefix = Join-Path $env:USERPROFILE 'miniconda3/envs/concrete-yolov10'
$taskPython = Join-Path $taskPrefix 'python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Environment not found. Run setup.ps1 first.'
}
$env:PATH = "$taskPrefix;$taskPrefix/Library/bin;$taskPrefix/Scripts;" + $env:PATH
switch ($Action) {
    'annotate' { & $taskPython annotate.py }
    'detect' { & $taskPython detect.py }
    'train' {
        $taskVersion = Get-Date -Format 'yyyyMMdd-HHmmss'
        $taskDataset = 'data/prepared/crack-seg-user-20261021'
        if (-not (Test-Path -LiteralPath "$taskDataset/data.yaml")) {
            & $taskPython prepare_external.py
            if ($LASTEXITCODE -ne 0) { throw 'Selected dataset preparation failed.' }
        }
        & $taskPython train.py --data $taskDataset --name "concrete-crack-$taskVersion"
    }
    'demo' {
        Write-Host 'CRACK-ONLY PIPELINE DEMO: accuracy is not validated. Yellow watermark stays visible.'
        & $taskPython detect.py --weights runs/pipeline-smoke-crack/weights/best.pt --imgsz 320 --allow-unvalidated
    }
}
if ($LASTEXITCODE -ne 0) { throw "Program exited with code $LASTEXITCODE" }
