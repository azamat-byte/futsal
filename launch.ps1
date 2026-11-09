$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '.cache/ultralytics'
$env:HF_HOME = Join-Path $PSScriptRoot '.cache/huggingface'
& (Join-Path $PSScriptRoot '.venv/Scripts/python.exe') -m futsal dashboard
