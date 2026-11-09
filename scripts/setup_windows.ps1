param([switch]$Cuda)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8='1'
New-Item -ItemType Directory -Force -Path '.cache/tmp','.cache/pip' | Out-Null
$env:TEMP=(Resolve-Path '.cache/tmp').Path
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=(Resolve-Path '.cache/pip').Path
if (!(Test-Path -LiteralPath '.venv/Scripts/python.exe')) { py -3.11 -m venv .venv }
$python=Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
if ($Cuda) {
    & $python -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
} else {
    & $python -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $python -m pip install -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed' }
& $python -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed' }
& $python -m futsal download-model yolo11n
& $python -m futsal doctor
