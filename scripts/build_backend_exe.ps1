$ErrorActionPreference = "Stop"
python -m pip install pyinstaller
python -m PyInstaller backend/packaging/pyinstaller.spec --distpath backend_dist --workpath backend_build
Write-Host "Built backend_dist/orchestrator.exe"
