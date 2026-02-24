# install_ibc.ps1 - Automates downloading and installing IBC (IB Controller) for headless IBKR Gateway

param(
    [string]$InstallDir = "C:\IBC"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Installing IBC (IB Controller)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Check if IBC is already installed
if (Test-Path "$InstallDir\StartGateway.bat") {
    Write-Host "IBC appears to already be installed at $InstallDir." -ForegroundColor Yellow
    exit 0
}

# 2. Find the latest release of IBC from GitHub
Write-Host "`n[1/3] Fetching latest IBC release from GitHub..." -ForegroundColor Green
try {
    $releasesUrl = "https://api.github.com/repos/IbcAlpha/IBC/releases/latest"
    $release = Invoke-RestMethod -Uri $releasesUrl -UseBasicParsing
    $zipAsset = $release.assets | Where-Object { $_.name -like "IBCWin*.zip" } | Select-Object -First 1

    if (-not $zipAsset) {
        Write-Error "Could not find a Windows ZIP release asset."
        exit 1
    }

    $downloadUrl = $zipAsset.browser_download_url
    $zipFile = Join-Path $env:TEMP $zipAsset.name

    Write-Host "Downloading version $($release.tag_name)..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
}
catch {
    Write-Error "Failed to download IBC: $_"
    exit 1
}

# 3. Extract the ZIP file
Write-Host "`n[2/3] Extracting IBC to $InstallDir..." -ForegroundColor Green
try {
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }
    Expand-Archive -Path $zipFile -DestinationPath $InstallDir -Force
}
catch {
    Write-Error "Failed to extract IBC: $_"
    exit 1
}

# 4. Create the config.ini file template
Write-Host "`n[3/3] Setting up configuration..." -ForegroundColor Green
$configPath = "$InstallDir\config.ini"
if (-not (Test-Path $configPath)) {
    # It might extract to a subfolder inside C:\IBC, so we should check
    $innerFolder = Get-ChildItem $InstallDir | Where-Object { $_.PSIsContainer -and (Test-Path "$($_.FullName)\config.ini") }
    if ($innerFolder) {
        # Move contents up one level
        Move-Item "$($innerFolder.FullName)\*" $InstallDir -Force
        Remove-Item $innerFolder.FullName -Force -Recurse
    }
}

if (Test-Path $configPath) {
    Write-Host "Configuration template found at $configPath." -ForegroundColor Green
}
else {
    Write-Host "WARNING: config.ini not found. You will need to create it manually in $InstallDir." -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " IBC Installation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNEXT STEPS:"
Write-Host "1. Open $InstallDir\config.ini in a text editor."
Write-Host "2. Find [IbLoginId] and [IbPassword] and enter your Paper Trading credentials."
Write-Host "3. Find TradingMode and set it to 'paper'."
Write-Host "4. Optionally, configure auto-restart properties."
Write-Host "========================================"
