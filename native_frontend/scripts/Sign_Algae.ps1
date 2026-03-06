# ─────────────────────────────────────────────────────────────────────
# Sign_Algae.ps1 — EV Authenticode Code Signing Pipeline
#
# Signs the compiled Algae_Live.exe with an Extended Validation (EV)
# certificate via Azure Key Vault HSM. Falls back to a physical
# YubiKey FIPS token in "Break-Glass" emergency mode.
#
# Usage:
#   .\Sign_Algae.ps1 -Target Live
#   .\Sign_Algae.ps1 -Target Live -BreakGlass   # Emergency USB token
# ─────────────────────────────────────────────────────────────────────
param(
    [ValidateSet("Live", "Sim")]
    [string]$Target = "Live",

    [string]$BuildDir = "$PSScriptRoot\..\build\Release",

    [switch]$BreakGlass,    # Emergency: use physical USB token instead of cloud HSM

    [string]$TimestampServer = "http://timestamp.digicert.com",

    [string]$AzureKeyVaultKey = "algae-ev-signing-key"
)

$ErrorActionPreference = "Stop"

# ── Locate signtool.exe ──────────────────────────────────────────────
$SignToolPaths = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe",
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
)
$SignTool = $SignToolPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $SignTool) {
    Write-Error "signtool.exe not found. Install Windows SDK."
    exit 1
}

# ── Resolve target executable ────────────────────────────────────────
$exeName = if ($Target -eq "Sim") { "Algae_Sim.exe" } else { "Algae_Live.exe" }
$exePath = Join-Path $BuildDir $exeName

if (-not (Test-Path $exePath)) {
    Write-Error "Executable not found: $exePath"
    exit 1
}

Write-Host "[Sign] Target: $exePath"
Write-Host "[Sign] Timestamp: $TimestampServer"

# ── Sign ─────────────────────────────────────────────────────────────
if ($BreakGlass) {
    # Emergency Break-Glass: physical YubiKey FIPS USB token
    Write-Host "[Sign] BREAK-GLASS MODE: Using physical USB token"
    Write-Host "[Sign] Insert your EV signing YubiKey and press Enter..."
    Read-Host

    & $SignTool sign /v /fd SHA256 `
        /tr $TimestampServer /td SHA256 `
        /a `
        $exePath
}
else {
    # Primary: Azure Key Vault Cloud HSM
    Write-Host "[Sign] Using Azure Key Vault HSM: $AzureKeyVaultKey"

    & $SignTool sign /v /fd SHA256 `
        /tr $TimestampServer /td SHA256 `
        /csp "Azure Key Vault Key Storage Provider" `
        /kc $AzureKeyVaultKey `
        $exePath
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "[Sign] Signing FAILED (exit code $LASTEXITCODE)"
    exit 1
}

# ── Verify ───────────────────────────────────────────────────────────
Write-Host "[Sign] Verifying signature chain..."
& $SignTool verify /pa /v $exePath

if ($LASTEXITCODE -ne 0) {
    Write-Error "[Sign] Verification FAILED"
    exit 1
}

# ── Output SHA-256 for EDR whitelisting ──────────────────────────────
$hash = (Get-FileHash -Path $exePath -Algorithm SHA256).Hash
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════"
Write-Host "  SIGNED SUCCESSFULLY"
Write-Host "  SHA-256: $hash"
Write-Host "  Add this hash to EDR global exclusion policies."
Write-Host "═══════════════════════════════════════════════════════"
