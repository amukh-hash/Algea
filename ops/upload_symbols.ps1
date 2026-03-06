# ─────────────────────────────────────────────────────────────────────
# upload_symbols.ps1 — CI/CD Symbol Server Indexing
#
# Indexes the compiled .pdb and .exe into the centralized symbol server
# using Microsoft's symstore.exe. This MUST execute before distributing
# the signed executable to the trading floor.
#
# WinDbg uses the GUID embedded by the MSVC linker to locate the exact
# matching PDB. Without symstore indexing, post-mortem crash triage
# resolves to opaque hexadecimal addresses.
#
# Usage:
#   .\upload_symbols.ps1 -BuildOutputDir "C:\build\Release" -AppVersion "1.2.0"
#   .\upload_symbols.ps1 -BuildOutputDir ".\build\Release" -AppVersion "1.2.0" -Target Sim
# ─────────────────────────────────────────────────────────────────────
param(
    [Parameter(Mandatory = $true)]
    [string]$BuildOutputDir,

    [Parameter(Mandatory = $true)]
    [string]$AppVersion,

    [ValidateSet("Live", "Sim")]
    [string]$Target = "Live",

    [string]$SymbolShare = "\\algae.internal\symbols"
)

$ErrorActionPreference = "Stop"

# ── Locate symstore.exe ──────────────────────────────────────────────
$SymStorePaths = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\Debuggers\x64\symstore.exe",
    "${env:ProgramFiles}\Windows Kits\10\Debuggers\x64\symstore.exe"
)
$SymStore = $SymStorePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $SymStore) {
    Write-Error "symstore.exe not found. Install Windows SDK Debugging Tools."
    exit 1
}

$exeName = if ($Target -eq "Sim") { "Algae_Sim" } else { "Algae_Live" }
$pdbPath = Join-Path $BuildOutputDir "$exeName.pdb"
$exePath = Join-Path $BuildOutputDir "$exeName.exe"

# ── Validate artifacts exist ─────────────────────────────────────────
if (-not (Test-Path $pdbPath)) {
    Write-Error "PDB not found: $pdbPath (was /Zi enabled in CMake Release?)"
    exit 1
}
if (-not (Test-Path $exePath)) {
    Write-Error "EXE not found: $exePath"
    exit 1
}

$pdbSize = [math]::Round((Get-Item $pdbPath).Length / 1MB, 1)
Write-Host "[Symbols] PDB: $pdbPath ($pdbSize MB)"
Write-Host "[Symbols] EXE: $exePath"
Write-Host "[Symbols] Target: $SymbolShare"
Write-Host "[Symbols] Version: $AppVersion"
Write-Host ""

# ── Index PDB ────────────────────────────────────────────────────────
Write-Host "[1/3] Indexing $exeName.pdb to symbol server..."
& $SymStore add /f $pdbPath /s $SymbolShare /t $exeName /v $AppVersion /c "CI/CD auto-index"

if ($LASTEXITCODE -ne 0) {
    Write-Error "FATAL: PDB indexing failed. Halting deployment pipeline."
    exit 1
}

# ── Index EXE (required for WinDbg absolute offset resolution) ───────
Write-Host "[2/3] Indexing $exeName.exe to symbol server..."
& $SymStore add /f $exePath /s $SymbolShare /t $exeName /v $AppVersion /c "CI/CD auto-index"

if ($LASTEXITCODE -ne 0) {
    Write-Error "FATAL: EXE indexing failed. Halting deployment pipeline."
    exit 1
}

# ── Verify ───────────────────────────────────────────────────────────
Write-Host "[3/3] Verifying symbol server contents..."
$indexed = Get-ChildItem -Path $SymbolShare -Filter "$exeName.pdb" -Recurse -ErrorAction SilentlyContinue
if ($indexed) {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════"
    Write-Host "  SYMBOL INDEXING COMPLETE"
    Write-Host "  Product: $exeName v$AppVersion"
    Write-Host "  Server:  $SymbolShare"
    Write-Host ""
    Write-Host "  SRE WinDbg config:"
    Write-Host "  _NT_SYMBOL_PATH=srv*C:\Symbols*$SymbolShare"
    Write-Host "═══════════════════════════════════════════════════════"
}
else {
    Write-Warning "Could not verify indexed symbols at $SymbolShare"
}
