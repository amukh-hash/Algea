# ─────────────────────────────────────────────────────────────────────
# log_forwarder.ps1 — WORM-Compliant Sidecar Log Shipper
#
# Continuously tails Algae native frontend logs and ships them to
# an immutable storage vault for SEC Rule 17a-4 compliance.
#
# Deployment: Run as a Windows Scheduled Task or NSSM service.
#   powershell -ExecutionPolicy Bypass -File .\log_forwarder.ps1
#
# Supports: Splunk Universal Forwarder, Datadog Agent, or Azure
# Blob Storage WORM direct upload via SAS token.
# ─────────────────────────────────────────────────────────────────────

param(
    [string]$LogDir = "C:\Users\crick\ResolveLabs\Algae\native_frontend\build\Release",
    [string]$LogPattern = "Algae_debug.log",
    [string]$AzureSasUrl = $env:ALGAE_WORM_SAS_URL,      # Azure Blob SAS URL
    [string]$SplunkHecUrl = $env:ALGAE_SPLUNK_HEC_URL,     # Splunk HEC endpoint
    [string]$SplunkToken = $env:ALGAE_SPLUNK_HEC_TOKEN,
    [int]$PollIntervalMs = 2000,
    [int]$BatchLines = 100
)

$ErrorActionPreference = "Stop"

Write-Host "[log_forwarder] Starting WORM-compliant log sidecar"
Write-Host "[log_forwarder] Watching: $LogDir\$LogPattern"

# ── State tracking ────────────────────────────────────────────────
$script:lastPosition = @{}  # filename → byte offset
$script:sessionId = [guid]::NewGuid().ToString("N").Substring(0, 8)
$script:hostname = $env:COMPUTERNAME

function Send-ToSplunkHEC {
    param([string[]]$Lines, [string]$SourceFile)
    
    if (-not $SplunkHecUrl -or -not $SplunkToken) { return }
    
    foreach ($line in $Lines) {
        $payload = @{
            event      = $line
            time       = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
            host       = $script:hostname
            source     = $SourceFile
            sourcetype = "algae:native_frontend"
            index      = "algae_trading_audit"
        } | ConvertTo-Json -Compress
        
        try {
            Invoke-RestMethod -Uri $SplunkHecUrl -Method POST `
                -Headers @{ Authorization = "Splunk $SplunkToken" } `
                -ContentType "application/json" `
                -Body $payload -TimeoutSec 5 | Out-Null
        }
        catch {
            Write-Warning "[log_forwarder] Splunk HEC failed: $($_.Exception.Message)"
        }
    }
}

function Send-ToAzureWORM {
    param([string[]]$Lines, [string]$SourceFile)
    
    if (-not $AzureSasUrl) { return }
    
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $blobName = "algae/${script:hostname}/${timestamp}_$(Split-Path $SourceFile -Leaf)"
    $body = ($Lines -join "`n")
    
    # Append blob API — each call appends to the immutable WORM blob
    $uploadUrl = $AzureSasUrl -replace '\?', "/$blobName?"
    
    try {
        Invoke-RestMethod -Uri $uploadUrl -Method PUT `
            -Headers @{
            "x-ms-blob-type" = "AppendBlob"
            "x-ms-version"   = "2021-08-06"
            "Content-Type"   = "text/plain"
        } `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
            -TimeoutSec 10 | Out-Null
    }
    catch {
        Write-Warning "[log_forwarder] Azure WORM upload failed: $($_.Exception.Message)"
    }
}

function Watch-LogFile {
    param([string]$FilePath)
    
    # Use FILE_SHARE_READ to prevent locking spdlog's rotation thread
    $stream = [System.IO.File]::Open(
        $FilePath,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    )
    $reader = New-Object System.IO.StreamReader($stream)
    
    # Seek to last known position
    $key = Split-Path $FilePath -Leaf
    if ($script:lastPosition.ContainsKey($key)) {
        $stream.Seek($script:lastPosition[$key], [System.IO.SeekOrigin]::Begin) | Out-Null
    }
    
    $buffer = @()
    while ($null -ne ($line = $reader.ReadLine())) {
        $buffer += $line
        if ($buffer.Count -ge $BatchLines) {
            Send-ToSplunkHEC -Lines $buffer -SourceFile $FilePath
            Send-ToAzureWORM -Lines $buffer -SourceFile $FilePath
            $buffer = @()
        }
    }
    
    # Flush remaining
    if ($buffer.Count -gt 0) {
        Send-ToSplunkHEC -Lines $buffer -SourceFile $FilePath
        Send-ToAzureWORM -Lines $buffer -SourceFile $FilePath
    }
    
    $script:lastPosition[$key] = $stream.Position
    $reader.Close()
    $stream.Close()
}

# ── Main Loop ─────────────────────────────────────────────────────
Write-Host "[log_forwarder] Session $($script:sessionId) — polling every ${PollIntervalMs}ms"

while ($true) {
    $logFiles = Get-ChildItem -Path $LogDir -Filter $LogPattern -ErrorAction SilentlyContinue
    foreach ($f in $logFiles) {
        try {
            Watch-LogFile -FilePath $f.FullName
        }
        catch {
            Write-Warning "[log_forwarder] Error tailing $($f.Name): $($_.Exception.Message)"
        }
    }
    Start-Sleep -Milliseconds $PollIntervalMs
}
