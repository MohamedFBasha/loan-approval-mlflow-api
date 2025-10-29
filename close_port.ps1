param (
    [Parameter(Mandatory = $true)]
    [int]$Port
)

# ابحث عن العملية اللي شغالة على البورت
$connection = netstat -ano | findstr ":$Port"

if (-not $connection) {
    Write-Host "⚠️ No process found using port $Port."
    exit
}

# استخرج الـ PID
$pids = @()
foreach ($line in $connection) {
    $parts = $line -split '\s+'
    $pid = $parts[-1]
    if ($pid -match '^[0-9]+$' -and ($pids -notcontains $pid)) {
        $pids += $pid
    }
}

# اقفل كل الـ PIDs اللي لقيتها
foreach ($pid in $pids) {
    Write-Host "🛑 Killing process with PID $pid on port $Port..."
    try {
        taskkill /PID $pid /F | Out-Null
        Write-Host "✅ Port $Port is now free."
    } catch {
        Write-Host "❌ Failed to kill PID $pid (maybe already closed)."
    }
}
