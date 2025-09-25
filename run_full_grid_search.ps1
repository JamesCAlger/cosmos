# PowerShell script to run FULL grid search with real data
# WARNING: Estimated cost: $2-5 with OpenAI API

param(
    [string]$ApiKey = "",
    [switch]$Force
)

Write-Host "`n================================" -ForegroundColor Red
Write-Host "  FULL GRID SEARCH (~$2-5)     " -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Red

# Check if in correct directory
if (-not (Test-Path ".\scripts\run_real_grid_search.py")) {
    Write-Host "ERROR: Not in auto-RAG directory!" -ForegroundColor Red
    Write-Host "Please cd to: C:\Users\alger\Documents\000. Projects\auto-RAG" -ForegroundColor Yellow
    exit 1
}

# Function to read .env file
function Read-EnvFile {
    param([string]$Path = ".\.env")

    if (Test-Path $Path) {
        Get-Content $Path | ForEach-Object {
            if ($_ -match '^([^#][^=]+)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim()
                if ($key -eq "OPENAI_API_KEY" -and $value) {
                    return $value
                }
            }
        }
    }
    return $null
}

# Check for API key (required for full search)
# Priority: 1) Parameter, 2) .env file, 3) Environment variable
if ($ApiKey) {
    $env:OPENAI_API_KEY = $ApiKey
    Write-Host "`n✓ API key set from parameter" -ForegroundColor Green
} else {
    # Try to read from .env file
    $envKey = Read-EnvFile
    if ($envKey) {
        $env:OPENAI_API_KEY = $envKey
        Write-Host "`n✓ API key loaded from .env file" -ForegroundColor Green
        $masked = $env:OPENAI_API_KEY.Substring(0, 10) + "..."
        Write-Host "  Key: $masked" -ForegroundColor Gray
    } elseif ($env:OPENAI_API_KEY) {
        Write-Host "`n✓ Using existing OPENAI_API_KEY from environment" -ForegroundColor Green
        $masked = $env:OPENAI_API_KEY.Substring(0, 10) + "..."
        Write-Host "  Key: $masked" -ForegroundColor Gray
    } else {
        Write-Host "`n✗ ERROR: Full grid search requires OpenAI API key!" -ForegroundColor Red
        Write-Host "`nTo set API key:" -ForegroundColor Yellow
        Write-Host "  Option 1: Add OPENAI_API_KEY to .env file" -ForegroundColor Gray
        Write-Host "  Option 2: .\run_full_grid_search.ps1 -ApiKey 'sk-...'" -ForegroundColor Gray
        Write-Host "  Option 3: `$env:OPENAI_API_KEY = 'sk-...'" -ForegroundColor Gray
        Write-Host "`nFor testing without API, use run_minimal_grid_search.ps1 instead" -ForegroundColor Cyan
        exit 1
    }
}

# Cost warning
Write-Host "`n⚠  COST WARNING ⚠" -ForegroundColor Red
Write-Host "================================" -ForegroundColor Yellow
Write-Host "This will use REAL MONEY!" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White
Write-Host "Configuration:" -ForegroundColor White
Write-Host "  • Documents: 100 (MS MARCO)" -ForegroundColor Gray
Write-Host "  • Queries: 20" -ForegroundColor Gray
Write-Host "  • Configurations: 27 (3x3x3)" -ForegroundColor Gray
Write-Host "  • Total API calls: ~540+" -ForegroundColor Gray
Write-Host "" -ForegroundColor White
Write-Host "Parameters tested:" -ForegroundColor White
Write-Host "  • Chunk sizes: 256, 512, 1024" -ForegroundColor DarkGray
Write-Host "  • Top-K: 3, 5, 10" -ForegroundColor DarkGray
Write-Host "  • Temperature: 0.0, 0.3, 0.7" -ForegroundColor DarkGray
Write-Host "" -ForegroundColor White
Write-Host "Estimated cost: $2-5" -ForegroundColor Red
Write-Host "Estimated time: 15-30 minutes" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

# Confirm unless -Force flag used
if (-not $Force) {
    Write-Host "`nThis will cost real money. Are you sure?" -ForegroundColor Red
    Write-Host "Type 'YES' (in capitals) to proceed: " -NoNewline -ForegroundColor Yellow
    $confirm = Read-Host

    if ($confirm -ne "YES") {
        Write-Host "`nCancelled. No charges incurred." -ForegroundColor Green
        Write-Host "Tip: Use run_minimal_grid_search.ps1 for low-cost testing (~$0.04)" -ForegroundColor Cyan
        exit 0
    }

    # Double confirmation for safety
    Write-Host "`nFinal confirmation - this will charge your OpenAI account $2-5" -ForegroundColor Red
    Write-Host "Press Enter to proceed or Ctrl+C to cancel..." -ForegroundColor Yellow
    Read-Host
}

# Create checkpoint directory
$checkpointDir = ".\grid_search_checkpoints"
if (-not (Test-Path $checkpointDir)) {
    New-Item -ItemType Directory -Path $checkpointDir | Out-Null
    Write-Host "`n✓ Created checkpoint directory" -ForegroundColor Green
}

# Start timestamp
$startTime = Get-Date
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "Starting Full Grid Search" -ForegroundColor White
Write-Host "Start time: $($startTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host "================================" -ForegroundColor Cyan

# Run the grid search
try {
    python scripts\run_real_grid_search.py

    if ($LASTEXITCODE -eq 0) {
        $endTime = Get-Date
        $duration = $endTime - $startTime

        Write-Host "`n================================" -ForegroundColor Green
        Write-Host "✓ GRID SEARCH COMPLETED!" -ForegroundColor Green
        Write-Host "================================" -ForegroundColor Green
        Write-Host "Duration: $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan

        # Show results
        if (Test-Path "real_grid_search_results.json") {
            $results = Get-Content "real_grid_search_results.json" | ConvertFrom-Json

            Write-Host "`nBest Configuration Found:" -ForegroundColor Yellow
            Write-Host "  Config ID: #$($results.best_config.config_id)" -ForegroundColor White
            Write-Host "  Score: $([math]::Round($results.best_config.score, 3))" -ForegroundColor White

            Write-Host "`nOptimal Parameters:" -ForegroundColor Yellow
            $results.best_config.params.PSObject.Properties | ForEach-Object {
                $name = $_.Name.Split('.')[-1]
                Write-Host "  • $($name): $($_.Value)" -ForegroundColor Gray
            }

            Write-Host "`nMetrics:" -ForegroundColor Yellow
            $results.best_config.metrics.PSObject.Properties | ForEach-Object {
                if ($_.Name -eq "total_cost") {
                    Write-Host "  • $($_.Name): `$$($_.Value)" -ForegroundColor Gray
                } elseif ($_.Name -eq "avg_latency") {
                    Write-Host "  • $($_.Name): $($_.Value)s" -ForegroundColor Gray
                } else {
                    Write-Host "  • $($_.Name): $([math]::Round($_.Value, 3))" -ForegroundColor Gray
                }
            }

            # Save backup with timestamp
            $backupName = "real_grid_search_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
            Copy-Item "real_grid_search_results.json" $backupName
            Write-Host "`nResults saved:" -ForegroundColor Cyan
            Write-Host "  • Main: real_grid_search_results.json" -ForegroundColor Gray
            Write-Host "  • Backup: $backupName" -ForegroundColor Gray

            # Check if checkpoints exist
            $checkpoints = Get-ChildItem "grid_search_checkpoint_*.json" -ErrorAction SilentlyContinue
            if ($checkpoints) {
                Write-Host "`nCheckpoints created: $($checkpoints.Count)" -ForegroundColor Cyan
                Move-Item $checkpoints $checkpointDir -Force
                Write-Host "  Moved to: $checkpointDir" -ForegroundColor Gray
            }
        }
    } else {
        Write-Host "`n✗ Grid search failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host "Check error messages above for details" -ForegroundColor Yellow

        # Check for partial results
        $checkpoints = Get-ChildItem "grid_search_checkpoint_*.json" -ErrorAction SilentlyContinue
        if ($checkpoints) {
            Write-Host "`nPartial results saved in checkpoints: $($checkpoints.Count) configs completed" -ForegroundColor Yellow
            Write-Host "You can analyze these partial results" -ForegroundColor Cyan
        }
    }
} catch {
    Write-Host "`n✗ Error running grid search: $_" -ForegroundColor Red
} finally {
    # Always show cost estimate
    Write-Host "`n================================" -ForegroundColor Yellow
    Write-Host "Cost Summary" -ForegroundColor White
    Write-Host "================================" -ForegroundColor Yellow

    $checkpoints = Get-ChildItem "grid_search_checkpoint_*.json" -ErrorAction SilentlyContinue
    if ($checkpoints) {
        $configsRun = $checkpoints.Count
        $estimatedCost = $configsRun * 0.07  # Rough estimate per config
        Write-Host "Configurations completed: $configsRun / 27" -ForegroundColor White
        Write-Host "Estimated cost incurred: `$$([math]::Round($estimatedCost, 2))" -ForegroundColor Yellow
    } else {
        Write-Host "No configurations completed - no costs incurred" -ForegroundColor Green
    }
}

Write-Host "`nPress any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")