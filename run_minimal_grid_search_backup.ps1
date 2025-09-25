# PowerShell script to run MINIMAL grid search
# Estimated cost: ~$0.04 with OpenAI API, Free in mock mode

param(
    [string]$ApiKey = "",
    [switch]$UseMock
)

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "  MINIMAL GRID SEARCH (~$0.04)  " -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Cyan

# Check if in correct directory
if (-not (Test-Path ".\scripts\run_minimal_real_grid_search.py")) {
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

# Set API key - Priority: 1) Parameter, 2) .env file, 3) Environment variable
if ($UseMock) {
    Write-Host "`n✓ Running in MOCK MODE (no API calls)" -ForegroundColor Yellow
    $env:OPENAI_API_KEY = ""
} elseif ($ApiKey) {
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
        Write-Host "`n⚠ No API key found - will run in MOCK MODE" -ForegroundColor Yellow
        Write-Host "  To use real API:" -ForegroundColor Gray
        Write-Host "    - Add OPENAI_API_KEY to .env file" -ForegroundColor Gray
        Write-Host "    - Or use -ApiKey parameter" -ForegroundColor Gray
        Write-Host "    - Or run with -UseMock to explicitly use mock mode" -ForegroundColor Gray
    }
}

# Show configuration
Write-Host "`nConfiguration:" -ForegroundColor White
Write-Host "  • Documents: 10" -ForegroundColor Gray
Write-Host "  • Queries: 3" -ForegroundColor Gray
Write-Host "  • Configurations: 27 (3x3x3)" -ForegroundColor Gray
Write-Host "  • Parameters tested:" -ForegroundColor Gray
Write-Host "    - Chunk sizes: 256, 512, 1024" -ForegroundColor DarkGray
Write-Host "    - Top-K: 3, 5, 10" -ForegroundColor DarkGray
Write-Host "    - Temperature: 0.0, 0.3, 0.7" -ForegroundColor DarkGray

if ($env:OPENAI_API_KEY) {
    Write-Host "`nEstimated cost: ~$0.04" -ForegroundColor Yellow
    Write-Host "Estimated time: 2-3 minutes" -ForegroundColor Cyan
} else {
    Write-Host "`nCost: FREE (mock mode)" -ForegroundColor Green
    Write-Host "Estimated time: 1-2 minutes" -ForegroundColor Cyan
}

Write-Host "`nPress Enter to start or Ctrl+C to cancel..." -ForegroundColor DarkGray
Read-Host

# Run the grid search
Write-Host "`nStarting minimal grid search..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor DarkGray

# Auto-respond 'y' to confirmation prompt
"y" | python scripts\run_minimal_real_grid_search.py

# Check results
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n================================" -ForegroundColor Green
    Write-Host "✓ Grid search completed!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green

    # Show results if file exists
    if (Test-Path "minimal_real_results.json") {
        $results = Get-Content "minimal_real_results.json" | ConvertFrom-Json

        Write-Host "`nBest Configuration Found:" -ForegroundColor Yellow
        Write-Host "  Config ID: #$($results.best.config_id)" -ForegroundColor White
        Write-Host "  Score: $([math]::Round($results.best.score, 3))" -ForegroundColor White

        Write-Host "`nOptimal Parameters:" -ForegroundColor Yellow
        $results.best.params.PSObject.Properties | ForEach-Object {
            $name = $_.Name.Split('.')[-1]
            Write-Host "  • $($name): $($_.Value)" -ForegroundColor Gray
        }

        Write-Host "`nResults saved to: minimal_real_results.json" -ForegroundColor Cyan
    }
} else {
    Write-Host "`n✗ Grid search failed!" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
}

Write-Host "`nPress any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")