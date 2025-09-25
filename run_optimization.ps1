# PowerShell script to run Week 5 optimization with 10,000 documents
# Run from project root: .\run_optimization.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AUTO-RAG OPTIMIZATION TEST" -ForegroundColor Cyan
Write-Host "  10,000 Documents | 10 Configurations" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonCheck = python --version 2>&1
if ($?) {
    Write-Host "Python detected: $pythonCheck" -ForegroundColor Green
} else {
    Write-Host "Python not found. Please ensure Python is installed." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (Test-Path ".env") {
    Write-Host ".env file found" -ForegroundColor Green
} else {
    Write-Host ".env file not found. API keys required." -ForegroundColor Red
    exit 1
}

# Check if OpenAI API key is set
$envContent = Get-Content .env | Select-String "OPENAI_API_KEY"
if ($envContent) {
    Write-Host "OpenAI API key configured" -ForegroundColor Green
} else {
    Write-Host "OpenAI API key not found in .env" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  * Documents: 10,000 (MS MARCO)"
Write-Host "  * Queries per config: 30"
Write-Host "  * Configurations to test: 10"
Write-Host "  * Estimated cost: ~$1.30"
Write-Host "  * Estimated time: 15-30 minutes"
Write-Host ""

# Ask for confirmation
$confirmation = Read-Host "Do you want to proceed? (Y/N)"
if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
    Write-Host "Optimization cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Starting optimization..." -ForegroundColor Green
Write-Host "Press Ctrl+C to cancel (progress will be saved)" -ForegroundColor Gray
Write-Host ""

# Create necessary directories
if (!(Test-Path "optimization_results")) {
    New-Item -ItemType Directory -Path "optimization_results" | Out-Null
    Write-Host "Created optimization_results directory" -ForegroundColor Gray
}

if (!(Test-Path "checkpoints")) {
    New-Item -ItemType Directory -Path "checkpoints" | Out-Null
    Write-Host "Created checkpoints directory" -ForegroundColor Gray
}

if (!(Test-Path "cache")) {
    New-Item -ItemType Directory -Path "cache" | Out-Null
    Write-Host "Created cache directory" -ForegroundColor Gray
}

Write-Host ""

# Run the optimization
try {
    # Set environment variable for better output encoding
    $env:PYTHONIOENCODING = "utf-8"

    # Run the optimization script
    python scripts/run_10k_optimization.py

    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  OPTIMIZATION COMPLETE!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Results saved in:" -ForegroundColor Cyan
        Write-Host "  * optimization_results\" -ForegroundColor White

        # Find the latest result file
        $latestResult = Get-ChildItem "optimization_results\*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestResult) {
            Write-Host "  * Latest: $($latestResult.Name)" -ForegroundColor White
        }
    } else {
        Write-Host ""
        Write-Host "Optimization ended with exit code: $exitCode" -ForegroundColor Yellow
    }

} catch {
    Write-Host ""
    Write-Host "Error occurred: $_" -ForegroundColor Red
    Write-Host "Check the error message above for details." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "To view results:" -ForegroundColor Cyan
Write-Host "  python scripts/view_metrics.py" -ForegroundColor White
Write-Host ""