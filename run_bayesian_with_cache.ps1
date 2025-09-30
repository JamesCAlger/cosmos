# Simple test script for Bayesian optimization with caching
# This runs a minimal search to verify the caching mechanism works

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "TESTING BAYESIAN SEARCH WITH CACHE" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Load environment variables from .env
if (Test-Path ".env") {
    Write-Host "`nLoading environment variables from .env..." -ForegroundColor Yellow
    Get-Content .env | ForEach-Object {
        if ($_ -match "^([^#=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value)
            if ($key -eq "OPENAI_API_KEY") {
                Write-Host "  - OpenAI API key loaded" -ForegroundColor Green
            }
        }
    }
} else {
    Write-Host "Warning: .env file not found" -ForegroundColor Yellow
}

# Configuration for simple test
$n_calls = 5          # Small number of evaluations
$num_docs = 10        # Reduced documents for faster testing
$num_queries = 3      # Reduced queries for faster testing
$cache_memory = 256   # MB for cache

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  - Evaluations: $n_calls" -ForegroundColor White
Write-Host "  - Documents: $num_docs" -ForegroundColor White
Write-Host "  - Queries: $num_queries" -ForegroundColor White
Write-Host "  - Cache Memory: ${cache_memory}MB" -ForegroundColor White
Write-Host "  - Cache Directory: .embedding_cache" -ForegroundColor White

# Check if we should use real API
$useRealApi = ""
if ($env:OPENAI_API_KEY) {
    Write-Host "  - Mode: Real API (OpenAI)" -ForegroundColor Green
    $useRealApi = "--real-api"
} else {
    Write-Host "  - Mode: Mock (no API calls)" -ForegroundColor Yellow
}

# Clear cache for clean test (optional - comment out to test with existing cache)
$clearCache = Read-Host "`nClear existing cache? (y/n)"
$clearCacheArg = ""
if ($clearCache -eq "y") {
    $clearCacheArg = "--clear-cache"
    Write-Host "Will clear cache before starting" -ForegroundColor Yellow
} else {
    Write-Host "Will use existing cache if available" -ForegroundColor Green
}

Write-Host "`n=================================" -ForegroundColor Green
Write-Host "STARTING BAYESIAN OPTIMIZATION" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Build command
$pythonScript = "scripts/run_bayesian_full_space_enhanced.py"
$arguments = @(
    "--n-calls", $n_calls,
    "--num-docs", $num_docs,
    "--num-queries", $num_queries,
    "--use-cache",
    "--cache-dir", ".embedding_cache",
    "--cache-memory-limit", $cache_memory
)

# Add optional arguments
if ($useRealApi) {
    $arguments += $useRealApi
}
if ($clearCacheArg) {
    $arguments += $clearCacheArg
}

Write-Host "`nCommand:" -ForegroundColor Cyan
Write-Host "python $pythonScript $($arguments -join ' ')" -ForegroundColor White
Write-Host ""

# Record start time
$startTime = Get-Date

try {
    # Run the optimization
    python $pythonScript @arguments

    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Write-Host "`n=================================" -ForegroundColor Green
        Write-Host "OPTIMIZATION COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Host "=================================" -ForegroundColor Green
    } else {
        Write-Host "`n=================================" -ForegroundColor Red
        Write-Host "OPTIMIZATION FAILED (Exit code: $exitCode)" -ForegroundColor Red
        Write-Host "=================================" -ForegroundColor Red
    }
} catch {
    Write-Host "`n=================================" -ForegroundColor Red
    Write-Host "ERROR DURING EXECUTION" -ForegroundColor Red
    Write-Host "=================================" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# Calculate duration
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`nTotal execution time: $($duration.TotalSeconds) seconds" -ForegroundColor Yellow

# Check cache statistics
Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "CACHE DIRECTORY INFO" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

$cacheDir = ".embedding_cache"
if (Test-Path $cacheDir) {
    $cacheFiles = Get-ChildItem -Path $cacheDir -File
    $totalSize = ($cacheFiles | Measure-Object -Property Length -Sum).Sum

    Write-Host "Cache files: $($cacheFiles.Count)" -ForegroundColor White
    if ($totalSize) {
        $sizeMB = [Math]::Round($totalSize / 1MB, 2)
        Write-Host "Total size: $sizeMB MB" -ForegroundColor White
    }

    # List cache files
    if ($cacheFiles.Count -gt 0) {
        Write-Host "`nCache contents:" -ForegroundColor Yellow
        foreach ($file in $cacheFiles | Select-Object -First 5) {
            $sizeKB = [Math]::Round($file.Length / 1KB, 2)
            Write-Host "  - $($file.Name) (${sizeKB} KB)" -ForegroundColor Gray
        }
        if ($cacheFiles.Count -gt 5) {
            Write-Host "  ... and $($cacheFiles.Count - 5) more files" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "No cache directory found" -ForegroundColor Yellow
}

# Check results file
Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "RESULTS" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

$resultsFile = "bayesian_enhanced_results.json"
if (Test-Path $resultsFile) {
    Write-Host "Results saved to: $resultsFile" -ForegroundColor Green

    # Parse and display key results
    try {
        $results = Get-Content $resultsFile | ConvertFrom-Json
        Write-Host "`nBest configuration found:" -ForegroundColor Yellow
        Write-Host "  - Best score: $($results.best_score)" -ForegroundColor White
        Write-Host "  - Evaluations: $($results.n_evaluations)" -ForegroundColor White
        Write-Host "  - Total time: $($results.total_time)s" -ForegroundColor White
    } catch {
        Write-Host "Could not parse results file" -ForegroundColor Yellow
    }
} else {
    Write-Host "No results file found" -ForegroundColor Yellow
}

Write-Host "`n=================================" -ForegroundColor Green
Write-Host "TEST COMPLETE" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")