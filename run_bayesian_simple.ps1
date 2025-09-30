#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Simple Bayesian Optimization runner for auto-RAG
.DESCRIPTION
    Direct script to run Bayesian optimization with minimal setup
#>

param(
    [int]$NCalls = 10,
    [int]$NumDocs = 20,
    [int]$NumQueries = 5
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BAYESIAN OPTIMIZATION FOR AUTO-RAG" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Load environment variables
if (Test-Path ".env") {
    Write-Host "Loading .env file..." -ForegroundColor Green
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#].+?)\s*=\s*(.+)") {
            $name = $matches[1]
            $value = $matches[2]
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "  - Set $name" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "No .env file found - will use mock mode" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  - Evaluations: $NCalls" -ForegroundColor White
Write-Host "  - Documents: $NumDocs" -ForegroundColor White
Write-Host "  - Queries: $NumQueries" -ForegroundColor White
Write-Host ""

# Check cache directory
$cacheDir = ".embedding_cache"
if (Test-Path $cacheDir) {
    $cacheFiles = Get-ChildItem $cacheDir -Filter "*.pkl" 2>$null
    Write-Host "Cache Status:" -ForegroundColor Cyan
    Write-Host "  - Files: $($cacheFiles.Count)" -ForegroundColor White
    Write-Host "  - Size: $([math]::Round(($cacheFiles | Measure-Object -Property Length -Sum).Sum / 1MB, 2)) MB" -ForegroundColor White
} else {
    Write-Host "Creating cache directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STARTING OPTIMIZATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Build and run command
$args = @(
    "scripts/run_bayesian_full_space_enhanced.py",
    "--n-calls", $NCalls,
    "--num-docs", $NumDocs,
    "--num-queries", $NumQueries,
    "--use-cache",
    "--cache-dir", $cacheDir,
    "--cache-memory-limit", "256",
    "--real-api"
)

$pythonCmd = "python " + ($args -join " ")
Write-Host "Command: $pythonCmd" -ForegroundColor Gray
Write-Host ""

# Run the optimization
& python $args

# Check results
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "OPTIMIZATION COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    # Display results if file exists
    $resultsFile = "bayesian_enhanced_results.json"
    if (Test-Path $resultsFile) {
        Write-Host ""
        Write-Host "Results saved to: $resultsFile" -ForegroundColor Green

        # Try to parse and show best config
        try {
            $results = Get-Content $resultsFile -Raw | ConvertFrom-Json
            Write-Host ""
            Write-Host "Best Score: $([math]::Round($results.best_score, 4))" -ForegroundColor Cyan
            Write-Host "Best Configuration:" -ForegroundColor Cyan
            Write-Host "  - Chunking: $($results.best_params.chunking_strategy)" -ForegroundColor White
            Write-Host "  - Chunk Size: $($results.best_params.chunk_size)" -ForegroundColor White
            Write-Host "  - Retrieval: $($results.best_params.retrieval_method)" -ForegroundColor White
            Write-Host "  - Reranking: $($results.best_params.reranking_enabled)" -ForegroundColor White

            if ($results.cache_stats) {
                Write-Host ""
                Write-Host "Cache Performance:" -ForegroundColor Cyan
                Write-Host "  - Hit Rate: $([math]::Round($results.cache_stats.hit_rate * 100, 1))%" -ForegroundColor Green
                Write-Host "  - API Calls Saved: $($results.cache_stats.embeddings_cached)" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "Could not parse results file" -ForegroundColor Yellow
        }
    }

    # Show final cache status
    Write-Host ""
    $cacheFiles = Get-ChildItem $cacheDir -Filter "*.pkl" 2>$null
    Write-Host "Final Cache Status:" -ForegroundColor Cyan
    Write-Host "  - Files: $($cacheFiles.Count)" -ForegroundColor White
    Write-Host "  - Size: $([math]::Round(($cacheFiles | Measure-Object -Property Length -Sum).Sum / 1MB, 2)) MB" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "OPTIMIZATION FAILED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")