#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Full Bayesian Optimization for auto-RAG system with caching and comprehensive evaluation
.DESCRIPTION
    This script runs a complete Bayesian optimization process for the auto-RAG system,
    including embedding caching, multiple evaluation metrics, and detailed progress reporting.
.PARAMETER NCalls
    Number of Bayesian optimization evaluations (default: 20)
.PARAMETER NumDocs
    Number of documents to process (default: 50)
.PARAMETER NumQueries
    Number of queries to evaluate (default: 10)
.PARAMETER UseCache
    Enable embedding cache (default: true)
.PARAMETER ClearCache
    Clear existing cache before starting (default: false)
.EXAMPLE
    .\run_full_bayesian_optimization.ps1 -NCalls 30 -NumDocs 100
#>

param(
    [int]$NCalls = 20,
    [int]$NumDocs = 50,
    [int]$NumQueries = 10,
    [bool]$UseCache = $true,
    [bool]$ClearCache = $false,
    [string]$OutputFile = "bayesian_enhanced_results.json"
)

# Colors for output
$Colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
    Metric = "Magenta"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White",
        [switch]$NoNewline
    )

    $params = @{
        ForegroundColor = $Colors[$Color]
        NoNewline = $NoNewline
    }
    Write-Host $Message @params
}

function Write-Header {
    param([string]$Title)

    $line = "=" * 80
    Write-ColorOutput "`n$line" "Header"
    Write-ColorOutput $Title "Header"
    Write-ColorOutput "$line`n" "Header"
}

function Write-Section {
    param([string]$Title)

    $line = "-" * 60
    Write-ColorOutput "`n$line" "Info"
    Write-ColorOutput $Title "Info"
    Write-ColorOutput "$line" "Info"
}

function Test-Environment {
    Write-Header "ENVIRONMENT CHECK"

    # Check Python
    Write-ColorOutput "Checking Python installation... " "Info" -NoNewline
    try {
        $pythonVersion = python --version 2>&1
        Write-ColorOutput "OK ($pythonVersion)" "Success"
    }
    catch {
        Write-ColorOutput "FAILED" "Error"
        Write-ColorOutput "Error: Python is not installed or not in PATH" "Error"
        return $false
    }

    # Check .env file
    Write-ColorOutput "Checking .env file... " "Info" -NoNewline
    if (Test-Path ".env") {
        Write-ColorOutput "OK" "Success"

        # Load and check API key
        $envContent = Get-Content ".env" -Raw
        if ($envContent -match "OPENAI_API_KEY=") {
            Write-ColorOutput "  - OpenAI API key found" "Success"
        }
        else {
            Write-ColorOutput "  - Warning: OPENAI_API_KEY not found in .env" "Warning"
            Write-ColorOutput "  - System will use mock mode" "Warning"
        }
    }
    else {
        Write-ColorOutput "NOT FOUND" "Warning"
        Write-ColorOutput "  - System will use mock mode" "Warning"
    }

    # Check cache directory
    if ($UseCache) {
        Write-ColorOutput "Checking cache directory... " "Info" -NoNewline
        if (Test-Path ".embedding_cache") {
            $cacheFiles = Get-ChildItem ".embedding_cache" -Filter "*.pkl" 2>$null
            $cacheSize = ($cacheFiles | Measure-Object -Property Length -Sum).Sum / 1MB
            Write-ColorOutput "OK" "Success"
            Write-ColorOutput "  - Cache files: $($cacheFiles.Count)" "Info"
            Write-ColorOutput "  - Cache size: $([math]::Round($cacheSize, 2)) MB" "Info"

            if ($ClearCache) {
                Write-ColorOutput "  - Clearing cache as requested..." "Warning" -NoNewline
                Remove-Item ".embedding_cache\*.pkl" -Force 2>$null
                Write-ColorOutput " DONE" "Success"
            }
        }
        else {
            Write-ColorOutput "NOT FOUND (will be created)" "Info"
            New-Item -ItemType Directory -Path ".embedding_cache" -Force | Out-Null
        }
    }

    # Check required packages
    Write-ColorOutput "`nChecking required Python packages..." "Info"

    # Package name mappings (pip name -> import name)
    $packageMappings = @{
        "numpy" = "numpy"
        "scikit-optimize" = "skopt"
        "openai" = "openai"
        "transformers" = "transformers"
        "sentence-transformers" = "sentence_transformers"
        "rank-bm25" = "rank_bm25"
        "python-dotenv" = "dotenv"
        "loguru" = "loguru"
        "tiktoken" = "tiktoken"
    }

    $missingPackages = @()
    foreach ($package in $packageMappings.Keys) {
        $importName = $packageMappings[$package]
        Write-ColorOutput "  - ${package}... " "Info" -NoNewline
        $result = python -c "import $importName" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "OK" "Success"
        }
        else {
            Write-ColorOutput "MISSING" "Error"
            $missingPackages += $package
        }
    }

    if ($missingPackages.Count -gt 0) {
        Write-ColorOutput "`nMissing packages detected. Install with:" "Error"
        Write-ColorOutput "  pip install $($missingPackages -join ' ')" "Warning"
        return $false
    }

    return $true
}

function Show-Configuration {
    Write-Header "OPTIMIZATION CONFIGURATION"

    Write-ColorOutput "Bayesian Optimization Settings:" "Info"
    Write-ColorOutput "  - Evaluations: $NCalls" "Metric"
    Write-ColorOutput "  - Search space: ~432 discrete combinations" "Metric"
    Write-ColorOutput "  - Reduction: $([math]::Round((1 - $NCalls/432) * 100, 1))%" "Metric"

    Write-ColorOutput "`nDataset Configuration:" "Info"
    Write-ColorOutput "  - Documents: $NumDocs" "Metric"
    Write-ColorOutput "  - Queries: $NumQueries" "Metric"
    Write-ColorOutput "  - Dataset: MS MARCO" "Metric"

    Write-ColorOutput "`nCache Configuration:" "Info"
    Write-ColorOutput "  - Enabled: $UseCache" "Metric"
    if ($UseCache) {
        Write-ColorOutput "  - Directory: .embedding_cache" "Metric"
        Write-ColorOutput "  - Memory limit: 256MB" "Metric"
        Write-ColorOutput "  - Clear before run: $ClearCache" "Metric"
    }

    Write-ColorOutput "`nOutput Configuration:" "Info"
    Write-ColorOutput "  - Results file: $OutputFile" "Metric"
    Write-ColorOutput "  - Log level: INFO" "Metric"

    # Estimate time and cost
    $estimatedTime = $NCalls * 10  # ~10 seconds per evaluation
    $estimatedCost = $NCalls * 0.002  # ~$0.002 per evaluation with cache

    Write-ColorOutput "`nEstimated Performance:" "Info"
    Write-ColorOutput "  - Time: ~$([math]::Round($estimatedTime/60, 1)) minutes" "Warning"
    Write-ColorOutput "  - Cost: ~`$$([math]::Round($estimatedCost, 3))" "Warning"
}

function Run-Optimization {
    Write-Header "STARTING BAYESIAN OPTIMIZATION"

    # Build command
    $pythonScript = "scripts/run_bayesian_full_space_enhanced.py"

    if (-not (Test-Path $pythonScript)) {
        Write-ColorOutput "Error: Script not found at $pythonScript" "Error"
        return $false
    }

    $arguments = @(
        $pythonScript,
        "--n-calls", $NCalls,
        "--num-docs", $NumDocs,
        "--num-queries", $NumQueries
    )

    if ($UseCache) {
        $arguments += @(
            "--use-cache",
            "--cache-dir", ".embedding_cache",
            "--cache-memory-limit", "256"
        )
    }

    # Check for real API mode
    if (Test-Path ".env") {
        $envContent = Get-Content ".env" -Raw
        if ($envContent -match "OPENAI_API_KEY=.+") {
            $arguments += "--real-api"
            Write-ColorOutput "Running in REAL API mode (OpenAI)" "Success"
        }
        else {
            Write-ColorOutput "Running in MOCK mode (no API key)" "Warning"
        }
    }
    else {
        Write-ColorOutput "Running in MOCK mode (no .env file)" "Warning"
    }

    Write-ColorOutput "`nCommand: python $($arguments -join ' ')" "Info"
    Write-ColorOutput "`nStarting optimization...`n" "Info"

    # Run the optimization
    $startTime = Get-Date

    try {
        # Use Start-Process for better output handling
        $process = Start-Process -FilePath "python" `
                                -ArgumentList $arguments `
                                -NoNewWindow `
                                -PassThru `
                                -Wait

        if ($process.ExitCode -eq 0) {
            $duration = (Get-Date) - $startTime
            Write-ColorOutput "`nOptimization completed successfully!" "Success"
            Write-ColorOutput "Duration: $([math]::Round($duration.TotalMinutes, 2)) minutes" "Info"
            return $true
        }
        else {
            Write-ColorOutput "`nOptimization failed with exit code: $($process.ExitCode)" "Error"
            return $false
        }
    }
    catch {
        Write-ColorOutput "`nError running optimization: $_" "Error"
        return $false
    }
}

function Show-Results {
    param([string]$ResultsFile)

    Write-Header "OPTIMIZATION RESULTS"

    if (-not (Test-Path $ResultsFile)) {
        Write-ColorOutput "Results file not found: $ResultsFile" "Error"
        return
    }

    try {
        $results = Get-Content $ResultsFile -Raw | ConvertFrom-Json

        # Best configuration
        Write-Section "Best Configuration"
        $best = $results.best_params
        Write-ColorOutput "Chunking Strategy: $($best.chunking_strategy)" "Metric"
        Write-ColorOutput "Chunk Size: $($best.chunk_size)" "Metric"
        Write-ColorOutput "Retrieval Method: $($best.retrieval_method)" "Metric"
        Write-ColorOutput "Retrieval Top-K: $($best.retrieval_top_k)" "Metric"
        if ($best.retrieval_method -eq "hybrid") {
            Write-ColorOutput "Hybrid Weight: $([math]::Round($best.hybrid_weight, 3))" "Metric"
        }
        Write-ColorOutput "Reranking: $($best.reranking_enabled)" "Metric"
        if ($best.reranking_enabled) {
            Write-ColorOutput "Reranking Top-K: $($best.top_k_rerank)" "Metric"
        }
        Write-ColorOutput "Temperature: $([math]::Round($best.temperature, 3))" "Metric"
        Write-ColorOutput "Max Tokens: $($best.max_tokens)" "Metric"

        # Performance metrics
        Write-Section "Performance Metrics"
        Write-ColorOutput "Best Score: $([math]::Round($results.best_score, 4))" "Success"
        Write-ColorOutput "Evaluations: $($results.n_calls)" "Metric"
        Write-ColorOutput "Total Time: $([math]::Round($results.total_time, 1))s" "Metric"

        # Cache statistics if available
        if ($results.cache_stats) {
            Write-Section "Cache Performance"
            $cache = $results.cache_stats
            Write-ColorOutput "Hit Rate: $([math]::Round($cache.hit_rate * 100, 1))%" "Success"
            Write-ColorOutput "Total Hits: $($cache.hits)" "Metric"
            Write-ColorOutput "Total Misses: $($cache.misses)" "Metric"
            Write-ColorOutput "API Calls Saved: $($cache.embeddings_cached)" "Metric"
            Write-ColorOutput "Cost Saved: `$$([math]::Round($cache.cost_saved, 4))" "Success"
            Write-ColorOutput "Time Saved: $([math]::Round($cache.time_saved, 1))s" "Metric"
        }

        # Show convergence
        Write-Section "Optimization Progress"
        $allScores = $results.all_scores
        for ($i = 0; $i -lt [Math]::Min(5, $allScores.Count); $i++) {
            $score = [math]::Round($allScores[$i], 4)
            Write-ColorOutput "  Eval $($i+1): $score" "Info"
        }
        if ($allScores.Count -gt 5) {
            Write-ColorOutput "  ..." "Info"
            for ($i = [Math]::Max(5, $allScores.Count - 3); $i -lt $allScores.Count; $i++) {
                $score = [math]::Round($allScores[$i], 4)
                Write-ColorOutput "  Eval $($i+1): $score" "Info"
            }
        }
    }
    catch {
        Write-ColorOutput "Error parsing results: $_" "Error"
    }
}

function Main {
    # Clear screen for better visibility
    Clear-Host

    Write-Header "AUTO-RAG BAYESIAN OPTIMIZATION SUITE"
    Write-ColorOutput "Advanced RAG Pipeline Optimization with Embedding Cache" "Info"
    Write-ColorOutput "Version: 2.0" "Info"

    # Test environment
    if (-not (Test-Environment)) {
        Write-ColorOutput "`nEnvironment check failed. Please fix the issues above." "Error"
        exit 1
    }

    # Show configuration
    Show-Configuration

    # Confirm with user
    Write-ColorOutput "`nPress Enter to start optimization or Ctrl+C to cancel..." "Warning" -NoNewline
    Read-Host

    # Run optimization
    $success = Run-Optimization

    if ($success) {
        # Show results
        Show-Results -ResultsFile $OutputFile

        # Cache report
        if ($UseCache -and (Test-Path ".embedding_cache")) {
            Write-Section "Final Cache Status"
            $cacheFiles = Get-ChildItem ".embedding_cache" -Filter "*.pkl"
            $totalSize = ($cacheFiles | Measure-Object -Property Length -Sum).Sum / 1MB
            Write-ColorOutput "Cache files: $($cacheFiles.Count)" "Info"
            Write-ColorOutput "Total size: $([math]::Round($totalSize, 2)) MB" "Info"

            # Show top 5 largest cache entries
            if ($cacheFiles.Count -gt 0) {
                Write-ColorOutput "`nLargest cache entries:" "Info"
                $cacheFiles | Sort-Object Length -Descending |
                    Select-Object -First 5 |
                    ForEach-Object {
                        $sizeMB = [math]::Round($_.Length / 1KB, 2)
                        Write-ColorOutput "  - $($_.Name): ${sizeMB} KB" "Info"
                    }
            }
        }

        Write-Header "OPTIMIZATION COMPLETE"
        Write-ColorOutput "Results saved to: $OutputFile" "Success"
        Write-ColorOutput "`nNext steps:" "Info"
        Write-ColorOutput "  1. Review the results in $OutputFile" "Info"
        Write-ColorOutput "  2. Apply the best configuration to your RAG pipeline" "Info"
        Write-ColorOutput "  3. Run validation on a held-out test set" "Info"
    }
    else {
        Write-ColorOutput "`nOptimization failed. Check the error messages above." "Error"
        exit 1
    }
}

# Run the main function
Main