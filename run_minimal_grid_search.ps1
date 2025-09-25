# PowerShell script to run MINIMAL grid search
# Estimated cost: ~$0.04 with OpenAI API, Free in mock mode

param(
    [string]$ApiKey = "",
    [switch]$UseMock,
    [string]$EvaluationMethod = "semantic_fixed",  # Evaluation method: "keyword" or "semantic_fixed"
    [float]$SemanticThreshold = 0.75  # Threshold for semantic similarity (0.0-1.0)
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

# Set API key - Priority: 1) UseMock flag, 2) Parameter, 3) .env file, 4) Environment variable
if ($UseMock) {
    Write-Host "`n[OK] Running in MOCK MODE (no API calls)" -ForegroundColor Yellow
    $env:OPENAI_API_KEY = ""
} elseif ($ApiKey) {
    $env:OPENAI_API_KEY = $ApiKey
    Write-Host "`n[OK] API key set from parameter" -ForegroundColor Green
} else {
    # Try to read from .env file
    $envKey = Read-EnvFile
    if ($envKey) {
        $env:OPENAI_API_KEY = $envKey
        Write-Host "`n[OK] API key loaded from .env file" -ForegroundColor Green
        $masked = $env:OPENAI_API_KEY.Substring(0, 10) + "..."
        Write-Host "  Key: $masked" -ForegroundColor Gray
    } elseif ($env:OPENAI_API_KEY) {
        Write-Host "`n[OK] Using existing OPENAI_API_KEY from environment" -ForegroundColor Green
        $masked = $env:OPENAI_API_KEY.Substring(0, 10) + "..."
        Write-Host "  Key: $masked" -ForegroundColor Gray
    } else {
        Write-Host "`n[WARNING] No API key found - will run in MOCK MODE" -ForegroundColor Yellow
        Write-Host "  To use real API:" -ForegroundColor Gray
        Write-Host "    - Add OPENAI_API_KEY to .env file" -ForegroundColor Gray
        Write-Host "    - Or use -ApiKey parameter" -ForegroundColor Gray
        Write-Host "    - Or run with -UseMock to explicitly use mock mode" -ForegroundColor Gray
    }
}

# Show configuration
Write-Host "`nConfiguration:" -ForegroundColor White
Write-Host "  - Documents: 10" -ForegroundColor Gray
Write-Host "  - Queries: 3" -ForegroundColor Gray
Write-Host "  - Configurations: 27 (3x3x3)" -ForegroundColor Gray
Write-Host "  - Evaluation Method: $EvaluationMethod" -ForegroundColor Gray
if ($EvaluationMethod -eq "semantic_fixed") {
    Write-Host "  - Semantic Threshold: $SemanticThreshold" -ForegroundColor Gray
}
Write-Host "  - Parameters tested:" -ForegroundColor Gray
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
$confirmation = "y"
$confirmation | python scripts\run_minimal_real_grid_search.py

# Check results
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n================================" -ForegroundColor Green
    Write-Host "[OK] Grid search completed!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green

    # Show results if file exists
    if (Test-Path "minimal_real_results.json") {
        $results = Get-Content "minimal_real_results.json" | ConvertFrom-Json

        # Display top 5 configurations
        Write-Host "`n================================" -ForegroundColor Cyan
        Write-Host "TOP 5 CONFIGURATIONS" -ForegroundColor Yellow
        Write-Host "================================" -ForegroundColor Cyan

        # Sort and get top 5
        $sortedResults = $results.results | Sort-Object -Property score -Descending | Select-Object -First 5

        $rank = 1
        foreach ($config in $sortedResults) {
            Write-Host "`n$rank. Configuration #$($config.config_id)" -ForegroundColor White
            Write-Host "   Overall Score:        $([math]::Round($config.score, 3))" -ForegroundColor Gray

            # Handle potentially missing metrics
            if ($config.metrics.accuracy) {
                Write-Host "   Accuracy:            $([math]::Round($config.metrics.accuracy, 3))" -ForegroundColor Gray
            }
            if ($config.metrics.semantic_similarity) {
                Write-Host "   Semantic Similarity: $([math]::Round($config.metrics.semantic_similarity, 3))" -ForegroundColor Gray
            }
            if ($config.metrics.retrieval_precision) {
                Write-Host "   Retrieval Precision: $([math]::Round($config.metrics.retrieval_precision, 3))" -ForegroundColor Gray
            }
            if ($config.metrics.answer_completeness) {
                Write-Host "   Answer Completeness: $([math]::Round($config.metrics.answer_completeness, 3))" -ForegroundColor Gray
            }
            if ($config.metrics.avg_latency) {
                Write-Host "   Average Latency:     $([math]::Round($config.metrics.avg_latency, 3))s" -ForegroundColor Gray
            }
            if ($config.metrics.total_latency) {
                Write-Host "   Total Latency:       $([math]::Round($config.metrics.total_latency, 3))s" -ForegroundColor Gray
            }
            if ($config.metrics.total_cost) {
                Write-Host "   Total Cost:          `$$([math]::Round($config.metrics.total_cost, 4))" -ForegroundColor Gray
            }

            Write-Host "   Parameters:" -ForegroundColor DarkGray
            $config.params.PSObject.Properties | ForEach-Object {
                $name = $_.Name.Split('.')[-1]
                Write-Host "      - $($name): $($_.Value)" -ForegroundColor DarkGray
            }
            $rank++
        }

        Write-Host "`n================================" -ForegroundColor Green
        Write-Host "BEST CONFIGURATION SUMMARY" -ForegroundColor Yellow
        Write-Host "================================" -ForegroundColor Green
        Write-Host "  Config ID: #$($results.best.config_id)" -ForegroundColor White
        Write-Host "  Score: $([math]::Round($results.best.score, 3))" -ForegroundColor White
        Write-Host "  Parameters:" -ForegroundColor White
        $results.best.params.PSObject.Properties | ForEach-Object {
            $name = $_.Name.Split('.')[-1]
            Write-Host "    - $($name): $($_.Value)" -ForegroundColor Gray
        }

        Write-Host "`nResults saved to: minimal_real_results.json" -ForegroundColor Cyan
    } else {
        Write-Host "`nNo results file found" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[ERROR] Grid search failed!" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
}

Write-Host "`nPress any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")