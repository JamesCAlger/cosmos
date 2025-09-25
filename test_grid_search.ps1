# PowerShell script to test the grid search with debug output
# This will help verify the system is working correctly

param(
    [int]$NumConfigs = 3,  # Test first N configurations (default 3)
    [switch]$FullTest,     # Run all 27 configurations
    [switch]$QuickTest,    # Run single configuration
    [switch]$Debug         # Show detailed debug output
)

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "  GRID SEARCH TEST SCRIPT" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Cyan

# Check if in correct directory
if (-not (Test-Path ".\scripts\run_minimal_real_grid_search.py")) {
    Write-Host "ERROR: Not in auto-RAG directory!" -ForegroundColor Red
    Write-Host "Please cd to: C:\Users\alger\Documents\000. Projects\auto-RAG" -ForegroundColor Yellow
    exit 1
}

# Check for .env file
if (-not (Test-Path ".\.env")) {
    Write-Host "WARNING: No .env file found!" -ForegroundColor Yellow
    Write-Host "API calls will fail without OPENAI_API_KEY" -ForegroundColor Red
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') { exit 0 }
} else {
    Write-Host "[OK] .env file found" -ForegroundColor Green

    # Check if API key is present
    $envContent = Get-Content ".\.env" | Select-String "OPENAI_API_KEY"
    if ($envContent) {
        $keyValue = ($envContent -split "=")[1].Trim()
        if ($keyValue -and $keyValue.StartsWith("sk-")) {
            Write-Host "[OK] OPENAI_API_KEY found (starts with sk-)" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] OPENAI_API_KEY may be invalid" -ForegroundColor Yellow
        }
    }
}

# Determine test mode
$testMode = "partial"
$configLimit = $NumConfigs

if ($FullTest) {
    $testMode = "full"
    $configLimit = 27
    Write-Host "`nMode: FULL TEST (27 configurations)" -ForegroundColor Magenta
    Write-Host "Estimated time: 2-3 minutes" -ForegroundColor Cyan
    Write-Host "Estimated cost: ~$0.04" -ForegroundColor Yellow
} elseif ($QuickTest) {
    $testMode = "quick"
    $configLimit = 1
    Write-Host "`nMode: QUICK TEST (1 configuration)" -ForegroundColor Magenta
    Write-Host "Estimated time: 5-10 seconds" -ForegroundColor Cyan
} else {
    Write-Host "`nMode: PARTIAL TEST ($NumConfigs configurations)" -ForegroundColor Magenta
    Write-Host "Estimated time: $([math]::Round($NumConfigs * 5 / 27 * 60)) seconds" -ForegroundColor Cyan
}

Write-Host "`nTest Configuration:" -ForegroundColor White
Write-Host "  - Configurations to test: $configLimit" -ForegroundColor Gray
Write-Host "  - Documents: 10" -ForegroundColor Gray
Write-Host "  - Queries per config: 3" -ForegroundColor Gray
Write-Host "  - Total API calls: $($configLimit * 3)" -ForegroundColor Gray
Write-Host "  - Delay between calls: 0.15s" -ForegroundColor Gray

# Create Python test script
$pythonScript = @"
import sys
sys.path.append('.')
from dotenv import load_dotenv
load_dotenv()

import os
import json
from scripts.run_minimal_real_grid_search import MinimalRealGridSearch, OpenAIMiniGenerator

# Reset counters
OpenAIMiniGenerator._api_call_count = 0
OpenAIMiniGenerator._success_count = 0
OpenAIMiniGenerator._error_count = 0
OpenAIMiniGenerator._first_error_config = None

print("\nStarting grid search test...")
print("=" * 60)

# Create grid search instance
gs = MinimalRealGridSearch(num_docs=10, num_queries=3, api_delay=0.15)

# Load data
docs, queries = gs.load_minimal_data()
print(f"Loaded {len(docs)} documents, {len(queries)} queries")

# Define search space
search_space = {
    "chunker.fixed_size.chunk_size": [256, 512, 1024],
    "retriever.simple.top_k": [3, 5, 10],
    "generator.openai_mini.temperature": [0.0, 0.3, 0.7]
}

# Generate configurations
from itertools import product
param_names = list(search_space.keys())
param_values = [search_space[name] for name in param_names]
all_combinations = list(product(*param_values))[:$configLimit]

print(f"Testing {len(all_combinations)} configurations...")
print("=" * 60)

results = []
for i, combination in enumerate(all_combinations, 1):
    params = dict(zip(param_names, combination))

    print(f"\nConfig {i}/{len(all_combinations)}:")
    print(f"  chunk_size: {params['chunker.fixed_size.chunk_size']}")
    print(f"  top_k: {params['retriever.simple.top_k']}")
    print(f"  temperature: {params['generator.openai_mini.temperature']}")

    metrics = gs.evaluate_configuration(params, docs, queries, config_num=i)

    print(f"  Result: Accuracy={metrics['accuracy']:.2f}, Latency={metrics['avg_latency']:.2f}s")

    results.append({
        "config_id": i,
        "params": params,
        "metrics": metrics,
        "score": metrics["accuracy"]
    })

print("\n" + "=" * 60)
print("TEST RESULTS SUMMARY")
print("=" * 60)
print(f"Configurations tested: {len(results)}")
print(f"Total API calls: {OpenAIMiniGenerator._api_call_count}")
print(f"Successful calls: {OpenAIMiniGenerator._success_count}")
print(f"Failed calls: {OpenAIMiniGenerator._error_count}")
if OpenAIMiniGenerator._error_count > 0:
    print(f"First error at call: #{OpenAIMiniGenerator._first_error_config}")
    print(f"Success rate: {OpenAIMiniGenerator._success_count / max(1, OpenAIMiniGenerator._api_call_count) * 100:.1f}%")
else:
    print("Success rate: 100% - No errors!")

# Show accuracy distribution
accuracies = [r['metrics']['accuracy'] for r in results]
avg_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAccuracy: Average={avg_accuracy:.2%}, Min={min(accuracies):.2%}, Max={max(accuracies):.2%}")

# Show latency stats
latencies = [r['metrics']['avg_latency'] for r in results]
avg_latency = sum(latencies) / len(latencies)
print(f"Latency: Average={avg_latency:.2f}s, Min={min(latencies):.2f}s, Max={max(latencies):.2f}s")

# Save test results
with open('test_results.json', 'w') as f:
    json.dump({
        'test_mode': '$testMode',
        'configs_tested': len(results),
        'results': results,
        'summary': {
            'total_calls': OpenAIMiniGenerator._api_call_count,
            'success_calls': OpenAIMiniGenerator._success_count,
            'failed_calls': OpenAIMiniGenerator._error_count,
            'avg_accuracy': avg_accuracy,
            'avg_latency': avg_latency
        }
    }, f, indent=2)

print("\nTest results saved to: test_results.json")
"@

# Save Python script
$pythonScript | Out-File -FilePath "test_grid_search_temp.py" -Encoding UTF8

Write-Host "`nPress Enter to start test or Ctrl+C to cancel..." -ForegroundColor DarkGray
Read-Host

Write-Host "`nRunning test..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor DarkGray

# Run the test
if ($Debug) {
    # Show all output including debug logs
    python test_grid_search_temp.py 2>&1
} else {
    # Filter output for cleaner display
    python test_grid_search_temp.py 2>&1 | Select-String -Pattern "(Starting|Loaded|Testing|Config \d+|Result:|TEST RESULTS|API calls|Success|Failed|Accuracy:|Latency:|saved to)" | ForEach-Object { $_.Line }
}

$exitCode = $LASTEXITCODE

# Cleanup
Remove-Item "test_grid_search_temp.py" -ErrorAction SilentlyContinue

# Check results
if ($exitCode -eq 0) {
    Write-Host "`n================================" -ForegroundColor Green
    Write-Host "[OK] Test completed successfully!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green

    # Display results if available
    if (Test-Path "test_results.json") {
        $results = Get-Content "test_results.json" | ConvertFrom-Json

        Write-Host "`nQuick Summary:" -ForegroundColor Yellow
        Write-Host "  - Configs tested: $($results.configs_tested)" -ForegroundColor White
        Write-Host "  - Success rate: $([math]::Round(($results.summary.success_calls / [math]::Max(1, $results.summary.total_calls)) * 100, 1))%" -ForegroundColor White
        Write-Host "  - Avg accuracy: $([math]::Round($results.summary.avg_accuracy * 100, 1))%" -ForegroundColor White
        Write-Host "  - Avg latency: $([math]::Round($results.summary.avg_latency, 2))s" -ForegroundColor White

        if ($results.summary.failed_calls -gt 0) {
            Write-Host "`n[WARNING] $($results.summary.failed_calls) API calls failed" -ForegroundColor Yellow
            Write-Host "This may indicate rate limiting or connection issues" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "`n================================" -ForegroundColor Red
    Write-Host "[ERROR] Test failed!" -ForegroundColor Red
    Write-Host "================================" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Yellow
}

Write-Host "`nTest Options:" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search.ps1 -QuickTest     # Test 1 config" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search.ps1 -NumConfigs 5  # Test 5 configs" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search.ps1 -FullTest      # Test all 27 configs" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search.ps1 -Debug         # Show debug output" -ForegroundColor DarkGray

Write-Host "`nPress any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")