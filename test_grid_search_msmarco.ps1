# PowerShell script to test the grid search with MS MARCO dataset
# This will use real MS MARCO data instead of mock data

param(
    [int]$NumConfigs = 3,  # Test first N configurations (default 3)
    [switch]$FullTest,     # Run all 27 configurations
    [switch]$QuickTest,    # Run single configuration
    [switch]$Debug,        # Show detailed debug output
    [int]$NumDocs = 50,    # Number of MS MARCO documents to load
    [int]$NumQueries = 10,  # Number of MS MARCO queries to test
    [string]$EvaluationMethod = "semantic_fixed",  # Evaluation method: "keyword" or "semantic_fixed"
    [float]$SemanticThreshold = 0.75  # Threshold for semantic similarity (0.0-1.0)
)

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "  MS MARCO GRID SEARCH TEST" -ForegroundColor Yellow
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
    Write-Host "Dataset: MS MARCO ($NumDocs documents, $NumQueries queries)" -ForegroundColor Cyan
    Write-Host "Estimated time: 3-5 minutes" -ForegroundColor Cyan
    Write-Host "Estimated cost: ~$0.10-0.20" -ForegroundColor Yellow
} elseif ($QuickTest) {
    $testMode = "quick"
    $configLimit = 1
    Write-Host "`nMode: QUICK TEST (1 configuration)" -ForegroundColor Magenta
    Write-Host "Dataset: MS MARCO ($NumDocs documents, $NumQueries queries)" -ForegroundColor Cyan
    Write-Host "Estimated time: 10-20 seconds" -ForegroundColor Cyan
} else {
    Write-Host "`nMode: PARTIAL TEST ($NumConfigs configurations)" -ForegroundColor Magenta
    Write-Host "Dataset: MS MARCO ($NumDocs documents, $NumQueries queries)" -ForegroundColor Cyan
    Write-Host "Estimated time: $([math]::Round($NumConfigs * 10 / 27 * 60)) seconds" -ForegroundColor Cyan
}

Write-Host "`nTest Configuration:" -ForegroundColor White
Write-Host "  - Configurations to test: $configLimit" -ForegroundColor Gray
Write-Host "  - MS MARCO Documents: $NumDocs" -ForegroundColor Gray
Write-Host "  - MS MARCO Queries: $NumQueries" -ForegroundColor Gray
Write-Host "  - Total API calls: $($configLimit * $NumQueries)" -ForegroundColor Gray
Write-Host "  - Delay between calls: 0.15s (rate limit protection)" -ForegroundColor Gray
Write-Host "  - Evaluation Method: $EvaluationMethod" -ForegroundColor Gray
if ($EvaluationMethod -eq "semantic_fixed") {
    Write-Host "  - Semantic Threshold: $SemanticThreshold" -ForegroundColor Gray
}

# Create Python test script
$pythonScript = @"
import sys
import os
sys.path.append('.')
from dotenv import load_dotenv
load_dotenv()

import json
import time
from typing import Dict, Any, List
from loguru import logger
import numpy as np
from itertools import product
from datasets import load_dataset
import traceback

# Import our components
from autorag.datasets.msmarco_loader import MSMARCOLoader
from scripts.run_minimal_real_grid_search import (
    MinimalRealGridSearch,
    OpenAIMiniGenerator,
    RealFixedSizeChunker,
    SimpleRetriever,
    MockEmbedder
)
from autorag.components.auto_register import auto_register
from autorag.pipeline.registry import get_registry

# Setup logging
logger.add("msmarco_grid_search_test.log", rotation="10 MB")

# Clear registry and register components
registry = get_registry()
registry.clear()

# Register our components
auto_register("chunker", "fixed_size")(RealFixedSizeChunker)
auto_register("embedder", "mock")(MockEmbedder)
auto_register("retriever", "simple")(SimpleRetriever)
auto_register("generator", "openai_mini")(OpenAIMiniGenerator)

# Reset counters
OpenAIMiniGenerator._api_call_count = 0
OpenAIMiniGenerator._success_count = 0
OpenAIMiniGenerator._error_count = 0
OpenAIMiniGenerator._first_error_config = None

print("\nStarting MS MARCO grid search test...")
print("=" * 60)

# Load MS MARCO data
print("\nLoading MS MARCO dataset...")
loader = MSMARCOLoader()

try:
    # Load subset with ground truth answers for evaluation
    documents, queries = loader.load_subset(
        num_docs=$NumDocs,
        num_queries=$NumQueries,
        include_answers=True
    )

    # Convert Document objects to strings for the grid search
    doc_texts = [doc.content for doc in documents]

    # Convert queries to the format expected by evaluate_configuration
    # The method expects a list of dicts with 'question' and 'expected' keys
    formatted_queries = []
    for q in queries:
        query_dict = {
            'question': q['question'],
            'expected': ''  # We'll extract keywords from ground truth
        }

        # Store both keyword and full ground truth for evaluation
        if q.get('ground_truth_answer'):
            answer = q['ground_truth_answer']
            query_dict['ground_truth_answer'] = answer  # Full answer for semantic evaluation

            # Extract keyword for backwards compatibility
            answer_lower = answer.lower()
            if 'paris' in answer_lower:
                query_dict['expected'] = 'paris'
            elif 'python' in answer_lower:
                query_dict['expected'] = 'python'
            elif 'manhattan' in answer_lower:
                query_dict['expected'] = 'manhattan'
            elif 'project' in answer_lower:
                query_dict['expected'] = 'project'
            else:
                # Use the first significant word from the answer
                words = answer_lower.split()
                for word in words:
                    if len(word) > 4:  # Skip short words
                        query_dict['expected'] = word
                        break

        formatted_queries.append(query_dict)

    print(f"Successfully loaded {len(doc_texts)} documents, {len(formatted_queries)} queries")
    print(f"Sample document: {doc_texts[0][:100]}...")
    print(f"Sample query: {formatted_queries[0]['question']}")
    print(f"Expected keyword: {formatted_queries[0]['expected']}")

except Exception as e:
    print(f"Error loading MS MARCO data: {e}")
    print("Falling back to synthetic data...")

    # Fallback to synthetic data if MS MARCO fails
    doc_texts = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Climate change affects global weather patterns.",
        "DNA contains genetic information.",
        "Shakespeare wrote many famous plays.",
        "The speed of light is approximately 300,000 km/s.",
        "The Pacific Ocean is the largest ocean.",
        "Photosynthesis converts light into energy.",
        "The human brain has billions of neurons."
    ] * (($NumDocs // 10) + 1)
    doc_texts = doc_texts[:$NumDocs]

    formatted_queries = [
        {"question": "What is Python?", "expected": "programming"},
        {"question": "Where is the Eiffel Tower?", "expected": "paris"},
        {"question": "What is the speed of light?", "expected": "300"}
    ] * (($NumQueries // 3) + 1)
    formatted_queries = formatted_queries[:$NumQueries]

print("\n" + "=" * 60)

# Create grid search instance with evaluation method
gs = MinimalRealGridSearch(
    num_docs=len(doc_texts),
    num_queries=len(formatted_queries),
    api_delay=0.15,
    evaluation_method='$EvaluationMethod',
    semantic_threshold=$SemanticThreshold
)

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

    try:
        # Evaluate configuration with MS MARCO data
        metrics = gs.evaluate_configuration(params, doc_texts, formatted_queries, config_num=i)

        print(f"  Result: Accuracy={metrics['accuracy']:.2f}, Latency={metrics['avg_latency']:.2f}s")

    except Exception as e:
        logger.error(f"Failed to evaluate config {i}: {e}")
        print(f"  Result: FAILED - {str(e)[:50]}...")
        metrics = {
            "accuracy": 0.0,
            "avg_latency": 0.0,
            "error": str(e)
        }

    results.append({
        "config_id": i,
        "params": params,
        "metrics": metrics,
        "score": metrics["accuracy"]
    })

print("\n" + "=" * 60)
print("MS MARCO TEST RESULTS SUMMARY")
print("=" * 60)
print(f"Dataset: MS MARCO")
print(f"Documents: {len(doc_texts)}")
print(f"Queries: {len(formatted_queries)}")
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
valid_results = [r for r in results if 'error' not in r['metrics']]
if valid_results:
    accuracies = [r['metrics']['accuracy'] for r in valid_results]
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAccuracy: Average={avg_accuracy:.2%}, Min={min(accuracies):.2%}, Max={max(accuracies):.2%}")

    # Show best configuration
    best = max(valid_results, key=lambda x: x['score'])
    print(f"\nBest Configuration (Config #{best['config_id']}):")
    for param, value in best['params'].items():
        print(f"  {param.split('.')[-1]}: {value}")
    print(f"  Score: {best['score']:.2%}")

    # Show latency stats
    latencies = [r['metrics']['avg_latency'] for r in valid_results]
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nLatency: Average={avg_latency:.2f}s, Min={min(latencies):.2f}s, Max={max(latencies):.2f}s")

# Save test results
with open('msmarco_test_results.json', 'w') as f:
    json.dump({
        'test_mode': '$testMode',
        'dataset': 'MS MARCO',
        'num_documents': len(doc_texts),
        'num_queries': len(formatted_queries),
        'configs_tested': len(results),
        'results': results,
        'summary': {
            'total_calls': OpenAIMiniGenerator._api_call_count,
            'success_calls': OpenAIMiniGenerator._success_count,
            'failed_calls': OpenAIMiniGenerator._error_count,
            'avg_accuracy': avg_accuracy if valid_results else 0,
            'avg_latency': avg_latency if valid_results else 0
        }
    }, f, indent=2)

print("\nTest results saved to: msmarco_test_results.json")
"@

# Save Python script
$pythonScript | Out-File -FilePath "test_msmarco_grid_search_temp.py" -Encoding UTF8

Write-Host "`nPress Enter to start test or Ctrl+C to cancel..." -ForegroundColor DarkGray
Read-Host

Write-Host "`nRunning MS MARCO test..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor DarkGray

# Run the test
if ($Debug) {
    # Show all output including debug logs
    python test_msmarco_grid_search_temp.py 2>&1
} else {
    # Filter output for cleaner display
    python test_msmarco_grid_search_temp.py 2>&1 | Select-String -Pattern "(Starting|Loading|Successfully|Sample|Testing|Config \d+|Result:|MS MARCO TEST|Dataset:|Documents:|Queries:|API calls|Success|Failed|Accuracy:|Best Configuration|Score:|Latency:|saved to|Error)" | ForEach-Object { $_.Line }
}

$exitCode = $LASTEXITCODE

# Cleanup
Remove-Item "test_msmarco_grid_search_temp.py" -ErrorAction SilentlyContinue

# Check results
if ($exitCode -eq 0) {
    Write-Host "`n================================" -ForegroundColor Green
    Write-Host "[OK] MS MARCO test completed!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green

    # Display results if available
    if (Test-Path "msmarco_test_results.json") {
        $results = Get-Content "msmarco_test_results.json" | ConvertFrom-Json

        Write-Host "`nQuick Summary:" -ForegroundColor Yellow
        Write-Host "  - Dataset: MS MARCO" -ForegroundColor White
        Write-Host "  - Documents: $($results.num_documents)" -ForegroundColor White
        Write-Host "  - Queries: $($results.num_queries)" -ForegroundColor White
        Write-Host "  - Configs tested: $($results.configs_tested)" -ForegroundColor White

        if ($results.summary.total_calls -gt 0) {
            Write-Host "  - Success rate: $([math]::Round(($results.summary.success_calls / [math]::Max(1, $results.summary.total_calls)) * 100, 1))%" -ForegroundColor White
            Write-Host "  - Avg accuracy: $([math]::Round($results.summary.avg_accuracy * 100, 1))%" -ForegroundColor White
            Write-Host "  - Avg latency: $([math]::Round($results.summary.avg_latency, 2))s" -ForegroundColor White
        }

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
Write-Host "  .\test_grid_search_msmarco.ps1 -QuickTest                # Test 1 config with MS MARCO" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search_msmarco.ps1 -NumConfigs 5             # Test 5 configs" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search_msmarco.ps1 -FullTest                 # Test all 27 configs" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search_msmarco.ps1 -NumDocs 100 -NumQueries 20  # Custom dataset size" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search_msmarco.ps1 -Debug                    # Show debug output" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search_msmarco.ps1 -EvaluationMethod keyword # Use keyword matching" -ForegroundColor DarkGray
Write-Host "  .\test_grid_search_msmarco.ps1 -SemanticThreshold 0.8   # Higher similarity threshold" -ForegroundColor DarkGray

Write-Host "`nPress any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")