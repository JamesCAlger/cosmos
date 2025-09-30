# Test script for Bayesian optimization with caching mechanism
# This script tests the caching functionality with different scenarios

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "BAYESIAN OPTIMIZATION CACHE TEST" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Load environment variables
if (Test-Path ".env") {
    Write-Host "`nLoading environment variables from .env..." -ForegroundColor Yellow
    Get-Content .env | ForEach-Object {
        if ($_ -match "^([^#=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
        }
    }
}

# Test scenarios
$testScenarios = @(
    @{
        Name = "Test 1: Fresh run with cache (clear existing cache)"
        Args = @(
            "--n-calls", "5",
            "--num-docs", "10",
            "--num-queries", "3",
            "--use-cache",
            "--clear-cache",
            "--cache-memory-limit", "512"
        )
        Description = "Clears cache and runs fresh - should show 0% hit rate"
    },
    @{
        Name = "Test 2: Second run with existing cache"
        Args = @(
            "--n-calls", "5",
            "--num-docs", "10",
            "--num-queries", "3",
            "--use-cache",
            "--cache-memory-limit", "512"
        )
        Description = "Uses existing cache - should show high hit rate"
    },
    @{
        Name = "Test 3: Run without cache (baseline)"
        Args = @(
            "--n-calls", "3",
            "--num-docs", "10",
            "--num-queries", "2",
            "--no-cache"
        )
        Description = "No caching - for performance comparison"
    }
)

# Function to run a test scenario
function Run-Test {
    param(
        [string]$Name,
        [array]$Args,
        [string]$Description
    )

    Write-Host "`n" -NoNewline
    Write-Host "=================================" -ForegroundColor Green
    Write-Host $Name -ForegroundColor Green
    Write-Host "=================================" -ForegroundColor Green
    Write-Host $Description -ForegroundColor Yellow
    Write-Host "`nRunning command:" -ForegroundColor Cyan
    Write-Host "python scripts/run_bayesian_full_space_enhanced.py $($Args -join ' ')" -ForegroundColor White
    Write-Host ""

    # Run the Python script
    $startTime = Get-Date
    python scripts/run_bayesian_full_space_enhanced.py @Args
    $endTime = Get-Date
    $duration = $endTime - $startTime

    Write-Host "`nTest completed in: $($duration.TotalSeconds) seconds" -ForegroundColor Yellow
    return $duration.TotalSeconds
}

# Main test execution
try {
    $timings = @{}

    # Run each test scenario
    foreach ($scenario in $testScenarios) {
        $duration = Run-Test -Name $scenario.Name -Args $scenario.Args -Description $scenario.Description
        $timings[$scenario.Name] = $duration

        # Pause between tests
        Write-Host "`nPress any key to continue to next test..." -ForegroundColor Magenta
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }

    # Summary report
    Write-Host "`n" -NoNewline
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host "CACHE TEST SUMMARY" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan

    foreach ($name in $timings.Keys) {
        Write-Host "$name : $([Math]::Round($timings[$name], 2))s" -ForegroundColor White
    }

    # Calculate cache speedup
    if ($timings.ContainsKey("Test 1: Fresh run with cache (clear existing cache)") -and
        $timings.ContainsKey("Test 2: Second run with existing cache")) {
        $speedup = $timings["Test 1: Fresh run with cache (clear existing cache)"] / $timings["Test 2: Second run with existing cache"]
        Write-Host "`nCache Speedup: $([Math]::Round($speedup, 2))x faster" -ForegroundColor Green
    }

    # Check cache directory
    Write-Host "`n" -NoNewline
    Write-Host "=================================" -ForegroundColor Cyan
    Write-Host "CACHE DIRECTORY ANALYSIS" -ForegroundColor Cyan
    Write-Host "=================================" -ForegroundColor Cyan

    $cacheDir = ".embedding_cache"
    if (Test-Path $cacheDir) {
        $cacheFiles = Get-ChildItem -Path $cacheDir -File
        Write-Host "Cache files found: $($cacheFiles.Count)" -ForegroundColor Yellow

        $totalSize = ($cacheFiles | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "Total cache size: $([Math]::Round($totalSize, 2)) MB" -ForegroundColor Yellow

        Write-Host "`nCache files:" -ForegroundColor White
        foreach ($file in $cacheFiles) {
            Write-Host "  - $($file.Name) ($([Math]::Round($file.Length / 1KB, 2)) KB)" -ForegroundColor Gray
        }
    } else {
        Write-Host "No cache directory found" -ForegroundColor Red
    }

    Write-Host "`n" -NoNewline
    Write-Host "=================================" -ForegroundColor Green
    Write-Host "ALL TESTS COMPLETED SUCCESSFULLY" -ForegroundColor Green
    Write-Host "=================================" -ForegroundColor Green

} catch {
    Write-Host "`nError occurred: $_" -ForegroundColor Red
    exit 1
}