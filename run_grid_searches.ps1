# PowerShell script to run grid searches for AutoRAG
# This script provides options to run different grid search configurations

param(
    [string]$Mode = "menu",
    [string]$ApiKey = "",
    [switch]$SkipInstall
)

# Colors for output
$Host.UI.RawUI.ForegroundColor = "White"

function Write-Title {
    param([string]$Text)
    Write-Host "`n$('=' * 80)" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Yellow
    Write-Host "$('=' * 80)" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Write-Error {
    param([string]$Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Write-Info {
    param([string]$Text)
    Write-Host "→ $Text" -ForegroundColor Cyan
}

function Write-Warning {
    param([string]$Text)
    Write-Host "⚠ $Text" -ForegroundColor Yellow
}

# Check if running in correct directory
function Check-Directory {
    if (-not (Test-Path ".\scripts\run_minimal_real_grid_search.py")) {
        Write-Error "Not in the auto-RAG project directory!"
        Write-Info "Please run this script from: C:\Users\alger\Documents\000. Projects\auto-RAG"
        exit 1
    }
    Write-Success "Found auto-RAG project files"
}

# Install dependencies
function Install-Dependencies {
    Write-Title "Installing Dependencies"

    $packages = @(
        "loguru",
        "numpy",
        "openai",
        "datasets",
        "chromadb",
        "tiktoken",
        "ragas",
        "langchain"
    )

    Write-Info "Installing required Python packages..."

    foreach ($package in $packages) {
        Write-Host "  Installing $package..." -NoNewline
        $output = pip install $package 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✓" -ForegroundColor Green
        } else {
            Write-Host " ✗" -ForegroundColor Red
            Write-Warning "Failed to install $package, continuing..."
        }
    }

    Write-Success "Dependencies installation complete"
}

# Check or set OpenAI API key
function Setup-ApiKey {
    param([string]$ProvidedKey)

    Write-Title "API Key Setup"

    # Check if key was provided as parameter
    if ($ProvidedKey) {
        $env:OPENAI_API_KEY = $ProvidedKey
        Write-Success "API key set from parameter"
        return $ProvidedKey
    }

    # Check if key exists in environment
    if ($env:OPENAI_API_KEY) {
        Write-Success "Found existing OPENAI_API_KEY in environment"
        $masked = $env:OPENAI_API_KEY.Substring(0, 10) + "..." + $env:OPENAI_API_KEY.Substring($env:OPENAI_API_KEY.Length - 4)
        Write-Info "Current key: $masked"

        $response = Read-Host "Use existing key? (Y/n)"
        if ($response -ne 'n' -and $response -ne 'N') {
            return $env:OPENAI_API_KEY
        }
    }

    # Ask for new key
    Write-Info "Enter your OpenAI API key (or press Enter to run in mock mode):"
    $key = Read-Host "API Key"

    if ($key) {
        $env:OPENAI_API_KEY = $key
        Write-Success "API key set successfully"

        # Ask if user wants to save it permanently
        $save = Read-Host "Save API key to user environment variables? (y/N)"
        if ($save -eq 'y' -or $save -eq 'Y') {
            [System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $key, "User")
            Write-Success "API key saved to user environment variables"
        }
    } else {
        Write-Warning "No API key provided - will run in mock mode"
        Write-Info "Mock mode uses simulated responses instead of real API calls"
    }

    return $key
}

# Run simulated grid search
function Run-SimulatedSearch {
    Write-Title "Running Simulated Grid Search"

    Write-Info "This runs a simulated grid search with no API calls"
    Write-Info "Testing 27 configurations (3x3x3 grid)..."

    python scripts\demo_grid_search_simulation.py

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Simulated grid search completed successfully"
        Write-Info "Results saved to: simulated_grid_search_results.json"
    } else {
        Write-Error "Simulated grid search failed"
    }
}

# Run minimal real grid search
function Run-MinimalRealSearch {
    Write-Title "Running Minimal Real Grid Search"

    if ($env:OPENAI_API_KEY) {
        Write-Info "Running with OpenAI API (estimated cost: ~$0.04)"
    } else {
        Write-Warning "Running in mock mode (no API calls)"
    }

    Write-Info "Configuration:"
    Write-Host "  • Documents: 10" -ForegroundColor Gray
    Write-Host "  • Queries: 3" -ForegroundColor Gray
    Write-Host "  • Configurations: 27 (3x3x3 grid)" -ForegroundColor Gray
    Write-Host "  • Estimated time: 2-3 minutes" -ForegroundColor Gray

    $confirm = Read-Host "`nProceed? (Y/n)"
    if ($confirm -eq 'n' -or $confirm -eq 'N') {
        Write-Warning "Cancelled by user"
        return
    }

    # Run the script and capture output
    Write-Info "Starting grid search..."

    # Auto-respond 'y' to the confirmation prompt in the Python script
    "y" | python scripts\run_minimal_real_grid_search.py

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Minimal grid search completed successfully"
        Write-Info "Results saved to: minimal_real_results.json"

        # Display results summary if file exists
        if (Test-Path "minimal_real_results.json") {
            Write-Title "Results Summary"
            $results = Get-Content "minimal_real_results.json" | ConvertFrom-Json
            if ($results.best) {
                Write-Info "Best Configuration: #$($results.best.config_id)"
                Write-Info "Best Score: $($results.best.score)"
                Write-Host "`nOptimal Parameters:" -ForegroundColor Yellow
                foreach ($param in $results.best.params.PSObject.Properties) {
                    Write-Host "  • $($param.Name): $($param.Value)" -ForegroundColor Gray
                }
            }
        }
    } else {
        Write-Error "Minimal grid search failed"
    }
}

# Run full real grid search
function Run-FullRealSearch {
    Write-Title "Running Full Real Grid Search"

    if (-not $env:OPENAI_API_KEY) {
        Write-Error "Full grid search requires OpenAI API key"
        Write-Info "Please set your API key first"
        return
    }

    Write-Warning "This will use real API calls with higher costs!"
    Write-Info "Configuration:"
    Write-Host "  • Documents: 100" -ForegroundColor Gray
    Write-Host "  • Queries: 20" -ForegroundColor Gray
    Write-Host "  • Configurations: 27 (3x3x3 grid)" -ForegroundColor Gray
    Write-Host "  • Estimated cost: $2-5" -ForegroundColor Yellow
    Write-Host "  • Estimated time: 15-30 minutes" -ForegroundColor Gray

    Write-Warning "This will cost real money. Are you sure?"
    $confirm = Read-Host "Type 'yes' to proceed"
    if ($confirm -ne 'yes') {
        Write-Warning "Cancelled by user"
        return
    }

    Write-Info "Starting full grid search..."
    python scripts\run_real_grid_search.py

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Full grid search completed successfully"
        Write-Info "Results saved to: real_grid_search_results.json"
    } else {
        Write-Error "Full grid search failed"
    }
}

# Run comparison
function Run-Comparison {
    Write-Title "Running All Grid Searches for Comparison"

    Write-Info "This will run all three grid search types in sequence:"
    Write-Host "  1. Simulated (no cost)" -ForegroundColor Gray
    Write-Host "  2. Minimal Real (~$0.04)" -ForegroundColor Gray
    Write-Host "  3. Full Real (~$2-5)" -ForegroundColor Gray

    $confirm = Read-Host "`nProceed with all? (y/N)"
    if ($confirm -ne 'y' -and $confirm -ne 'Y') {
        Write-Warning "Cancelled by user"
        return
    }

    # Run simulated
    Write-Title "Step 1/3: Simulated Grid Search"
    Run-SimulatedSearch
    Start-Sleep -Seconds 2

    # Run minimal
    Write-Title "Step 2/3: Minimal Real Grid Search"
    Run-MinimalRealSearch
    Start-Sleep -Seconds 2

    # Ask before running full
    Write-Title "Step 3/3: Full Real Grid Search"
    Write-Warning "The full search costs $2-5. Skip it? (Y/n)"
    $skip = Read-Host "Skip full search"
    if ($skip -ne 'n' -and $skip -ne 'N') {
        Write-Info "Skipping full search"
    } else {
        Run-FullRealSearch
    }

    # Compare results
    Write-Title "Comparison Summary"

    $files = @(
        @{Name="Simulated"; File="simulated_grid_search_results.json"},
        @{Name="Minimal Real"; File="minimal_real_results.json"},
        @{Name="Full Real"; File="real_grid_search_results.json"}
    )

    foreach ($item in $files) {
        if (Test-Path $item.File) {
            $data = Get-Content $item.File | ConvertFrom-Json
            $best = if ($data.best) { $data.best } else { $data.best_config }

            Write-Host "`n$($item.Name):" -ForegroundColor Yellow
            if ($best) {
                Write-Host "  Best Config: #$($best.config_id)" -ForegroundColor Gray
                Write-Host "  Best Score: $($best.score)" -ForegroundColor Gray

                if ($best.metrics.total_cost) {
                    Write-Host "  Total Cost: `$$($best.metrics.total_cost)" -ForegroundColor Gray
                }
            }
        } else {
            Write-Host "`n$($item.Name): Not run" -ForegroundColor DarkGray
        }
    }
}

# Main menu
function Show-Menu {
    Clear-Host
    Write-Title "AutoRAG Grid Search Runner"

    Write-Host "`nSelect an option:" -ForegroundColor White
    Write-Host "  [1] " -NoNewline -ForegroundColor Cyan; Write-Host "Simulated Grid Search (No API, Free)"
    Write-Host "  [2] " -NoNewline -ForegroundColor Cyan; Write-Host "Minimal Real Grid Search (~`$0.04)"
    Write-Host "  [3] " -NoNewline -ForegroundColor Cyan; Write-Host "Full Real Grid Search (~`$2-5)"
    Write-Host "  [4] " -NoNewline -ForegroundColor Cyan; Write-Host "Run All for Comparison"
    Write-Host "  [5] " -NoNewline -ForegroundColor Cyan; Write-Host "Setup API Key"
    Write-Host "  [6] " -NoNewline -ForegroundColor Cyan; Write-Host "Install Dependencies"
    Write-Host "  [Q] " -NoNewline -ForegroundColor Cyan; Write-Host "Quit"

    Write-Host "`nCurrent Status:" -ForegroundColor White
    if ($env:OPENAI_API_KEY) {
        Write-Success "API Key is set"
    } else {
        Write-Warning "No API Key (will run in mock mode)"
    }

    $choice = Read-Host "`nEnter choice"

    switch ($choice) {
        '1' { Run-SimulatedSearch }
        '2' { Run-MinimalRealSearch }
        '3' { Run-FullRealSearch }
        '4' { Run-Comparison }
        '5' { Setup-ApiKey }
        '6' { Install-Dependencies }
        'Q' { exit }
        'q' { exit }
        default {
            Write-Warning "Invalid choice"
            Start-Sleep -Seconds 2
        }
    }

    Write-Host "`nPress any key to return to menu..." -ForegroundColor DarkGray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Show-Menu
}

# Main execution
Write-Title "AutoRAG Grid Search Runner"

# Check we're in the right directory
Check-Directory

# Install dependencies if requested
if (-not $SkipInstall) {
    Write-Info "Checking dependencies..."
    $checkDeps = Read-Host "Install/update dependencies? (y/N)"
    if ($checkDeps -eq 'y' -or $checkDeps -eq 'Y') {
        Install-Dependencies
    }
}

# Setup API key if provided
if ($ApiKey) {
    Setup-ApiKey -ProvidedKey $ApiKey
}

# Handle different modes
switch ($Mode) {
    "menu" { Show-Menu }
    "simulated" { Run-SimulatedSearch }
    "minimal" { Run-MinimalRealSearch }
    "full" { Run-FullRealSearch }
    "all" { Run-Comparison }
    default { Show-Menu }
}

Write-Host "`nScript completed" -ForegroundColor Green