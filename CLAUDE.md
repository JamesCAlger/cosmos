# Claude Code Configuration Guide

## Project: auto-RAG System

### OpenAI API Configuration
- **API Key Location**: The OpenAI API key is stored in the `.env` file in the project root
- **File Path**: `C:\Users\alger\Documents\000. Projects\auto-RAG\.env`
- **Key Variable**: `OPENAI_API_KEY`

### Important Notes for Grid Search

When running the grid search scripts, the API key needs to be properly loaded:

1. **PowerShell Script Issue**: The `run_minimal_grid_search.ps1` script reads the `.env` file but the Python subprocess doesn't always inherit the environment variable properly.

2. **Direct Python Execution**: For reliable API key usage, run the Python script directly:
   ```bash
   cd "C:\Users\alger\Documents\000. Projects\auto-RAG"
   python scripts/run_minimal_real_grid_search.py
   ```

3. **Mock Mode**: If the API key isn't detected, the system automatically falls back to mock mode with pre-defined responses.

### Testing Commands

**Lint/Type Check Commands**:
- Not currently configured. Ask user for the specific commands if code quality checks are needed.

**Run Grid Search**:
```powershell
# With PowerShell script (may not load API key properly)
.\run_minimal_grid_search.ps1

# Direct Python (recommended for API usage)
python scripts/run_minimal_real_grid_search.py
```

### Known Issues - RESOLVED

1. **API Key Loading**: ~~The PowerShell script doesn't always pass the API key to the Python subprocess correctly.~~ **FIXED**: Added `dotenv` loading to the Python script.

2. **Rate Limiting**: **FIXED**: Implemented proper rate limiting for OpenAI free tier (500 RPM)
   - Delay between calls: 0.15 seconds (safe margin above 0.12s minimum)
   - Shared client across all generator instances to avoid connection pooling issues
   - Retry logic with exponential backoff for rate limit errors
   - 20-second timeout for API calls with max 2 retries

3. **Solution**: The script now automatically loads the `.env` file and respects rate limits:
   - PowerShell script: `.\run_minimal_grid_search.ps1`
   - Python script directly: `python scripts/run_minimal_real_grid_search.py`
   - Both methods will now properly load the API key from `.env` and handle rate limiting

### File Structure
```
auto-RAG/
├── .env                                    # Contains OPENAI_API_KEY
├── run_minimal_grid_search.ps1            # PowerShell wrapper script
├── scripts/
│   └── run_minimal_real_grid_search.py    # Main grid search implementation
├── minimal_real_results.json              # Grid search results
└── CLAUDE.md                               # This file
```