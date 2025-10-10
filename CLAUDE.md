# Claude Code Configuration - auto-RAG System

## Project Overview

**auto-RAG** is an advanced Retrieval-Augmented Generation (RAG) system with two sophisticated optimization approaches:

1. **Bayesian Optimization** (Phase 1): Direct hyperparameter optimization of complete RAG pipelines using Bayesian search with embedding caching.
2. **COSMOS Framework** (Phase 2 - Current): Compositional optimization that breaks the circular dependency problem by optimizing components sequentially with context passing.

### Evolution Timeline
- **Initial Implementation**: Basic RAG with grid search
- **Phase 1 Complete**: Bayesian optimization with caching (`scripts/bayesian_with_cache/run_optimization.py`)
- **Phase 2-6 Complete**: Full COSMOS framework implementation
  - Phase 1: Component Metrics Extraction
  - Phase 2: COSMOSComponent Wrapper
  - Phase 3: Component Evaluators
  - Phase 4: Optimization Framework
  - Phase 5: Compositional Optimizer
  - Phase 6: End-to-End Demonstration Script

### Current Status
✅ COSMOS framework fully operational with RAG as test case
✅ Both Bayesian and Random search strategies implemented
✅ Component-intrinsic metrics with caching support
🔄 Framework ready for generalization to other architectures

---

## 📚 Hierarchical Documentation

This project uses **focused CLAUDE.md files in subfolders** for efficient context management. When working on a specific area, **read the relevant subfolder CLAUDE.md first** for architectural context before diving into code.

### Documentation Navigation Guide

**Start here**: You're reading the right doc! This file covers project setup, running optimizations, and common tasks.

**What are you trying to do?**

```
┌────────────────────────────────────────────────────────────────┐
│ TASK                              → READ THIS FIRST            │
├────────────────────────────────────────────────────────────────┤
│ Setup project & install deps     → Root CLAUDE.md (below)     │
│ Run optimizations (COSMOS/Bayes) → Root CLAUDE.md (below)     │
│ Understand project structure     → Root CLAUDE.md (below)     │
├────────────────────────────────────────────────────────────────┤
│ Add new component implementation → autorag/components/        │
│ Understand component architecture → autorag/components/        │
│ Learn design patterns (wrapper)  → autorag/components/        │
├────────────────────────────────────────────────────────────────┤
│ Understand COSMOS framework      → autorag/cosmos/            │
│ Add component to COSMOS          → autorag/cosmos/            │
│ Debug optimization issues        → autorag/cosmos/            │
├────────────────────────────────────────────────────────────────┤
│ Work with rerankers specifically → autorag/components/rerankers│
│ Integrate reranker into COSMOS  → autorag/cosmos/ first, then │
│                                    autorag/components/rerankers│
└────────────────────────────────────────────────────────────────┘
```

**Example: Adding Reranker to COSMOS**
1. `autorag/cosmos/CLAUDE.md` → understand sequential optimization & context passing
2. `autorag/components/rerankers/CLAUDE.md` → understand reranker purpose & metrics
3. Jump to specific files with full architectural context

### Available Documentation

- **Root `CLAUDE.md`** (this file): Project overview, setup, running optimizations, common tasks
- **`autorag/components/CLAUDE.md`**: Component architecture, base classes, data structures, design patterns
- **`autorag/cosmos/CLAUDE.md`**: COSMOS framework, sequential optimization, adding new component types
- **`autorag/components/rerankers/CLAUDE.md`**: Reranker specifics, when to use, integration guide

### Benefits of Hierarchical Docs

- **80% context reduction**: Read 2-5KB of targeted docs instead of scanning 50-100KB of code
- **Faster onboarding**: Understand architectural decisions and patterns in minutes, not hours
- **Better maintainability**: Documentation lives next to the code it describes

---

## Project Structure

```
auto-RAG/
├── autorag/
│   ├── components/           # RAG components (chunkers, retrievers, generators, embedders)
│   │   ├── base.py           # Base classes for all components
│   │   ├── chunkers/         # Text chunking strategies (fixed, semantic, sliding)
│   │   ├── embedders/        # Embedding models (OpenAI, cached, mock)
│   │   ├── generators/       # Answer generators (OpenAI, mock)
│   │   ├── retrievers/       # Retrieval methods (dense, BM25, hybrid)
│   │   ├── rerankers/        # Document reranking (cross-encoder)
│   │   └── vector_stores/    # Vector storage backends (simple, FAISS)
│   │
│   ├── cosmos/               # 🌟 COSMOS Framework - Compositional Optimization
│   │   ├── component_wrapper.py         # COSMOSComponent wrapper for metrics
│   │   ├── metrics/
│   │   │   └── component_metrics.py     # Component-intrinsic metrics
│   │   └── optimization/
│   │       ├── compositional_optimizer.py  # Main orchestrator
│   │       ├── evaluators.py            # Component evaluators
│   │       ├── strategy.py              # Base optimization strategy
│   │       ├── bayesian_strategy.py     # Bayesian search strategy
│   │       ├── random_strategy.py       # Random search strategy
│   │       └── task.py                  # Optimization task definition
│   │
│   ├── optimization/         # Bayesian optimization framework
│   │   ├── bayesian_search.py          # Core Bayesian optimizer
│   │   ├── cache_manager.py            # Embedding cache manager
│   │   ├── grid_search.py              # Grid search implementation
│   │   └── search_space.py             # Search space definitions
│   │
│   ├── evaluation/           # Evaluation metrics and services
│   │   ├── semantic_metrics.py         # Semantic similarity (gte-large-en-v1.5)
│   │   ├── external_metrics.py         # Multi-metric evaluation
│   │   └── ragas_evaluator.py          # RAGAS metrics
│   │
│   ├── pipeline/             # Pipeline orchestration
│   │   ├── rag_pipeline.py             # Full RAG pipeline
│   │   └── simple_rag.py               # Simplified RAG for testing
│   │
│   └── data/                 # Dataset loaders
│       ├── loaders.py                  # MS MARCO, BEIR dataset loaders
│       └── registry.py                 # Dataset registry
│
├── scripts/
│   ├── run_cosmos_optimization.py         # 🌟 Main COSMOS demo script
│   ├── bayesian_with_cache/
│   │   └── run_optimization.py            # Bayesian optimization with caching
│   └── run_minimal_real_grid_search.py    # Grid search baseline
│
├── .env                      # Environment variables (OPENAI_API_KEY)
├── requirements.txt          # Python dependencies
├── CLAUDE.md                 # This file
└── CHANGELOG.md              # 📝 Feature changelog (updated by Claude)
```

---

## Environment Setup

### Required Environment Variables

**Location**: `.env` file in project root
**Required Variables**:
- `OPENAI_API_KEY` - Your OpenAI API key for embeddings and generation

**Example `.env`**:
```bash
OPENAI_API_KEY=sk-...your-key-here...
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Running Optimizations

### 🌟 COSMOS Framework (Recommended)

**Main Script**: `scripts/run_cosmos_optimization.py`

The COSMOS framework optimizes RAG components sequentially, breaking circular dependencies:

```bash
# Basic usage - optimize chunker and retriever with mock data
python scripts/run_cosmos_optimization.py --components chunker retriever --budget 15

# Full pipeline optimization with MS MARCO dataset
python scripts/run_cosmos_optimization.py \
  --components chunker retriever generator \
  --strategy bayesian \
  --budget 30 \
  --dataset marco \
  --num-docs 50 \
  --num-queries 10

# Use BEIR SciFact dataset
python scripts/run_cosmos_optimization.py \
  --components retriever generator \
  --dataset beir/scifact \
  --budget 20 \
  --num-docs 100

# Disable caching (not recommended)
python scripts/run_cosmos_optimization.py \
  --components chunker retriever \
  --no-cache \
  --budget 15
```

**Available Datasets**:
- `mock` - Built-in test data (default)
- `marco` - MS MARCO v2.1
- `beir/scifact` - BEIR SciFact dataset
- `beir/nfcorpus` - BEIR NFCorpus dataset

**Optimization Strategies**:
- `random` - Random search (default, faster)
- `bayesian` - Bayesian optimization (smarter, but requires more initial points)

**Key Parameters**:
- `--components`: Which components to optimize (chunker, retriever, generator)
- `--budget`: Total evaluation budget across all components
- `--strategy`: Optimization strategy (random or bayesian)
- `--num-docs`: Number of documents from dataset
- `--num-queries`: Number of test queries per evaluation
- `--dataset`: Dataset to use for optimization
- `--use-cache` / `--no-cache`: Enable/disable embedding cache (default: enabled)

**Output**: Results saved to `cosmos_results.json`

### Bayesian Optimization (Original Approach)

**Main Script**: `scripts/bayesian_with_cache/run_optimization.py`

Traditional Bayesian optimization of complete pipeline configurations:

```bash
# Run with real API and caching
python scripts/bayesian_with_cache/run_optimization.py \
  --n-calls 20 \
  --real-api \
  --num-docs 50 \
  --num-queries 10 \
  --cache-dir .embedding_cache

# Clear cache and start fresh
python scripts/bayesian_with_cache/run_optimization.py \
  --n-calls 20 \
  --real-api \
  --clear-cache
```

**Key Parameters**:
- `--n-calls`: Number of configurations to evaluate
- `--real-api`: Use real OpenAI API (otherwise uses mock)
- `--num-docs`: Number of documents from MS MARCO
- `--num-queries`: Number of queries per configuration
- `--cache-dir`: Directory for embedding cache
- `--clear-cache`: Clear cache before starting

**Output**: Results saved to `bayesian_with_cache_results.json`

---

## Important Implementation Notes

### Rate Limiting (OpenAI API)

The system implements comprehensive rate limiting for OpenAI's free tier (500 RPM):
- **Delay**: 0.15s between calls (safe margin above 0.12s minimum)
- **Shared client**: Single client instance across all generators
- **Retry logic**: Exponential backoff on rate limit errors
- **Timeout**: 20-second timeout with max 2 retries

**Location**: `autorag/components/generators/openai.py:120-150`

### Embedding Cache

Embedding caching significantly reduces API costs and speeds up optimization:
- **Cache manager**: `autorag/optimization/cache_manager.py`
- **Cached embedder**: `autorag/components/embedders/cached.py`
- **Default location**: `.embedding_cache/` or `.embedding_cache_marco/`
- **Benefits**: ~50-80% reduction in API calls during optimization

**Cache is automatically used in**:
- COSMOS optimization (via `--use-cache` flag, enabled by default)
- Bayesian optimization (via `--cache-dir` parameter)

### Component-Intrinsic Metrics (COSMOS)

COSMOS uses component-specific metrics that don't require full pipeline context:

**Chunking Metrics**:
- Chunk count, average length, size variance
- Optional: semantic coherence within chunks

**Retrieval Metrics**:
- Latency, retrieval rate
- Coverage score (document utilization)
- Optional: precision@k (requires ground truth)

**Generation Metrics**:
- Semantic similarity to ground truth
- Generation latency
- Answer length

**Location**: `autorag/cosmos/metrics/component_metrics.py`

### Search Spaces

**COSMOS Framework** (`scripts/run_cosmos_optimization.py:146-173`):
```python
{
    'chunker': {
        'chunking_strategy': ['fixed', 'semantic'],
        'chunk_size': [128, 256, 512],
        'overlap': [0, 25, 50]
    },
    'retriever': {
        'retrieval_method': ['sparse', 'dense'],
        'retrieval_top_k': [3, 5, 7]
    },
    'generator': {
        'use_real_api': [True],
        'temperature': [0.3, 0.5, 0.7]
    }
}
```

**Bayesian Optimization** (`scripts/bayesian_with_cache/run_optimization.py:360-369`):
```python
{
    'chunking_strategy': ['fixed', 'semantic'],
    'chunk_size': [128, 256, 512],
    'retrieval_method': ['bm25', 'dense', 'hybrid'],
    'retrieval_top_k': [3, 5, 10],
    'reranking_enabled': [False, True],
    'temperature': [0.3, 0.7, 1.0],
    'max_tokens': [256, 512]
}
```

---

## Testing and Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v tests/
```

### Code Quality

Currently not configured. For linting/formatting:
```bash
# Format code
black autorag/ scripts/

# Type checking (if configured)
mypy autorag/
```

---

## Common Tasks

### Adding a New Component

1. **Create component class** inheriting from base:
   - Chunkers: `autorag/components/base.py:BaseChunker`
   - Retrievers: `autorag/components/base.py:BaseRetriever`
   - Generators: `autorag/components/base.py:BaseGenerator`

2. **Implement required methods**:
   - Chunker: `chunk(documents) -> List[Chunk]`
   - Retriever: `retrieve(query, top_k) -> List[QueryResult]`
   - Generator: `generate(query, context) -> str`

3. **Add to COSMOS search space** if needed:
   - Edit `scripts/run_cosmos_optimization.py:define_search_spaces()`

4. **Update CHANGELOG.md** with the new feature

### Extending to New Architectures

The COSMOS framework is designed for generalization. To apply to non-RAG architectures:

1. **Define components** for your architecture (e.g., encoder, decoder, attention)
2. **Implement component metrics** in `autorag/cosmos/metrics/`
3. **Create component evaluators** in `autorag/cosmos/optimization/evaluators.py`
4. **Update search spaces** in your optimization script
5. **Run COSMOS optimization** with your components

**Reference implementation**: Current RAG implementation in `scripts/run_cosmos_optimization.py`

---

## Best Practices

### When Using COSMOS

1. **Start small**: Use mock data or small datasets first (`--num-docs 10 --num-queries 3`)
2. **Enable caching**: Always use `--use-cache` (default) for real API calls
3. **Budget allocation**: For 3 components, budget of 30-45 is reasonable (10-15 per component)
4. **Component order**: Optimize in forward order (chunker → retriever → generator)
5. **Strategy selection**:
   - Use `random` for quick experiments or small search spaces
   - Use `bayesian` for larger search spaces where sample efficiency matters

### When Using Bayesian Optimization

1. **Use caching**: Always specify `--cache-dir` to avoid redundant API calls
2. **API key**: Ensure `OPENAI_API_KEY` is set in `.env` when using `--real-api`
3. **Start small**: Test with `--n-calls 10 --num-queries 5` before scaling up
4. **Monitor rate limits**: Watch for rate limit errors and adjust delays if needed

### Cost Management

1. **Use mock mode** for development: Don't pass `--real-api` flag
2. **Enable caching**: Reduces API calls by 50-80% in optimization runs
3. **Start with small datasets**: Use `--num-docs 10-20` for initial experiments
4. **Limit queries**: Use `--num-queries 3-5` for quick iterations

---

## Changelog

**Important**: All new features, bug fixes, and significant changes should be documented in `CHANGELOG.md`. Claude will automatically update this file when implementing changes.

**See**: [CHANGELOG.md](./CHANGELOG.md) for detailed project history.

---

## Known Issues & Solutions

### ✅ RESOLVED: API Key Loading
**Issue**: PowerShell scripts don't always pass environment variables to Python subprocesses
**Solution**: Python scripts now use `python-dotenv` to load `.env` directly

### ✅ RESOLVED: Rate Limiting
**Issue**: OpenAI free tier rate limits (500 RPM) cause API failures
**Solution**: Implemented 0.15s delays, retry logic, and shared client instances

### ✅ RESOLVED: Embedding Costs
**Issue**: Repeated embeddings during optimization are expensive
**Solution**: Implemented `EmbeddingCacheManager` with disk-based caching

### Current Considerations

- **Large datasets**: Memory usage can be high with 200+ documents and caching enabled
- **Long optimization runs**: Bayesian optimization with 50+ calls can take 1-2 hours with real API
- **Mock vs Real API**: Mock mode is useful for testing but won't reflect real performance

---

## Quick Reference

### Most Common Commands

```bash
# COSMOS optimization (recommended workflow)
python scripts/run_cosmos_optimization.py --components chunker retriever --budget 15

# With real dataset
python scripts/run_cosmos_optimization.py --components chunker retriever generator \
  --dataset marco --num-docs 50 --budget 30

# Bayesian optimization (original approach)
python scripts/bayesian_with_cache/run_optimization.py --n-calls 20 --real-api
```

### Key File Locations

- **API Key**: `.env` (create from `.env.example`)
- **Main COSMOS Script**: `scripts/run_cosmos_optimization.py`
- **Bayesian Script**: `scripts/bayesian_with_cache/run_optimization.py`
- **Component Base Classes**: `autorag/components/base.py`
- **COSMOS Framework**: `autorag/cosmos/`
- **Optimization Results**: `cosmos_results.json`, `bayesian_with_cache_results.json`

---

## Additional Resources

- **README.md**: High-level project overview and results
- **CHANGELOG.md**: Detailed feature and change history
- **Git History**: Phase-by-phase commit history showing framework evolution

---

**Last Updated**: 2025-10-01
**Project Phase**: COSMOS Framework (Phase 6 Complete)
**Next Steps**: Generalize COSMOS to other architectures beyond RAG
