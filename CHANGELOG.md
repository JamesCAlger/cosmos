# Changelog

All notable changes to the auto-RAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed - 2025-10-01

- **Generator evaluation in multi-component optimization** (`autorag/cosmos/optimization/evaluators.py:426-467`)
  - Fixed critical bug where upstream retriever was not properly configured when used in generator evaluation
  - Issue: Retriever component from optimization doesn't persist embedder/vector store dependencies
  - Symptom: Generator scored 0.0 in 3-component runs due to `ValueError: Embedder and vector store must be set before indexing`
  - Solution: Added automatic embedder and vector store setup for upstream retriever in generator evaluation
  - Impact: Generator now scores correctly (0.71) in full pipeline optimization instead of 0.0
  - Context passing mechanism now fully operational for all component types

### Note to Claude
**When implementing new features or fixes, add entries here under the appropriate section (Added, Changed, Fixed, etc.) with:**
- Brief description of the change
- Affected files (use relative paths from project root)
- Any breaking changes or migration notes
- Date of change

---

## [0.3.0] - Phase 6: COSMOS Framework Complete - 2025-09-30

### Added - Phase 6
- **End-to-End COSMOS Demonstration Script** (`scripts/run_cosmos_optimization.py`)
  - Full compositional optimization workflow
  - Support for multiple datasets (mock, MS MARCO, BEIR)
  - Configurable components, strategies, and budgets
  - Comprehensive CLI with examples and help text
  - Integration with embedding cache for cost reduction
  - Detailed progress reporting and statistics

### Changed - Phase 6
- **Enhanced cache manager** (`autorag/optimization/cache_manager.py`)
  - Added statistics tracking (hits, misses, time saved, cost saved)
  - Improved memory management
  - Better error handling

---

## [0.2.0] - Phase 2-5: COSMOS Framework Implementation - 2025-09-28

### Added - Phase 5: Compositional Optimizer
- **CompositionalOptimizer** (`autorag/cosmos/optimization/compositional_optimizer.py`)
  - Sequential component optimization with context passing
  - Budget allocation across components
  - Integration with optimization strategies
  - Cache-aware optimization
- **CompositionalOptimizerBuilder** (`autorag/cosmos/optimization/__init__.py`)
  - Factory methods for creating optimizers with different strategies

### Added - Phase 4: Optimization Framework
- **Optimization Strategy Interface** (`autorag/cosmos/optimization/strategy.py`)
  - Abstract base class for optimization strategies
  - Unified interface for Bayesian and Random search
- **Bayesian Strategy** (`autorag/cosmos/optimization/bayesian_strategy.py`)
  - Bayesian optimization using scikit-optimize
  - Support for categorical and continuous parameters
- **Random Strategy** (`autorag/cosmos/optimization/random_strategy.py`)
  - Simple random search baseline
- **Optimization Task** (`autorag/cosmos/optimization/task.py`)
  - Task definition for component optimization
  - Parameter space and evaluation function encapsulation

### Added - Phase 3: Component Evaluators
- **Component Evaluators** (`autorag/cosmos/optimization/evaluators.py`)
  - Standalone evaluators for chunker, retriever, and generator
  - Evaluation without full pipeline assembly
  - Support for context passing between components
  - Integration with embedding cache
  - Component builder functions

### Added - Phase 2: COSMOSComponent Wrapper
- **COSMOSComponent Wrapper** (`autorag/cosmos/component_wrapper.py`)
  - Wraps existing components to add `process_with_metrics()` capability
  - Component-specific metric collection during execution
  - No modification to original component implementations
  - Metrics history tracking

### Added - Phase 1: Component Metrics
- **ComponentMetrics** (`autorag/cosmos/metrics/component_metrics.py`)
  - Component-intrinsic metrics for chunker, retriever, and generator
  - Semantic similarity evaluation using sentence transformers
  - Metrics computation without full pipeline context
  - Support for optional ground truth (precision@k for retrieval)
- **Dataset Loader Registry** (`autorag/data/`)
  - Unified dataset loading interface
  - Support for MS MARCO and BEIR datasets
  - Query and document data classes
  - Dataset registry for easy extension

### Changed - Phase 2-5
- **Project Structure**: Organized COSMOS framework in `autorag/cosmos/` module
- **Embedding Cache**: Extended to support COSMOS optimization workflow
- **Semantic Metrics**: Updated to use `all-MiniLM-L6-v2` as default for component metrics

---

## [0.1.0] - Phase 1: Bayesian Optimization Framework - 2025-09-25

### Added - Bayesian Optimization
- **Bayesian Optimization with Caching** (`scripts/bayesian_with_cache/run_optimization.py`)
  - Intelligent hyperparameter search using scikit-optimize
  - Support for full RAG pipeline configuration optimization
  - Integration with MS MARCO dataset
  - Comprehensive evaluation metrics
- **SimpleBayesianOptimizer** (`autorag/optimization/bayesian_search.py`)
  - Bayesian optimization for discrete and continuous parameters
  - Support for multi-objective optimization
  - Result tracking and history
- **EmbeddingCacheManager** (`autorag/optimization/cache_manager.py`)
  - Disk-based caching for embeddings
  - Automatic cache invalidation
  - Memory limit enforcement
  - Cost and time savings tracking
- **CachedEmbedder** (`autorag/components/embedders/cached.py`)
  - Transparent caching layer for any embedder
  - Integration with EmbeddingCacheManager
  - Automatic cache key generation
- **Semantic Metrics** (`autorag/evaluation/semantic_metrics.py`)
  - Semantic similarity using sentence transformers
  - Support for multiple similarity models
  - Default: `gte-large-en-v1.5` for high-quality embeddings
- **External Metrics Collector** (`autorag/evaluation/external_metrics.py`)
  - Multi-metric evaluation (semantic similarity, ROUGE, retrieval metrics)
  - Aggregation strategies (mean, median)
  - Cost tracking

### Added - RAG Components
- **Hybrid Retriever** (`autorag/components/retrievers/hybrid.py`)
  - Combines dense and sparse retrieval
  - Configurable weighting and fusion methods
- **Dense Retriever** (`autorag/components/retrievers/dense.py`)
  - Vector similarity search
  - Integration with embedders and vector stores
- **BM25 Retriever** (`autorag/components/retrievers/bm25.py`)
  - Sparse keyword-based retrieval
- **Cross-Encoder Reranker** (`autorag/components/rerankers/cross_encoder.py`)
  - Document reranking using cross-encoder models
  - Score normalization
- **Advanced Chunkers**:
  - Semantic chunking (`autorag/components/chunkers/semantic.py`)
  - Sliding window (`autorag/components/chunkers/sliding_window.py`)
  - Hierarchical chunking (`autorag/components/chunkers/hierarchical.py`)
  - Document-aware chunking (`autorag/components/chunkers/document_aware.py`)

### Changed - Bayesian Phase
- **OpenAI Generator**: Added rate limiting for free tier (500 RPM)
  - 0.15s delay between calls
  - Exponential backoff retry logic
  - Shared client instance
  - Timeout handling
- **Environment Loading**: All scripts now use `python-dotenv` to load `.env` automatically

### Fixed - Bayesian Phase
- **API Key Loading**: Python scripts now reliably load API keys from `.env` file
- **Rate Limit Errors**: Implemented delays and retry logic to handle OpenAI rate limits
- **Connection Pooling**: Fixed issues with multiple generator instances by sharing client

---

## [0.0.1] - Initial Release - 2025-09-15

### Added - Initial Implementation
- **Core RAG Pipeline** (`autorag/pipeline/rag_pipeline.py`)
  - Modular pipeline architecture
  - Component registry system
  - Support for chunking, embedding, retrieval, and generation
- **Grid Search Optimization** (`autorag/optimization/grid_search.py`)
  - Exhaustive search through parameter combinations
  - Result tracking and comparison
- **Basic Components**:
  - Fixed-size chunker (`autorag/components/chunkers/fixed_size.py`)
  - Mock chunker, embedder, generator for testing
  - OpenAI embedder and generator
  - Simple vector store (`autorag/components/vector_stores/simple.py`)
- **Evaluation Framework** (`autorag/evaluation/`)
  - Traditional metrics (ROUGE, BLEU)
  - RAGAS evaluator
  - Progressive evaluation
- **MS MARCO Integration** (`autorag/datasets/msmarco_loader.py`)
  - Dataset loading and preprocessing
  - Query and document extraction
- **Configuration System** (`autorag/config/loader.py`)
  - YAML-based configuration
  - Component auto-registration
- **Testing Suite**:
  - Unit tests for components
  - Integration tests for pipeline
  - Test scripts for validation

### Infrastructure
- **Environment Configuration**: `.env` file support for API keys
- **Logging**: Loguru integration for structured logging
- **Requirements**: Comprehensive dependency list in `requirements.txt`
- **Documentation**: Initial README with project overview and usage

---

## Template for Future Entries

When adding new changes, use this template:

```markdown
## [X.Y.Z] - Feature Name - YYYY-MM-DD

### Added
- **Feature Name** (`path/to/file.py`)
  - Description of what was added
  - Key capabilities
  - Related files

### Changed
- **Component Name** (`path/to/file.py`)
  - What was changed and why
  - Breaking changes (if any)
  - Migration notes (if needed)

### Fixed
- **Bug Description** (`path/to/file.py:line`)
  - What was broken
  - How it was fixed
  - Related issues

### Removed
- **Deprecated Feature** (`path/to/file.py`)
  - What was removed and why
  - Migration path (if applicable)
```

---

**Note**: This changelog is maintained by Claude and should be updated with every significant change to the codebase. Users can reference this file to understand the evolution of the project and track new features.
