# Documentation Index

**Quick Start**: New to the project? Read `CLAUDE.md` (root) for setup and overview.

---

## 📚 Documentation Files

| Doc | What It Covers | When to Read |
|-----|----------------|--------------|
| **`CLAUDE.md`** (root) | Project setup, running optimizations, common tasks | First read, general usage |
| **`autorag/components/CLAUDE.md`** | Component architecture, base classes, design patterns | Adding/modifying components |
| **`autorag/cosmos/CLAUDE.md`** | COSMOS framework, sequential optimization, adding components | Understanding/extending COSMOS |
| **`autorag/components/rerankers/CLAUDE.md`** | Reranker specifics, when to use, integration | Working with rerankers |

---

## 📁 Key Directories

### Core Framework
- **`autorag/components/`** - All RAG components
  - `base.py` - Abstract base classes (Chunker, Retriever, Generator, Reranker, etc.)
  - `chunkers/` - Text chunking strategies (fixed, semantic, sliding)
  - `embedders/` - Embedding models (OpenAI, cached, mock)
  - `retrievers/` - Retrieval methods (dense, BM25, hybrid)
  - `rerankers/` - Document reranking (cross-encoder)
  - `generators/` - Answer generation (OpenAI, mock)
  - `vector_stores/` - Vector storage (simple, FAISS)

- **`autorag/cosmos/`** - COSMOS optimization framework
  - `component_wrapper.py` - COSMOSComponent wrapper for metrics
  - `metrics/` - Component-intrinsic metrics
  - `optimization/` - Compositional optimizer, evaluators, strategies

### Optimization & Evaluation
- **`autorag/optimization/`** - Bayesian optimization framework
  - `bayesian_search.py` - Core Bayesian optimizer
  - `cache_manager.py` - Embedding cache manager
  - `search_space.py` - Search space definitions

- **`autorag/evaluation/`** - Evaluation metrics
  - `semantic_metrics.py` - Semantic similarity
  - `external_metrics.py` - Multi-metric evaluation
  - `ragas_evaluator.py` - RAGAS metrics

### Pipeline & Data
- **`autorag/pipeline/`** - Pipeline orchestration
  - `rag_pipeline.py` - Full RAG pipeline
  - `simple_rag.py` - Simplified RAG for testing

- **`autorag/data/`** - Dataset handling
  - `loaders.py` - MS MARCO, BEIR dataset loaders
  - `registry.py` - Dataset registry

### Scripts
- **`scripts/run_cosmos_optimization.py`** - Main COSMOS demo script
- **`scripts/bayesian_with_cache/`** - Bayesian optimization with caching
- **`scripts/run_minimal_real_grid_search.py`** - Grid search baseline

---

## 🎯 Quick Task Lookup

| I Want To... | Start Here |
|--------------|------------|
| Set up the project | `CLAUDE.md` → Environment Setup |
| Run optimizations | `CLAUDE.md` → Running Optimizations (COSMOS or Bayesian) |
| Add a new chunker/retriever/generator | `autorag/components/CLAUDE.md` |
| Understand how COSMOS works | `autorag/cosmos/CLAUDE.md` |
| Add a component to COSMOS | `autorag/cosmos/CLAUDE.md` → "How to Add a New Component Type" |
| Integrate reranker | `autorag/components/rerankers/CLAUDE.md` |
| Modify search spaces | `scripts/run_cosmos_optimization.py` (COSMOS) or `scripts/bayesian_with_cache/run_optimization.py` (Bayesian) |
| Change metrics | `autorag/cosmos/metrics/component_metrics.py` |
| Add new optimization strategy | `autorag/cosmos/optimization/` (create new strategy class) |

---

## 🗂️ Component Type Reference

| Component | Base Class | Implementations | Doc Location |
|-----------|------------|-----------------|--------------|
| **Chunker** | `BaseChunker` | FixedSize, Semantic, SlidingWindow | `autorag/components/CLAUDE.md` |
| **Embedder** | `BaseEmbedder` | OpenAI, Cached, Mock | `autorag/components/CLAUDE.md` |
| **Retriever** | `BaseRetriever` | Dense, BM25, Hybrid | `autorag/components/CLAUDE.md` |
| **Reranker** | `BaseReranker` | CrossEncoder | `autorag/components/rerankers/CLAUDE.md` |
| **Generator** | `BaseGenerator` | OpenAI, Mock | `autorag/components/CLAUDE.md` |
| **VectorStore** | `BaseVectorStore` | Simple, FAISS | `autorag/components/CLAUDE.md` |

---

**Last Updated**: 2025-10-06
