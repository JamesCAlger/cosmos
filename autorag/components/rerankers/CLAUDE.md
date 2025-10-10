# Rerankers - Component Architecture

## When to Read This

**Read this doc if you're**:
- ✅ Understanding what rerankers are and when to use them
- ✅ Integrating reranker into your RAG pipeline
- ✅ Adding reranker support to COSMOS framework
- ✅ Comparing cross-encoder vs bi-encoder approaches

**Skip this doc if you're**:
- ❌ Not using reranking in your pipeline
- ❌ Just understanding general component architecture → see `autorag/components/CLAUDE.md`
- ❌ Understanding COSMOS first → see `autorag/cosmos/CLAUDE.md`

**Prerequisites**:
- Understanding of retrieval basics (embeddings, similarity search)
- Familiarity with RAG pipeline flow

**Recommended reading order for COSMOS integration**:
1. Read `autorag/cosmos/CLAUDE.md` first → understand framework
2. Read this doc → understand reranker specifics
3. Follow integration steps in both docs

---

## Purpose

Rerankers **improve retrieval quality** by re-scoring retrieved documents using more sophisticated models than the initial retriever. They sit **between retrieval and generation** in the RAG pipeline.

## Why Reranking?

**Problem**: Fast retrievers (BM25, dense embeddings) prioritize speed over accuracy
**Solution**: Rerankers use slower but more accurate cross-encoders to refine top-k results

**Trade-off**: ~100-500ms latency increase for 10-30% relevance improvement

## Architecture

### Base Class

**File**: `autorag/components/base.py` → `Reranker` class (abstract base)

```python
class Reranker(Component):
    @abstractmethod
    def rerank(self, query: str, results: List[QueryResult], top_k: int = 5) -> List[QueryResult]:
        """Rerank retrieval results"""
        pass
```

**Key method**: `rerank(query, results, top_k)` takes retriever output and returns reranked results

### Implementations

#### `CrossEncoderReranker` (`cross_encoder.py`)
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
- **Method**: Computes relevance scores for (query, document) pairs jointly
- **Normalization**: Applies sigmoid to convert scores to [0, 1] range
- **Metadata**: Preserves original scores, adds `reranker_score` and `reranker_model`

**Configuration parameters**:
- `model_name`: Cross-encoder model path
- `normalize_scores`: Apply sigmoid normalization (default: True)
- `batch_size`: Batch size for prediction (default: 32)
- `device`: CPU or GPU (default: "cpu")

## Example Usage

### Basic Usage

```python
from autorag.components.rerankers.cross_encoder import CrossEncoderReranker
from autorag.components.retrievers.dense import DenseRetriever

# Initialize reranker
reranker = CrossEncoderReranker({
    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'normalize_scores': True
})

# Over-retrieve candidates
results = retriever.retrieve(query, top_k=10)  # Get 10 candidates

# Rerank to refine results
reranked = reranker.rerank(query, results, top_k=5)  # Refine to top 5

# Use reranked results for generation
answer = generator.generate(query, reranked)
```

### Impact Example: Before vs After Reranking

```python
# Query: "What causes climate change?"

# BEFORE RERANKING (retriever scores - dense embeddings):
# Results sorted by cosine similarity:
# 1. Doc A: 0.72 - "Climate change refers to long-term shifts..." (relevant ✓)
# 2. Doc B: 0.68 - "Climate models use temperature data..." (tangential ~)
# 3. Doc C: 0.65 - "Greenhouse gases trap heat in atmosphere..." (highly relevant ✓✓)
# 4. Doc D: 0.63 - "Weather patterns vary seasonally..." (not relevant ✗)
# 5. Doc E: 0.61 - "Carbon dioxide emissions from industry..." (relevant ✓)

# AFTER RERANKING (cross-encoder scores - joint query-doc encoding):
# Results re-scored and re-sorted:
# 1. Doc C: 0.89 - "Greenhouse gases trap heat..." (promoted from #3!)
# 2. Doc E: 0.84 - "Carbon dioxide emissions..." (promoted from #5!)
# 3. Doc A: 0.78 - "Climate change refers to..." (stayed relevant)
# 4. Doc B: 0.45 - "Climate models use..." (demoted, less relevant)
# 5. Doc D: 0.21 - "Weather patterns vary..." (demoted, not relevant)

# Result: Most relevant docs (C, E) now at top for better answer generation
```

### Complete Pipeline Example

```python
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.components.retrievers.dense import DenseRetriever
from autorag.components.rerankers.cross_encoder import CrossEncoderReranker
from autorag.components.generators.openai import OpenAIGenerator

# Initialize pipeline with reranker
chunker = FixedSizeChunker({'chunk_size': 256, 'overlap': 50})
retriever = DenseRetriever({'top_k': 10})  # Over-retrieve for reranking
reranker = CrossEncoderReranker({'normalize_scores': True})
generator = OpenAIGenerator({'model': 'gpt-3.5-turbo'})

# Index documents
chunks = chunker.chunk(documents)
retriever.index(chunks)

# Query with reranking
query = "What causes climate change?"
candidates = retriever.retrieve(query, top_k=10)  # Over-retrieve
refined = reranker.rerank(query, candidates, top_k=5)  # Refine
answer = generator.generate(query, refined)
```

## Pipeline Integration

### Flow Position
```
Retriever → [top-k=10 results] → Reranker → [top-k=5 reranked] → Generator
```

**Typical pattern**:
1. Retriever gets 10-20 candidates (over-retrieve)
2. Reranker re-scores and returns top-5
3. Generator uses refined context

### Optional Component
Reranking is **optional** and controlled by configuration:
- `reranking_enabled: False` → skip reranking
- `reranking_enabled: True` → apply reranker

## When to Use Reranking

✅ **Use when**:
- Retrieval quality is critical
- Latency budget allows (extra 100-500ms acceptable)
- Initial retriever is fast but imprecise (e.g., BM25)

❌ **Skip when**:
- Latency is critical (real-time chat)
- Retrieval is already high quality (hybrid retriever + good embeddings)
- Small document collections (<100 docs)

## Cross-Encoder vs Bi-Encoder

| Aspect | Bi-Encoder (Retriever) | Cross-Encoder (Reranker) |
|--------|------------------------|-------------------------|
| **Speed** | Fast (precomputed embeddings) | Slow (joint encoding) |
| **Accuracy** | Good | Excellent |
| **Scalability** | Millions of docs | Hundreds of candidates |
| **Usage** | Initial retrieval | Refinement |

**Why not use cross-encoder for retrieval?**
Cross-encoders must encode each (query, doc) pair on-the-fly → infeasible for large collections.

## COSMOS Integration Status

⚠️ **Not yet integrated** into COSMOS framework:
- `COSMOSComponent` wrapper lacks reranker handler
- Component evaluators don't include reranker metrics
- Search space definitions don't include reranker parameters

**To add reranker support to COSMOS**, see:
- `autorag/cosmos/CLAUDE.md` - Framework architecture and step-by-step guide
- `autorag/cosmos/component_wrapper.py` → `COSMOSComponent.process_with_metrics()` → add 'reranker' elif branch
- `scripts/bayesian_with_cache/run_optimization.py` → search space definition → shows `'reranking_enabled': [False, True]` pattern

## Metrics for Optimization

**Component-intrinsic metrics** (for COSMOS):
- **Latency**: Reranking time per query
- **Score change**: Mean absolute difference between original and reranked scores
- **Rank correlation**: Kendall's tau between original and reranked rankings
- **Optional (with ground truth)**: NDCG@k improvement over baseline

**End-to-end metrics** (for Bayesian):
- Answer quality (semantic similarity, RAGAS)
- Retrieval precision improvement
- Overall pipeline latency

## Key Implementation Details

1. **Score normalization**: CrossEncoder outputs raw logits; sigmoid converts to probabilities
2. **Metadata preservation**: Original retriever scores stored in `original_score` field
3. **Top-k handling**: Reranker can return fewer results than input (filtering low scores)
4. **Batch processing**: Supports batch reranking for multiple queries via `batch_rerank()`

## Reference Implementation

**File**: `scripts/bayesian_with_cache/run_optimization.py` → `SimplePipeline.query()` method → shows reranker integration:

```python
if self.reranker:
    results = self.reranker.rerank(query_text, results, top_k)
```

---

**Last Updated**: 2025-10-03
**Related Docs**: `autorag/components/CLAUDE.md`, `autorag/cosmos/CLAUDE.md`
