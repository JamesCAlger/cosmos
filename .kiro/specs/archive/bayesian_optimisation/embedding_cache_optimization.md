# Embedding Cache Optimization for Bayesian Search

## Problem Statement

The current Bayesian optimization implementation (`run_bayesian_full_space_enhanced.py`) redundantly computes embeddings for the same document chunks across multiple configurations. With 100 trials evaluating 5000 documents, the system:

- Performs **100 identical chunking operations** when parameters match
- Makes **100 identical embedding API calls** for the same chunks
- Wastes **~$7+ and 23+ hours** on duplicate computations
- Becomes **practically infeasible** for large-scale optimization

## Optimization Opportunity

### Key Insight
Multiple pipeline configurations share:
1. **Same documents** (constant across all trials)
2. **Same chunking parameters** (only 6 unique combinations in search space)
3. **Same embedding model** (text-embedding-ada-002 for all configs)

### Potential Impact
- **Cost reduction**: 94% fewer embedding API calls
- **Time reduction**: 90% faster for large document sets
- **Scalability**: Makes 10,000+ document experiments feasible

## Implementation Architecture

### Where to Implement

You are correct that this should be implemented **primarily in `run_bayesian_full_space_enhanced.py`**, but with a modular design:

```
autorag/
├── optimization/
│   ├── bayesian_search.py         # Existing optimizer
│   └── cache_manager.py           # NEW: Embedding cache module
└── scripts/
    └── run_bayesian_full_space_enhanced.py  # Modified to use cache
```

### Cache Manager Component

```python
# autorag/optimization/cache_manager.py

class EmbeddingCacheManager:
    """
    Manages cached embeddings and chunks across optimization trials.

    Cache Structure:
    {
        (chunking_strategy, chunk_size, overlap): {
            'chunks': List[Chunk],
            'embeddings': np.ndarray,
            'chunk_texts': List[str],
            'timestamp': datetime,
            'doc_hash': str  # To verify documents haven't changed
        }
    }
    """

    def __init__(self, cache_dir: str = ".embedding_cache"):
        self.memory_cache = {}
        self.cache_dir = cache_dir
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saved_api_calls': 0,
            'saved_time': 0.0
        }

    def get_cache_key(self, chunking_config: Dict) -> Tuple:
        """Generate deterministic cache key from chunking parameters"""
        return (
            chunking_config.get('strategy'),
            chunking_config.get('chunk_size'),
            chunking_config.get('overlap', 0)
        )

    def get_or_compute_embeddings(self,
                                  documents: List[str],
                                  chunking_config: Dict,
                                  embedder: Embedder,
                                  chunker: Chunker) -> Tuple[List, np.ndarray]:
        """
        Returns cached embeddings if available, otherwise computes and caches.

        Returns:
            chunks: List of chunk objects
            embeddings: numpy array of embedding vectors
        """
        cache_key = self.get_cache_key(chunking_config)
        doc_hash = self._hash_documents(documents)

        # Check memory cache
        if cache_key in self.memory_cache:
            cached = self.memory_cache[cache_key]
            if cached['doc_hash'] == doc_hash:
                self.stats['hits'] += 1
                logger.info(f"Cache HIT for {cache_key}")
                return cached['chunks'], cached['embeddings']

        # Check disk cache if enabled
        if self.cache_dir and self._disk_cache_exists(cache_key, doc_hash):
            cached = self._load_from_disk(cache_key)
            self.memory_cache[cache_key] = cached
            self.stats['hits'] += 1
            return cached['chunks'], cached['embeddings']

        # Cache MISS - compute embeddings
        self.stats['misses'] += 1
        logger.info(f"Cache MISS for {cache_key} - computing embeddings...")

        start_time = time.time()
        chunks = chunker.chunk(documents)
        chunk_texts = [c.content for c in chunks]
        embeddings = embedder.embed(chunk_texts)
        compute_time = time.time() - start_time

        # Store in cache
        cache_entry = {
            'chunks': chunks,
            'embeddings': embeddings,
            'chunk_texts': chunk_texts,
            'timestamp': datetime.now(),
            'doc_hash': doc_hash,
            'compute_time': compute_time
        }

        self.memory_cache[cache_key] = cache_entry
        if self.cache_dir:
            self._save_to_disk(cache_key, cache_entry)

        return chunks, embeddings

    def get_cache_stats(self) -> Dict:
        """Return cache performance statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(total_requests, 1)

        return {
            'hit_rate': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'estimated_cost_saved': self.stats['saved_api_calls'] * 0.0001,
            'time_saved_hours': self.stats['saved_time'] / 3600
        }
```

### Modified Evaluator Integration

```python
# In run_bayesian_full_space_enhanced.py

class EnhancedMetricsCollector:
    def __init__(self, show_details: bool = True, use_cache: bool = True):
        self.show_details = show_details
        self.semantic_evaluator = SemanticMetrics(...)

        # Initialize cache manager
        self.cache_manager = EmbeddingCacheManager() if use_cache else None
        self.query_cache = {}  # Cache query embeddings separately

    def evaluate_with_detailed_metrics(self, pipeline, query: str,
                                      documents: List[str], ground_truth: Dict):

        # Extract chunking config from pipeline
        chunking_config = {
            'strategy': pipeline.chunker.strategy,
            'chunk_size': pipeline.chunker.chunk_size,
            'overlap': getattr(pipeline.chunker, 'overlap', 0)
        }

        if self.cache_manager:
            # Use cached embeddings if available
            chunks, embeddings = self.cache_manager.get_or_compute_embeddings(
                documents,
                chunking_config,
                pipeline.embedder,
                pipeline.chunker
            )

            # Inject cached data into pipeline
            pipeline.retriever.vector_store.clear()
            pipeline.retriever.vector_store.add_with_embeddings(chunks, embeddings)
        else:
            # Original flow without caching
            chunks = pipeline.chunker.chunk(doc_objects)
            pipeline.retriever.index(chunks)

        # Continue with retrieval and generation...
```

### Query Embedding Cache

```python
class QueryEmbeddingCache:
    """Separate cache for query embeddings (much smaller, always in memory)"""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.cache = {}

    def get_embedding(self, query: str) -> np.ndarray:
        if query not in self.cache:
            self.cache[query] = self.embedder.embed([query])[0]
        return self.cache[query]
```

## Implementation Steps

### Phase 1: Basic Memory Cache (Quick Win)
1. Add `EmbeddingCacheManager` class to `run_bayesian_full_space_enhanced.py`
2. Modify `evaluator` function to use cache for chunking/embedding
3. Add cache statistics to final report
4. Test with medium-size dataset (500 docs, 50 configs)

### Phase 2: Persistent Disk Cache
1. Implement disk serialization using pickle or HDF5
2. Add cache warmup from previous runs
3. Implement cache versioning and invalidation
4. Add compression for large embedding matrices

### Phase 3: Advanced Optimizations
1. **Partial chunk reuse**: When only overlap changes, reuse common chunks
2. **Incremental indexing**: Add only new chunks to vector stores
3. **Parallel preprocessing**: Pre-compute all unique chunking configs upfront
4. **Distributed cache**: Share cache across multiple optimization runs

## Configuration Options

```python
# Command line arguments
parser.add_argument('--use-cache', action='store_true', default=True,
                   help='Enable embedding cache (default: True)')
parser.add_argument('--cache-dir', type=str, default='.embedding_cache',
                   help='Directory for persistent cache storage')
parser.add_argument('--cache-memory-limit', type=int, default=4096,
                   help='Maximum memory for cache in MB')
parser.add_argument('--precompute-embeddings', action='store_true',
                   help='Precompute all unique embeddings before optimization')
```

## Expected Performance Improvements

### Small Dataset (250 docs, 25 queries)
- Without cache: $2.20, 75 minutes
- With cache: $0.30, 15 minutes
- **Improvement**: 86% cost reduction, 80% time reduction

### Large Dataset (5000 docs, 25 queries)
- Without cache: $9.30, 30 hours
- With cache: $0.80, 3 hours
- **Improvement**: 91% cost reduction, 90% time reduction

### Very Large Dataset (10000 docs, 50 queries)
- Without cache: **Infeasible** (60+ hours, $20+)
- With cache: $2.00, 6 hours
- **Improvement**: Makes it actually possible!

## Memory Considerations

### Memory Requirements
```
Cache size = num_unique_configs × num_chunks × embedding_dim × 4 bytes

For 5000 docs:
- 6 configs × 15000 chunks × 1536 dims × 4 bytes = 550 MB

For 10000 docs:
- 6 configs × 30000 chunks × 1536 dims × 4 bytes = 1.1 GB
```

### Memory Management Strategies
1. **LRU eviction**: Remove least recently used configs when limit reached
2. **Disk offloading**: Move cold entries to disk automatically
3. **Compression**: Use float16 instead of float32 for embeddings
4. **Selective caching**: Only cache expensive operations (embeddings, not BM25)

## Validation & Testing

### Test Scenarios
1. **Cache correctness**: Verify cached results match non-cached
2. **Memory limits**: Test behavior when cache exceeds limits
3. **Concurrent access**: Multiple trials accessing same cache
4. **Cache invalidation**: Documents change between runs

### Monitoring
```python
# Add to final report
print("\nCache Performance:")
print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
print(f"  API Calls Saved: {cache_stats['saved_api_calls']}")
print(f"  Estimated Cost Saved: ${cache_stats['estimated_cost_saved']:.2f}")
print(f"  Time Saved: {cache_stats['time_saved_hours']:.1f} hours")
```

## Alternative Architectures

### Option 1: Separate Preprocessing Step
```bash
# First: preprocess and cache all embeddings
python preprocess_embeddings.py --docs 5000 --configs config_space.yaml

# Then: run optimization using cached embeddings
python run_bayesian_full_space_enhanced.py --use-preprocessed --n-calls 100
```

### Option 2: Shared Cache Service
- Run a separate cache service (Redis/Memcached)
- Multiple optimization runs share the same cache
- Better for distributed/parallel optimization

### Option 3: Database-Backed Cache
- Store embeddings in PostgreSQL with pgvector
- Enables SQL queries on embeddings
- Better for production systems

## Conclusion

The embedding cache optimization is **essential** for making Bayesian optimization practical at scale. It should be implemented in `run_bayesian_full_space_enhanced.py` with a modular design that can be extended to other optimization scripts.

The implementation is relatively straightforward but provides dramatic improvements in both cost and runtime, transforming the system from a prototype to a production-ready optimization tool.