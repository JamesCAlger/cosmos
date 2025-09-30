# Embedding Cache Implementation Tasks

## Overview
Implementation plan for adding embedding cache to Bayesian optimization, reducing redundant API calls by 90%+ and cutting costs from $9+ to under $1 for typical experiments.

## Phase 1: Evaluator-Level Cache (Quick Win)
*Goal: Implement caching at the metrics collector level with minimal changes to existing components*

### 1.1 Core Cache Implementation
- [ ] Create `EmbeddingCacheManager` class in `run_bayesian_full_space_enhanced.py`
  - [ ] Implement cache key generation from chunking config
  - [ ] Add document hashing for change detection
  - [ ] Implement get_or_compute_embeddings() method
  - [ ] Add memory cache with LRU eviction (1GB limit)
  - [ ] Add cache statistics tracking

### 1.2 Storage Implementation
- [ ] Implement file-based persistence using pickle/numpy
  - [ ] Create cache directory structure (.embedding_cache/)
  - [ ] Implement save/load methods for embeddings
  - [ ] Add compression for large arrays (optional: float32→float16)
  - [ ] Implement cache versioning for format changes

### 1.3 Evaluator Integration
- [ ] Modify `EnhancedMetricsCollector.evaluate_with_detailed_metrics()`
  - [ ] Extract chunking configuration into standardized format
  - [ ] Add cache lookup before embedding generation
  - [ ] Implement fallback for cache misses
  - [ ] Add cache statistics to output

### 1.4 Retriever Modifications
- [ ] Add `index_with_embeddings()` method to `DenseRetriever`
  - [ ] Accept pre-computed embeddings
  - [ ] Bypass embedding generation when provided
  - [ ] Maintain backward compatibility
- [ ] Update `HybridRetriever` to support cached embeddings
  - [ ] Pass cached embeddings to dense component
  - [ ] Ensure sparse component still works independently

### 1.5 Testing - Phase 1
- [ ] Unit tests for cache manager
  - [ ] Test cache key generation consistency
  - [ ] Test memory limit enforcement
  - [ ] Test file persistence and loading
  - [ ] Test cache hit/miss statistics
- [ ] Integration tests
  - [ ] Compare cached vs non-cached results (must be identical)
  - [ ] Test with different chunking configurations
  - [ ] Test cache invalidation on document changes
  - [ ] Test memory limits with large datasets
- [ ] Performance benchmarks
  - [ ] Measure speedup with 100 trials, 5000 docs
  - [ ] Measure memory usage patterns
  - [ ] Validate 90%+ cache hit rate

### 1.6 Configuration & Documentation
- [ ] Add command-line arguments
  - [ ] --use-cache (default: True)
  - [ ] --cache-dir (default: .embedding_cache)
  - [ ] --cache-memory-limit (default: 1024 MB)
- [ ] Update run documentation
  - [ ] Document cache behavior
  - [ ] Add troubleshooting guide
  - [ ] Include cache clearing instructions

### 1.7 Validation & Monitoring
- [ ] Add cache statistics to final report
  - [ ] Hit rate percentage
  - [ ] API calls saved
  - [ ] Estimated cost savings
  - [ ] Time saved
- [ ] Add logging for cache operations
  - [ ] Log cache hits/misses
  - [ ] Log eviction events
  - [ ] Log storage errors

## Phase 2: Component-Level Integration
*Goal: Create reusable cached components for use across all scripts*

### 2.1 Cache Manager Module
- [ ] Extract `EmbeddingCacheManager` to `autorag/optimization/cache_manager.py`
  - [ ] Generalize for different embedding models
  - [ ] Add support for multiple cache backends
  - [ ] Implement cache warming from previous runs
  - [ ] Add distributed cache support (optional)

### 2.2 Cached Components
- [ ] Create `CachedEmbedder` wrapper class
  - [ ] Wrap any embedder implementation
  - [ ] Transparent cache integration
  - [ ] Maintain original embedder interface
  - [ ] Add cache bypass option
- [ ] Create `CachedChunker` wrapper (optional)
  - [ ] Cache chunking results separately
  - [ ] Enable partial chunk reuse
  - [ ] Handle overlap variations efficiently

### 2.3 Query Cache Implementation
- [ ] Implement `QueryEmbeddingCache` class
  - [ ] Separate cache for query embeddings
  - [ ] Always in-memory (small size)
  - [ ] Share across all evaluations
  - [ ] TTL-based expiration

### 2.4 Advanced Retriever Support
- [ ] Extend all retriever types
  - [ ] Update `BM25Retriever` (cache term frequencies)
  - [ ] Update `DenseRetriever` (full embedding cache)
  - [ ] Update `HybridRetriever` (coordinate both caches)
- [ ] Add incremental indexing
  - [ ] Detect unchanged chunks
  - [ ] Reuse existing vector store entries
  - [ ] Update only modified chunks

### 2.5 Testing - Phase 2
- [ ] Component unit tests
  - [ ] Test CachedEmbedder with different embedders
  - [ ] Test cache wrapper transparency
  - [ ] Test query cache functionality
  - [ ] Test component composition
- [ ] Integration tests
  - [ ] Test with grid search script
  - [ ] Test with other optimization scripts
  - [ ] Test concurrent access patterns
  - [ ] Test distributed cache scenarios
- [ ] Regression tests
  - [ ] Ensure Phase 1 code still works
  - [ ] Validate backward compatibility
  - [ ] Test migration path

### 2.6 Performance Optimizations
- [ ] Implement parallel preprocessing
  - [ ] Pre-compute all unique configs upfront
  - [ ] Batch embedding generation
  - [ ] Parallel chunk processing
- [ ] Memory optimizations
  - [ ] Implement memory-mapped arrays
  - [ ] Add chunk-level deduplication
  - [ ] Optimize cache key storage

### 2.7 Production Readiness
- [ ] Error handling
  - [ ] Graceful cache corruption recovery
  - [ ] Automatic cache rebuilding
  - [ ] Fallback to non-cached operation
- [ ] Monitoring and metrics
  - [ ] Add cache metrics to logs
  - [ ] Create cache analysis tools
  - [ ] Add cache health checks
- [ ] Documentation
  - [ ] API documentation for cached components
  - [ ] Migration guide from Phase 1
  - [ ] Performance tuning guide

## Success Metrics

### Phase 1 Success Criteria
- ✓ 90%+ reduction in embedding API calls
- ✓ 80%+ reduction in optimization time
- ✓ Zero difference in optimization results
- ✓ Less than 1GB memory overhead
- ✓ Works with existing scripts unchanged

### Phase 2 Success Criteria
- ✓ All optimization scripts can use cached components
- ✓ 95%+ cache hit rate across different scripts
- ✓ Cache sharing between parallel runs
- ✓ Production-ready error handling
- ✓ Full backward compatibility maintained

## Timeline Estimate

### Phase 1: 2-3 days
- Day 1: Core implementation and storage
- Day 2: Integration and testing
- Day 3: Documentation and validation

### Phase 2: 3-4 days
- Day 1: Extract and generalize cache manager
- Day 2: Create cached components
- Day 3: Advanced features and optimizations
- Day 4: Testing and documentation

## Risk Mitigation

### Phase 1 Risks
- **Risk**: Cache key collisions
  - **Mitigation**: Include all config parameters, use SHA-256 hashing
- **Risk**: Memory overflow
  - **Mitigation**: Strict memory limits, LRU eviction
- **Risk**: Cache corruption
  - **Mitigation**: Checksums, automatic cache clearing on error

### Phase 2 Risks
- **Risk**: Breaking API changes
  - **Mitigation**: Wrapper pattern, maintain original interfaces
- **Risk**: Performance regression
  - **Mitigation**: Comprehensive benchmarks, feature flags
- **Risk**: Complex debugging
  - **Mitigation**: Detailed logging, cache inspection tools

## Notes

- Start with Phase 1 for immediate gains
- Phase 1 code becomes the foundation for Phase 2
- All Phase 1 improvements carry forward to Phase 2
- Consider running Phase 1 in production for validation before Phase 2
- Cache format should be versioned from the start for future compatibility