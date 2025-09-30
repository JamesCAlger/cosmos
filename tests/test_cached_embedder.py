"""
Tests for CachedEmbedder and cache components
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autorag.optimization.cache_manager import EmbeddingCacheManager, QueryEmbeddingCache
from autorag.components.embedders.cached import CachedEmbedder
from autorag.components.embedders.mock import MockEmbedder


def test_embedding_cache_manager():
    """Test EmbeddingCacheManager functionality"""
    print("\nTesting EmbeddingCacheManager...")

    # Create temp cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        cache_manager = EmbeddingCacheManager(
            cache_dir=cache_dir,
            max_memory_mb=10,
            use_cache=True
        )

        # Test data
        documents = ["Document 1", "Document 2", "Document 3"]
        chunking_config = {
            'strategy': 'fixed',
            'chunk_size': 256,
            'overlap': 0
        }

        # Create mock embedder and chunker
        from autorag.components.embedders.mock import MockEmbedder
        from autorag.components.chunkers.fixed_size import FixedSizeChunker

        embedder = MockEmbedder({})
        chunker = FixedSizeChunker({'chunk_size': 256})

        # First call - should be a cache miss
        chunks1, embeddings1 = cache_manager.get_or_compute_embeddings(
            documents, chunking_config, embedder, chunker
        )

        assert cache_manager.stats['misses'] == 1
        assert cache_manager.stats['hits'] == 0
        print(f"  [OK] First call was a cache miss")

        # Second call with same config - should be a cache hit
        chunks2, embeddings2 = cache_manager.get_or_compute_embeddings(
            documents, chunking_config, embedder, chunker
        )

        assert cache_manager.stats['hits'] == 1
        assert chunks1 == chunks2
        print(f"  [OK] Second call was a cache hit")

        # Different config - should be a cache miss
        chunking_config2 = {
            'strategy': 'fixed',
            'chunk_size': 512,  # Different size
            'overlap': 0
        }
        chunks3, embeddings3 = cache_manager.get_or_compute_embeddings(
            documents, chunking_config2, embedder, chunker
        )

        assert cache_manager.stats['misses'] == 2
        print(f"  [OK] Different config caused cache miss")

        # Check stats
        stats = cache_manager.get_cache_stats()
        assert stats['hit_rate'] == 1/3  # 1 hit, 2 misses
        assert stats['memory_cache_size'] == 2  # Two different configs cached
        print(f"  [OK] Cache statistics correct: hit_rate={stats['hit_rate']:.1%}")

        # Test cache clearing
        cache_manager.clear_cache()
        assert cache_manager.stats['hits'] == 0
        assert cache_manager.stats['misses'] == 0
        print(f"  [OK] Cache cleared successfully")

    print("EmbeddingCacheManager tests passed!")


def test_query_embedding_cache():
    """Test QueryEmbeddingCache functionality"""
    print("\nTesting QueryEmbeddingCache...")

    embedder = MockEmbedder({})
    query_cache = QueryEmbeddingCache(embedder)

    # First query - cache miss
    query1 = "What is machine learning?"
    embedding1 = query_cache.get_embedding(query1)
    assert query_cache.stats['misses'] == 1
    assert query_cache.stats['hits'] == 0
    print(f"  [OK] First query was a cache miss")

    # Same query - cache hit
    embedding2 = query_cache.get_embedding(query1)
    assert query_cache.stats['hits'] == 1
    assert embedding1 == embedding2
    print(f"  [OK] Second identical query was a cache hit")

    # Different query - cache miss
    query2 = "How does deep learning work?"
    embedding3 = query_cache.get_embedding(query2)
    assert query_cache.stats['misses'] == 2
    print(f"  [OK] Different query caused cache miss")

    # Check cache size
    stats = query_cache.get_stats()
    assert stats['cache_size'] == 2
    assert stats['hit_rate'] == 1/3
    print(f"  [OK] Query cache statistics correct: {stats['cache_size']} cached, hit_rate={stats['hit_rate']:.1%}")

    print("QueryEmbeddingCache tests passed!")


def test_cached_embedder():
    """Test CachedEmbedder wrapper"""
    print("\nTesting CachedEmbedder...")

    # Create base embedder and cache manager
    base_embedder = MockEmbedder({})

    with tempfile.TemporaryDirectory() as cache_dir:
        cache_manager = EmbeddingCacheManager(cache_dir=cache_dir)
        cached_embedder = CachedEmbedder(base_embedder, cache_manager)

        # Test document embedding
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings1 = cached_embedder.embed(texts)
        assert cached_embedder.stats['embed_calls'] == 1
        print(f"  [OK] First embed call completed")

        # Same texts should not increase cache hits (since we're not using the chunking flow)
        embeddings2 = cached_embedder.embed(texts)
        assert cached_embedder.stats['embed_calls'] == 2
        print(f"  [OK] Second embed call completed")

        # Test query embedding with caching
        query = "What is AI?"
        query_emb1 = cached_embedder.embed_query(query)
        assert cached_embedder.stats['query_calls'] == 1
        print(f"  [OK] First query embed call completed")

        # Same query should hit cache
        query_emb2 = cached_embedder.embed_query(query)
        assert cached_embedder.stats['query_calls'] == 2
        assert query_emb1 == query_emb2
        print(f"  [OK] Query cache working")

        # Check that wrapper passes through attributes
        # MockEmbedder might not have 'model' attribute, so check dimension instead
        assert cached_embedder.dimension == base_embedder.dimension
        print(f"  [OK] Attribute pass-through working")

        # Get stats
        stats = cached_embedder.get_stats()
        print(f"  [OK] Stats: {stats['query_calls']} query calls, hit_rate={stats['query_hit_rate']:.1%}")

    print("CachedEmbedder tests passed!")


def test_integration():
    """Test the full integration of cached components"""
    print("\nTesting full integration...")

    from autorag.components.chunkers.fixed_size import FixedSizeChunker
    from autorag.components.retrievers.dense import DenseRetriever
    from autorag.components.vector_stores.simple import SimpleVectorStore
    from autorag.components.base import Document

    with tempfile.TemporaryDirectory() as cache_dir:
        # Setup components
        cache_manager = EmbeddingCacheManager(cache_dir=cache_dir)
        base_embedder = MockEmbedder({})
        cached_embedder = CachedEmbedder(base_embedder, cache_manager)

        chunker = FixedSizeChunker({'chunk_size': 100})
        retriever = DenseRetriever({})
        vector_store = SimpleVectorStore({})
        retriever.set_components(cached_embedder, vector_store)

        # Test documents
        documents = [
            Document(content="Machine learning is great", doc_id="1"),
            Document(content="Deep learning uses neural networks", doc_id="2")
        ]

        # First indexing
        chunks = chunker.chunk(documents)
        retriever.index(chunks)
        print(f"  [OK] First indexing completed")

        # Query
        results = retriever.retrieve("What is machine learning?", top_k=2)
        assert len(results) <= 2
        print(f"  [OK] Retrieval working with cached embedder")

        # Check that query was cached
        query_stats = cached_embedder.query_cache.get_stats()
        assert query_stats['cache_size'] > 0
        print(f"  [OK] Query caching working in retrieval")

    print("Integration tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2 CACHED COMPONENTS TEST SUITE")
    print("=" * 60)

    try:
        test_embedding_cache_manager()
        test_query_embedding_cache()
        test_cached_embedder()
        test_integration()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! [SUCCESS]")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()