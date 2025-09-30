"""
Cached Embedder Wrapper

This module provides a wrapper that adds caching functionality to any embedder.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from ...components.base import Embedder
from ...optimization.cache_manager import QueryEmbeddingCache


class CachedEmbedder(Embedder):
    """
    Wrapper class that adds caching to any embedder implementation.

    This wrapper transparently caches embeddings to avoid redundant API calls.
    It can wrap any embedder (OpenAI, local models, etc.) without changing the interface.
    """

    def __init__(self, embedder: Embedder, cache_manager=None, config: Dict[str, Any] = None):
        """
        Initialize cached embedder wrapper.

        Args:
            embedder: The underlying embedder to wrap
            cache_manager: Optional EmbeddingCacheManager for document embeddings
            config: Additional configuration
        """
        super().__init__(config or {})
        self.embedder = embedder
        self.cache_manager = cache_manager
        self.query_cache = QueryEmbeddingCache(embedder)

        # Copy important attributes from wrapped embedder
        self.model = getattr(embedder, 'model', 'unknown')
        self.dimension = getattr(embedder, 'dimension', None)
        self.batch_size = getattr(embedder, 'batch_size', 100)

        # Statistics
        self.stats = {
            'embed_calls': 0,
            'embed_cache_hits': 0,
            'query_calls': 0,
            'query_cache_hits': 0,
            'time_saved': 0.0
        }

        logger.info(f"CachedEmbedder initialized wrapping {type(embedder).__name__}")

    def _get_text_hash(self, texts: List[str]) -> str:
        """Generate hash for a list of texts."""
        combined = '|'.join(texts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self.stats['embed_calls'] += 1

        # If we have a cache manager, check if these exact texts are cached
        if self.cache_manager and hasattr(self.cache_manager, 'memory_cache'):
            text_hash = self._get_text_hash(texts)

            # Check if we have these exact embeddings cached
            for cache_key, cache_entry in self.cache_manager.memory_cache.items():
                if 'chunk_texts' in cache_entry:
                    cached_texts = cache_entry['chunk_texts']
                    if cached_texts == texts:
                        self.stats['embed_cache_hits'] += 1
                        self.stats['time_saved'] += cache_entry.get('compute_time', 0)
                        logger.info(f"CachedEmbedder: Direct cache HIT for {len(texts)} texts")
                        return cache_entry['embeddings']

        # Not cached, compute embeddings
        start_time = time.time()
        embeddings = self.embedder.embed(texts)
        compute_time = time.time() - start_time

        logger.debug(f"CachedEmbedder: Computed {len(texts)} embeddings in {compute_time:.3f}s")

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query with caching.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector for the query
        """
        self.stats['query_calls'] += 1

        # Use query cache
        embedding = self.query_cache.get_embedding(query)

        if self.query_cache.stats['hits'] > self.stats['query_cache_hits']:
            self.stats['query_cache_hits'] = self.query_cache.stats['hits']

        return embedding

    def get_stats(self) -> Dict:
        """Get caching statistics."""
        embed_hit_rate = self.stats['embed_cache_hits'] / max(self.stats['embed_calls'], 1)
        query_stats = self.query_cache.get_stats()

        return {
            'embed_calls': self.stats['embed_calls'],
            'embed_cache_hits': self.stats['embed_cache_hits'],
            'embed_hit_rate': embed_hit_rate,
            'query_calls': self.stats['query_calls'],
            'query_cache_hits': self.stats['query_cache_hits'],
            'query_hit_rate': query_stats['hit_rate'],
            'time_saved': self.stats['time_saved'],
            'query_cache_size': query_stats['cache_size']
        }

    def clear_cache(self):
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_cache()
        self.query_cache.clear()
        logger.info("CachedEmbedder: All caches cleared")

    # Pass through any other methods/attributes to the wrapped embedder
    def __getattr__(self, name):
        """Pass through any unknown attributes to the wrapped embedder."""
        return getattr(self.embedder, name)