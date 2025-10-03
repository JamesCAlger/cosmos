"""
Embedding Cache Manager for RAG Optimization

This module provides caching functionality for embeddings and chunks to avoid
redundant API calls during optimization runs.
"""

import hashlib
import json
import pickle
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class EmbeddingCacheManager:
    """
    Manages cached embeddings and chunks across optimization trials.

    Cache Structure:
    {
        (chunking_strategy, chunk_size, overlap, embedder_model): {
            'chunks': List[Chunk],
            'embeddings': np.ndarray or List[List[float]],
            'chunk_texts': List[str],
            'timestamp': datetime,
            'doc_hash': str,  # To verify documents haven't changed
            'compute_time': float
        }
    }
    """

    def __init__(self, cache_dir: str = ".embedding_cache", max_memory_mb: int = 1024, use_cache: bool = True):
        """
        Initialize the embedding cache manager.

        Args:
            cache_dir: Directory for persistent cache storage
            max_memory_mb: Maximum memory usage in MB (default 1GB)
            use_cache: Whether to use caching (can be disabled for testing)
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_mb = max_memory_mb
        self.use_cache = use_cache
        self.memory_cache = OrderedDict()  # LRU cache
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saved_api_calls': 0,
            'saved_time': 0.0,
            'total_embeddings_computed': 0,
            'total_embeddings_cached': 0
        }

        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            self._load_cache_index()

        logger.info(f"EmbeddingCacheManager initialized: cache={'enabled' if use_cache else 'disabled'}, "
                   f"dir={cache_dir}, max_memory={max_memory_mb}MB")

    def _load_cache_index(self):
        """Load cache index from disk to know what's available."""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self.cache_index = json.load(f)
                logger.info(f"Loaded cache index with {len(self.cache_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.cache_index = {}
        else:
            self.cache_index = {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        index_path = self.cache_dir / "cache_index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def get_cache_key(self, chunking_config: Dict, embedder_config: Dict,
                      dataset_name: Optional[str] = None, dataset_size: Optional[int] = None) -> str:
        """
        Generate deterministic cache key from chunking, embedder, and dataset parameters.

        Args:
            chunking_config: Chunking configuration
            embedder_config: Embedder configuration
            dataset_name: Name of dataset (e.g., 'marco', 'beir/scifact')
            dataset_size: Number of documents in dataset

        Returns:
            String cache key
        """
        # Extract relevant parameters
        key_parts = {
            'dataset_name': dataset_name or 'unknown',
            'dataset_size': dataset_size or 0,
            'chunking_strategy': chunking_config.get('strategy', 'unknown'),
            'chunk_size': chunking_config.get('chunk_size', 0),
            'overlap': chunking_config.get('overlap', 0),
            'embedder_model': embedder_config.get('model', 'unknown'),
            'embedder_type': embedder_config.get('type', 'unknown')
        }

        # Add semantic chunker threshold if applicable
        if key_parts['chunking_strategy'] == 'semantic':
            key_parts['semantic_threshold'] = chunking_config.get('threshold', 0.5)

        # Create deterministic string representation
        key_str = json.dumps(key_parts, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]  # Use first 16 chars for readability

    def _hash_documents(self, documents: List[str]) -> str:
        """Generate hash of document contents to detect changes."""
        doc_str = '|'.join(sorted(documents))  # Sort for consistency
        return hashlib.sha256(doc_str.encode()).hexdigest()[:16]

    def _get_memory_usage_mb(self) -> float:
        """Estimate current memory usage of cache in MB."""
        total_bytes = 0
        for cache_entry in self.memory_cache.values():
            if 'embeddings' in cache_entry and cache_entry['embeddings'] is not None:
                # Estimate embedding array size
                embeddings = cache_entry['embeddings']
                if isinstance(embeddings, np.ndarray):
                    total_bytes += embeddings.nbytes
                elif isinstance(embeddings, list):
                    # Rough estimation for list of lists
                    total_bytes += len(embeddings) * len(embeddings[0] if embeddings else []) * 4

            # Add some overhead for other data
            total_bytes += len(str(cache_entry.get('chunks', []))) + 1024

        return total_bytes / (1024 * 1024)  # Convert to MB

    def _evict_lru(self):
        """Evict least recently used items if memory limit exceeded."""
        current_mb = self._get_memory_usage_mb()

        while current_mb > self.max_memory_mb and len(self.memory_cache) > 0:
            # Remove oldest item (first in OrderedDict)
            evicted_key, evicted_value = self.memory_cache.popitem(last=False)
            logger.debug(f"Evicted cache entry {evicted_key} to stay under memory limit")
            current_mb = self._get_memory_usage_mb()

    def _save_to_disk(self, cache_key: str, cache_entry: Dict):
        """Save cache entry to disk."""
        if not self.use_cache:
            return

        file_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(cache_entry, f)

            # Update index
            self.cache_index[cache_key] = {
                'timestamp': cache_entry['timestamp'].isoformat() if isinstance(cache_entry['timestamp'], datetime) else cache_entry['timestamp'],
                'doc_hash': cache_entry['doc_hash'],
                'num_chunks': len(cache_entry.get('chunks', [])),
                'compute_time': cache_entry.get('compute_time', 0),
                'dataset_name': cache_entry.get('dataset_name', 'unknown'),
                'dataset_size': cache_entry.get('dataset_size', 0)
            }
            self._save_cache_index()

            logger.debug(f"Saved cache entry {cache_key} to disk")
        except Exception as e:
            logger.error(f"Failed to save cache entry {cache_key} to disk: {e}")

    def _load_from_disk(self, cache_key: str) -> Optional[Dict]:
        """Load cache entry from disk."""
        if not self.use_cache:
            return None

        file_path = self.cache_dir / f"{cache_key}.pkl"
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'rb') as f:
                cache_entry = pickle.load(f)
            logger.debug(f"Loaded cache entry {cache_key} from disk")
            return cache_entry
        except Exception as e:
            logger.error(f"Failed to load cache entry {cache_key} from disk: {e}")
            return None

    def get_or_compute_embeddings(self,
                                  documents: List[str],
                                  chunking_config: Dict,
                                  embedder,
                                  chunker,
                                  dataset_name: Optional[str] = None,
                                  dataset_size: Optional[int] = None) -> Tuple[List, Any]:
        """
        Returns cached embeddings if available, otherwise computes and caches.

        Args:
            documents: List of document strings
            chunking_config: Configuration for chunking
            embedder: Embedder instance
            chunker: Chunker instance
            dataset_name: Name of dataset (e.g., 'marco', 'beir/scifact')
            dataset_size: Number of documents in dataset

        Returns:
            chunks: List of chunk objects
            embeddings: Array of embedding vectors (numpy array or list)
        """
        from ..components.base import Document

        if not self.use_cache:
            # Cache disabled, compute directly
            doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
            chunks = chunker.chunk(doc_objects)
            chunk_texts = [c.content if hasattr(c, 'content') else str(c) for c in chunks]
            embeddings = embedder.embed(chunk_texts)
            return chunks, embeddings

        # Generate cache key
        embedder_config = {
            'model': getattr(embedder, 'model', 'unknown'),
            'type': type(embedder).__name__
        }
        # Include dataset info in cache key
        dataset_size = dataset_size or len(documents)
        cache_key = self.get_cache_key(chunking_config, embedder_config, dataset_name, dataset_size)
        doc_hash = self._hash_documents(documents)

        # Check memory cache first
        if cache_key in self.memory_cache:
            cached = self.memory_cache[cache_key]
            if cached['doc_hash'] == doc_hash:
                self.stats['hits'] += 1
                self.stats['saved_api_calls'] += len(cached.get('chunks', []))
                self.stats['saved_time'] += cached.get('compute_time', 0)
                self.stats['total_embeddings_cached'] += len(cached.get('chunks', []))

                # Move to end (most recently used)
                self.memory_cache.move_to_end(cache_key)

                logger.info(f"Cache HIT for {cache_key} - saved {len(cached.get('chunks', []))} API calls")
                return cached['chunks'], cached['embeddings']

        # Check disk cache
        if cache_key in self.cache_index:
            if self.cache_index[cache_key]['doc_hash'] == doc_hash:
                cached = self._load_from_disk(cache_key)
                if cached:
                    self.memory_cache[cache_key] = cached
                    self._evict_lru()  # Check memory limits

                    self.stats['hits'] += 1
                    self.stats['saved_api_calls'] += len(cached.get('chunks', []))
                    self.stats['saved_time'] += cached.get('compute_time', 0)
                    self.stats['total_embeddings_cached'] += len(cached.get('chunks', []))

                    logger.info(f"Cache HIT (disk) for {cache_key} - saved {len(cached.get('chunks', []))} API calls")
                    return cached['chunks'], cached['embeddings']

        # Cache MISS - compute embeddings
        self.stats['misses'] += 1
        logger.info(f"Cache MISS for {cache_key} - computing embeddings...")

        start_time = time.time()

        # Convert to Document objects and chunk
        doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
        chunks = chunker.chunk(doc_objects)
        chunk_texts = [c.content if hasattr(c, 'content') else str(c) for c in chunks]

        # Generate embeddings
        embeddings = embedder.embed(chunk_texts)

        compute_time = time.time() - start_time
        self.stats['total_embeddings_computed'] += len(chunks)

        # Store in cache
        cache_entry = {
            'chunks': chunks,
            'embeddings': embeddings,
            'chunk_texts': chunk_texts,
            'timestamp': datetime.now(),
            'doc_hash': doc_hash,
            'compute_time': compute_time,
            'dataset_name': dataset_name or 'unknown',
            'dataset_size': dataset_size
        }

        # Add to memory cache
        self.memory_cache[cache_key] = cache_entry
        self._evict_lru()  # Check memory limits

        # Save to disk
        self._save_to_disk(cache_key, cache_entry)

        logger.info(f"Computed and cached {len(chunks)} embeddings in {compute_time:.2f}s")

        return chunks, embeddings

    def get_cache_stats(self) -> Dict:
        """Return cache performance statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(total_requests, 1)

        # Estimate cost savings (OpenAI ada-002 pricing)
        estimated_cost_saved = self.stats['saved_api_calls'] * 0.0001  # $0.0001 per 1K tokens

        return {
            'hit_rate': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'total_requests': total_requests,
            'embeddings_computed': self.stats['total_embeddings_computed'],
            'embeddings_cached': self.stats['total_embeddings_cached'],
            'estimated_cost_saved': estimated_cost_saved,
            'time_saved_seconds': self.stats['saved_time'],
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_size': len(self.cache_index),
            'memory_usage_mb': self._get_memory_usage_mb()
        }

    def clear_cache(self):
        """Clear all cache entries."""
        self.memory_cache.clear()
        self.cache_index.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saved_api_calls': 0,
            'saved_time': 0.0,
            'total_embeddings_computed': 0,
            'total_embeddings_cached': 0
        }

        # Remove cache files
        if self.cache_dir.exists():
            for file_path in self.cache_dir.glob("*.pkl"):
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cache file {file_path}: {e}")

        self._save_cache_index()
        logger.info("Cache cleared")


class QueryEmbeddingCache:
    """
    Separate cache for query embeddings (much smaller, always in memory).

    Query embeddings are small and frequently reused, so we keep them all in memory.
    """

    def __init__(self, embedder=None):
        """
        Initialize query embedding cache.

        Args:
            embedder: Optional embedder instance to use for generating embeddings
        """
        self.embedder = embedder
        self.cache = {}
        self.stats = {
            'hits': 0,
            'misses': 0
        }

    def set_embedder(self, embedder):
        """Set or update the embedder instance."""
        self.embedder = embedder

    def get_embedding(self, query: str) -> Any:
        """
        Get cached embedding for query or compute if not cached.

        Args:
            query: Query string to embed

        Returns:
            Query embedding vector
        """
        if query not in self.cache:
            if not self.embedder:
                raise ValueError("Embedder not set for QueryEmbeddingCache")

            self.stats['misses'] += 1

            # Check if embedder has embed_query method
            if hasattr(self.embedder, 'embed_query'):
                self.cache[query] = self.embedder.embed_query(query)
            else:
                # Fallback to regular embed method
                embeddings = self.embedder.embed([query])
                self.cache[query] = embeddings[0] if embeddings else None

            logger.debug(f"Query embedding cache MISS for: {query[:50]}...")
        else:
            self.stats['hits'] += 1
            logger.debug(f"Query embedding cache HIT for: {query[:50]}...")

        return self.cache[query]

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(total, 1)

        return {
            'hit_rate': hit_rate,
            'total_hits': self.stats['hits'],
            'total_misses': self.stats['misses'],
            'cache_size': len(self.cache)
        }

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.stats = {'hits': 0, 'misses': 0}
        logger.info("Query embedding cache cleared")