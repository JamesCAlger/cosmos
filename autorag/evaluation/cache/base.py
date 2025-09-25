"""Base caching system for evaluation results"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import time
from loguru import logger


class CacheKey:
    """Generate cache keys for evaluation results"""

    @staticmethod
    def generate(config: Dict, query: str, context: str, answer: str) -> str:
        """Generate a unique cache key"""
        # Create a deterministic string representation
        key_data = {
            "config": json.dumps(config, sort_keys=True),
            "query": query,
            "context": context,
            "answer": answer
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    @staticmethod
    def generate_batch(config: Dict, batch_id: str) -> str:
        """Generate cache key for batch evaluation"""
        key_data = {
            "config": json.dumps(config, sort_keys=True),
            "batch_id": batch_id
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class CacheBackend(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear entire cache"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get cache size in bytes"""
        pass


class FileCache(CacheBackend):
    """File-based cache implementation (lowest overhead)"""

    def __init__(self, cache_dir: str = "cache/evaluation"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self._save_metadata()

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Use first 2 chars as subdirectory for better file system performance
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return None

        # Check TTL if set
        if key in self.metadata:
            ttl = self.metadata[key].get("ttl")
            if ttl and time.time() > ttl:
                self.delete(key)
                return None

        try:
            with open(file_path, "rb") as f:
                value = pickle.load(f)

            # Update access time
            if key in self.metadata:
                self.metadata[key]["last_access"] = time.time()
                self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1
                self._save_metadata()

            return value
        except Exception as e:
            logger.error(f"Error loading cache key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        file_path = self._get_file_path(key)

        try:
            with open(file_path, "wb") as f:
                pickle.dump(value, f)

            # Update metadata
            self.metadata[key] = {
                "created": time.time(),
                "last_access": time.time(),
                "access_count": 0,
                "size": file_path.stat().st_size
            }

            if ttl:
                self.metadata[key]["ttl"] = time.time() + ttl

            self._save_metadata()
            return True

        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return False

        # Check TTL
        if key in self.metadata:
            ttl = self.metadata[key].get("ttl")
            if ttl and time.time() > ttl:
                self.delete(key)
                return False

        return True

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        file_path = self._get_file_path(key)

        try:
            if file_path.exists():
                file_path.unlink()

            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()

            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear entire cache"""
        try:
            # Remove all cache files
            for key in list(self.metadata.keys()):
                self.delete(key)

            # Clear metadata
            self.metadata = {}
            self._save_metadata()

            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for key, meta in self.metadata.items():
            total_size += meta.get("size", 0)
        return total_size

    def prune(self, max_size: Optional[int] = None, max_age: Optional[int] = None):
        """Prune cache based on size or age"""
        if max_age:
            current_time = time.time()
            for key, meta in list(self.metadata.items()):
                if current_time - meta["created"] > max_age:
                    self.delete(key)

        if max_size and self.size() > max_size:
            # Delete least recently used items
            sorted_keys = sorted(
                self.metadata.items(),
                key=lambda x: x[1]["last_access"]
            )

            while self.size() > max_size and sorted_keys:
                key, _ = sorted_keys.pop(0)
                self.delete(key)


class MemoryCache(CacheBackend):
    """In-memory cache for frequently accessed items"""

    def __init__(self, max_items: int = 1000):
        self.cache: Dict[str, Tuple[Any, Optional[float], float]] = {}
        self.max_items = max_items
        self.access_counts: Dict[str, int] = {}

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache"""
        if key not in self.cache:
            return None

        value, ttl, _ = self.cache[key]

        # Check TTL
        if ttl and time.time() > ttl:
            del self.cache[key]
            return None

        # Update access count
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        # Evict if at capacity
        if len(self.cache) >= self.max_items and key not in self.cache:
            self._evict()

        ttl_timestamp = time.time() + ttl if ttl else None
        self.cache[key] = (value, ttl_timestamp, time.time())
        self.access_counts[key] = 0

        return True

    def _evict(self):
        """Evict least recently used item"""
        if not self.cache:
            return

        # Find least accessed item
        min_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
        del self.cache[min_key]
        del self.access_counts[min_key]

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if key not in self.cache:
            return False

        _, ttl, _ = self.cache[key]
        if ttl and time.time() > ttl:
            del self.cache[key]
            return False

        return True

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
            return True
        return False

    def clear(self) -> bool:
        """Clear entire cache"""
        self.cache.clear()
        self.access_counts.clear()
        return True

    def size(self) -> int:
        """Get approximate cache size in bytes"""
        # This is an approximation
        import sys
        return sum(sys.getsizeof(v[0]) for v in self.cache.values())


class TieredCache:
    """Tiered caching system: Memory → Disk"""

    def __init__(self, cache_dir: str = "cache/evaluation", memory_items: int = 100):
        self.memory_cache = MemoryCache(max_items=memory_items)
        self.file_cache = FileCache(cache_dir)

    def get(self, key: str) -> Optional[Any]:
        """Try memory first, then disk"""
        # Check memory cache
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Check file cache
        value = self.file_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)

        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store in both tiers"""
        # Store in memory
        self.memory_cache.set(key, value, ttl)

        # Store on disk
        return self.file_cache.set(key, value, ttl)

    def exists(self, key: str) -> bool:
        """Check both tiers"""
        return self.memory_cache.exists(key) or self.file_cache.exists(key)

    def delete(self, key: str) -> bool:
        """Delete from both tiers"""
        memory_deleted = self.memory_cache.delete(key)
        file_deleted = self.file_cache.delete(key)
        return memory_deleted or file_deleted

    def clear(self) -> bool:
        """Clear both tiers"""
        memory_cleared = self.memory_cache.clear()
        file_cleared = self.file_cache.clear()
        return memory_cleared and file_cleared

    def size(self) -> int:
        """Get total size across tiers"""
        return self.memory_cache.size() + self.file_cache.size()