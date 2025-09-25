"""Mock embedder for testing"""

from typing import List, Dict, Any
import random
from ...components.base import Embedder
from loguru import logger


class MockEmbedder(Embedder):
    """Mock embedder that generates random embeddings"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.dimension = self.config.get("dimension", 384)
        self.seed = self.config.get("seed", 42)
        random.seed(self.seed)
        logger.info(f"MockEmbedder initialized with dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for texts"""
        embeddings = []
        for text in texts:
            # Generate deterministic "random" embedding based on text
            random.seed(hash(text) % (2**32))
            embedding = [random.random() for _ in range(self.dimension)]
            embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} mock embeddings")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate mock embedding for a query"""
        random.seed(hash(query) % (2**32))
        return [random.random() for _ in range(self.dimension)]