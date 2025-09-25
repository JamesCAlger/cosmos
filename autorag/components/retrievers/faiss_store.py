"""FAISS vector store implementation"""

from typing import List, Dict, Any
import numpy as np
import faiss
from ...components.base import VectorStore, Chunk, QueryResult
from loguru import logger


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for similarity search"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.dimension = self.config.get("dimension", 1536)
        self.index_type = self.config.get("index_type", "flat")  # flat, ivf, hnsw

        # Initialize FAISS index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            # For now, just use flat index
            self.index = faiss.IndexFlatL2(self.dimension)

        self.chunks: List[Chunk] = []
        logger.info(f"FAISSVectorStore initialized with dimension: {self.dimension}, type: {self.index_type}")

    def add(self, embeddings: List[List[float]], chunks: List[Chunk]) -> None:
        """Add embeddings and chunks to the store"""
        if not embeddings or not chunks:
            return

        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")

        # Convert to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Add to index
        self.index.add(embeddings_np)
        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks to vector store (total: {len(self.chunks)})")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        """Search for similar vectors"""
        if not self.chunks:
            logger.warning("Vector store is empty")
            return []

        # Convert to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)

        # Search
        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_np, k)

        # Convert to QueryResult
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                # Convert L2 distance to similarity score (inverse)
                score = 1.0 / (1.0 + float(distance))
                result = QueryResult(
                    chunk=self.chunks[idx],
                    score=score,
                    metadata={"distance": float(distance)}
                )
                results.append(result)

        logger.debug(f"Found {len(results)} results for query")
        return results

    def clear(self) -> None:
        """Clear the vector store"""
        self.index.reset()
        self.chunks = []
        logger.info("Vector store cleared")