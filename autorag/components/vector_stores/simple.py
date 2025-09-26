"""Simple in-memory vector store implementation"""

import numpy as np
from typing import List, Dict, Any
from ...components.base import VectorStore, QueryResult, Chunk
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class SimpleVectorStore(VectorStore):
    """Simple in-memory vector store using numpy arrays"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.embeddings = None
        self.chunks = []
        logger.info("SimpleVectorStore initialized")

    def add(self, embeddings: List[List[float]], chunks: List[Chunk]) -> None:
        """Add embeddings and chunks to store"""
        if self.embeddings is None:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.chunks.extend(chunks)
        logger.debug(f"Added {len(chunks)} chunks to vector store")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        """Search for most similar vectors"""
        if self.embeddings is None or len(self.chunks) == 0:
            logger.warning("Vector store is empty")
            return []

        # Calculate similarities
        query_vec = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Create results
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                result = QueryResult(
                    chunk=chunk,
                    score=float(similarities[idx]),
                    metadata={'similarity_type': 'cosine'}
                )
                results.append(result)

        logger.debug(f"Vector search returned {len(results)} results")
        return results

    def clear(self) -> None:
        """Clear the vector store"""
        self.embeddings = None
        self.chunks = []
        logger.debug("Vector store cleared")