"""Retrieval component for Week 1 minimal implementation"""

import faiss
import numpy as np
from typing import List, Tuple
from loguru import logger
from .document_processor import Chunk


class FAISSRetriever:
    """Simple FAISS-based dense retriever with cosine similarity"""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        logger.info(f"Initialized FAISS Retriever with dimension={dimension}")

    def index_chunks(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """Index embeddings and store associated chunks"""
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) must have same length")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index (using Inner Product for cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(f"Indexed {len(chunks)} chunks")

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k most similar chunks"""
        if self.index is None:
            raise ValueError("Index not initialized. Call index_chunks first.")

        # Reshape and normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # FAISS returns -1 for empty results
                results.append((self.chunks[idx], float(score)))

        logger.debug(f"Retrieved {len(results)} chunks")
        return results