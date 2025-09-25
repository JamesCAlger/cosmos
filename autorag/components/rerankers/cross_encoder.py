"""Cross-encoder reranker implementation"""

from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import CrossEncoder
from ...components.base import Reranker, QueryResult
from loguru import logger
import numpy as np


class CrossEncoderReranker(Reranker):
    """Reranker using cross-encoder models from sentence-transformers"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Model configuration
        self.model_name = self.config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = self.config.get("device", "cpu")
        self.batch_size = self.config.get("batch_size", 32)
        self.max_length = self.config.get("max_length", 512)
        self.show_progress = self.config.get("show_progress", False)

        # Score normalization
        self.normalize_scores = self.config.get("normalize_scores", True)

        # Initialize model
        self.model = None
        self._initialize_model()

        logger.info(f"CrossEncoderReranker initialized with model: {self.model_name}")

    def _initialize_model(self):
        """Initialize the cross-encoder model"""
        try:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            logger.warning("Reranker will operate in passthrough mode")
            self.model = None

    def rerank(self, query: str, results: List[QueryResult], top_k: int = 5) -> List[QueryResult]:
        """Rerank retrieval results using cross-encoder"""
        if not results:
            return []

        if not self.model:
            logger.warning("No model loaded, returning original results")
            return results[:top_k]

        # Prepare query-document pairs
        pairs = [(query, result.chunk.content) for result in results]

        # Get scores from cross-encoder
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress
            )
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results[:top_k]

        # Normalize scores if configured
        if self.normalize_scores:
            # Convert to 0-1 range using sigmoid
            scores = 1 / (1 + np.exp(-scores))

        # Create reranked results with new scores
        reranked_results = []
        for result, score in zip(results, scores):
            # Create new QueryResult with updated score
            reranked = QueryResult(
                chunk=result.chunk,
                score=float(score),
                metadata={
                    **(result.metadata or {}),
                    "original_score": result.score,
                    "reranker_score": float(score),
                    "reranker_model": self.model_name
                }
            )
            reranked_results.append(reranked)

        # Sort by reranker score
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        # Return top-k
        final_results = reranked_results[:top_k]

        logger.debug(f"Reranked {len(results)} results, returning top {len(final_results)}")
        return final_results

    def batch_rerank(self, queries: List[str], results_list: List[List[QueryResult]],
                    top_k: int = 5) -> List[List[QueryResult]]:
        """Batch reranking for multiple queries"""
        reranked_list = []
        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k)
            reranked_list.append(reranked)
        return reranked_list