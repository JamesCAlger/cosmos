"""Hybrid retrieval combining dense and sparse methods"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ...components.base import Retriever, QueryResult, Chunk, Document
from loguru import logger
from collections import defaultdict


class HybridRetriever(Retriever):
    """Hybrid retriever combining multiple retrieval methods"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Retrieval components
        self.dense_retriever = None
        self.sparse_retriever = None

        # Fusion configuration
        self.fusion_method = self.config.get("fusion_method", "weighted_sum")  # weighted_sum, rrf, parallel
        self.dense_weight = self.config.get("dense_weight", 0.5)
        self.sparse_weight = self.config.get("sparse_weight", 0.5)
        self.rrf_k = self.config.get("rrf_k", 60)  # Reciprocal Rank Fusion constant
        self.normalization = self.config.get("normalization", "min_max")  # min_max, z_score, none

        logger.info(f"HybridRetriever initialized with fusion={self.fusion_method}, "
                   f"weights=[dense:{self.dense_weight}, sparse:{self.sparse_weight}]")

    def set_retrievers(self, dense_retriever: Retriever, sparse_retriever: Retriever) -> None:
        """Set the dense and sparse retrievers"""
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        logger.info("Retrievers set for hybrid retrieval")

    def _normalize_scores(self, scores: List[float], method: str = "min_max") -> List[float]:
        """Normalize scores using specified method"""
        if not scores or method == "none":
            return scores

        scores_array = np.array(scores)

        if method == "min_max":
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score > min_score:
                normalized = (scores_array - min_score) / (max_score - min_score)
            else:
                normalized = np.ones_like(scores_array)

        elif method == "z_score":
            mean = scores_array.mean()
            std = scores_array.std()
            if std > 0:
                normalized = (scores_array - mean) / std
                # Convert to 0-1 range using sigmoid
                normalized = 1 / (1 + np.exp(-normalized))
            else:
                normalized = np.ones_like(scores_array) * 0.5

        else:
            normalized = scores_array

        return normalized.tolist()

    def _weighted_sum_fusion(self, dense_results: List[QueryResult],
                           sparse_results: List[QueryResult],
                           top_k: int) -> List[QueryResult]:
        """Combine results using weighted sum of normalized scores"""
        chunk_scores = defaultdict(lambda: {"dense": 0.0, "sparse": 0.0, "chunk": None})

        # Normalize dense scores
        if dense_results:
            dense_scores = [r.score for r in dense_results]
            normalized_dense = self._normalize_scores(dense_scores, self.normalization)

            for result, norm_score in zip(dense_results, normalized_dense):
                chunk_id = result.chunk.chunk_id
                chunk_scores[chunk_id]["dense"] = norm_score * self.dense_weight
                chunk_scores[chunk_id]["chunk"] = result.chunk

        # Normalize sparse scores
        if sparse_results:
            sparse_scores = [r.score for r in sparse_results]
            normalized_sparse = self._normalize_scores(sparse_scores, self.normalization)

            for result, norm_score in zip(sparse_results, normalized_sparse):
                chunk_id = result.chunk.chunk_id
                chunk_scores[chunk_id]["sparse"] = norm_score * self.sparse_weight
                if chunk_scores[chunk_id]["chunk"] is None:
                    chunk_scores[chunk_id]["chunk"] = result.chunk

        # Calculate combined scores
        combined_results = []
        for chunk_id, scores in chunk_scores.items():
            combined_score = scores["dense"] + scores["sparse"]
            result = QueryResult(
                chunk=scores["chunk"],
                score=combined_score,
                metadata={
                    "fusion_method": "weighted_sum",
                    "dense_score": scores["dense"],
                    "sparse_score": scores["sparse"],
                    "combined_score": combined_score
                }
            )
            combined_results.append(result)

        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]

    def _rrf_fusion(self, dense_results: List[QueryResult],
                   sparse_results: List[QueryResult],
                   top_k: int) -> List[QueryResult]:
        """Reciprocal Rank Fusion (RRF) for combining results"""
        chunk_rrf_scores = defaultdict(lambda: {"score": 0.0, "chunk": None})

        # Add RRF scores from dense results
        for rank, result in enumerate(dense_results, 1):
            chunk_id = result.chunk.chunk_id
            chunk_rrf_scores[chunk_id]["score"] += 1.0 / (self.rrf_k + rank)
            chunk_rrf_scores[chunk_id]["chunk"] = result.chunk
            if "dense_rank" not in chunk_rrf_scores[chunk_id]:
                chunk_rrf_scores[chunk_id]["dense_rank"] = rank

        # Add RRF scores from sparse results
        for rank, result in enumerate(sparse_results, 1):
            chunk_id = result.chunk.chunk_id
            chunk_rrf_scores[chunk_id]["score"] += 1.0 / (self.rrf_k + rank)
            if chunk_rrf_scores[chunk_id]["chunk"] is None:
                chunk_rrf_scores[chunk_id]["chunk"] = result.chunk
            if "sparse_rank" not in chunk_rrf_scores[chunk_id]:
                chunk_rrf_scores[chunk_id]["sparse_rank"] = rank

        # Create QueryResult objects
        combined_results = []
        for chunk_id, data in chunk_rrf_scores.items():
            result = QueryResult(
                chunk=data["chunk"],
                score=data["score"],
                metadata={
                    "fusion_method": "rrf",
                    "rrf_score": data["score"],
                    "dense_rank": data.get("dense_rank", -1),
                    "sparse_rank": data.get("sparse_rank", -1)
                }
            )
            combined_results.append(result)

        # Sort by RRF score and return top-k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]

    def _parallel_fusion(self, dense_results: List[QueryResult],
                        sparse_results: List[QueryResult],
                        top_k: int) -> List[QueryResult]:
        """Parallel retrieval - keep results separate with source tracking"""
        all_results = []

        # Add dense results with source tracking
        for result in dense_results[:top_k//2]:  # Take half from each
            result.metadata = result.metadata or {}
            result.metadata["source"] = "dense"
            all_results.append(result)

        # Add sparse results with source tracking
        for result in sparse_results[:top_k//2]:
            result.metadata = result.metadata or {}
            result.metadata["source"] = "sparse"
            all_results.append(result)

        # If we need more results to reach top_k
        if len(all_results) < top_k:
            # Add remaining from dense
            remaining_dense = dense_results[top_k//2:]
            for result in remaining_dense[:top_k - len(all_results)]:
                result.metadata = result.metadata or {}
                result.metadata["source"] = "dense"
                all_results.append(result)

        return all_results[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """Retrieve using hybrid approach"""
        if not self.dense_retriever or not self.sparse_retriever:
            logger.error("Retrievers not set for hybrid retrieval")
            return []

        # Get results from both retrievers (fetch more for better fusion)
        fetch_k = min(top_k * 3, 20)  # Fetch 3x top_k for better reranking

        dense_results = self.dense_retriever.retrieve(query, fetch_k)
        sparse_results = self.sparse_retriever.retrieve(query, fetch_k)

        logger.debug(f"Dense retriever returned {len(dense_results)} results")
        logger.debug(f"Sparse retriever returned {len(sparse_results)} results")

        # Apply fusion method
        if self.fusion_method == "weighted_sum":
            results = self._weighted_sum_fusion(dense_results, sparse_results, top_k)
        elif self.fusion_method == "rrf":
            results = self._rrf_fusion(dense_results, sparse_results, top_k)
        elif self.fusion_method == "parallel":
            results = self._parallel_fusion(dense_results, sparse_results, top_k)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}, using weighted_sum")
            results = self._weighted_sum_fusion(dense_results, sparse_results, top_k)

        logger.info(f"Hybrid retrieval returned {len(results)} results using {self.fusion_method}")
        return results

    def index(self, chunks: List[Chunk]) -> None:
        """Index chunks in both retrievers"""
        if self.dense_retriever and hasattr(self.dense_retriever, 'index'):
            self.dense_retriever.index(chunks)
        if self.sparse_retriever and hasattr(self.sparse_retriever, 'index'):
            self.sparse_retriever.index(chunks)

    def clear(self) -> None:
        """Clear both retrievers"""
        if self.dense_retriever and hasattr(self.dense_retriever, 'clear'):
            self.dense_retriever.clear()
        if self.sparse_retriever and hasattr(self.sparse_retriever, 'clear'):
            self.sparse_retriever.clear()