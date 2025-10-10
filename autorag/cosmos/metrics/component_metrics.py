"""
Component Metrics Computation

Isolated metric computation for each RAG component type.
Extracted from EnhancedMetricsCollector for reusability.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class ComponentMetrics:
    """
    Compute component-intrinsic metrics without full pipeline execution.

    This class provides isolated metric computation for each component type,
    enabling component-level optimization without circular dependencies.
    """

    def __init__(self, semantic_evaluator=None):
        """
        Initialize component metrics calculator

        Args:
            semantic_evaluator: SemanticMetrics instance for semantic similarity computation
        """
        self.semantic_evaluator = semantic_evaluator

    def compute_chunking_metrics(self,
                                  chunks: List,
                                  latency: float,
                                  compute_coherence: bool = True) -> Dict[str, float]:
        """
        Compute chunking quality metrics

        Extracted from EnhancedMetricsCollector lines 119-145

        Args:
            chunks: List of chunk objects (with .content attribute)
            latency: Time taken to chunk in seconds
            compute_coherence: Whether to compute semantic coherence (slower)

        Returns:
            Dictionary with metrics:
            - time: Chunking latency in seconds
            - chunks_created: Number of chunks produced
            - avg_chunk_size: Average chunk size in words
            - size_variance: Standard deviation of chunk sizes
            - semantic_coherence: Semantic similarity between chunks (if computed)
        """
        if not chunks:
            return {
                'time': latency,
                'chunks_created': 0,
                'avg_chunk_size': 0.0,
                'size_variance': 0.0,
                'semantic_coherence': 0.0
            }

        # Extract text content from chunks
        chunk_texts = []
        for c in chunks:
            if hasattr(c, 'content'):
                chunk_texts.append(c.content)
            else:
                chunk_texts.append(str(c))

        # Calculate chunk sizes in words
        chunk_sizes = [len(text.split()) for text in chunk_texts]

        # Semantic coherence: How similar are chunks to each other
        # (should be moderate - too similar = redundant, too different = incoherent)
        coherence = 0.5  # Default if can't compute

        if compute_coherence and self.semantic_evaluator and len(chunk_texts) > 1:
            try:
                # Limit to first 10 chunks for speed
                sample_texts = chunk_texts[:10]
                chunk_embeddings = self.semantic_evaluator.model.encode(sample_texts)

                # Compute pairwise similarities
                similarities = cosine_similarity(chunk_embeddings)

                # Average of upper triangle (excluding diagonal)
                coherence = float(np.mean(similarities[np.triu_indices_from(similarities, k=1)]))
            except Exception as e:
                logger.warning(f"Could not compute semantic coherence: {e}")
                coherence = 0.5

        metrics = {
            'time': float(latency),
            'chunks_created': len(chunks),
            'avg_chunk_size': float(np.mean(chunk_sizes)) if chunk_sizes else 0.0,
            'size_variance': float(np.std(chunk_sizes)) if chunk_sizes else 0.0,
            'semantic_coherence': float(coherence)
        }

        logger.debug(f"Chunking metrics: {metrics}")
        return metrics

    def compute_retrieval_metrics(self,
                                   query: str,
                                   results: List,
                                   latency: float,
                                   ground_truth: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute retrieval quality metrics

        Extracted from EnhancedMetricsCollector lines 166-190

        Args:
            query: Query string
            results: List of retrieval results (QueryResult objects)
            latency: Time taken to retrieve in seconds
            ground_truth: Optional ground truth data with 'relevant_chunks' field

        Returns:
            Dictionary with metrics:
            - time: Retrieval latency in seconds
            - docs_retrieved: Number of documents retrieved
            - avg_relevance: Average semantic relevance to query
            - max_relevance: Best relevance score
            - min_relevance: Worst relevance score
            - precision: Precision if ground truth available
            - score_spread: Difference between max and min relevance
        """
        if not results:
            return {
                'time': latency,
                'docs_retrieved': 0,
                'avg_relevance': 0.0,
                'max_relevance': 0.0,
                'min_relevance': 0.0,
                'precision': 0.0,
                'score_spread': 0.0
            }

        # Extract text from results
        retrieved_texts = []
        for r in results:
            if hasattr(r, 'content'):
                retrieved_texts.append(r.content)
            elif hasattr(r, 'chunk') and hasattr(r.chunk, 'content'):
                retrieved_texts.append(r.chunk.content)
            else:
                retrieved_texts.append(str(r))

        # Calculate semantic relevance to query
        relevance_scores = []
        if self.semantic_evaluator and query and retrieved_texts:
            try:
                query_embedding = self.semantic_evaluator.model.encode([query])
                retrieved_embeddings = self.semantic_evaluator.model.encode(retrieved_texts)

                # Compute cosine similarity
                relevance_scores = cosine_similarity(query_embedding, retrieved_embeddings)[0]
                relevance_scores = [float(s) for s in relevance_scores]
            except Exception as e:
                logger.warning(f"Could not compute relevance scores: {e}")
                relevance_scores = [0.5] * len(retrieved_texts)
        else:
            # Fallback: use uniform scores
            relevance_scores = [0.5] * len(retrieved_texts)

        # Calculate precision if ground truth available
        precision = 0.0
        if ground_truth and 'relevant_chunks' in ground_truth:
            relevant_indices = set(ground_truth['relevant_chunks'])
            retrieved_indices = set()
            for i, r in enumerate(results):
                if hasattr(r, 'doc_id'):
                    retrieved_indices.add(r.doc_id)
                elif hasattr(r, 'chunk') and hasattr(r.chunk, 'doc_id'):
                    retrieved_indices.add(r.chunk.doc_id)
                else:
                    retrieved_indices.add(str(i))

            if retrieved_indices:
                precision = len(relevant_indices & retrieved_indices) / len(retrieved_indices)
        else:
            # Use semantic similarity as proxy for precision
            # Count proportion of retrieved docs with high relevance (> 0.7)
            if relevance_scores:
                high_relevance_count = sum(1 for s in relevance_scores if s > 0.7)
                precision = float(high_relevance_count) / len(relevance_scores)
            else:
                precision = 0.0

        metrics = {
            'time': float(latency),
            'docs_retrieved': len(results),
            'avg_relevance': float(np.mean(relevance_scores)) if relevance_scores else 0.0,
            'max_relevance': float(np.max(relevance_scores)) if relevance_scores else 0.0,
            'min_relevance': float(np.min(relevance_scores)) if relevance_scores else 0.0,
            'precision': float(precision),
            'score_spread': float(np.max(relevance_scores) - np.min(relevance_scores)) if relevance_scores else 0.0
        }

        logger.debug(f"Retrieval metrics: {metrics}")
        return metrics

    def compute_reranking_metrics(self,
                                   query: str,
                                   original_results: List,
                                   reranked_results: List,
                                   latency: float) -> Dict[str, float]:
        """
        Compute reranking quality metrics

        Args:
            query: Query string
            original_results: Results before reranking (List[QueryResult])
            reranked_results: Results after reranking (List[QueryResult])
            latency: Time taken to rerank in seconds

        Returns:
            Dictionary with metrics:
            - latency: Reranking time in seconds
            - score_change: Mean absolute score change [0-1]
            - rank_correlation: Kendall's tau correlation [-1 to 1]
            - top_k_overlap: Proportion of original top-k retained [0-1]

        Metric Interpretation (what is "good"):
            rank_correlation:
                * -0.8 to 0.8 = effective reordering ✓
                * > 0.8 = no-op passthrough ✗
                * < -0.9 = extreme inverse (overfitting?) ~

            score_change:
                * 0.2 to 0.5 = meaningful rescoring ✓
                * < 0.1 = minimal change ✗
                * > 0.7 = aggressive rescoring ~

            top_k_overlap:
                * 0.4 to 0.7 = effective refinement ✓
                * > 0.9 = minimal reordering ✗
                * < 0.3 = complete replacement ~

            latency:
                * < 500ms = acceptable ✓
                * > 1000ms = too slow ✗
        """
        # Handle edge cases
        if not original_results or not reranked_results:
            return {
                'latency': latency,
                'score_change': 0.0,
                'rank_correlation': 0.0,
                'top_k_overlap': 0.0
            }

        # Single result - can't compute correlation
        if len(original_results) < 2:
            return {
                'latency': latency,
                'score_change': 0.0,
                'rank_correlation': 1.0,  # Perfect correlation (trivial)
                'top_k_overlap': 1.0
            }

        # Extract scores and IDs
        orig_scores = np.array([r.score for r in original_results])
        orig_ids = [r.chunk.chunk_id if hasattr(r.chunk, 'chunk_id') else id(r.chunk)
                    for r in original_results]

        reranked_ids = [r.chunk.chunk_id if hasattr(r.chunk, 'chunk_id') else id(r.chunk)
                        for r in reranked_results]

        # 1. Absolute score change
        # Match reranked results to original by ID
        # Use absolute change for normalized scores [0,1]
        # Cross-encoder outputs are sigmoid-normalized, so absolute difference is meaningful
        score_changes = []
        for i, orig_id in enumerate(orig_ids):
            if orig_id in reranked_ids:
                rerank_idx = reranked_ids.index(orig_id)
                orig_score = orig_scores[i]
                new_score = reranked_results[rerank_idx].score

                # Absolute change (appropriate for normalized [0,1] scores)
                absolute_change = abs(new_score - orig_score)
                score_changes.append(absolute_change)

        mean_score_change = float(np.mean(score_changes)) if score_changes else 0.0

        # 2. Rank correlation (Kendall's tau)
        from scipy.stats import kendalltau

        # Map chunk IDs to original ranks
        orig_rank_map = {chunk_id: rank for rank, chunk_id in enumerate(orig_ids)}

        # Get ranks for reranked results (only for chunks that appear in both)
        common_ids = [rid for rid in reranked_ids if rid in orig_rank_map]

        if len(common_ids) >= 2:
            orig_ranks = [orig_rank_map[rid] for rid in common_ids]
            reranked_ranks = list(range(len(common_ids)))  # 0, 1, 2, ...

            tau, _ = kendalltau(orig_ranks, reranked_ranks)
            rank_correlation = float(tau) if not np.isnan(tau) else 0.0
        else:
            rank_correlation = 0.0

        # 3. Top-k overlap (proportion of original top-k retained)
        k = min(5, len(original_results), len(reranked_results))
        orig_top_k_ids = set(orig_ids[:k])
        reranked_top_k_ids = set(reranked_ids[:k])

        overlap = len(orig_top_k_ids & reranked_top_k_ids) / k if k > 0 else 0.0

        metrics = {
            'latency': float(latency),
            'score_change': float(mean_score_change),
            'rank_correlation': float(rank_correlation),
            'top_k_overlap': float(overlap)
        }

        logger.debug(f"Reranking metrics: {metrics}")
        return metrics

    def compute_generation_metrics(self,
                                    query: str,
                                    answer: str,
                                    context: List,
                                    latency: float,
                                    ground_truth_answer: Optional[str] = None) -> Dict[str, float]:
        """
        Compute generation quality metrics

        Extracted from EnhancedMetricsCollector lines 228-257

        Args:
            query: Query string
            answer: Generated answer string
            context: List of context documents/chunks used
            latency: Time taken to generate in seconds
            ground_truth_answer: Optional ground truth answer for accuracy

        Returns:
            Dictionary with metrics:
            - time: Generation latency in seconds
            - answer_length: Length of answer in words
            - answer_relevance: Semantic similarity to query
            - context_utilization: Proportion of context words used in answer
            - accuracy: Semantic similarity to ground truth (if available)
        """
        if not answer or len(answer) < 10:
            return {
                'time': latency,
                'answer_length': len(answer.split()) if answer else 0,
                'answer_relevance': 0.0,
                'context_utilization': 0.0,
                'accuracy': 0.0
            }

        # Answer relevance to query
        answer_relevance = 0.5  # Default
        if self.semantic_evaluator and query:
            try:
                answer_emb = self.semantic_evaluator.model.encode([answer])
                query_emb = self.semantic_evaluator.model.encode([query])
                answer_relevance = float(self.semantic_evaluator.compute_similarity(
                    answer_emb[0], query_emb[0]
                ))
            except Exception as e:
                logger.warning(f"Could not compute answer relevance: {e}")

        # Context utilization: How much of the context is reflected in the answer
        context_utilization = 0.0
        if context:
            try:
                # Extract text from context
                context_texts = []
                for c in context:
                    if isinstance(c, str):
                        context_texts.append(c)
                    elif hasattr(c, 'content'):
                        context_texts.append(c.content)
                    elif hasattr(c, 'chunk') and hasattr(c.chunk, 'content'):
                        context_texts.append(c.chunk.content)
                    else:
                        context_texts.append(str(c))

                context_text = ' '.join(context_texts)

                # Word-level overlap
                answer_words = set(answer.lower().split())
                context_words = set(context_text.lower().split())

                if answer_words:
                    context_utilization = len(answer_words & context_words) / len(answer_words)
            except Exception as e:
                logger.warning(f"Could not compute context utilization: {e}")

        # Accuracy against ground truth
        accuracy = answer_relevance  # Default to relevance if no ground truth
        if ground_truth_answer and self.semantic_evaluator:
            try:
                answer_emb = self.semantic_evaluator.model.encode([answer])
                truth_emb = self.semantic_evaluator.model.encode([ground_truth_answer])
                accuracy = float(self.semantic_evaluator.compute_similarity(
                    answer_emb[0], truth_emb[0]
                ))
            except Exception as e:
                logger.warning(f"Could not compute accuracy: {e}")

        metrics = {
            'time': float(latency),
            'answer_length': len(answer.split()),
            'answer_relevance': float(answer_relevance),
            'context_utilization': float(context_utilization),
            'accuracy': float(accuracy)
        }

        logger.debug(f"Generation metrics: {metrics}")
        return metrics

    def compute_quality_score(self,
                             component_type: str,
                             metrics: Dict[str, float]) -> float:
        """
        Compute overall quality score for a component based on its metrics

        This is a heuristic scoring function that combines multiple metrics
        into a single score for optimization.

        Args:
            component_type: Type of component ('chunker', 'retriever', 'generator')
            metrics: Dictionary of metrics from compute_*_metrics methods

        Returns:
            Quality score in range [0, 1] where higher is better
        """
        if component_type == 'chunker':
            # Chunker quality: balance chunk size and coherence
            # Target: moderate chunk sizes (200-400 words), good coherence
            target_size = 300
            size_score = 1.0 - min(abs(metrics['avg_chunk_size'] - target_size) / target_size, 1.0)

            # Penalize high variance (inconsistent chunking)
            consistency_score = 1.0 - min(metrics['size_variance'] / 100, 1.0)

            # Coherence should be moderate (not too high = redundant, not too low = incoherent)
            coherence_score = metrics['semantic_coherence']

            # Weighted combination
            quality = 0.4 * size_score + 0.3 * consistency_score + 0.3 * coherence_score

        elif component_type == 'retriever':
            # Retriever quality: primarily relevance and precision
            quality = 0.7 * metrics['avg_relevance'] + 0.3 * metrics['precision']

        elif component_type == 'reranker':
            # Reranker quality: effective reordering with meaningful rescoring
            rank_corr = metrics.get('rank_correlation', 0.0)
            score_change = metrics.get('score_change', 0.0)
            latency = metrics.get('latency', 0.0)

            # Reordering quality: reward moderate changes, penalize extremes
            # - High positive correlation (>0.8): no-op passthrough = bad
            # - Extreme negative correlation (<-0.9): possible overfitting = questionable
            # - Sweet spot (-0.8 to 0.8): effective reordering = good
            if rank_corr > 0.8:
                # No-op: minimal reordering
                reordering_quality = 0.2
            elif rank_corr < -0.9:
                # Extreme inverse: possible overfitting
                reordering_quality = 0.6
            else:
                # Sweet spot: effective reordering
                # Quality peaks at moderate correlation (around 0)
                reordering_quality = 0.8 + 0.2 * (1 - abs(rank_corr) / 0.9)

            # Rescoring quality: target moderate score changes (0.2-0.5)
            # Too low = passthrough, too high = overfitting
            target_change = 0.35
            normalized_change = min(score_change, 1.0)  # Clamp to [0,1]
            rescoring_quality = 1.0 - min(abs(normalized_change - target_change) / target_change, 1.0)

            # Base quality: prioritize reordering (70%) over rescoring (30%)
            base_quality = 0.7 * reordering_quality + 0.3 * rescoring_quality

            # Additional penalty for extreme no-op (belt and suspenders)
            if abs(rank_corr) > 0.95 and score_change < 0.05:
                base_quality *= 0.5

            # Latency penalty (>500ms)
            latency_penalty = max(0, (latency - 0.5) * 0.1)

            quality = max(0.0, base_quality - latency_penalty)

        elif component_type == 'generator':
            # Generator quality: accuracy and relevance, penalize poor utilization
            accuracy = metrics['accuracy']
            relevance = metrics['answer_relevance']
            utilization = metrics['context_utilization']

            # Base quality from accuracy/relevance
            base_quality = 0.6 * accuracy + 0.4 * relevance

            # Penalize very low or very high utilization
            # Low = not using context, high = just copying context
            utilization_penalty = 0.0
            if utilization < 0.1 or utilization > 0.9:
                utilization_penalty = 0.1

            quality = base_quality - utilization_penalty

        else:
            raise ValueError(f"Unknown component type: {component_type}")

        # Ensure score is in [0, 1], handling NaN values properly
        # NaN values are converted to 0.0 (failed evaluation)
        quality = float(np.clip(np.nan_to_num(quality, nan=0.0), 0.0, 1.0))

        return quality