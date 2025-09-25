"""
External metrics collection for RAG pipeline components.
Collects metrics without modifying existing components.
This is the quickest way to add metrics to the pipeline for Bayesian optimization.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Container for stage-specific metrics"""
    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time(self) -> float:
        """Calculate execution time for this stage"""
        return self.end_time - self.start_time


class ExternalMetricsCollector:
    """
    Collects metrics without modifying existing components.
    This is the quickest way to add metrics to the pipeline.
    """

    def __init__(self, pipeline):
        """
        Initialize the metrics collector.

        Args:
            pipeline: The RAG pipeline to collect metrics from
        """
        self.pipeline = pipeline
        self.metrics = {}
        self.stage_metrics = []

    def evaluate_with_metrics(self, query: str, documents: List[str],
                             ground_truth: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Wraps pipeline execution with metric collection.

        Args:
            query: The query string
            documents: List of documents to process
            ground_truth: Optional ground truth data for evaluation

        Returns:
            answer: Generated answer
            metrics: Dict with stage-wise metrics
        """
        metrics = {}

        # Track overall pipeline start
        pipeline_start = time.time()

        # Chunking metrics
        chunk_metrics = self._collect_chunking_metrics(documents)
        metrics['chunking'] = chunk_metrics
        chunks = chunk_metrics.pop('chunks', [])  # Remove chunks from metrics

        # Retrieval metrics
        retrieval_metrics = self._collect_retrieval_metrics(query, chunks, ground_truth)
        metrics['retrieval'] = retrieval_metrics
        retrieved = retrieval_metrics.pop('retrieved_docs', [])  # Remove docs from metrics

        # Generation metrics
        generation_metrics, answer = self._collect_generation_metrics(query, retrieved)
        metrics['generation'] = generation_metrics

        # Overall pipeline metrics
        pipeline_end = time.time()
        metrics['total'] = {
            'pipeline_time': pipeline_end - pipeline_start,
            'estimated_cost': self._estimate_cost(metrics),
            'success': bool(answer),
            'stages_completed': 3
        }

        # Add quality metrics if we have them
        if hasattr(self.pipeline, 'evaluator'):
            quality_metrics = self._collect_quality_metrics(query, answer, retrieved, ground_truth)
            metrics['quality'] = quality_metrics

        return answer, metrics

    def _collect_chunking_metrics(self, documents: List[str]) -> Dict[str, Any]:
        """Collect chunking stage metrics"""
        start = time.time()

        # Get chunks from pipeline chunker
        if hasattr(self.pipeline, 'chunker'):
            # Convert strings to Document objects if needed
            from autorag.components.base import Document
            if documents and isinstance(documents[0], str):
                doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
            else:
                doc_objects = documents

            chunk_objects = self.pipeline.chunker.chunk(doc_objects)
            # Convert chunk objects back to strings
            chunks = [c.content if hasattr(c, 'content') else str(c) for c in chunk_objects]
        else:
            # Fallback for pipelines without explicit chunker
            chunks = documents

        end = time.time()

        # Calculate chunk statistics
        chunk_sizes = [len(c.split()) for c in chunks] if chunks else []

        metrics = {
            'time': end - start,
            'chunks_count': len(chunks),
            'avg_size': np.mean(chunk_sizes) if chunk_sizes else 0,
            'size_std': np.std(chunk_sizes) if chunk_sizes else 0,
            'total_chars': sum(len(c) for c in chunks) if chunks else 0,
            'min_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_size': max(chunk_sizes) if chunk_sizes else 0,
            'chunks': chunks  # Will be removed after use
        }

        # Size coefficient of variation (consistency metric)
        if metrics['avg_size'] > 0:
            metrics['size_cv'] = metrics['size_std'] / metrics['avg_size']
        else:
            metrics['size_cv'] = 0

        return metrics

    def _collect_retrieval_metrics(self, query: str, chunks: List[str],
                                  ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Collect retrieval stage metrics"""
        start = time.time()

        # Get retrieved documents
        if hasattr(self.pipeline, 'retriever'):
            # Try to get scored results
            try:
                # Index chunks first if retriever needs it
                from autorag.components.base import Document
                if hasattr(self.pipeline.retriever, 'index'):
                    chunk_docs = [Document(content=c, doc_id=str(i)) for i, c in enumerate(chunks)]
                    self.pipeline.retriever.index(chunk_docs)

                retrieved = self.pipeline.retriever.retrieve(query, top_k=min(10, len(chunks)))
                # Check if results have scores
                if retrieved and hasattr(retrieved[0], 'score'):
                    scores = [r.score for r in retrieved]
                    retrieved_texts = [r.content if hasattr(r, 'content') else
                                     r.text if hasattr(r, 'text') else str(r) for r in retrieved]
                else:
                    # No scores, treat as simple list
                    scores = list(range(len(retrieved), 0, -1))  # Fake descending scores
                    retrieved_texts = [r.content if hasattr(r, 'content') else str(r) for r in retrieved]
            except Exception as e:
                logger.warning(f"Error during retrieval: {e}")
                retrieved_texts = chunks[:5] if chunks else []
                scores = list(range(len(retrieved_texts), 0, -1))
        else:
            # Fallback: use first few chunks
            retrieved_texts = chunks[:5] if chunks else []
            scores = list(range(len(retrieved_texts), 0, -1))

        end = time.time()

        metrics = {
            'time': end - start,
            'docs_retrieved': len(retrieved_texts),
            'top_score': scores[0] if scores else 0,
            'avg_score': np.mean(scores) if scores else 0,
            'score_spread': max(scores) - min(scores) if len(scores) > 1 else 0,
            'score_std': np.std(scores) if scores else 0,
            'retrieved_docs': retrieved_texts  # Will be removed after use
        }

        # Add recall metrics if ground truth is available
        if ground_truth and 'relevant_docs' in ground_truth:
            relevant_set = set(ground_truth['relevant_docs'])
            retrieved_set = set(range(len(retrieved_texts)))  # Use indices

            if relevant_set:
                metrics['recall@k'] = len(relevant_set & retrieved_set) / len(relevant_set)
                metrics['precision@k'] = len(relevant_set & retrieved_set) / len(retrieved_set) if retrieved_set else 0

        return metrics

    def _collect_generation_metrics(self, query: str, retrieved_docs: List[str]) -> Tuple[Dict[str, Any], str]:
        """Collect generation stage metrics"""
        start = time.time()

        # Generate answer
        if hasattr(self.pipeline, 'generator'):
            # Limit context to top 5 docs
            context = retrieved_docs[:5] if retrieved_docs else []
            answer = self.pipeline.generator.generate(query, context)
        else:
            # Fallback
            answer = f"Based on the retrieved information: {retrieved_docs[0][:100]}..." if retrieved_docs else "No answer generated."

        end = time.time()

        # Estimate token usage
        input_text = query + ' '.join(retrieved_docs[:5]) if retrieved_docs else query
        estimated_input_tokens = len(input_text.split()) * 1.3  # Rough estimate
        estimated_output_tokens = len(answer.split()) * 1.3

        metrics = {
            'time': end - start,
            'answer_length': len(answer.split()),
            'answer_chars': len(answer),
            'estimated_input_tokens': int(estimated_input_tokens),
            'estimated_output_tokens': int(estimated_output_tokens),
            'estimated_total_tokens': int(estimated_input_tokens + estimated_output_tokens),
            'context_docs_used': min(5, len(retrieved_docs)),
            'answer_generated': bool(answer and answer != "No answer generated.")
        }

        return metrics, answer

    def _collect_quality_metrics(self, query: str, answer: str, context: List[str],
                                ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Collect quality metrics (if evaluator is available)"""
        metrics = {}

        # Basic quality checks
        metrics['has_answer'] = bool(answer and len(answer) > 10)
        metrics['uses_context'] = self._check_context_usage(answer, context)
        metrics['answer_relevance_to_query'] = self._check_query_relevance(query, answer)

        # If we have ground truth answer
        if ground_truth and 'answer' in ground_truth:
            expected = ground_truth['answer'].lower()
            actual = answer.lower()

            # Simple similarity check
            common_words = set(expected.split()) & set(actual.split())
            metrics['answer_overlap'] = len(common_words) / max(len(expected.split()), 1)

        return metrics

    def _check_context_usage(self, answer: str, context: List[str]) -> float:
        """Check how much the answer uses the provided context"""
        if not context or not answer:
            return 0.0

        answer_words = set(answer.lower().split())
        context_words = set(' '.join(context).lower().split())

        if not context_words:
            return 0.0

        overlap = len(answer_words & context_words)
        return min(1.0, overlap / (len(answer_words) + 1))

    def _check_query_relevance(self, query: str, answer: str) -> float:
        """Check if answer addresses the query"""
        if not query or not answer:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on', 'at', 'to', 'for'}
        query_words -= stop_words

        if not query_words:
            return 0.5  # Neutral score for queries with only stop words

        overlap = len(query_words & answer_words)
        return min(1.0, overlap / len(query_words))

    def _estimate_cost(self, metrics: Dict[str, Any]) -> float:
        """
        Estimate the cost of the pipeline execution.
        Based on OpenAI pricing (rough estimates).
        """
        # Rough cost estimates (in dollars)
        # GPT-3.5-turbo: $0.001 per 1K input tokens, $0.002 per 1K output tokens
        # text-embedding-ada-002: $0.0001 per 1K tokens

        cost = 0.0

        # Embedding cost (for retrieval)
        if 'retrieval' in metrics:
            embedding_tokens = metrics['retrieval'].get('docs_retrieved', 0) * 100  # Rough estimate
            cost += (embedding_tokens / 1000) * 0.0001

        # Generation cost
        if 'generation' in metrics:
            input_tokens = metrics['generation'].get('estimated_input_tokens', 0)
            output_tokens = metrics['generation'].get('estimated_output_tokens', 0)
            cost += (input_tokens / 1000) * 0.001
            cost += (output_tokens / 1000) * 0.002

        return round(cost, 6)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        if not self.stage_metrics:
            return {}

        summary = {
            'total_evaluations': len(self.stage_metrics),
            'avg_pipeline_time': np.mean([m.execution_time for m in self.stage_metrics]),
            'total_cost': sum(m.metrics.get('cost', 0) for m in self.stage_metrics),
            'success_rate': np.mean([m.metrics.get('success', False) for m in self.stage_metrics])
        }

        return summary