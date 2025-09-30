"""
Unit tests for ComponentMetrics

Tests that metric computation works correctly for each component type.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from autorag.cosmos.metrics import ComponentMetrics
from autorag.components.base import Chunk, QueryResult


class TestChunkingMetrics:
    """Test chunking metric computation"""

    def test_compute_chunking_metrics_basic(self):
        """Test basic chunking metrics without semantic evaluator"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        # Create mock chunks
        chunks = [
            Mock(content="This is the first chunk with some words"),
            Mock(content="This is the second chunk with different words"),
            Mock(content="This is the third chunk")
        ]

        latency = 0.5
        metrics = metrics_calc.compute_chunking_metrics(chunks, latency, compute_coherence=False)

        # Verify metrics structure
        assert 'time' in metrics
        assert 'chunks_created' in metrics
        assert 'avg_chunk_size' in metrics
        assert 'size_variance' in metrics
        assert 'semantic_coherence' in metrics

        # Verify values
        assert metrics['time'] == 0.5
        assert metrics['chunks_created'] == 3
        assert metrics['avg_chunk_size'] > 0
        assert metrics['size_variance'] >= 0
        assert 0.0 <= metrics['semantic_coherence'] <= 1.0

    def test_compute_chunking_metrics_empty(self):
        """Test chunking metrics with empty input"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        metrics = metrics_calc.compute_chunking_metrics([], 0.1)

        assert metrics['chunks_created'] == 0
        assert metrics['avg_chunk_size'] == 0.0
        assert metrics['size_variance'] == 0.0

    def test_compute_chunking_metrics_with_coherence(self):
        """Test chunking metrics with semantic coherence computation"""
        # Mock semantic evaluator
        mock_evaluator = Mock()
        mock_evaluator.model = Mock()
        mock_evaluator.model.encode = Mock(return_value=np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
            [0.12, 0.22, 0.32]
        ]))

        metrics_calc = ComponentMetrics(semantic_evaluator=mock_evaluator)

        chunks = [
            Mock(content="First chunk"),
            Mock(content="Second chunk"),
            Mock(content="Third chunk")
        ]

        metrics = metrics_calc.compute_chunking_metrics(chunks, 0.5, compute_coherence=True)

        # Should have computed coherence
        assert mock_evaluator.model.encode.called
        assert 0.0 <= metrics['semantic_coherence'] <= 1.0


class TestRetrievalMetrics:
    """Test retrieval metric computation"""

    def test_compute_retrieval_metrics_basic(self):
        """Test basic retrieval metrics"""
        # Mock semantic evaluator
        mock_evaluator = Mock()
        mock_evaluator.model = Mock()
        mock_evaluator.model.encode = Mock(side_effect=[
            np.array([[0.1, 0.2, 0.3]]),  # Query embedding
            np.array([  # Retrieved embeddings
                [0.15, 0.25, 0.35],
                [0.12, 0.22, 0.32],
                [0.18, 0.28, 0.38]
            ])
        ])

        metrics_calc = ComponentMetrics(semantic_evaluator=mock_evaluator)

        # Create mock results
        results = [
            Mock(content="Retrieved document 1", doc_id="1"),
            Mock(content="Retrieved document 2", doc_id="2"),
            Mock(content="Retrieved document 3", doc_id="3")
        ]

        query = "test query"
        latency = 0.3

        metrics = metrics_calc.compute_retrieval_metrics(query, results, latency)

        # Verify metrics structure
        assert 'time' in metrics
        assert 'docs_retrieved' in metrics
        assert 'avg_relevance' in metrics
        assert 'max_relevance' in metrics
        assert 'min_relevance' in metrics
        assert 'precision' in metrics
        assert 'score_spread' in metrics

        # Verify values
        assert metrics['time'] == 0.3
        assert metrics['docs_retrieved'] == 3
        assert 0.0 <= metrics['avg_relevance'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0

    def test_compute_retrieval_metrics_empty(self):
        """Test retrieval metrics with no results"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        metrics = metrics_calc.compute_retrieval_metrics("query", [], 0.1)

        assert metrics['docs_retrieved'] == 0
        assert metrics['avg_relevance'] == 0.0
        assert metrics['precision'] == 0.0

    def test_compute_retrieval_metrics_with_ground_truth(self):
        """Test retrieval metrics with ground truth"""
        mock_evaluator = Mock()
        mock_evaluator.model = Mock()
        mock_evaluator.model.encode = Mock(side_effect=[
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0.15, 0.25, 0.35], [0.12, 0.22, 0.32]])
        ])

        metrics_calc = ComponentMetrics(semantic_evaluator=mock_evaluator)

        results = [
            Mock(content="Doc 1", doc_id="1", chunk=Mock(doc_id="1")),
            Mock(content="Doc 2", doc_id="2", chunk=Mock(doc_id="2"))
        ]

        ground_truth = {'relevant_chunks': ['1', '3']}

        metrics = metrics_calc.compute_retrieval_metrics(
            "query", results, 0.2, ground_truth=ground_truth
        )

        # Should compute precision based on ground truth
        # 1 relevant out of 2 retrieved = 0.5 precision
        assert metrics['precision'] == 0.5


class TestGenerationMetrics:
    """Test generation metric computation"""

    def test_compute_generation_metrics_basic(self):
        """Test basic generation metrics"""
        # Mock semantic evaluator
        mock_evaluator = Mock()
        mock_evaluator.model = Mock()
        mock_evaluator.model.encode = Mock(side_effect=[
            np.array([[0.1, 0.2, 0.3]]),  # Answer
            np.array([[0.15, 0.25, 0.35]]),  # Query
        ])
        mock_evaluator.compute_similarity = Mock(return_value=0.85)

        metrics_calc = ComponentMetrics(semantic_evaluator=mock_evaluator)

        query = "What is machine learning?"
        answer = "Machine learning is a subset of AI that enables systems to learn from data."
        context = [
            Mock(content="ML is part of AI"),
            Mock(content="It learns from data")
        ]
        latency = 1.2

        metrics = metrics_calc.compute_generation_metrics(query, answer, context, latency)

        # Verify metrics structure
        assert 'time' in metrics
        assert 'answer_length' in metrics
        assert 'answer_relevance' in metrics
        assert 'context_utilization' in metrics
        assert 'accuracy' in metrics

        # Verify values
        assert metrics['time'] == 1.2
        assert metrics['answer_length'] > 0
        assert 0.0 <= metrics['answer_relevance'] <= 1.0
        assert 0.0 <= metrics['context_utilization'] <= 1.0

    def test_compute_generation_metrics_empty_answer(self):
        """Test generation metrics with empty answer"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        metrics = metrics_calc.compute_generation_metrics("query", "", [], 0.5)

        assert metrics['answer_length'] == 0
        assert metrics['answer_relevance'] == 0.0
        assert metrics['context_utilization'] == 0.0

    def test_compute_generation_metrics_with_ground_truth(self):
        """Test generation metrics with ground truth answer"""
        mock_evaluator = Mock()
        mock_evaluator.model = Mock()
        mock_evaluator.model.encode = Mock(side_effect=[
            np.array([[0.1, 0.2, 0.3]]),  # Answer for relevance
            np.array([[0.15, 0.25, 0.35]]),  # Query
            np.array([[0.1, 0.2, 0.3]]),  # Answer for accuracy
            np.array([[0.11, 0.21, 0.31]])  # Ground truth
        ])
        mock_evaluator.compute_similarity = Mock(side_effect=[0.75, 0.90])

        metrics_calc = ComponentMetrics(semantic_evaluator=mock_evaluator)

        query = "What is AI?"
        answer = "AI is artificial intelligence"
        ground_truth = "Artificial intelligence is the simulation of human intelligence"
        context = [Mock(content="AI context")]

        metrics = metrics_calc.compute_generation_metrics(
            query, answer, context, 1.0, ground_truth_answer=ground_truth
        )

        # Should compute accuracy against ground truth
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 0.90


class TestQualityScore:
    """Test overall quality score computation"""

    def test_quality_score_chunker(self):
        """Test quality score for chunker"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        # Good chunker metrics
        metrics = {
            'avg_chunk_size': 300,  # Target size
            'size_variance': 50,  # Low variance
            'semantic_coherence': 0.7  # Good coherence
        }

        score = metrics_calc.compute_quality_score('chunker', metrics)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be good quality

    def test_quality_score_retriever(self):
        """Test quality score for retriever"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        # Good retriever metrics
        metrics = {
            'avg_relevance': 0.8,
            'precision': 0.7
        }

        score = metrics_calc.compute_quality_score('retriever', metrics)

        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high quality

    def test_quality_score_generator(self):
        """Test quality score for generator"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        # Good generator metrics
        metrics = {
            'accuracy': 0.85,
            'answer_relevance': 0.80,
            'context_utilization': 0.5  # Moderate utilization
        }

        score = metrics_calc.compute_quality_score('generator', metrics)

        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high quality

    def test_quality_score_invalid_type(self):
        """Test quality score with invalid component type"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        with pytest.raises(ValueError):
            metrics_calc.compute_quality_score('invalid_type', {})

    def test_quality_score_bounds(self):
        """Test that quality scores are always in [0, 1]"""
        metrics_calc = ComponentMetrics(semantic_evaluator=None)

        # Extreme bad metrics
        bad_metrics = {
            'avg_chunk_size': 10000,  # Way too large
            'size_variance': 1000,  # Very inconsistent
            'semantic_coherence': 0.1  # Poor coherence
        }

        score = metrics_calc.compute_quality_score('chunker', bad_metrics)
        assert 0.0 <= score <= 1.0


class TestIntegration:
    """Integration tests for component metrics"""

    def test_full_chunking_pipeline(self):
        """Test complete chunking metrics computation"""
        from autorag.components.chunkers.fixed_size import FixedSizeChunker
        from autorag.components.base import Document
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Create real components
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metrics_calc = ComponentMetrics(semantic_evaluator=semantic_eval)

        chunker = FixedSizeChunker({'chunk_size': 50, 'overlap': 10, 'unit': 'tokens'})
        documents = [
            Document(content="This is a test document. " * 20, doc_id="1")
        ]

        # Time the chunking
        import time
        start = time.time()
        chunks = chunker.chunk(documents)
        latency = time.time() - start

        # Compute metrics
        metrics = metrics_calc.compute_chunking_metrics(chunks, latency, compute_coherence=True)

        # Verify all metrics are present and valid
        assert metrics['chunks_created'] > 0
        assert metrics['avg_chunk_size'] > 0
        assert metrics['time'] > 0
        assert 0.0 <= metrics['semantic_coherence'] <= 1.0

        # Compute quality score
        quality = metrics_calc.compute_quality_score('chunker', metrics)
        assert 0.0 <= quality <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])