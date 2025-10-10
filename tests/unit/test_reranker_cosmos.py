"""
Unit tests for reranker COSMOS integration

Tests the reranker component wrapper, metrics computation, and quality scoring.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from autorag.cosmos.component_wrapper import COSMOSComponent
from autorag.cosmos.metrics import ComponentMetrics
from autorag.cosmos.optimization.evaluators import build_component
from autorag.components.base import QueryResult, Chunk


@dataclass
class MockChunk:
    """Mock chunk for testing"""
    content: str
    chunk_id: str
    doc_id: str = "doc_0"


def create_mock_query_result(chunk_id: str, score: float) -> QueryResult:
    """Create a mock QueryResult for testing"""
    chunk = MockChunk(
        content=f"Content for chunk {chunk_id}",
        chunk_id=chunk_id
    )
    return QueryResult(chunk=chunk, score=score, metadata={})


class TestCosmosRerankerWrapper:
    """Test COSMOSComponent wrapper for reranker"""

    def test_cosmos_reranker_wrapper_basic(self):
        """Test basic reranker wrapper functionality"""
        # Create mock reranker
        mock_reranker = Mock()
        mock_reranker.config = {}

        # Mock rerank method
        original_results = [
            create_mock_query_result("chunk_1", 0.8),
            create_mock_query_result("chunk_2", 0.6),
            create_mock_query_result("chunk_3", 0.4),
        ]

        reranked_results = [
            create_mock_query_result("chunk_3", 0.9),
            create_mock_query_result("chunk_1", 0.7),
            create_mock_query_result("chunk_2", 0.5),
        ]

        mock_reranker.rerank = Mock(return_value=reranked_results)

        # Create metrics collector
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Wrap with COSMOS
        cosmos_reranker = COSMOSComponent(mock_reranker, 'reranker', metrics_collector)

        # Process with metrics
        query = "test query"
        results, metrics = cosmos_reranker.process_with_metrics(query, original_results, top_k=3)

        # Verify output
        assert isinstance(results, list)
        assert len(results) == 3
        assert isinstance(metrics, dict)
        assert 'latency' in metrics
        assert 'score_change' in metrics
        assert 'rank_correlation' in metrics
        assert 'top_k_overlap' in metrics

        # Verify rerank was called
        mock_reranker.rerank.assert_called_once_with(query, original_results, 3)

    def test_cosmos_reranker_handles_empty_results(self):
        """Test reranker wrapper with empty results"""
        mock_reranker = Mock()
        mock_reranker.config = {}

        metrics_collector = ComponentMetrics(semantic_evaluator=None)
        cosmos_reranker = COSMOSComponent(mock_reranker, 'reranker', metrics_collector)

        # Process with empty results
        results, metrics = cosmos_reranker.process_with_metrics("query", [], top_k=5)

        # Verify empty results handled gracefully
        assert results == []
        assert metrics['latency'] == 0.0
        assert metrics['score_change'] == 0.0
        assert metrics['rank_correlation'] == 0.0
        assert metrics['top_k_overlap'] == 0.0

        # Verify rerank was NOT called
        mock_reranker.rerank.assert_not_called()

    def test_cosmos_reranker_handles_single_result(self):
        """Test reranker with single result (edge case)"""
        mock_reranker = Mock()
        mock_reranker.config = {}

        original = [create_mock_query_result("chunk_1", 0.8)]
        reranked = [create_mock_query_result("chunk_1", 0.9)]

        mock_reranker.rerank = Mock(return_value=reranked)

        metrics_collector = ComponentMetrics(semantic_evaluator=None)
        cosmos_reranker = COSMOSComponent(mock_reranker, 'reranker', metrics_collector)

        results, metrics = cosmos_reranker.process_with_metrics("query", original, top_k=1)

        # Verify metrics computed for single result
        assert len(results) == 1
        assert metrics['rank_correlation'] == 1.0  # Trivial case


class TestRerankerMetricsComputation:
    """Test reranking metrics computation"""

    def test_cosmos_reranker_metrics_computation(self):
        """Test compute_reranking_metrics returns expected keys"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        original = [
            create_mock_query_result("chunk_1", 0.8),
            create_mock_query_result("chunk_2", 0.6),
            create_mock_query_result("chunk_3", 0.4),
        ]

        reranked = [
            create_mock_query_result("chunk_3", 0.9),
            create_mock_query_result("chunk_1", 0.7),
            create_mock_query_result("chunk_2", 0.5),
        ]

        metrics = metrics_collector.compute_reranking_metrics(
            "test query", original, reranked, latency=0.15
        )

        # Verify all expected keys present
        assert 'latency' in metrics
        assert 'score_change' in metrics
        assert 'rank_correlation' in metrics
        assert 'top_k_overlap' in metrics

        # Verify values are reasonable
        assert metrics['latency'] == 0.15
        assert 0.0 <= metrics['score_change'] <= 10.0
        assert -1.0 <= metrics['rank_correlation'] <= 1.0
        assert 0.0 <= metrics['top_k_overlap'] <= 1.0

    def test_reranker_score_change_detection(self):
        """Test score change metric computation"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Create results with known score changes
        original = [
            create_mock_query_result("chunk_1", 0.8),
            create_mock_query_result("chunk_2", 0.6),
        ]

        # Reranked with significant score changes
        reranked = [
            create_mock_query_result("chunk_1", 0.4),  # Absolute change: |0.8-0.4| = 0.4
            create_mock_query_result("chunk_2", 0.9),  # Absolute change: |0.6-0.9| = 0.3
        ]

        metrics = metrics_collector.compute_reranking_metrics(
            "query", original, reranked, latency=0.1
        )

        # Score change should be non-zero
        assert metrics['score_change'] > 0.0
        # Average absolute change: (0.4 + 0.3) / 2 = 0.35
        assert 0.3 < metrics['score_change'] < 0.4

    def test_reranker_rank_correlation_computation(self):
        """Test Kendall's tau rank correlation computation"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Perfect inverse order (should have negative correlation)
        original = [
            create_mock_query_result("chunk_1", 0.9),
            create_mock_query_result("chunk_2", 0.7),
            create_mock_query_result("chunk_3", 0.5),
            create_mock_query_result("chunk_4", 0.3),
        ]

        reranked = [
            create_mock_query_result("chunk_4", 0.9),
            create_mock_query_result("chunk_3", 0.7),
            create_mock_query_result("chunk_2", 0.5),
            create_mock_query_result("chunk_1", 0.3),
        ]

        metrics = metrics_collector.compute_reranking_metrics(
            "query", original, reranked, latency=0.1
        )

        # Perfect inverse should have correlation close to -1.0
        assert metrics['rank_correlation'] < -0.9

    def test_reranker_top_k_overlap_metric(self):
        """Test top-k overlap proportion computation"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # 10 results, top-3 overlap = 2/3
        original = [
            create_mock_query_result(f"chunk_{i}", 0.9 - i * 0.1)
            for i in range(10)
        ]

        # Reranked: chunk_0 and chunk_1 stay in top-3, chunk_2 drops out
        reranked = [
            create_mock_query_result("chunk_1", 0.95),
            create_mock_query_result("chunk_5", 0.90),
            create_mock_query_result("chunk_0", 0.85),
        ] + [create_mock_query_result(f"chunk_{i}", 0.5) for i in range(2, 10)]

        metrics = metrics_collector.compute_reranking_metrics(
            "query", original, reranked, latency=0.1
        )

        # Top-5 overlap (k=min(5, len(results))):
        # Original top-5: {chunk_0, chunk_1, chunk_2, chunk_3, chunk_4}
        # Reranked top-5: {chunk_1, chunk_5, chunk_0, chunk_2, chunk_3}
        # Intersection: {chunk_0, chunk_1, chunk_2, chunk_3} = 4/5 = 0.8
        assert 0.75 < metrics['top_k_overlap'] < 0.85


class TestRerankerQualityScore:
    """Test reranker quality score computation"""

    def test_reranker_quality_score_computation(self):
        """Test quality score is in [0, 1] range"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Various metric combinations
        test_cases = [
            {'rank_correlation': 0.0, 'score_change': 0.35, 'latency': 0.2},  # Ideal
            {'rank_correlation': 0.9, 'score_change': 0.1, 'latency': 0.1},   # Poor reordering
            {'rank_correlation': -0.5, 'score_change': 0.5, 'latency': 0.3},  # Good reordering
            {'rank_correlation': 0.0, 'score_change': 1.0, 'latency': 0.1},   # High score change
        ]

        for metrics in test_cases:
            score = metrics_collector.compute_quality_score('reranker', metrics)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for metrics {metrics}"

    def test_reranker_no_op_penalty(self):
        """Test that no-op reranking (passthrough) is penalized"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # No-op reranking (high correlation, low score change)
        no_op_metrics = {
            'rank_correlation': 0.98,
            'score_change': 0.02,
            'latency': 0.1
        }

        # Good reranking (low correlation, moderate score change)
        good_metrics = {
            'rank_correlation': 0.1,
            'score_change': 0.35,
            'latency': 0.1
        }

        no_op_score = metrics_collector.compute_quality_score('reranker', no_op_metrics)
        good_score = metrics_collector.compute_quality_score('reranker', good_metrics)

        # Good reranking should score higher than no-op
        assert good_score > no_op_score

        # No-op should be penalized (score < 0.5)
        assert no_op_score < 0.5


class TestBuildRerankerComponent:
    """Test building reranker component from config"""

    def test_build_reranker_component(self):
        """Test build_component returns CrossEncoderReranker instance"""
        config = {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'normalize_scores': True,
            'batch_size': 32
        }

        reranker = build_component('reranker', config)

        # Verify correct type
        from autorag.components.rerankers.cross_encoder import CrossEncoderReranker
        assert isinstance(reranker, CrossEncoderReranker)

        # Verify configuration applied
        assert reranker.model_name == config['model_name']
        assert reranker.normalize_scores == config['normalize_scores']
        assert reranker.batch_size == config['batch_size']

    def test_build_reranker_with_different_configs(self):
        """Test building reranker with various configurations"""
        configs = [
            {'model_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2', 'normalize_scores': False},
            {'normalize_scores': True, 'batch_size': 64},
            {},  # Empty config (should use defaults)
        ]

        for config in configs:
            reranker = build_component('reranker', config)
            assert reranker is not None
            assert hasattr(reranker, 'rerank')


class TestRerankerMetricsEdgeCases:
    """Test edge cases in reranking metrics"""

    def test_metrics_with_empty_results(self):
        """Test metrics computation with empty results"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        metrics = metrics_collector.compute_reranking_metrics(
            "query", [], [], latency=0.0
        )

        # All metrics should be zero
        assert metrics['latency'] == 0.0
        assert metrics['score_change'] == 0.0
        assert metrics['rank_correlation'] == 0.0
        assert metrics['top_k_overlap'] == 0.0

    def test_metrics_with_single_result(self):
        """Test metrics with single result (correlation undefined)"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        original = [create_mock_query_result("chunk_1", 0.8)]
        reranked = [create_mock_query_result("chunk_1", 0.9)]

        metrics = metrics_collector.compute_reranking_metrics(
            "query", original, reranked, latency=0.1
        )

        # Single result: perfect correlation (trivial), full overlap
        assert metrics['rank_correlation'] == 1.0
        assert metrics['top_k_overlap'] == 1.0

    def test_metrics_with_mismatched_chunk_ids(self):
        """Test metrics when reranked results have different chunks"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        original = [
            create_mock_query_result("chunk_1", 0.8),
            create_mock_query_result("chunk_2", 0.6),
            create_mock_query_result("chunk_3", 0.4),
        ]

        # Completely different chunks in reranked
        reranked = [
            create_mock_query_result("chunk_4", 0.9),
            create_mock_query_result("chunk_5", 0.7),
            create_mock_query_result("chunk_6", 0.5),
        ]

        metrics = metrics_collector.compute_reranking_metrics(
            "query", original, reranked, latency=0.1
        )

        # No overlap in chunk IDs
        assert metrics['top_k_overlap'] == 0.0
        # No common chunks to compute correlation
        assert metrics['rank_correlation'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
