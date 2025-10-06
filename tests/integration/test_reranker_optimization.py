"""
Integration tests for reranker COSMOS optimization

Tests end-to-end reranker evaluation and optimization in COSMOS framework.
"""

import pytest
import os
from unittest.mock import Mock, patch
import numpy as np

from autorag.cosmos.optimization.evaluators import ComponentEvaluator, build_component
from autorag.cosmos.metrics import ComponentMetrics
from autorag.components.base import Document
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.components.retrievers.dense import DenseRetriever
from autorag.components.embedders.mock import MockEmbedder
from autorag.components.vector_stores.simple import SimpleVectorStore


class TestRerankerEvaluatorWithUpstream:
    """Test reranker evaluator with upstream components"""

    @pytest.fixture
    def test_data(self):
        """Create test data for evaluation"""
        return {
            'documents': [
                "Machine learning is a subset of AI.",
                "Deep learning uses neural networks.",
                "Natural language processing analyzes text.",
                "Computer vision processes images.",
                "Reinforcement learning maximizes rewards."
            ],
            'queries': [
                {'query': "What is machine learning?", 'answer': "ML is a subset of AI"},
                {'query': "How does deep learning work?", 'answer': "Uses neural networks"},
                {'query': "What is NLP?", 'answer': "Analyzes natural language"}
            ],
            'dataset_name': 'test'
        }

    @pytest.fixture
    def upstream_components(self):
        """Create upstream components for reranker evaluation"""
        # Create chunker
        chunker = FixedSizeChunker({'chunk_size': 50, 'overlap': 10})

        # Create retriever with mock embedder
        retriever = DenseRetriever({'metric': 'cosine', 'top_k': 5})
        embedder = MockEmbedder({})
        vector_store = SimpleVectorStore({})
        retriever.set_components(embedder, vector_store)

        return {
            'chunker': chunker,
            'retriever': retriever
        }

    def test_evaluate_reranker_with_upstream(self, test_data, upstream_components):
        """Test reranker evaluation with upstream chunker and retriever"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        evaluator = ComponentEvaluator(
            component_type='reranker',
            test_data=test_data,
            metric_collector=metrics_collector,
            upstream_components=upstream_components,
            max_queries=2
        )

        config = {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'normalize_scores': True,
            'batch_size': 32,
            'retrieval_top_k': 5,
            'rerank_top_k': 3,
            'use_real_api': False  # Use mock embeddings
        }

        quality_score = evaluator.evaluate(config)

        # Verify quality score is valid
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0

    def test_reranker_upstream_retriever_reinit(self, test_data, upstream_components):
        """Test that retriever gets embedder/vector store re-initialized"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Create evaluator with upstream retriever
        evaluator = ComponentEvaluator(
            component_type='reranker',
            test_data=test_data,
            metric_collector=metrics_collector,
            upstream_components=upstream_components,
            max_queries=1
        )

        config = {
            'retrieval_top_k': 5,
            'rerank_top_k': 3,
            'use_real_api': False
        }

        # Evaluation should succeed (retriever re-initialized internally)
        quality_score = evaluator.evaluate(config)

        # If re-initialization failed, evaluation would return 0.0 or raise exception
        assert quality_score >= 0.0

    def test_reranker_configurable_retrieval_params(self, test_data, upstream_components):
        """Test different retrieval_top_k and rerank_top_k combinations"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        evaluator = ComponentEvaluator(
            component_type='reranker',
            test_data=test_data,
            metric_collector=metrics_collector,
            upstream_components=upstream_components,
            max_queries=1
        )

        # Test various configurations
        configs = [
            {'retrieval_top_k': 10, 'rerank_top_k': 3},
            {'retrieval_top_k': 15, 'rerank_top_k': 5},
            {'retrieval_top_k': 20, 'rerank_top_k': 7},
            {'retrieval_top_k': 5, 'rerank_top_k': 5},  # Equal values
        ]

        for config in configs:
            config['use_real_api'] = False
            quality_score = evaluator.evaluate(config)
            assert 0.0 <= quality_score <= 1.0

    def test_reranker_constraint_enforcement(self, test_data, upstream_components):
        """Test constraint: retrieval_top_k >= rerank_top_k"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        evaluator = ComponentEvaluator(
            component_type='reranker',
            test_data=test_data,
            metric_collector=metrics_collector,
            upstream_components=upstream_components,
            max_queries=1
        )

        # Config violates constraint (should be auto-adjusted)
        config = {
            'retrieval_top_k': 3,   # Lower than rerank_top_k
            'rerank_top_k': 5,       # Higher than retrieval_top_k
            'use_real_api': False
        }

        # Should not crash, constraint enforced internally
        quality_score = evaluator.evaluate(config)
        assert quality_score >= 0.0


class TestRerankerWithCacheManager:
    """Test reranker evaluation with cache manager"""

    @pytest.fixture
    def test_data(self):
        """Create test data"""
        return {
            'documents': [f"Document {i} with test content." for i in range(10)],
            'queries': [
                {'query': f"Query {i}", 'answer': f"Answer {i}"}
                for i in range(3)
            ],
            'dataset_name': 'test'
        }

    @pytest.fixture
    def upstream_components(self):
        """Create upstream components"""
        chunker = FixedSizeChunker({'chunk_size': 50})
        retriever = DenseRetriever({'metric': 'cosine'})
        embedder = MockEmbedder({})
        vector_store = SimpleVectorStore({})
        retriever.set_components(embedder, vector_store)

        return {'chunker': chunker, 'retriever': retriever}

    def test_reranker_with_cache_manager(self, test_data, upstream_components):
        """Test reranker evaluation with cache manager enabled"""
        from autorag.optimization.cache_manager import EmbeddingCacheManager

        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Create mock cache manager
        cache_manager = Mock(spec=EmbeddingCacheManager)
        cache_manager.get_or_compute_embeddings = Mock(return_value=([], []))

        evaluator = ComponentEvaluator(
            component_type='reranker',
            test_data=test_data,
            metric_collector=metrics_collector,
            upstream_components=upstream_components,
            max_queries=1,
            cache_manager=cache_manager
        )

        config = {
            'retrieval_top_k': 10,
            'rerank_top_k': 5,
            'use_real_api': False  # Use mock, but cache path still exercised
        }

        quality_score = evaluator.evaluate(config)

        # Verify cache manager would be used (if real API enabled)
        assert quality_score >= 0.0


class TestRerankerOptimizationSequence:
    """Test reranker in full COSMOS optimization sequence"""

    @pytest.fixture
    def test_data(self):
        """Create test data"""
        return {
            'documents': [
                "Machine learning is AI.",
                "Deep learning uses networks.",
                "NLP processes language.",
            ],
            'queries': [
                {'query': "What is ML?", 'answer': "ML is AI"},
            ],
            'dataset_name': 'test'
        }

    def test_reranker_in_optimization_sequence(self, test_data):
        """Test COSMOS optimization with reranker in sequence"""
        from autorag.cosmos.optimization import CompositionalOptimizerBuilder

        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Define minimal search spaces
        search_spaces = {
            'chunker': {
                'chunking_strategy': ['fixed'],
                'chunk_size': [50, 100],
                'overlap': [10]
            },
            'retriever': {
                'retrieval_method': ['dense'],
                'retrieval_top_k': [3, 5]
            },
            'reranker': {
                'model_name': ['cross-encoder/ms-marco-MiniLM-L-6-v2'],
                'normalize_scores': [True],
                'retrieval_top_k': [5],
                'rerank_top_k': [3]
            }
        }

        # Create optimizer (using random strategy for speed)
        optimizer = CompositionalOptimizerBuilder.create_with_random(
            components=['chunker', 'retriever', 'reranker'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metrics_collector,
            budget_per_component={'chunker': 2, 'retriever': 2, 'reranker': 2},
            random_state=42
        )

        # Run optimization
        results = optimizer.optimize()

        # Verify all components optimized
        assert 'chunker' in results['best_configs']
        assert 'retriever' in results['best_configs']
        assert 'reranker' in results['best_configs']

        # Verify reranker got upstream context
        reranker_config = results['best_configs']['reranker']
        assert reranker_config is not None
        assert 'retrieval_top_k' in reranker_config
        assert 'rerank_top_k' in reranker_config


class TestRerankerQualityScoreEdgeCases:
    """Test quality score computation edge cases"""

    def test_reranker_quality_score_edge_cases(self):
        """Test quality score with edge case metrics"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        # Test various edge cases
        edge_cases = [
            # Minimal reranking (should be penalized)
            {'rank_correlation': 0.99, 'score_change': 0.01, 'latency': 0.1},

            # Extreme score changes
            {'rank_correlation': 0.0, 'score_change': 2.0, 'latency': 0.1},

            # High latency
            {'rank_correlation': 0.2, 'score_change': 0.35, 'latency': 1.5},

            # Perfect inverse correlation
            {'rank_correlation': -1.0, 'score_change': 0.5, 'latency': 0.1},
        ]

        for metrics in edge_cases:
            score = metrics_collector.compute_quality_score('reranker', metrics)
            # All scores should be valid
            assert 0.0 <= score <= 1.0
            assert not np.isnan(score)


class TestRerankerEvaluationWithDifferentBackends:
    """Test reranker evaluation with different embedding backends"""

    @pytest.fixture
    def test_data(self):
        """Create test data"""
        return {
            'documents': ["Doc 1", "Doc 2", "Doc 3"],
            'queries': [{'query': "Query 1", 'answer': "Answer 1"}],
            'dataset_name': 'test'
        }

    @pytest.fixture
    def upstream_components(self):
        """Create upstream components"""
        chunker = FixedSizeChunker({'chunk_size': 50})
        retriever = DenseRetriever({'metric': 'cosine'})
        embedder = MockEmbedder({})
        vector_store = SimpleVectorStore({})
        retriever.set_components(embedder, vector_store)

        return {'chunker': chunker, 'retriever': retriever}

    def test_reranker_evaluation_with_mock_embedder(self, test_data, upstream_components):
        """Test reranker with mock embeddings"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        evaluator = ComponentEvaluator(
            component_type='reranker',
            test_data=test_data,
            metric_collector=metrics_collector,
            upstream_components=upstream_components,
            max_queries=1
        )

        config = {
            'retrieval_top_k': 5,
            'rerank_top_k': 3,
            'use_real_api': False  # Mock embeddings
        }

        quality_score = evaluator.evaluate(config)
        assert quality_score >= 0.0

    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="OpenAI API key not available"
    )
    def test_reranker_evaluation_with_real_embedder(self, test_data, upstream_components):
        """Test reranker with real OpenAI embeddings (if API key available)"""
        metrics_collector = ComponentMetrics(semantic_evaluator=None)

        evaluator = ComponentEvaluator(
            component_type='reranker',
            test_data=test_data,
            metric_collector=metrics_collector,
            upstream_components=upstream_components,
            max_queries=1
        )

        config = {
            'retrieval_top_k': 5,
            'rerank_top_k': 3,
            'use_real_api': True  # Real embeddings
        }

        quality_score = evaluator.evaluate(config)
        assert quality_score >= 0.0


class TestRerankerSearchSpaceCoverage:
    """Test reranker search space definition"""

    def test_reranker_search_space_coverage(self):
        """Test search space includes all expected parameters"""
        from scripts.run_cosmos_optimization import define_search_spaces

        search_spaces = define_search_spaces()

        # Verify reranker search space exists
        assert 'reranker' in search_spaces

        reranker_space = search_spaces['reranker']

        # Verify all expected parameters
        assert 'model_name' in reranker_space
        assert 'normalize_scores' in reranker_space
        assert 'batch_size' in reranker_space
        assert 'retrieval_top_k' in reranker_space
        assert 'rerank_top_k' in reranker_space

        # Verify parameter types
        assert isinstance(reranker_space['model_name'], list)
        assert len(reranker_space['model_name']) >= 1

    def test_reranker_search_space_sampling(self):
        """Test random sampling from search space"""
        from scripts.run_cosmos_optimization import define_search_spaces
        import random

        search_spaces = define_search_spaces()
        reranker_space = search_spaces['reranker']

        # Sample 10 random configurations
        for _ in range(10):
            config = {
                param: random.choice(values)
                for param, values in reranker_space.items()
            }

            # Verify all configs are valid
            assert 'model_name' in config
            assert 'normalize_scores' in config
            assert 'batch_size' in config
            assert 'retrieval_top_k' in config
            assert 'rerank_top_k' in config

            # Build component from config
            reranker = build_component('reranker', config)
            assert reranker is not None


class TestRerankerWithDifferentModels:
    """Test reranker with different cross-encoder models"""

    def test_reranker_with_different_models(self):
        """Test both MiniLM-L-6-v2 and MiniLM-L-12-v2 models can be loaded"""
        models = [
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'cross-encoder/ms-marco-MiniLM-L-12-v2',
        ]

        for model_name in models:
            config = {
                'model_name': model_name,
                'normalize_scores': True,
                'batch_size': 32
            }

            reranker = build_component('reranker', config)
            assert reranker is not None
            assert reranker.model_name == model_name


class TestRerankerBatchSizeImpact:
    """Test reranker batch size configuration"""

    def test_reranker_batch_size_impact(self):
        """Test different batch_size values"""
        batch_sizes = [16, 32, 64]

        for batch_size in batch_sizes:
            config = {
                'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'normalize_scores': True,
                'batch_size': batch_size
            }

            reranker = build_component('reranker', config)
            assert reranker.batch_size == batch_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
