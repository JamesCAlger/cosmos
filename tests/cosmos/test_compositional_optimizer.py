"""
Unit tests for Compositional Optimizer

Tests the orchestration of component-by-component optimization.
"""

import pytest
from unittest.mock import Mock, MagicMock
from autorag.cosmos.optimization import (
    CompositionalOptimizer,
    CompositionalOptimizerBuilder,
    RandomSearchStrategy,
    BayesianStrategy
)
from autorag.cosmos.metrics import ComponentMetrics
from autorag.optimization.bayesian_search import OptimizationResult


class TestCompositionalOptimizerBasic:
    """Test basic compositional optimizer functionality"""

    def test_initialization(self):
        """Test optimizer initialization"""
        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        assert optimizer.strategy == strategy
        assert optimizer.component_results == {}
        assert optimizer.upstream_components == {}

    def test_optimize_single_component(self):
        """Test optimizing a single component"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Setup
        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test document. " * 20] * 3,
            'queries': []
        }

        search_spaces = {
            'chunker': {
                'chunking_strategy': ['fixed'],
                'chunk_size': [128, 256],
                'overlap': [0, 25]
            }
        }

        # Optimize
        results = optimizer.optimize(
            components=['chunker'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=4
        )

        # Verify
        assert 'chunker' in results
        assert results['chunker'].best_config is not None
        assert 0.0 <= results['chunker'].best_score <= 1.0
        assert results['chunker'].n_evaluations == 4

    def test_optimize_two_components(self):
        """Test optimizing chunker then retriever"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Setup
        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Machine learning is great. " * 10] * 3,
            'queries': [
                {'query': 'What is machine learning?'}
            ]
        }

        search_spaces = {
            'chunker': {
                'chunking_strategy': ['fixed'],
                'chunk_size': [128, 256],
                'overlap': [0, 25]
            },
            'retriever': {
                'retrieval_method': ['sparse'],  # Use sparse for simplicity
                'retrieval_top_k': [3, 5]
            }
        }

        # Optimize
        results = optimizer.optimize(
            components=['chunker', 'retriever'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=8  # 4 per component
        )

        # Verify
        assert 'chunker' in results
        assert 'retriever' in results
        assert results['chunker'].n_evaluations == 4
        assert results['retriever'].n_evaluations == 4

        # Check that retriever used chunker
        assert 'chunker' in optimizer.upstream_components

    def test_optimize_three_components(self):
        """Test full pipeline optimization"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Setup
        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Python is a programming language. " * 10] * 2,
            'queries': [
                {'query': 'What is Python?', 'answer': 'Python is a programming language'}
            ]
        }

        search_spaces = {
            'chunker': {
                'chunking_strategy': ['fixed'],
                'chunk_size': [128, 256]
            },
            'retriever': {
                'retrieval_method': ['sparse'],
                'retrieval_top_k': [3, 5]
            },
            'generator': {
                'use_real_api': [False],  # Mock generator
                'temperature': [0.5, 0.7]
            }
        }

        # Optimize
        results = optimizer.optimize(
            components=['chunker', 'retriever', 'generator'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=9  # 3 per component
        )

        # Verify
        assert len(results) == 3
        assert 'chunker' in results
        assert 'retriever' in results
        assert 'generator' in results

        # Check upstream propagation
        assert 'chunker' in optimizer.upstream_components
        assert 'retriever' in optimizer.upstream_components


class TestBudgetAllocation:
    """Test budget allocation strategies"""

    def test_equal_budget_split(self):
        """Test equal budget allocation"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test. " * 20] * 2,
            'queries': [{'query': 'test?'}]
        }

        search_spaces = {
            'chunker': {'chunking_strategy': ['fixed'], 'chunk_size': [128, 256]},
            'retriever': {'retrieval_method': ['sparse'], 'retrieval_top_k': [3, 5]}
        }

        results = optimizer.optimize(
            components=['chunker', 'retriever'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=10
        )

        # With total_budget=10 and 2 components, each gets 5
        assert results['chunker'].n_evaluations == 5
        assert results['retriever'].n_evaluations == 5

    def test_custom_budget_allocation(self):
        """Test custom budget per component"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test. " * 20] * 2,
            'queries': [{'query': 'test?'}]
        }

        search_spaces = {
            'chunker': {'chunking_strategy': ['fixed'], 'chunk_size': [128, 256]},
            'retriever': {'retrieval_method': ['sparse'], 'retrieval_top_k': [3, 5]}
        }

        # Custom allocation: more budget for retriever
        budget_allocation = {
            'chunker': 3,
            'retriever': 7
        }

        results = optimizer.optimize(
            components=['chunker', 'retriever'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=10,
            budget_allocation=budget_allocation
        )

        assert results['chunker'].n_evaluations == 3
        assert results['retriever'].n_evaluations == 7


class TestContextPassing:
    """Test context passing between components"""

    def test_upstream_components_passed(self):
        """Test that upstream components are passed to downstream"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test. " * 20] * 2,
            'queries': [{'query': 'test?'}]
        }

        search_spaces = {
            'chunker': {'chunking_strategy': ['fixed'], 'chunk_size': [256]},
            'retriever': {'retrieval_method': ['sparse'], 'retrieval_top_k': [5]}
        }

        results = optimizer.optimize(
            components=['chunker', 'retriever'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=4
        )

        # Check upstream was populated
        assert 'chunker' in optimizer.upstream_components
        assert hasattr(optimizer.upstream_components['chunker'], 'chunk')


class TestResultMethods:
    """Test result retrieval methods"""

    def test_get_best_pipeline_config(self):
        """Test getting best pipeline configuration"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test. " * 20] * 2,
            'queries': []
        }

        search_spaces = {
            'chunker': {'chunking_strategy': ['fixed'], 'chunk_size': [128, 256]}
        }

        optimizer.optimize(
            components=['chunker'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=2
        )

        pipeline_config = optimizer.get_best_pipeline_config()

        assert 'chunker' in pipeline_config
        assert 'chunk_size' in pipeline_config['chunker']

    def test_get_optimization_summary(self):
        """Test getting optimization summary"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test. " * 20] * 2,
            'queries': []
        }

        search_spaces = {
            'chunker': {'chunking_strategy': ['fixed'], 'chunk_size': [256]}
        }

        optimizer.optimize(
            components=['chunker'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=2
        )

        summary = optimizer.get_optimization_summary()

        assert summary['status'] == 'complete'
        assert 'chunker' in summary['components_optimized']
        assert summary['total_evaluations'] == 2
        assert 'average_score' in summary
        assert 'component_scores' in summary

    def test_summary_before_optimization(self):
        """Test summary when no optimization has been run"""
        strategy = RandomSearchStrategy()
        optimizer = CompositionalOptimizer(strategy)

        summary = optimizer.get_optimization_summary()

        assert summary['status'] == 'not_run'

    def test_clear_results(self):
        """Test clearing optimization results"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        strategy = RandomSearchStrategy(random_state=42)
        optimizer = CompositionalOptimizer(strategy)

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test. " * 20] * 2,
            'queries': []
        }

        search_spaces = {
            'chunker': {'chunking_strategy': ['fixed'], 'chunk_size': [256]}
        }

        optimizer.optimize(
            components=['chunker'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=2
        )

        assert len(optimizer.component_results) > 0

        optimizer.clear_results()

        assert len(optimizer.component_results) == 0
        assert len(optimizer.upstream_components) == 0


class TestCompositionalOptimizerBuilder:
    """Test builder convenience methods"""

    def test_create_with_bayesian(self):
        """Test creating optimizer with Bayesian strategy"""
        optimizer = CompositionalOptimizerBuilder.create_with_bayesian(
            n_initial_points=3,
            random_state=42
        )

        assert isinstance(optimizer, CompositionalOptimizer)
        assert isinstance(optimizer.strategy, BayesianStrategy)
        assert optimizer.strategy.n_initial_points == 3
        assert optimizer.strategy.random_state == 42

    def test_create_with_random(self):
        """Test creating optimizer with random strategy"""
        optimizer = CompositionalOptimizerBuilder.create_with_random(random_state=123)

        assert isinstance(optimizer, CompositionalOptimizer)
        assert isinstance(optimizer.strategy, RandomSearchStrategy)
        assert optimizer.strategy.random_state == 123

    def test_create_with_custom_strategy(self):
        """Test creating optimizer with custom strategy"""
        custom_strategy = RandomSearchStrategy(random_state=999)
        optimizer = CompositionalOptimizerBuilder.create_with_custom_strategy(custom_strategy)

        assert isinstance(optimizer, CompositionalOptimizer)
        assert optimizer.strategy == custom_strategy


class TestStrategyComparison:
    """Test using different strategies with same problem"""

    def test_bayesian_vs_random(self):
        """Test that both Bayesian and Random work on same problem"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test document. " * 15] * 2,
            'queries': []
        }

        search_spaces = {
            'chunker': {
                'chunking_strategy': ['fixed'],
                'chunk_size': [128, 256, 512]
            }
        }

        # Random strategy
        random_optimizer = CompositionalOptimizerBuilder.create_with_random(random_state=42)
        random_results = random_optimizer.optimize(
            components=['chunker'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=6
        )

        # Bayesian strategy
        bayesian_optimizer = CompositionalOptimizerBuilder.create_with_bayesian(
            n_initial_points=3,
            random_state=42
        )
        bayesian_results = bayesian_optimizer.optimize(
            components=['chunker'],
            search_spaces=search_spaces,
            test_data=test_data,
            metric_collector=metric_collector,
            total_budget=6
        )

        # Both should complete successfully
        assert random_results['chunker'].n_evaluations == 6
        assert bayesian_results['chunker'].n_evaluations == 6
        assert random_results['chunker'].best_score > 0.0
        assert bayesian_results['chunker'].best_score > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])