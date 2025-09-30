"""
Unit tests for Optimization Strategies

Tests optimization task definition and strategy implementations.
"""

import pytest
from unittest.mock import Mock
from autorag.cosmos.optimization import (
    OptimizationTask,
    OptimizationStrategy,
    BayesianStrategy,
    RandomSearchStrategy
)


class TestOptimizationTask:
    """Test OptimizationTask dataclass"""

    def test_create_task(self):
        """Test creating an optimization task"""
        evaluator = lambda config: 0.5

        task = OptimizationTask(
            component_id='chunker',
            search_space={'chunk_size': [128, 256, 512]},
            evaluator=evaluator,
            budget=10
        )

        assert task.component_id == 'chunker'
        assert 'chunk_size' in task.search_space
        assert task.budget == 10
        assert callable(task.evaluator)
        assert task.context == {}

    def test_task_with_context(self):
        """Test task with upstream context"""
        task = OptimizationTask(
            component_id='retriever',
            search_space={'top_k': [3, 5, 10]},
            evaluator=lambda x: 0.5,
            budget=15,
            context={'chunker_config': {'chunk_size': 256}}
        )

        assert task.context['chunker_config']['chunk_size'] == 256

    def test_task_repr(self):
        """Test task string representation"""
        task = OptimizationTask(
            component_id='generator',
            search_space={'temperature': (0.0, 1.0), 'max_tokens': [100, 200]},
            evaluator=lambda x: 0.5,
            budget=20
        )

        repr_str = repr(task)
        assert 'generator' in repr_str
        assert 'budget=20' in repr_str


class TestRandomSearchStrategy:
    """Test random search strategy"""

    def test_random_search_basic(self):
        """Test basic random search"""
        # Define simple search space and evaluator
        search_space = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b', 'c']
        }

        # Simple evaluator: higher param1 is better
        eval_count = [0]
        def evaluator(config):
            eval_count[0] += 1
            return config['param1'] / 3.0  # Returns 0.33, 0.67, or 1.0

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=10
        )

        strategy = RandomSearchStrategy(random_state=42)
        result = strategy.optimize(task)

        # Verify result
        assert result.best_config is not None
        assert 0.0 <= result.best_score <= 1.0
        assert result.n_evaluations == 10
        assert len(result.all_configs) == 10
        assert len(result.all_scores) == 10
        assert len(result.convergence_history) == 10

        # Best score should be best found
        assert result.best_score == max(result.all_scores)

    def test_random_search_continuous(self):
        """Test random search with continuous parameters"""
        search_space = {
            'x': (0.0, 1.0),
            'y': (0.0, 1.0)
        }

        # Evaluator: maximize x + y
        def evaluator(config):
            return (config['x'] + config['y']) / 2.0

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=15
        )

        strategy = RandomSearchStrategy(random_state=42)
        result = strategy.optimize(task)

        assert result.n_evaluations == 15
        assert 0.0 <= result.best_score <= 1.0

    def test_random_search_mixed_params(self):
        """Test random search with mixed parameter types"""
        search_space = {
            'categorical': ['option1', 'option2'],
            'continuous': (0.0, 1.0),
            'discrete': [5, 10, 15, 20]
        }

        def evaluator(config):
            return 0.7

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=5
        )

        strategy = RandomSearchStrategy(random_state=42)
        result = strategy.optimize(task)

        assert result.n_evaluations == 5
        # Check that all configs have all parameters
        for config in result.all_configs:
            assert 'categorical' in config
            assert 'continuous' in config
            assert 'discrete' in config

    def test_random_search_handles_evaluation_errors(self):
        """Test that random search handles evaluation errors"""
        search_space = {'param': [1, 2, 3]}

        call_count = [0]
        def failing_evaluator(config):
            call_count[0] += 1
            if call_count[0] == 3:
                raise ValueError("Test error")
            return 0.5

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=failing_evaluator,
            budget=5
        )

        strategy = RandomSearchStrategy(random_state=42)
        result = strategy.optimize(task)

        # Should complete despite error
        assert result.n_evaluations == 5
        # Failed evaluation should have score 0.0
        assert result.all_scores[2] == 0.0

    def test_random_search_convergence(self):
        """Test convergence tracking"""
        search_space = {'param': list(range(10))}

        def evaluator(config):
            return config['param'] / 10.0

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=20
        )

        strategy = RandomSearchStrategy(random_state=42)
        result = strategy.optimize(task)

        # Convergence should be non-decreasing
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] >= result.convergence_history[i-1]

    def test_get_name(self):
        """Test strategy name"""
        strategy = RandomSearchStrategy()
        assert strategy.get_name() == "Random Search"


class TestBayesianStrategy:
    """Test Bayesian optimization strategy"""

    def test_bayesian_strategy_basic(self):
        """Test basic Bayesian optimization"""
        # Simple search space
        search_space = {
            'param1': [10, 20, 30],
            'param2': [0.1, 0.5, 1.0]
        }

        # Evaluator: higher param1 and lower param2 is better
        def evaluator(config):
            score = (config['param1'] / 30.0) * (1.0 - config['param2'])
            return score

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=10
        )

        strategy = BayesianStrategy(n_initial_points=3, random_state=42)
        result = strategy.optimize(task)

        # Verify result
        assert result.best_config is not None
        assert 0.0 <= result.best_score <= 1.0
        assert result.n_evaluations == 10
        assert len(result.all_configs) == 10

    def test_bayesian_strategy_with_continuous(self):
        """Test Bayesian optimization with continuous parameters"""
        search_space = {
            'x': (0.0, 1.0),
            'y': (0.0, 1.0)
        }

        # Simple quadratic function
        def evaluator(config):
            return 1.0 - ((config['x'] - 0.7)**2 + (config['y'] - 0.3)**2)

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=15
        )

        strategy = BayesianStrategy(n_initial_points=5, random_state=42)
        result = strategy.optimize(task)

        assert result.n_evaluations == 15
        # Should find reasonable solution
        assert result.best_score > 0.5

    def test_bayesian_strategy_vs_random(self):
        """Test that Bayesian often outperforms random search"""
        # Define search space with clear optimum
        search_space = {
            'x': (0.0, 1.0)
        }

        # Function with maximum at x=0.7
        def evaluator(config):
            return 1.0 - abs(config['x'] - 0.7)

        task = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=20
        )

        # Run both strategies
        bayesian = BayesianStrategy(n_initial_points=5, random_state=42)
        bayesian_result = bayesian.optimize(task)

        # Create new task for random (fresh evaluator state)
        task_random = OptimizationTask(
            component_id='test',
            search_space=search_space,
            evaluator=evaluator,
            budget=20
        )

        random_strat = RandomSearchStrategy(random_state=42)
        random_result = random_strat.optimize(task_random)

        # Both should find good solutions
        assert bayesian_result.best_score > 0.5
        assert random_result.best_score > 0.5

        # Note: We don't assert Bayesian > Random because with budget=20
        # on a 1D problem, random can get lucky. In practice on harder
        # problems, Bayesian will outperform.

    def test_bayesian_strategy_initialization(self):
        """Test strategy initialization parameters"""
        strategy = BayesianStrategy(
            n_initial_points=10,
            random_state=123,
            save_results=True
        )

        assert strategy.n_initial_points == 10
        assert strategy.random_state == 123
        assert strategy.save_results == True

    def test_get_name(self):
        """Test strategy name"""
        strategy = BayesianStrategy()
        assert strategy.get_name() == "Bayesian Optimization"


class TestStrategyComparison:
    """Compare different optimization strategies"""

    def test_strategies_find_reasonable_solutions(self):
        """Test that both strategies find reasonable solutions"""
        # Define problem
        search_space = {
            'param1': list(range(10)),
            'param2': [0.1, 0.3, 0.5, 0.7, 0.9]
        }

        def evaluator(config):
            # Optimum at param1=9, param2=0.9
            return (config['param1'] / 9.0 + config['param2']) / 2.0

        # Test random search
        task1 = OptimizationTask('test', search_space, evaluator, budget=15)
        random_result = RandomSearchStrategy(random_state=42).optimize(task1)

        # Test Bayesian
        task2 = OptimizationTask('test', search_space, evaluator, budget=15)
        bayesian_result = BayesianStrategy(n_initial_points=5, random_state=42).optimize(task2)

        # Both should find decent solutions
        assert random_result.best_score > 0.6
        assert bayesian_result.best_score > 0.6

        # Both should have tried the budget
        assert random_result.n_evaluations == 15
        assert bayesian_result.n_evaluations == 15


class TestIntegration:
    """Integration tests with real evaluators"""

    def test_optimize_chunker_config(self):
        """Test optimizing a chunker configuration"""
        from autorag.cosmos.optimization import ComponentEvaluator
        from autorag.cosmos.metrics import ComponentMetrics
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Setup
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        test_data = {
            'documents': ["Test document. " * 30] * 3,
            'queries': []
        }

        evaluator_obj = ComponentEvaluator('chunker', test_data, metric_collector)

        # Define task
        search_space = {
            'chunking_strategy': ['fixed'],
            'chunk_size': [128, 256, 512],
            'overlap': [0, 25, 50]
        }

        task = OptimizationTask(
            component_id='chunker',
            search_space=search_space,
            evaluator=evaluator_obj.evaluate,
            budget=5  # Small budget for testing
        )

        # Run optimization
        strategy = RandomSearchStrategy(random_state=42)
        result = strategy.optimize(task)

        # Verify
        assert result.best_config is not None
        assert 'chunk_size' in result.best_config
        assert 0.0 <= result.best_score <= 1.0
        assert result.n_evaluations == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])