"""
Bayesian Optimization Strategy

Wraps SimpleBayesianOptimizer for use in COSMOS framework.
"""

from typing import Dict, Any
from loguru import logger
from autorag.optimization.bayesian_search import SimpleBayesianOptimizer, OptimizationResult
from autorag.cosmos.optimization.strategy import OptimizationStrategy
from autorag.cosmos.optimization.task import OptimizationTask


class BayesianStrategy(OptimizationStrategy):
    """
    Bayesian optimization strategy using SimpleBayesianOptimizer

    This strategy wraps the existing SimpleBayesianOptimizer to work
    with the COSMOS OptimizationTask abstraction.
    """

    def __init__(self,
                 n_initial_points: int = 5,
                 random_state: int = 42,
                 save_results: bool = False):
        """
        Initialize Bayesian optimization strategy

        Args:
            n_initial_points: Number of random exploration points before GP
            random_state: Random seed for reproducibility
            save_results: Whether to save intermediate results
        """
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.save_results = save_results

        logger.info(f"BayesianStrategy initialized: n_initial={n_initial_points}, seed={random_state}")

    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        """
        Run Bayesian optimization on the task

        Args:
            task: OptimizationTask with search space and evaluator

        Returns:
            OptimizationResult with best configuration
        """
        logger.info(f"Starting Bayesian optimization for {task.component_id}")
        logger.info(f"  Search space: {list(task.search_space.keys())}")
        logger.info(f"  Budget: {task.budget} evaluations")

        # Create optimizer
        optimizer = SimpleBayesianOptimizer(
            search_space=task.search_space,
            evaluator=task.evaluator,
            n_calls=task.budget,
            n_initial_points=self.n_initial_points,
            objective='quality',  # We're optimizing the quality score
            minimize=False,  # Higher quality is better
            random_state=self.random_state,
            save_results=self.save_results,
            results_dir=f'cosmos_results/{task.component_id}' if self.save_results else 'bayesian_results'
        )

        # Run optimization
        result = optimizer.optimize()

        logger.info(f"Bayesian optimization complete: best_score={result.best_score:.3f}")
        return result

    def get_name(self) -> str:
        """Get strategy name"""
        return "Bayesian Optimization"