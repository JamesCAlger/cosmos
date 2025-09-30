"""
Random Search Strategy

Baseline strategy for comparison with Bayesian optimization.
"""

import random
import numpy as np
from typing import Dict, Any, List
from loguru import logger
from autorag.optimization.bayesian_search import OptimizationResult
from autorag.cosmos.optimization.strategy import OptimizationStrategy
from autorag.cosmos.optimization.task import OptimizationTask


class RandomSearchStrategy(OptimizationStrategy):
    """
    Random search baseline strategy

    Randomly samples configurations from the search space.
    Useful for comparing against more sophisticated strategies.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize random search strategy

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)

        logger.info(f"RandomSearchStrategy initialized: seed={random_state}")

    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        """
        Run random search on the task

        Args:
            task: OptimizationTask with search space and evaluator

        Returns:
            OptimizationResult with best configuration found
        """
        logger.info(f"Starting random search for {task.component_id}")
        logger.info(f"  Search space: {list(task.search_space.keys())}")
        logger.info(f"  Budget: {task.budget} evaluations")

        all_configs = []
        all_scores = []
        best_score = float('-inf')
        best_config = None

        # Sample and evaluate configurations
        for i in range(task.budget):
            # Sample random configuration
            config = self._sample_config(task.search_space)
            all_configs.append(config)

            # Evaluate
            try:
                score = task.evaluator(config)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                score = 0.0

            all_scores.append(score)

            # Track best
            if score > best_score:
                best_score = score
                best_config = config
                logger.info(f"Evaluation {i+1}/{task.budget}: New best score: {score:.3f}")
            elif (i + 1) % 5 == 0:
                logger.info(f"Evaluation {i+1}/{task.budget}: score={score:.3f}, best={best_score:.3f}")

        # Create result
        result = OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            all_configs=all_configs,
            all_scores=all_scores,
            n_evaluations=len(all_configs),
            convergence_history=self._compute_convergence(all_scores)
        )

        logger.info(f"Random search complete: best_score={best_score:.3f}")
        return result

    def _sample_config(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample a random configuration from search space

        Args:
            search_space: Parameter search space

        Returns:
            Random configuration dictionary
        """
        config = {}

        for param_name, param_spec in search_space.items():
            if isinstance(param_spec, list):
                # Categorical: pick random element
                config[param_name] = random.choice(param_spec)
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                # Range: sample uniformly
                if all(isinstance(v, int) for v in param_spec):
                    # Integer range
                    config[param_name] = random.randint(param_spec[0], param_spec[1])
                else:
                    # Float range
                    config[param_name] = random.uniform(param_spec[0], param_spec[1])
            else:
                raise ValueError(f"Unknown parameter specification: {param_spec}")

        return config

    def _compute_convergence(self, all_scores: List[float]) -> List[float]:
        """
        Compute convergence history (best score so far at each step)

        Args:
            all_scores: List of scores in evaluation order

        Returns:
            List of best scores so far
        """
        convergence = []
        best_so_far = float('-inf')

        for score in all_scores:
            if score > best_so_far:
                best_so_far = score
            convergence.append(best_so_far)

        return convergence

    def get_name(self) -> str:
        """Get strategy name"""
        return "Random Search"