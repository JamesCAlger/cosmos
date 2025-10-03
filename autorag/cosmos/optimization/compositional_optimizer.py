"""
Compositional Optimizer

Orchestrates component-by-component optimization with context passing.
This is the main class that brings together all COSMOS components.
"""

import time
from typing import Dict, Any, List, Optional
from loguru import logger

from autorag.cosmos.optimization.strategy import OptimizationStrategy
from autorag.cosmos.optimization.task import OptimizationTask
from autorag.cosmos.optimization.evaluators import ComponentEvaluator, build_component
from autorag.cosmos.metrics import ComponentMetrics
from autorag.optimization.bayesian_search import OptimizationResult


class CompositionalOptimizer:
    """
    Orchestrates component-by-component optimization

    This is the core COSMOS class that enables compositional optimization by:
    1. Optimizing components sequentially (forward order)
    2. Passing best upstream components to downstream optimization
    3. Breaking the circular dependency problem
    4. Allocating budget across components
    """

    def __init__(self, optimization_strategy: OptimizationStrategy):
        """
        Initialize compositional optimizer

        Args:
            optimization_strategy: Strategy to use for each component optimization
                                  (BayesianStrategy, RandomSearchStrategy, etc.)
        """
        self.strategy = optimization_strategy
        self.component_results: Dict[str, OptimizationResult] = {}
        self.upstream_components: Dict[str, Any] = {}

        logger.info(f"CompositionalOptimizer initialized with strategy: {self.strategy.get_name()}")

    def optimize(self,
                 components: List[str],
                 search_spaces: Dict[str, Dict[str, Any]],
                 test_data: Dict[str, Any],
                 metric_collector: ComponentMetrics,
                 total_budget: int,
                 budget_allocation: Optional[Dict[str, int]] = None,
                 max_queries: int = 10,
                 cache_manager: Optional[Any] = None,
                 dataset_name: Optional[str] = None) -> Dict[str, OptimizationResult]:
        """
        Optimize components sequentially with context passing

        Args:
            components: List of component IDs in optimization order
                       e.g., ['chunker', 'retriever', 'generator']
            search_spaces: Dict mapping component_id -> search_space
            test_data: Test data with 'documents' and 'queries'
            metric_collector: ComponentMetrics instance for evaluation
            total_budget: Total number of evaluations across all components
            budget_allocation: Optional custom budget per component
                             If None, splits budget equally
            max_queries: Maximum number of queries to use per evaluation (default: 10)
            cache_manager: Optional EmbeddingCacheManager for caching embeddings
            dataset_name: Name of dataset being used (e.g., 'marco', 'beir/scifact')

        Returns:
            Dict mapping component_id -> OptimizationResult
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPOSITIONAL OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Components: {components}")
        logger.info(f"Total budget: {total_budget}")
        logger.info(f"Strategy: {self.strategy.get_name()}")
        if cache_manager:
            logger.info(f"Cache: ENABLED (dataset={dataset_name}, docs={len(test_data.get('documents', []))})")
        else:
            logger.info("Cache: DISABLED")

        # Allocate budget across components
        if budget_allocation is None:
            budget_per_component = total_budget // len(components)
            budget_allocation = {comp: budget_per_component for comp in components}
            logger.info(f"Budget allocation (equal split): {budget_per_component} per component")
        else:
            logger.info(f"Budget allocation (custom): {budget_allocation}")

        # Track overall timing
        overall_start = time.time()

        # Optimize each component in sequence
        for i, component_id in enumerate(components):
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"OPTIMIZING COMPONENT {i+1}/{len(components)}: {component_id}")
            logger.info("=" * 80)

            # Get budget for this component
            budget = budget_allocation.get(component_id, total_budget // len(components))

            # Create evaluator with upstream context
            evaluator = ComponentEvaluator(
                component_type=component_id,
                test_data=test_data,
                metric_collector=metric_collector,
                upstream_components=self.upstream_components.copy(),
                max_queries=max_queries,
                cache_manager=cache_manager,
                dataset_name=dataset_name
            )

            # Create optimization task
            task = OptimizationTask(
                component_id=component_id,
                search_space=search_spaces[component_id],
                evaluator=evaluator.evaluate,
                budget=budget,
                context={'upstream_results': self.component_results.copy()}
            )

            # Log task details
            logger.info(f"Search space parameters: {list(task.search_space.keys())}")
            logger.info(f"Budget: {budget} evaluations")
            if self.upstream_components:
                logger.info(f"Using upstream: {list(self.upstream_components.keys())}")

            # Run optimization
            component_start = time.time()
            result = self.strategy.optimize(task)
            component_time = time.time() - component_start

            # Store result
            self.component_results[component_id] = result

            # Build best component for downstream use
            logger.info("")
            logger.info(f"Building best {component_id} for downstream use...")
            best_component = build_component(component_id, result.best_config)
            self.upstream_components[component_id] = best_component

            # Log results
            logger.info("")
            logger.info(f"COMPONENT {component_id} OPTIMIZATION COMPLETE")
            logger.info(f"  Best score: {result.best_score:.4f}")
            logger.info(f"  Evaluations: {result.n_evaluations}")
            logger.info(f"  Time: {component_time:.2f}s")
            logger.info(f"  Best config: {result.best_config}")

        # Overall summary
        total_time = time.time() - overall_start
        total_evaluations = sum(r.n_evaluations for r in self.component_results.values())

        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPOSITIONAL OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Total evaluations: {total_evaluations}")
        logger.info("")
        logger.info("Best configurations found:")
        for component_id, result in self.component_results.items():
            logger.info(f"  {component_id}: score={result.best_score:.4f}, config={result.best_config}")

        return self.component_results

    def get_best_pipeline_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get combined best configuration for entire pipeline

        Returns:
            Dict mapping component_id -> best_config
        """
        pipeline_config = {}
        for component_id, result in self.component_results.items():
            pipeline_config[component_id] = result.best_config
        return pipeline_config

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary

        Returns:
            Dictionary with summary statistics
        """
        if not self.component_results:
            return {'status': 'not_run'}

        total_evaluations = sum(r.n_evaluations for r in self.component_results.values())
        avg_score = sum(r.best_score for r in self.component_results.values()) / len(self.component_results)

        summary = {
            'status': 'complete',
            'strategy': self.strategy.get_name(),
            'components_optimized': list(self.component_results.keys()),
            'total_evaluations': total_evaluations,
            'average_score': avg_score,
            'component_scores': {
                comp_id: result.best_score
                for comp_id, result in self.component_results.items()
            },
            'component_evaluations': {
                comp_id: result.n_evaluations
                for comp_id, result in self.component_results.items()
            }
        }

        return summary

    def clear_results(self):
        """Clear optimization results and reset state"""
        self.component_results = {}
        self.upstream_components = {}
        logger.info("Cleared optimization results")


class CompositionalOptimizerBuilder:
    """
    Builder class for creating CompositionalOptimizer with common configurations

    Provides convenience methods for setting up the optimizer.
    """

    @staticmethod
    def create_with_bayesian(n_initial_points: int = 5, random_state: int = 42) -> CompositionalOptimizer:
        """
        Create optimizer with Bayesian strategy

        Args:
            n_initial_points: Number of random exploration points
            random_state: Random seed

        Returns:
            CompositionalOptimizer configured with BayesianStrategy
        """
        from autorag.cosmos.optimization.bayesian_strategy import BayesianStrategy

        strategy = BayesianStrategy(
            n_initial_points=n_initial_points,
            random_state=random_state,
            save_results=False
        )

        return CompositionalOptimizer(strategy)

    @staticmethod
    def create_with_random(random_state: int = 42) -> CompositionalOptimizer:
        """
        Create optimizer with random search strategy

        Args:
            random_state: Random seed

        Returns:
            CompositionalOptimizer configured with RandomSearchStrategy
        """
        from autorag.cosmos.optimization.random_strategy import RandomSearchStrategy

        strategy = RandomSearchStrategy(random_state=random_state)

        return CompositionalOptimizer(strategy)

    @staticmethod
    def create_with_custom_strategy(strategy: OptimizationStrategy) -> CompositionalOptimizer:
        """
        Create optimizer with custom strategy

        Args:
            strategy: Custom OptimizationStrategy implementation

        Returns:
            CompositionalOptimizer with the provided strategy
        """
        return CompositionalOptimizer(strategy)