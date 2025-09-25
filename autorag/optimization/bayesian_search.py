"""
Simple Bayesian Optimization implementation for auto-RAG system.
Phase 1: Basic single-objective optimization using scikit-optimize.
"""

import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging

try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Install with: pip install scikit-optimize")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_config: Dict[str, Any]
    best_score: float
    all_configs: List[Dict[str, Any]] = field(default_factory=list)
    all_scores: List[float] = field(default_factory=list)
    all_metrics: List[Dict[str, Any]] = field(default_factory=list)
    n_evaluations: int = 0
    total_time: float = 0.0
    convergence_history: List[float] = field(default_factory=list)


class SimpleBayesianOptimizer:
    """
    Initial Bayesian optimizer using scikit-optimize.
    Single-objective optimization for quick implementation.
    """

    def __init__(self,
                 search_space: Any,
                 evaluator: Callable,
                 n_calls: int = 50,
                 n_initial_points: int = 10,
                 objective: str = "accuracy",
                 minimize: bool = False,
                 random_state: int = 42,
                 save_results: bool = True,
                 results_dir: str = "bayesian_results"):
        """
        Initialize the Bayesian optimizer.

        Args:
            search_space: SearchSpace object or dict defining the parameter space
            evaluator: Function that takes a config and returns metrics
            n_calls: Total number of evaluations
            n_initial_points: Number of random exploration points
            objective: Metric to optimize (e.g., 'accuracy', 'f1_score')
            minimize: Whether to minimize the objective (False for accuracy)
            random_state: Random seed for reproducibility
            save_results: Whether to save intermediate results
            results_dir: Directory to save results
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")

        self.search_space = search_space
        self.evaluator = evaluator
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.objective = objective
        self.minimize = minimize
        self.random_state = random_state
        self.save_results = save_results
        self.results_dir = Path(results_dir)

        # Initialize results storage
        self.results = []
        self.best_score = float('inf') if minimize else float('-inf')
        self.best_config = None
        self.convergence_history = []

        # Create results directory if needed
        if save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SimpleBayesianOptimizer initialized: {n_calls} calls, {n_initial_points} initial points")

    def optimize(self) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Returns:
            OptimizationResult object with best configuration and metrics
        """
        logger.info("Starting Bayesian optimization...")
        start_time = time.time()

        # Convert search space to skopt format
        skopt_space = self._convert_search_space()

        # Track all evaluations
        all_configs = []
        all_scores = []
        all_metrics = []

        # Define objective function
        @use_named_args(skopt_space)
        def objective(**params):
            # Convert params to configuration
            config = self._params_to_config(params)
            all_configs.append(config)

            # Evaluate configuration
            try:
                result = self.evaluator(config)

                # Extract the objective metric
                if isinstance(result, dict):
                    if 'metrics' in result:
                        score = result['metrics'].get(self.objective, 0)
                        all_metrics.append(result['metrics'])
                    else:
                        score = result.get(self.objective, 0)
                        all_metrics.append(result)
                else:
                    score = float(result)
                    all_metrics.append({'score': score})

            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                score = float('inf') if self.minimize else float('-inf')
                all_metrics.append({'error': str(e)})

            all_scores.append(score)

            # Update best score
            if self.minimize:
                is_better = score < self.best_score
            else:
                is_better = score > self.best_score

            if is_better:
                self.best_score = score
                self.best_config = config
                logger.info(f"New best score: {score:.4f}")

            # Update convergence history (best score so far)
            if self.convergence_history:
                if self.minimize:
                    self.convergence_history.append(min(self.best_score, self.convergence_history[-1]))
                else:
                    self.convergence_history.append(max(self.best_score, self.convergence_history[-1]))
            else:
                self.convergence_history.append(self.best_score)

            # Save intermediate results
            if self.save_results and len(all_configs) % 5 == 0:
                self._save_intermediate_results(all_configs, all_scores, all_metrics)

            # Log progress
            n_evals = len(all_configs)
            if n_evals % 10 == 0 or n_evals <= 5:
                logger.info(f"Evaluation {n_evals}/{self.n_calls}: score={score:.4f}, best={self.best_score:.4f}")

            # Return negative score for maximization (skopt minimizes)
            return score if self.minimize else -score

        # Run optimization
        logger.info(f"Running GP minimization with {self.n_calls} calls...")
        result = gp_minimize(
            func=objective,
            dimensions=skopt_space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func='EI',  # Expected Improvement
            random_state=self.random_state,
            verbose=False
        )

        total_time = time.time() - start_time

        # Create final result
        optimization_result = OptimizationResult(
            best_config=self.best_config,
            best_score=self.best_score,
            all_configs=all_configs,
            all_scores=all_scores,
            all_metrics=all_metrics,
            n_evaluations=len(all_configs),
            total_time=total_time,
            convergence_history=self.convergence_history
        )

        # Save final results
        if self.save_results:
            self._save_final_results(optimization_result)

        # Log summary
        logger.info(f"Optimization complete in {total_time:.1f}s")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Total evaluations: {len(all_configs)}")

        return optimization_result

    def _convert_search_space(self) -> List:
        """
        Convert SearchSpace object to scikit-optimize format.

        Returns:
            List of skopt dimension objects
        """
        skopt_dimensions = []

        # Handle different search space formats
        if hasattr(self.search_space, 'components'):
            # auto-RAG SearchSpace object
            for component_name, component_space in self.search_space.components.items():
                for param_range in component_space.parameters:
                    dim_name = f"{component_name}.{param_range.name}"

                    if param_range.parameter_type == "categorical":
                        dimension = Categorical(param_range.values, name=dim_name)
                    elif param_range.parameter_type == "numerical":
                        if all(isinstance(v, int) for v in param_range.values):
                            dimension = Integer(
                                min(param_range.values),
                                max(param_range.values),
                                name=dim_name
                            )
                        else:
                            dimension = Real(
                                min(param_range.values),
                                max(param_range.values),
                                name=dim_name
                            )
                    elif param_range.parameter_type == "boolean":
                        dimension = Categorical([True, False], name=dim_name)
                    else:
                        # Default to categorical
                        dimension = Categorical(param_range.values, name=dim_name)

                    skopt_dimensions.append(dimension)

        elif isinstance(self.search_space, dict):
            # Simple dict format
            for param_name, param_spec in self.search_space.items():
                if isinstance(param_spec, list):
                    # Categorical
                    dimension = Categorical(param_spec, name=param_name)
                elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                    # Range
                    if all(isinstance(v, int) for v in param_spec):
                        dimension = Integer(param_spec[0], param_spec[1], name=param_name)
                    else:
                        dimension = Real(param_spec[0], param_spec[1], name=param_name)
                else:
                    raise ValueError(f"Unknown parameter specification: {param_spec}")

                skopt_dimensions.append(dimension)
        else:
            raise ValueError(f"Unknown search space format: {type(self.search_space)}")

        logger.info(f"Converted search space: {len(skopt_dimensions)} dimensions")
        return skopt_dimensions

    def _params_to_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert skopt parameters to configuration format.

        Args:
            params: Parameters from skopt

        Returns:
            Configuration dict for the evaluator
        """
        config = {}

        # Group parameters by component
        for param_name, param_value in params.items():
            if '.' in param_name:
                # Component.parameter format
                component, param = param_name.split('.', 1)
                if component not in config:
                    config[component] = {}
                config[component][param] = param_value
            else:
                # Flat format
                config[param_name] = param_value

        return config

    def _save_intermediate_results(self, configs: List, scores: List, metrics: List):
        """Save intermediate results to file"""
        intermediate_file = self.results_dir / f"intermediate_{len(configs)}.json"

        # Convert configs to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results = {
            'n_evaluations': len(configs),
            'best_score': float(self.best_score),
            'best_config': make_serializable(self.best_config),
            'last_5_configs': make_serializable(configs[-5:]),
            'last_5_scores': [float(s) for s in scores[-5:]],
            'convergence': [float(s) for s in self.convergence_history[-20:]]
        }

        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)

    def _save_final_results(self, result: OptimizationResult):
        """Save final optimization results"""
        final_file = self.results_dir / "bayesian_optimization_results.json"

        # Convert to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Prepare serializable results
        results = {
            'best_config': make_serializable(result.best_config),
            'best_score': float(result.best_score),
            'n_evaluations': result.n_evaluations,
            'total_time': result.total_time,
            'convergence_history': [float(s) for s in result.convergence_history],
            'all_scores': [float(s) for s in result.all_scores],
            'summary': {
                'avg_score': float(np.mean(result.all_scores)),
                'std_score': float(np.std(result.all_scores)),
                'min_score': float(min(result.all_scores)),
                'max_score': float(max(result.all_scores)),
                'improvement': float(result.best_score - result.all_scores[0])
            }
        }

        with open(final_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {final_file}")

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot the convergence history.

        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Convergence plot
            ax1.plot(self.convergence_history, 'b-', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel(f'{self.objective.capitalize()}')
            ax1.set_title('Optimization Convergence')
            ax1.grid(True, alpha=0.3)

            # Score distribution
            ax2.hist(self.results, bins=20, edgecolor='black', alpha=0.7)
            ax2.axvline(self.best_score, color='red', linestyle='--', label=f'Best: {self.best_score:.4f}')
            ax2.set_xlabel(f'{self.objective.capitalize()}')
            ax2.set_ylabel('Count')
            ax2.set_title('Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"Convergence plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")