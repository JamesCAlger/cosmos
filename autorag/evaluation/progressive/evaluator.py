"""Progressive evaluation system with configurable levels"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from loguru import logger
import numpy as np


class EvaluationLevel(Enum):
    """Evaluation levels with increasing comprehensiveness"""
    SMOKE = 0     # Quick validation
    QUICK = 1     # Fast pass/fail
    STANDARD = 2  # Normal evaluation
    COMPREHENSIVE = 3  # Thorough evaluation
    EXHAUSTIVE = 4  # Complete evaluation


@dataclass
class LevelConfig:
    """Configuration for an evaluation level"""
    name: str
    num_queries: int
    max_duration_seconds: float
    estimated_cost: float
    description: str
    confidence_threshold: float = 0.95
    early_stop_threshold: Optional[float] = None


class ProgressiveEvaluator:
    """Progressive evaluation system that saves costs by failing fast"""

    # Default evaluation levels - can be overridden
    DEFAULT_LEVELS = {
        EvaluationLevel.SMOKE: LevelConfig(
            name="Smoke Test",
            num_queries=5,
            max_duration_seconds=30,
            estimated_cost=0.01,
            description="Catch configuration errors",
            early_stop_threshold=0.3  # Fail if accuracy < 30%
        ),
        EvaluationLevel.QUICK: LevelConfig(
            name="Quick Evaluation",
            num_queries=20,
            max_duration_seconds=120,
            estimated_cost=0.05,
            description="Fast pass/fail decision",
            early_stop_threshold=0.4  # Fail if accuracy < 40%
        ),
        EvaluationLevel.STANDARD: LevelConfig(
            name="Standard Evaluation",
            num_queries=100,
            max_duration_seconds=600,
            estimated_cost=0.25,
            description="Statistical comparison",
            confidence_threshold=0.95
        ),
        EvaluationLevel.COMPREHENSIVE: LevelConfig(
            name="Comprehensive Evaluation",
            num_queries=500,
            max_duration_seconds=3600,
            estimated_cost=1.25,
            description="Thorough validation",
            confidence_threshold=0.99
        ),
        EvaluationLevel.EXHAUSTIVE: LevelConfig(
            name="Exhaustive Evaluation",
            num_queries=1000,
            max_duration_seconds=7200,
            estimated_cost=2.50,
            description="Complete evaluation",
            confidence_threshold=0.999
        ),
    }

    def __init__(self,
                 level_configs: Optional[Dict[EvaluationLevel, LevelConfig]] = None,
                 cost_per_token: float = 0.000002,  # Default GPT-3.5 pricing
                 auto_progress: bool = True):
        """
        Initialize progressive evaluator

        Args:
            level_configs: Custom level configurations
            cost_per_token: Cost per token for estimation
            auto_progress: Automatically progress to next level if current passes
        """
        self.levels = level_configs or self.DEFAULT_LEVELS
        self.cost_per_token = cost_per_token
        self.auto_progress = auto_progress
        self.evaluation_history: List[Dict] = []

    def add_level(self, level: EvaluationLevel, config: LevelConfig):
        """Add or update an evaluation level"""
        self.levels[level] = config
        logger.info(f"Added evaluation level {level.name}: {config.name}")

    def evaluate(self,
                 pipeline_func: Callable,
                 test_queries: List[Dict],
                 start_level: EvaluationLevel = EvaluationLevel.SMOKE,
                 target_level: Optional[EvaluationLevel] = None,
                 metrics_func: Optional[Callable] = None,
                 early_stop_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Progressively evaluate a pipeline

        Args:
            pipeline_func: Function that takes queries and returns results
            test_queries: List of test queries
            start_level: Starting evaluation level
            target_level: Target level to reach (None = auto-progress)
            metrics_func: Function to calculate metrics from results
            early_stop_func: Custom early stopping logic

        Returns:
            Evaluation results with level reached and metrics
        """
        results = {
            "levels_completed": [],
            "final_level": None,
            "total_cost": 0.0,
            "total_time": 0.0,
            "metrics": {},
            "stopped_early": False,
            "stop_reason": None
        }

        current_level = start_level
        cumulative_results = []

        while current_level is not None:
            level_config = self.levels.get(current_level)
            if not level_config:
                logger.warning(f"Level {current_level} not configured, skipping")
                break

            logger.info(f"Starting {level_config.name} ({level_config.num_queries} queries)")

            # Run evaluation for this level
            level_result = self._evaluate_level(
                pipeline_func,
                test_queries,
                level_config,
                metrics_func
            )

            # Add to history
            self.evaluation_history.append(level_result)
            cumulative_results.extend(level_result["results"])

            # Update aggregate results
            results["levels_completed"].append(current_level.name)
            results["total_cost"] += level_result["cost"]
            results["total_time"] += level_result["duration"]
            results["metrics"][current_level.name] = level_result["metrics"]
            results["final_level"] = current_level

            # Check early stopping conditions
            should_stop, reason = self._check_early_stop(
                level_result,
                level_config,
                early_stop_func
            )

            if should_stop:
                results["stopped_early"] = True
                results["stop_reason"] = reason
                logger.warning(f"Early stopping: {reason}")
                break

            # Check if we've reached target level
            if target_level and current_level == target_level:
                logger.info(f"Reached target level: {target_level.name}")
                break

            # Progress to next level if auto_progress is enabled
            if self.auto_progress:
                next_level = self._get_next_level(current_level)
                if next_level:
                    # Check if metrics are good enough to continue
                    if self._should_continue(level_result, level_config):
                        current_level = next_level
                    else:
                        logger.info("Metrics don't warrant progression to next level")
                        break
                else:
                    break
            else:
                break

        # Calculate final aggregated metrics
        if metrics_func and cumulative_results:
            results["final_metrics"] = metrics_func(cumulative_results)

        return results

    def _evaluate_level(self,
                        pipeline_func: Callable,
                        test_queries: List[Dict],
                        config: LevelConfig,
                        metrics_func: Optional[Callable]) -> Dict:
        """Evaluate at a specific level"""
        start_time = time.time()

        # Sample queries for this level
        sampled_queries = self._sample_queries(test_queries, config.num_queries)

        # Run pipeline
        results = []
        total_tokens = 0

        for query in sampled_queries:
            try:
                result = pipeline_func(query)
                results.append(result)

                # Estimate tokens (rough approximation)
                if isinstance(result, dict):
                    total_tokens += len(str(result).split()) * 1.3  # Rough token estimate

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                results.append({"error": str(e)})

            # Check timeout
            if time.time() - start_time > config.max_duration_seconds:
                logger.warning(f"Level {config.name} timeout reached")
                break

        duration = time.time() - start_time

        # Calculate metrics
        metrics = {}
        if metrics_func:
            metrics = metrics_func(results)

        # Calculate cost
        estimated_cost = total_tokens * self.cost_per_token

        return {
            "level": config.name,
            "num_queries": len(results),
            "duration": duration,
            "cost": estimated_cost,
            "metrics": metrics,
            "results": results,
            "config": config
        }

    def _sample_queries(self, queries: List[Dict], n: int) -> List[Dict]:
        """Sample queries for evaluation"""
        if len(queries) <= n:
            return queries

        # Use deterministic sampling for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(queries), n, replace=False)
        return [queries[i] for i in indices]

    def _check_early_stop(self,
                          level_result: Dict,
                          config: LevelConfig,
                          custom_func: Optional[Callable]) -> Tuple[bool, Optional[str]]:
        """Check if evaluation should stop early"""

        # Custom early stopping logic
        if custom_func:
            should_stop, reason = custom_func(level_result)
            if should_stop:
                return True, reason

        # Check threshold-based early stopping
        if config.early_stop_threshold:
            metrics = level_result.get("metrics", {})

            # Check if any key metric is below threshold
            for metric_name, value in metrics.items():
                if "accuracy" in metric_name.lower() or "f1" in metric_name.lower():
                    if isinstance(value, (int, float)) and value < config.early_stop_threshold:
                        return True, f"{metric_name} ({value:.3f}) below threshold ({config.early_stop_threshold})"

        # Check for critical errors
        errors = sum(1 for r in level_result["results"] if "error" in r)
        error_rate = errors / len(level_result["results"]) if level_result["results"] else 0

        if error_rate > 0.5:
            return True, f"High error rate: {error_rate:.1%}"

        return False, None

    def _should_continue(self, level_result: Dict, config: LevelConfig) -> bool:
        """Determine if evaluation should continue to next level"""
        metrics = level_result.get("metrics", {})

        # Check if metrics meet minimum requirements
        for metric_name, value in metrics.items():
            if "accuracy" in metric_name.lower():
                if isinstance(value, (int, float)):
                    # Continue if accuracy is reasonable
                    return value > 0.5

        # Default to continuing if no clear signal
        return True

    def _get_next_level(self, current: EvaluationLevel) -> Optional[EvaluationLevel]:
        """Get the next evaluation level"""
        level_values = list(EvaluationLevel)
        current_idx = level_values.index(current)

        if current_idx < len(level_values) - 1:
            return level_values[current_idx + 1]

        return None

    def estimate_cost(self, level: EvaluationLevel) -> float:
        """Estimate cost for a given evaluation level"""
        config = self.levels.get(level)
        return config.estimated_cost if config else 0.0

    def estimate_time(self, level: EvaluationLevel) -> float:
        """Estimate time for a given evaluation level"""
        config = self.levels.get(level)
        return config.max_duration_seconds if config else 0.0

    def get_optimal_level(self, budget: float, time_limit: float) -> Optional[EvaluationLevel]:
        """Get the highest evaluation level within budget and time constraints"""
        best_level = None

        for level in EvaluationLevel:
            config = self.levels.get(level)
            if config:
                if config.estimated_cost <= budget and config.max_duration_seconds <= time_limit:
                    best_level = level

        return best_level

    def get_confidence_interval(self, level: EvaluationLevel) -> float:
        """Get expected confidence interval for a level"""
        config = self.levels.get(level)
        if not config:
            return 0.0

        # Rough estimate based on sample size
        # Using standard error approximation
        n = config.num_queries
        if n < 30:
            # Small sample - wider interval
            return 0.2
        elif n < 100:
            return 0.1
        elif n < 500:
            return 0.05
        else:
            return 0.02

    def summary(self) -> str:
        """Get summary of evaluation levels"""
        lines = ["Progressive Evaluation Levels:"]
        lines.append("-" * 60)

        for level in EvaluationLevel:
            config = self.levels.get(level)
            if config:
                lines.append(f"\n{level.name} - {config.name}:")
                lines.append(f"  Queries: {config.num_queries}")
                lines.append(f"  Max Duration: {config.max_duration_seconds}s")
                lines.append(f"  Estimated Cost: ${config.estimated_cost:.2f}")
                lines.append(f"  Description: {config.description}")
                if config.early_stop_threshold:
                    lines.append(f"  Early Stop Threshold: {config.early_stop_threshold}")

        return "\n".join(lines)