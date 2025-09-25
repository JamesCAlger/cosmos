"""Grid search optimizer for systematic configuration exploration"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import pandas as pd

from .search_space import SearchSpace
from .config_generator import ConfigurationGenerator
from .result_manager import ResultManager


class GridSearchOptimizer:
    """Systematic grid search over configuration space with budget management"""

    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: Callable,
                 budget_limit: float = 5.0,
                 parallel_workers: int = 1,
                 early_stopping_threshold: float = 0.2,
                 checkpoint_dir: Optional[str] = None):
        """
        Initialize grid search optimizer

        Args:
            search_space: Search space defining parameter ranges
            evaluator: Function to evaluate a configuration
            budget_limit: Maximum budget in dollars (default: $5)
            parallel_workers: Number of parallel evaluation workers
            early_stopping_threshold: Stop if performance below this threshold
            checkpoint_dir: Directory for saving checkpoints
        """
        self.search_space = search_space
        self.evaluator = evaluator
        self.budget_limit = budget_limit
        self.parallel_workers = parallel_workers
        self.early_stopping_threshold = early_stopping_threshold
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")

        self.config_generator = ConfigurationGenerator(search_space)
        self.result_manager = ResultManager()

        self.total_cost = 0.0
        self.configurations_evaluated = 0
        self.start_time = None
        self.is_stopped = False

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"GridSearchOptimizer initialized with budget=${budget_limit}, "
                   f"workers={parallel_workers}")

    def search(self,
              max_configurations: Optional[int] = None,
              resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """
        Execute grid search over configuration space

        Args:
            max_configurations: Maximum number of configurations to evaluate
            resume_from_checkpoint: Resume from previous checkpoint

        Returns:
            Dictionary with best configuration and results
        """
        self.start_time = time.time()
        logger.info("Starting grid search optimization")

        # Load checkpoint if resuming
        if resume_from_checkpoint:
            self._load_checkpoint()

        # Generate configurations
        all_configs = self.config_generator.generate_all_configurations()

        if max_configurations and max_configurations < len(all_configs):
            # Sample subset if needed
            configs_to_eval = self.config_generator.generate_subset(
                max_configurations, method="grid"
            )
        else:
            configs_to_eval = all_configs

        # Filter out already evaluated configurations
        configs_to_eval = self._filter_evaluated_configs(configs_to_eval)

        logger.info(f"Will evaluate {len(configs_to_eval)} configurations")

        # Execute search
        if self.parallel_workers > 1:
            results = self._parallel_search(configs_to_eval)
        else:
            results = self._sequential_search(configs_to_eval)

        # Get best configuration
        best_config = self.result_manager.get_best_configuration()

        # Generate final report
        report = self._generate_report(best_config)

        # Save final checkpoint
        self._save_checkpoint()

        logger.info(f"Grid search complete. Best score: {best_config['score']:.4f}")
        logger.info(f"Total cost: ${self.total_cost:.2f}")

        return report

    def _sequential_search(self, configurations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute sequential grid search"""
        results = []

        for i, config in enumerate(configurations):
            if self.is_stopped:
                logger.warning("Search stopped early")
                break

            logger.info(f"Evaluating configuration {i+1}/{len(configurations)}")

            # Check budget
            if self.total_cost >= self.budget_limit:
                logger.warning(f"Budget limit (${self.budget_limit}) reached")
                break

            # Evaluate configuration
            result = self._evaluate_configuration(config)

            if result:
                results.append(result)
                self.result_manager.add_result(result)

                # Check for early stopping
                if self._should_stop_early(result):
                    logger.warning(f"Early stopping triggered (score {result['score']:.4f} "
                                 f"< threshold {self.early_stopping_threshold})")
                    self.is_stopped = True

                # Save checkpoint periodically
                if (i + 1) % 10 == 0:
                    self._save_checkpoint()

        return results

    def _parallel_search(self, configurations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute parallel grid search"""
        results = []

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self._evaluate_configuration, config): config
                for config in configurations
            }

            # Process completed tasks
            for future in as_completed(future_to_config):
                if self.is_stopped:
                    # Cancel remaining tasks
                    for f in future_to_config:
                        f.cancel()
                    break

                if self.total_cost >= self.budget_limit:
                    logger.warning(f"Budget limit (${self.budget_limit}) reached")
                    for f in future_to_config:
                        f.cancel()
                    break

                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.result_manager.add_result(result)

                        # Check for early stopping
                        if self._should_stop_early(result):
                            logger.warning("Early stopping triggered")
                            self.is_stopped = True

                except Exception as e:
                    logger.error(f"Error evaluating configuration: {e}")

                # Save checkpoint periodically
                if len(results) % 10 == 0:
                    self._save_checkpoint()

        return results

    def _evaluate_configuration(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single configuration"""
        try:
            config_id = config["metadata"]["config_id"]
            logger.debug(f"Evaluating configuration {config_id}")

            # Track evaluation start
            eval_start = time.time()

            # Run evaluation
            eval_result = self.evaluator(config)

            # Track evaluation time
            eval_time = time.time() - eval_start

            # Extract metrics and cost
            metrics = eval_result.get("metrics", {})
            cost = eval_result.get("cost", 0.0)

            # Update total cost
            self.total_cost += cost
            self.configurations_evaluated += 1

            # Prepare result
            result = {
                "config_id": config_id,
                "configuration": config,
                "parameters": config["metadata"]["parameters"],
                "metrics": metrics,
                "score": self._calculate_score(metrics),
                "cost": cost,
                "evaluation_time": eval_time,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Config {config_id}: score={result['score']:.4f}, "
                       f"cost=${cost:.3f}, time={eval_time:.1f}s")

            return result

        except Exception as e:
            logger.error(f"Failed to evaluate configuration: {e}")
            return None

    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall score from metrics"""
        # Prioritize RAGAS metrics if available
        if "ragas_metrics" in metrics:
            ragas = metrics["ragas_metrics"]
            # Weighted average of RAGAS scores
            score = (
                ragas.get("faithfulness", 0) * 0.3 +
                ragas.get("answer_relevancy", 0) * 0.4 +
                ragas.get("context_relevance", 0) * 0.3
            )
        else:
            # Fallback to simple metrics
            score = metrics.get("retrieval_success", 0) * metrics.get("answer_generated", 0)

        return score

    def _should_stop_early(self, result: Dict[str, Any]) -> bool:
        """Check if early stopping should be triggered"""
        # Only stop early after evaluating at least 5 configurations
        if self.configurations_evaluated < 5:
            return False

        # Check if recent results are consistently poor
        recent_results = self.result_manager.get_recent_results(5)
        if not recent_results:
            return False

        avg_score = sum(r["score"] for r in recent_results) / len(recent_results)
        return avg_score < self.early_stopping_threshold

    def _filter_evaluated_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out already evaluated configurations"""
        evaluated_ids = self.result_manager.get_evaluated_config_ids()
        return [
            config for config in configs
            if config["metadata"]["config_id"] not in evaluated_ids
        ]

    def _save_checkpoint(self):
        """Save current search state"""
        checkpoint = {
            "configurations_evaluated": self.configurations_evaluated,
            "total_cost": self.total_cost,
            "results": self.result_manager.get_all_results(),
            "timestamp": datetime.now().isoformat()
        }

        checkpoint_file = self.checkpoint_dir / "grid_search_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.debug(f"Checkpoint saved: {self.configurations_evaluated} configs evaluated")

    def _load_checkpoint(self):
        """Load search state from checkpoint"""
        checkpoint_file = self.checkpoint_dir / "grid_search_checkpoint.json"

        if not checkpoint_file.exists():
            logger.warning("No checkpoint found to resume from")
            return

        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)

        self.configurations_evaluated = checkpoint["configurations_evaluated"]
        self.total_cost = checkpoint["total_cost"]

        # Restore results
        for result in checkpoint["results"]:
            self.result_manager.add_result(result)

        logger.info(f"Resumed from checkpoint: {self.configurations_evaluated} configs evaluated")

    def _generate_report(self, best_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization report"""
        all_results = self.result_manager.get_all_results()

        report = {
            "summary": {
                "configurations_evaluated": self.configurations_evaluated,
                "total_cost": self.total_cost,
                "total_time": time.time() - self.start_time if self.start_time else 0,
                "best_score": best_config["score"],
                "best_config_id": best_config["config_id"],
                "improvement_over_baseline": self._calculate_improvement()
            },
            "best_configuration": best_config,
            "top_5_configurations": self.result_manager.get_top_configurations(5),
            "parameter_importance": self._analyze_parameter_importance(),
            "cost_breakdown": self._get_cost_breakdown(),
            "timestamp": datetime.now().isoformat()
        }

        # Save report
        report_file = self.checkpoint_dir / f"optimization_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def _calculate_improvement(self) -> float:
        """Calculate improvement over baseline configuration"""
        results = self.result_manager.get_all_results()
        if not results:
            return 0.0

        # Find baseline (first configuration)
        baseline_score = results[0]["score"] if results else 0
        best_score = self.result_manager.get_best_configuration()["score"]

        if baseline_score > 0:
            return ((best_score - baseline_score) / baseline_score) * 100
        return 0.0

    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze which parameters have the most impact on performance"""
        results = self.result_manager.get_all_results()
        if len(results) < 2:
            return {}

        # Convert to DataFrame for analysis
        df = pd.DataFrame([r["parameters"] for r in results])
        scores = [r["score"] for r in results]

        importance = {}

        # Calculate correlation between each parameter and score
        for component in df.columns:
            component_df = pd.json_normalize(df[component])
            for param in component_df.columns:
                # Convert categorical to numeric if needed
                if component_df[param].dtype == 'object':
                    component_df[param] = pd.Categorical(component_df[param]).codes

                # Calculate correlation
                correlation = component_df[param].corr(pd.Series(scores))
                importance[f"{component}.{param}"] = abs(correlation) if not pd.isna(correlation) else 0

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _get_cost_breakdown(self) -> Dict[str, Any]:
        """Get cost breakdown by configuration type"""
        results = self.result_manager.get_all_results()

        breakdown = {
            "total": self.total_cost,
            "average_per_config": self.total_cost / max(1, len(results)),
            "by_component": {}
        }

        # Group by retrieval method
        for result in results:
            method = result["parameters"].get("retrieval", {}).get("method", "unknown")
            if method not in breakdown["by_component"]:
                breakdown["by_component"][method] = {"count": 0, "total_cost": 0}

            breakdown["by_component"][method]["count"] += 1
            breakdown["by_component"][method]["total_cost"] += result.get("cost", 0)

        return breakdown

    async def async_search(self, configurations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Asynchronous version of grid search for better performance"""
        results = []

        # Create tasks for all configurations
        tasks = [
            self._async_evaluate_configuration(config)
            for config in configurations
        ]

        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(self.parallel_workers)

        async def bounded_evaluate(config):
            async with semaphore:
                return await self._async_evaluate_configuration(config)

        bounded_tasks = [bounded_evaluate(config) for config in configurations]

        # Gather results
        completed = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, Exception):
                logger.error(f"Async evaluation failed: {result}")
            elif result:
                results.append(result)
                self.result_manager.add_result(result)

        return results

    async def _async_evaluate_configuration(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async version of configuration evaluation"""
        # This would need an async version of the evaluator
        # For now, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._evaluate_configuration, config)