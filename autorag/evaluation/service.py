"""Standalone evaluation service for RAG pipelines"""

from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import all evaluation components
from .cache.base import TieredCache, CacheKey
from .progressive.evaluator import ProgressiveEvaluator, EvaluationLevel, LevelConfig
from .statistics.analyzer import StatisticalAnalyzer
from .cost_tracker import CostTracker
from .reporters.base import CompositeReporter, JSONReporter, HTMLReporter, MarkdownReporter
from .ragas_evaluator import RAGASEvaluator


class EvaluationService:
    """
    Standalone evaluation service that can evaluate any RAG pipeline.

    This service is designed to be pipeline-agnostic and can evaluate
    any system that follows the standard interface.
    """

    def __init__(self,
                 cache_dir: str = "cache/evaluation",
                 enable_caching: bool = True,
                 cost_tracking: bool = True,
                 budget_limit: Optional[float] = None,
                 progressive_eval: bool = True,
                 statistical_analysis: bool = True,
                 reporter_formats: Optional[List[str]] = None):
        """
        Initialize evaluation service

        Args:
            cache_dir: Directory for caching evaluation results
            enable_caching: Whether to use caching
            cost_tracking: Whether to track costs
            budget_limit: Maximum budget for evaluation
            progressive_eval: Whether to use progressive evaluation
            statistical_analysis: Whether to perform statistical analysis
            reporter_formats: List of report formats to generate
        """
        # Initialize components
        self.cache = TieredCache(cache_dir) if enable_caching else None
        self.cost_tracker = CostTracker(budget_limit=budget_limit) if cost_tracking else None
        self.progressive_evaluator = ProgressiveEvaluator() if progressive_eval else None
        self.statistical_analyzer = StatisticalAnalyzer() if statistical_analysis else None
        self.ragas_evaluator = RAGASEvaluator()

        # Configure reporters
        if reporter_formats is None:
            reporter_formats = ["json", "html", "markdown"]

        reporters = []
        if "json" in reporter_formats:
            reporters.append(JSONReporter())
        if "html" in reporter_formats:
            reporters.append(HTMLReporter())
        if "markdown" in reporter_formats:
            reporters.append(MarkdownReporter())

        self.reporter = CompositeReporter(reporters)

        # Evaluation state
        self.current_evaluation = None
        self.evaluation_history = []

        logger.info("Evaluation service initialized")

    def evaluate_pipeline(self,
                           pipeline: Any,
                           test_queries: List[Dict],
                           config: Optional[Dict] = None,
                           evaluation_name: str = "evaluation",
                           progressive_levels: Optional[List[EvaluationLevel]] = None,
                           metrics_to_track: Optional[List[str]] = None,
                           output_dir: str = "experiments") -> Dict[str, Any]:
        """
        Evaluate a RAG pipeline

        Args:
            pipeline: The pipeline to evaluate (must have query() method)
            test_queries: List of test queries with ground truth
            config: Pipeline configuration
            evaluation_name: Name for this evaluation
            progressive_levels: Which evaluation levels to run
            metrics_to_track: Specific metrics to track
            output_dir: Directory for output reports

        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting evaluation: {evaluation_name}")
        start_time = time.time()

        # Initialize evaluation state
        self.current_evaluation = {
            "name": evaluation_name,
            "config": config or {},
            "start_time": datetime.now().isoformat(),
            "test_queries": len(test_queries)
        }

        results = {}

        try:
            # Progressive evaluation if enabled
            if self.progressive_evaluator and progressive_levels:
                results["progressive"] = self._run_progressive_evaluation(
                    pipeline,
                    test_queries,
                    progressive_levels
                )
            else:
                # Standard evaluation
                results["standard"] = self._run_standard_evaluation(
                    pipeline,
                    test_queries,
                    metrics_to_track
                )

            # Statistical analysis if multiple configurations
            if self.statistical_analyzer and len(self.evaluation_history) > 0:
                results["statistical_comparison"] = self._run_statistical_comparison(
                    results,
                    self.evaluation_history[-1]
                )

            # Cost summary
            if self.cost_tracker:
                results["cost_summary"] = self.cost_tracker.get_summary()

            # Add metadata
            results["metadata"] = {
                "evaluation_name": evaluation_name,
                "config": config,
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

            # Generate reports
            output_path = Path(output_dir) / evaluation_name
            reports = self.reporter.report(results, str(output_path))
            results["reports_generated"] = list(reports.keys())

            # Save to history
            self.evaluation_history.append(results)

            logger.info(f"Evaluation completed in {time.time() - start_time:.2f}s")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results["error"] = str(e)

        finally:
            self.current_evaluation = None

        return results

    def evaluate_multiple_configs(self,
                                   configs: List[Dict],
                                   pipeline_factory: Callable,
                                   test_queries: List[Dict],
                                   parallel: bool = True,
                                   max_workers: int = 4) -> Dict[str, Any]:
        """
        Evaluate multiple pipeline configurations

        Args:
            configs: List of configurations to evaluate
            pipeline_factory: Function that creates pipeline from config
            test_queries: Test queries
            parallel: Whether to run evaluations in parallel
            max_workers: Maximum parallel workers

        Returns:
            Comparison results
        """
        logger.info(f"Evaluating {len(configs)} configurations")

        results = []

        if parallel:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, config in enumerate(configs):
                    future = executor.submit(
                        self._evaluate_single_config,
                        config,
                        pipeline_factory,
                        test_queries,
                        f"config_{i}"
                    )
                    futures[future] = i

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed evaluation {idx + 1}/{len(configs)}")
                    except Exception as e:
                        logger.error(f"Evaluation {idx} failed: {e}")
                        results.append({"error": str(e), "config": configs[idx]})
        else:
            # Sequential evaluation
            for i, config in enumerate(configs):
                result = self._evaluate_single_config(
                    config,
                    pipeline_factory,
                    test_queries,
                    f"config_{i}"
                )
                results.append(result)
                logger.info(f"Completed evaluation {i + 1}/{len(configs)}")

        # Statistical comparison
        comparison_results = self._compare_all_results(results)

        return {
            "individual_results": results,
            "comparison": comparison_results,
            "best_config": self._find_best_config(results)
        }

    def _run_progressive_evaluation(self,
                                     pipeline: Any,
                                     test_queries: List[Dict],
                                     levels: List[EvaluationLevel]) -> Dict:
        """Run progressive evaluation"""

        def pipeline_func(query: Dict) -> Dict:
            """Wrapper for pipeline query function"""
            # Check cache
            if self.cache:
                cache_key = CacheKey.generate(
                    self.current_evaluation.get("config", {}),
                    query["question"],
                    "",  # No context for cache key
                    ""   # No answer for cache key
                )
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    return cached_result

            # Run pipeline
            result = pipeline.query(query["question"])

            # Track cost if enabled
            if self.cost_tracker:
                self.cost_tracker.estimate_cost(
                    query["question"],
                    model=self.current_evaluation.get("config", {}).get("generation", {}).get("model", "gpt-3.5-turbo"),
                    operation="generation",
                    output_text=result.get("answer", "")
                )

            # Cache result
            if self.cache:
                self.cache.set(cache_key, result)

            return result

        def metrics_func(results: List[Dict]) -> Dict:
            """Calculate metrics from results"""
            # Use RAGAS evaluator
            ground_truths = [q.get("ground_truth_answer") for q in test_queries[:len(results)]]
            return self.ragas_evaluator.evaluate(results, ground_truths)

        # Run progressive evaluation
        return self.progressive_evaluator.evaluate(
            pipeline_func,
            test_queries,
            start_level=levels[0] if levels else EvaluationLevel.SMOKE,
            target_level=levels[-1] if len(levels) > 1 else None,
            metrics_func=metrics_func
        )

    def _run_standard_evaluation(self,
                                  pipeline: Any,
                                  test_queries: List[Dict],
                                  metrics_to_track: Optional[List[str]]) -> Dict:
        """Run standard (non-progressive) evaluation"""
        results = []
        ground_truths = []

        for query in test_queries:
            # Check cache
            if self.cache:
                cache_key = CacheKey.generate(
                    self.current_evaluation.get("config", {}),
                    query["question"],
                    "",
                    ""
                )
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    results.append(cached_result)
                    ground_truths.append(query.get("ground_truth_answer"))
                    continue

            # Run pipeline
            try:
                result = pipeline.query(query["question"])
                results.append(result)
                ground_truths.append(query.get("ground_truth_answer"))

                # Track cost
                if self.cost_tracker:
                    self.cost_tracker.estimate_cost(
                        query["question"],
                        model=self.current_evaluation.get("config", {}).get("generation", {}).get("model", "gpt-3.5-turbo"),
                        operation="generation",
                        output_text=result.get("answer", "")
                    )

                # Cache result
                if self.cache:
                    self.cache.set(cache_key, result)

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                results.append({"error": str(e)})
                ground_truths.append(query.get("ground_truth_answer"))

        # Evaluate with RAGAS
        evaluation_results = self.ragas_evaluator.evaluate(results, ground_truths)

        # Add statistical analysis
        if self.statistical_analyzer:
            evaluation_results["variance_analysis"] = self.statistical_analyzer.calculate_variance_components(
                [{"accuracy": r.get("accuracy", 0)} for r in results if "accuracy" in r]
            )

        return evaluation_results

    def _run_statistical_comparison(self,
                                     current_results: Dict,
                                     previous_results: Dict) -> Dict:
        """Run statistical comparison between evaluations"""
        # Extract comparable metrics
        current_metrics = self._extract_comparable_metrics(current_results)
        previous_metrics = self._extract_comparable_metrics(previous_results)

        # Run comparison
        return self.statistical_analyzer.compare_configurations(
            current_metrics,
            previous_metrics,
            "Current",
            "Previous"
        )

    def _evaluate_single_config(self,
                                 config: Dict,
                                 pipeline_factory: Callable,
                                 test_queries: List[Dict],
                                 name: str) -> Dict:
        """Evaluate a single configuration"""
        try:
            # Create pipeline
            pipeline = pipeline_factory(config)

            # Run evaluation
            return self.evaluate_pipeline(
                pipeline,
                test_queries,
                config=config,
                evaluation_name=name
            )
        except Exception as e:
            logger.error(f"Failed to evaluate config {name}: {e}")
            return {"error": str(e), "config": config}

    def _compare_all_results(self, results: List[Dict]) -> Dict:
        """Compare all evaluation results"""
        if not self.statistical_analyzer or len(results) < 2:
            return {}

        comparisons = {}

        # Pairwise comparisons
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if "error" not in results[i] and "error" not in results[j]:
                    comparison_key = f"config_{i}_vs_config_{j}"
                    comparisons[comparison_key] = self.statistical_analyzer.compare_configurations(
                        self._extract_comparable_metrics(results[i]),
                        self._extract_comparable_metrics(results[j]),
                        f"Config {i}",
                        f"Config {j}"
                    )

        # ANOVA if more than 2 configs
        if len(results) > 2:
            groups = {}
            for i, result in enumerate(results):
                if "error" not in result:
                    metrics = self._extract_comparable_metrics(result)
                    if metrics:
                        groups[f"Config {i}"] = list(metrics[0].values()) if isinstance(metrics, list) else []

            if len(groups) > 1:
                comparisons["anova"] = self.statistical_analyzer.run_anova(groups)

        return comparisons

    def _find_best_config(self, results: List[Dict]) -> Dict:
        """Find the best configuration based on metrics"""
        best_config = None
        best_score = -float('inf')
        best_idx = -1

        for i, result in enumerate(results):
            if "error" in result:
                continue

            # Calculate composite score (customize as needed)
            score = 0
            metrics = self._extract_comparable_metrics(result)

            if isinstance(metrics, list) and metrics:
                # Average accuracy-like metrics
                accuracy_metrics = [m.get("accuracy", 0) for m in metrics if "accuracy" in m]
                if accuracy_metrics:
                    score = sum(accuracy_metrics) / len(accuracy_metrics)

            if score > best_score:
                best_score = score
                best_config = result.get("config", {})
                best_idx = i

        return {
            "config": best_config,
            "index": best_idx,
            "score": best_score
        }

    def _extract_comparable_metrics(self, results: Dict) -> List[Dict]:
        """Extract metrics suitable for comparison"""
        metrics = []

        # Try different result structures
        if "standard" in results and "ragas_metrics" in results["standard"]:
            metrics.append(results["standard"]["ragas_metrics"])
        elif "progressive" in results and "final_metrics" in results["progressive"]:
            metrics.append(results["progressive"]["final_metrics"])
        elif "metrics" in results:
            metrics.append(results["metrics"])

        return metrics

    async def evaluate_pipeline_async(self,
                                       pipeline: Any,
                                       test_queries: List[Dict],
                                       **kwargs) -> Dict:
        """Async version of evaluate_pipeline"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.evaluate_pipeline,
            pipeline,
            test_queries,
            **kwargs
        )

    def reset(self):
        """Reset evaluation service state"""
        self.current_evaluation = None
        self.evaluation_history = []
        if self.cost_tracker:
            self.cost_tracker.reset()
        if self.cache:
            self.cache.clear()

    def save_state(self, filepath: str):
        """Save evaluation service state"""
        state = {
            "evaluation_history": self.evaluation_history,
            "cost_summary": self.cost_tracker.get_summary() if self.cost_tracker else None,
            "timestamp": datetime.now().isoformat()
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, filepath: str):
        """Load evaluation service state"""
        if not Path(filepath).exists():
            logger.warning(f"State file not found: {filepath}")
            return

        with open(filepath, "r") as f:
            state = json.load(f)

        self.evaluation_history = state.get("evaluation_history", [])
        if self.cost_tracker and "cost_summary" in state:
            # Restore cost tracker state
            summary = state["cost_summary"]
            self.cost_tracker.total_cost = summary.get("total_cost", 0)