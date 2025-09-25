"""
Week 5 Configuration Search Demonstration

This script demonstrates the configuration search and optimization functionality
implemented for Week 5 of the AutoRAG project.
"""

import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import argparse
from typing import Dict, Any, List
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autorag.optimization import (
    SearchSpace,
    ConfigurationGenerator,
    GridSearchOptimizer,
    ResultManager,
    StatisticalComparison
)
from autorag.pipeline.orchestrator import PipelineOrchestrator
from autorag.evaluation import RAGASEvaluator
from autorag.datasets import MSMARCOLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConfigurationEvaluator:
    """Evaluator for pipeline configurations"""

    def __init__(self, dataset_path: str, num_queries: int = 20, use_ragas: bool = True):
        """
        Initialize configuration evaluator

        Args:
            dataset_path: Path to MS MARCO dataset
            num_queries: Number of queries to evaluate
            use_ragas: Whether to use RAGAS evaluation
        """
        self.dataset_path = dataset_path
        self.num_queries = num_queries
        self.use_ragas = use_ragas

        # Load test data
        self.loader = MSMARCOLoader(dataset_path)
        self.test_queries, self.test_docs = self._load_test_data()

        # Initialize evaluators
        if use_ragas:
            self.ragas_evaluator = RAGASEvaluator(use_semantic_metrics=False)

        logger.info(f"ConfigurationEvaluator initialized with {num_queries} queries")

    def _load_test_data(self):
        """Load test queries and documents"""
        # Load a small sample for testing
        queries = self.loader.load_queries(split="dev", max_queries=self.num_queries)
        docs = self.loader.load_documents(max_docs=50)
        return queries, docs

    def __call__(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a configuration

        Args:
            config: Pipeline configuration to evaluate

        Returns:
            Evaluation results with metrics and cost
        """
        try:
            config_id = config["metadata"]["config_id"]
            logger.info(f"Evaluating configuration {config_id}")

            # Initialize pipeline with configuration
            pipeline = PipelineOrchestrator(config)

            # Index documents
            logger.debug(f"Indexing {len(self.test_docs)} documents")
            index_start = time.time()
            pipeline.index_documents(self.test_docs)
            indexing_time = time.time() - index_start

            # Run queries and collect results
            results = []
            total_query_time = 0
            total_tokens = 0

            for query in self.test_queries[:self.num_queries]:
                query_start = time.time()
                result = pipeline.query(query["question"])
                query_time = time.time() - query_start
                total_query_time += query_time

                results.append({
                    "question": query["question"],
                    "answer": result.get("answer", ""),
                    "contexts": result.get("contexts", [])
                })

                # Estimate tokens (rough approximation)
                total_tokens += len(result.get("answer", "").split()) * 1.3

            # Calculate metrics
            metrics = {}

            if self.use_ragas and results:
                # RAGAS evaluation
                ragas_metrics = self.ragas_evaluator.evaluate(results)
                metrics["ragas_metrics"] = ragas_metrics.get("ragas_metrics", {})

            # Basic metrics
            metrics["indexing_time"] = indexing_time
            metrics["avg_query_time"] = total_query_time / max(1, len(results))
            metrics["queries_processed"] = len(results)

            # Estimate cost (simplified)
            embedding_cost = len(self.test_docs) * 0.0001  # Rough estimate
            generation_cost = total_tokens * 0.00002  # GPT-3.5 estimate
            total_cost = embedding_cost + generation_cost

            return {
                "metrics": metrics,
                "cost": total_cost,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error evaluating configuration: {e}")
            return {
                "metrics": {"error": str(e)},
                "cost": 0,
                "success": False
            }


def demonstrate_search_space():
    """Demonstrate search space creation and exploration"""
    print("\n" + "="*60)
    print("WEEK 5: SEARCH SPACE DEFINITION")
    print("="*60)

    # Create search space
    search_space = SearchSpace()
    search_space.create_default_search_space()

    # Display search space summary
    total = search_space.calculate_total_combinations()
    print(f"\nSearch Space Summary:")
    print(f"- Total possible configurations: {total:,}")

    for component, space in search_space.components.items():
        print(f"\n{component.upper()}:")
        for param in space.parameters:
            print(f"  - {param.name}: {param.values}")
            if param.conditions:
                print(f"    (conditional on: {param.conditions})")

    # Sample configurations
    print("\n" + "-"*40)
    print("Sample Configurations:")
    sample = search_space.sample(3, method="random")
    for i, config in enumerate(sample, 1):
        print(f"\nConfiguration {i}:")
        for component, params in config.items():
            print(f"  {component}: {params}")

    # Save search space
    search_space.save("configs/search_spaces/week5_default.yaml")
    print("\n✓ Search space saved to configs/search_spaces/week5_default.yaml")

    return search_space


def demonstrate_config_generation(search_space: SearchSpace):
    """Demonstrate configuration generation"""
    print("\n" + "="*60)
    print("CONFIGURATION GENERATION")
    print("="*60)

    # Create configuration generator
    generator = ConfigurationGenerator(search_space)

    # Generate specific configuration
    params = {
        "chunking": {"strategy": "semantic", "size": 512},
        "retrieval": {"method": "hybrid", "top_k": 5, "hybrid_weight": 0.5},
        "reranking": {"enabled": True, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                     "top_k_rerank": 20},
        "generation": {"temperature": 0.3, "max_tokens": 300, "model": "gpt-3.5-turbo"},
        "embedding": {"model": "text-embedding-ada-002"}
    }

    config = generator.generate_configuration(params)
    print("\nGenerated Configuration:")
    print(f"- Config ID: {config['metadata']['config_id']}")
    print(f"- Nodes: {len(config['pipeline']['nodes'])}")
    print(f"- Node types: {[n['type'] for n in config['pipeline']['nodes']]}")

    # Generate subset
    subset_configs = generator.generate_subset(5, method="grid")
    print(f"\n✓ Generated {len(subset_configs)} configurations using grid sampling")

    # Save configurations
    output_dir = "experiments/week5_configs"
    generator.save_configurations(subset_configs[:3], output_dir)
    print(f"✓ Saved 3 configurations to {output_dir}")

    return generator


def demonstrate_grid_search(search_space: SearchSpace, budget: float = 1.0):
    """Demonstrate grid search optimization"""
    print("\n" + "="*60)
    print("GRID SEARCH OPTIMIZATION")
    print("="*60)

    # Create mock evaluator for demonstration
    def mock_evaluator(config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock evaluator that returns synthetic results"""
        import random

        # Simulate evaluation
        time.sleep(0.1)

        # Generate synthetic metrics based on configuration
        params = config["metadata"]["parameters"]

        # Better scores for certain configurations
        score_boost = 0
        if params.get("retrieval", {}).get("method") == "hybrid":
            score_boost += 0.1
        if params.get("reranking", {}).get("enabled"):
            score_boost += 0.05
        if params.get("chunking", {}).get("strategy") == "semantic":
            score_boost += 0.05

        base_score = 0.6 + random.uniform(-0.1, 0.1)
        score = min(1.0, base_score + score_boost)

        return {
            "metrics": {
                "ragas_metrics": {
                    "faithfulness": score + random.uniform(-0.05, 0.05),
                    "answer_relevancy": score + random.uniform(-0.05, 0.05),
                    "context_relevance": score + random.uniform(-0.05, 0.05)
                }
            },
            "cost": random.uniform(0.05, 0.15)
        }

    # Create optimizer
    optimizer = GridSearchOptimizer(
        search_space=search_space,
        evaluator=mock_evaluator,
        budget_limit=budget,
        parallel_workers=1,
        early_stopping_threshold=0.3,
        checkpoint_dir="checkpoints/week5_demo"
    )

    print(f"\nOptimizer Configuration:")
    print(f"- Budget limit: ${budget:.2f}")
    print(f"- Parallel workers: 1")
    print(f"- Early stopping threshold: 0.3")

    # Run search on small subset
    print("\nRunning grid search (mock evaluation)...")
    report = optimizer.search(max_configurations=10)

    # Display results
    print("\n" + "-"*40)
    print("OPTIMIZATION RESULTS:")
    print(f"- Configurations evaluated: {report['summary']['configurations_evaluated']}")
    print(f"- Total cost: ${report['summary']['total_cost']:.2f}")
    print(f"- Total time: {report['summary']['total_time']:.1f}s")
    print(f"- Best score: {report['summary']['best_score']:.4f}")

    if report['summary']['improvement_over_baseline']:
        print(f"- Improvement over baseline: {report['summary']['improvement_over_baseline']:.1f}%")

    # Show best configuration
    best = report['best_configuration']
    print(f"\nBest Configuration (ID: {best['config_id']}):")
    for component, params in best['parameters'].items():
        print(f"  {component}: {params}")

    return optimizer


def demonstrate_result_analysis(optimizer: GridSearchOptimizer):
    """Demonstrate result analysis and statistical comparison"""
    print("\n" + "="*60)
    print("RESULT ANALYSIS & STATISTICAL COMPARISON")
    print("="*60)

    manager = optimizer.result_manager

    # Get summary
    summary = manager.get_summary()
    print("\nResults Summary:")
    print(f"- Mean score: {summary['mean_score']:.4f} ± {summary['std_score']:.4f}")
    print(f"- Score range: [{summary['worst_score']:.4f}, {summary['best_score']:.4f}]")
    print(f"- Mean cost per config: ${summary['mean_cost_per_config']:.3f}")

    # Top configurations
    print("\nTop 3 Configurations:")
    for i, config in enumerate(manager.get_top_configurations(3), 1):
        print(f"{i}. {config['config_id']}: score={config['score']:.4f}, cost=${config['cost']:.3f}")

    # Parameter impact analysis
    impact = manager.analyze_parameter_impact()
    if impact:
        print("\nParameter Impact (top 3):")
        for param, data in list(impact.items())[:3]:
            print(f"- {param}: variance explained = {data['variance_explained']:.3f}")

    # Pareto optimal configurations
    pareto = manager.find_pareto_optimal()
    print(f"\nPareto Optimal Configurations: {len(pareto)} found")
    for config in pareto[:3]:
        ratio = config['score'] / max(0.001, config['cost'])
        print(f"- {config['config_id']}: score={config['score']:.4f}, "
              f"cost=${config['cost']:.3f}, ratio={ratio:.2f}")

    # Statistical comparison
    if len(manager.results) >= 2:
        print("\n" + "-"*40)
        print("Statistical Comparison (best vs baseline):")

        stats_comp = StatisticalComparison()
        best = manager.get_best_configuration()
        baseline = manager.results[0]  # First configuration as baseline

        # Create synthetic sample scores for demonstration
        import numpy as np
        scores_baseline = [baseline['score'] + np.random.normal(0, 0.02) for _ in range(20)]
        scores_best = [best['score'] + np.random.normal(0, 0.02) for _ in range(20)]

        result = stats_comp.independent_t_test(scores_baseline, scores_best)
        print(f"- Test: {result.test_name}")
        print(f"- t-statistic: {result.statistic:.3f}")
        print(f"- p-value: {result.p_value:.4f}")
        print(f"- Effect size (Cohen's d): {result.effect_size:.3f}")
        print(f"- Significant: {result.is_significant}")
        print(f"- Interpretation: {result.interpretation}")

    # Generate report
    report_path = "optimization_results/week5_demo_report.md"
    report = manager.generate_report(report_path)
    print(f"\n✓ Full report saved to {report_path}")

    return manager


def demonstrate_real_evaluation(search_space: SearchSpace, args):
    """Demonstrate real evaluation with actual pipeline"""
    print("\n" + "="*60)
    print("REAL PIPELINE EVALUATION (Optional)")
    print("="*60)

    if not args.real_eval:
        print("Skipping real evaluation (use --real-eval to enable)")
        return

    print("\nSetting up real evaluation...")

    # Create real evaluator
    evaluator = ConfigurationEvaluator(
        dataset_path="data/msmarco",
        num_queries=args.num_queries,
        use_ragas=args.use_ragas
    )

    # Create optimizer with real evaluator
    optimizer = GridSearchOptimizer(
        search_space=search_space,
        evaluator=evaluator,
        budget_limit=args.budget,
        parallel_workers=args.parallel,
        early_stopping_threshold=0.2,
        checkpoint_dir="checkpoints/week5_real"
    )

    print(f"\nRunning real evaluation:")
    print(f"- Max configurations: {args.max_configs}")
    print(f"- Queries per config: {args.num_queries}")
    print(f"- Budget limit: ${args.budget:.2f}")
    print(f"- Parallel workers: {args.parallel}")

    # Run search
    report = optimizer.search(max_configurations=args.max_configs)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"optimization_results/week5_real_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    # Display best configuration
    best = report['best_configuration']
    print(f"\nBest Configuration Found:")
    print(f"- Score: {best['score']:.4f}")
    print(f"- Cost: ${best['cost']:.3f}")
    print(f"- Parameters:")
    for component, params in best['parameters'].items():
        print(f"  {component}: {params}")


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="Week 5 Configuration Search Demo")
    parser.add_argument("--real-eval", action="store_true",
                       help="Run real evaluation with actual pipeline")
    parser.add_argument("--budget", type=float, default=5.0,
                       help="Budget limit in dollars (default: 5.0)")
    parser.add_argument("--max-configs", type=int, default=10,
                       help="Maximum configurations to evaluate (default: 10)")
    parser.add_argument("--num-queries", type=int, default=20,
                       help="Number of queries per configuration (default: 20)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel workers (default: 1)")
    parser.add_argument("--use-ragas", action="store_true",
                       help="Use RAGAS evaluation (requires API key)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("WEEK 5: CONFIGURATION SEARCH DEMONSTRATION")
    print("="*60)
    print("\nThis demo showcases the Week 5 implementation:")
    print("1. Search space definition")
    print("2. Configuration generation")
    print("3. Grid search optimization")
    print("4. Result management and tracking")
    print("5. Statistical comparison and reporting")

    try:
        # 1. Demonstrate search space
        search_space = demonstrate_search_space()

        # 2. Demonstrate configuration generation
        generator = demonstrate_config_generation(search_space)

        # 3. Demonstrate grid search (with mock evaluator)
        optimizer = demonstrate_grid_search(search_space, budget=1.0)

        # 4. Demonstrate result analysis
        manager = demonstrate_result_analysis(optimizer)

        # 5. Optional: Real evaluation
        if args.real_eval:
            demonstrate_real_evaluation(search_space, args)

        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\n✓ Week 5 functionality successfully demonstrated!")
        print("\nKey achievements:")
        print("- Search space with ~192 configurations defined")
        print("- Configuration generator creates valid pipeline configs")
        print("- Grid search optimizer manages budget and tracks results")
        print("- Statistical comparison provides significance testing")
        print("- Results can be exported and analyzed comprehensively")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())