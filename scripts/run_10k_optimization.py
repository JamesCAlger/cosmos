"""
Run Week 5 Optimization with 10,000 MS MARCO Documents

Configured for 10 configurations with statistical comparison.
"""

import os
import sys
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np
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
from autorag.evaluation.traditional_metrics import TraditionalMetrics
from autorag.datasets import MSMARCOLoader
from autorag.evaluation.cost_tracker import CostTracker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LargeDocumentEvaluator:
    """Evaluator configured for 10,000 documents"""

    def __init__(self, num_docs: int = 10000, num_queries: int = 30, use_ragas: bool = True):
        """Initialize with large document set"""
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.use_ragas = use_ragas

        logger.info(f"Loading {num_docs:,} documents and {num_queries} queries...")

        # Initialize MS MARCO loader and load data
        self.loader = MSMARCOLoader()
        self.test_docs, self.test_queries = self.loader.load_subset(
            num_docs=num_docs,
            num_queries=num_queries * 2,  # Load extra for variety
            include_answers=True  # Include ground truth for comparison
        )

        # Trim to exact number of queries
        self.test_queries = self.test_queries[:num_queries]

        # Initialize evaluators
        if use_ragas:
            self.ragas_evaluator = RAGASEvaluator(use_semantic_metrics=False)

        # Always initialize traditional metrics for ground truth comparison
        self.traditional_metrics = TraditionalMetrics()

        self.cost_tracker = CostTracker()

        logger.info(f"Evaluator ready with {len(self.test_docs):,} documents and {len(self.test_queries)} queries")

    def __call__(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a configuration"""
        try:
            config_id = config["metadata"]["config_id"]
            params = config["metadata"]["parameters"]

            logger.info(f"\nEvaluating config {config_id}:")
            logger.info(f"  Chunking: {params.get('chunking', {})}")
            logger.info(f"  Retrieval: {params.get('retrieval', {})}")
            logger.info(f"  Reranking: {params.get('reranking', {})}")

            start_time = time.time()

            # Initialize pipeline
            pipeline = PipelineOrchestrator(config)

            # Index documents
            logger.info(f"  Indexing {len(self.test_docs):,} documents...")
            index_start = time.time()

            # Index in batches
            batch_size = 1000
            for i in range(0, len(self.test_docs), batch_size):
                batch = self.test_docs[i:i+batch_size]
                pipeline.index_documents(batch)
                if i > 0 and i % 5000 == 0:
                    logger.debug(f"    Indexed {i:,} documents...")

            indexing_time = time.time() - index_start
            logger.info(f"  Indexing completed in {indexing_time:.1f}s")

            # Estimate indexing cost
            embedding_cost = (self.num_docs * 100 / 1000) * 0.0001  # ~$0.10

            # Run queries
            results = []
            predictions = []
            ground_truths = []
            generation_tokens = 0

            logger.info(f"  Running {len(self.test_queries)} queries...")
            for idx, query in enumerate(self.test_queries):
                try:
                    query_text = query.get("question", query) if isinstance(query, dict) else str(query)
                    result = pipeline.query(query_text)

                    results.append({
                        "question": query_text,
                        "answer": result.get("answer", ""),
                        "contexts": result.get("contexts", [])
                    })

                    # Collect for traditional metrics
                    predictions.append(result.get("answer", ""))
                    if isinstance(query, dict) and "ground_truth_answer" in query:
                        ground_truths.append(query["ground_truth_answer"])

                    # Estimate tokens
                    generation_tokens += len(result.get("answer", "").split()) * 1.3

                except Exception as e:
                    logger.warning(f"Query {idx} failed: {e}")

            # Calculate costs
            generation_cost = (generation_tokens / 1000) * 0.002
            total_cost = embedding_cost + generation_cost

            # Calculate metrics
            metrics = {
                "queries_successful": len(results),
                "queries_failed": self.num_queries - len(results),
                "indexing_time": indexing_time,
                "total_time": time.time() - start_time
            }

            # RAGAS evaluation if enabled
            if self.use_ragas and results:
                try:
                    logger.info("  Running RAGAS evaluation...")
                    ragas_result = self.ragas_evaluator.evaluate(results)
                    metrics["ragas_metrics"] = ragas_result.get("ragas_metrics", {})
                except Exception as e:
                    logger.warning(f"RAGAS evaluation failed: {e}")

            # Traditional metrics if ground truth available
            if predictions and ground_truths and len(predictions) == len(ground_truths):
                try:
                    logger.info("  Running traditional metrics evaluation...")
                    trad_metrics = self.traditional_metrics.evaluate(predictions, ground_truths)
                    metrics["traditional_metrics"] = trad_metrics
                except Exception as e:
                    logger.warning(f"Traditional metrics failed: {e}")

            # Calculate composite score
            score = self._calculate_score(metrics)

            logger.info(f"  Score: {score:.4f}, Cost: ${total_cost:.2f}")

            return {
                "metrics": metrics,
                "cost": total_cost,
                "score": score,
                "success": True
            }

        except Exception as e:
            logger.error(f"Configuration evaluation failed: {e}")
            return {
                "metrics": {"error": str(e)},
                "cost": 0,
                "score": 0,
                "success": False
            }

    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite score from all available metrics"""
        score = 0.0
        weight_used = 0.0

        # RAGAS metrics (50% weight if available)
        if "ragas_metrics" in metrics:
            ragas = metrics["ragas_metrics"]
            ragas_score = (
                ragas.get("faithfulness", 0) * 0.3 +
                ragas.get("answer_relevancy", 0) * 0.4 +
                ragas.get("context_relevance", 0) * 0.3
            )
            score += ragas_score * 0.5
            weight_used += 0.5

        # Traditional metrics (50% weight if available)
        if "traditional_metrics" in metrics:
            trad = metrics["traditional_metrics"]
            trad_score = (
                trad.get("exact_match_accuracy", 0) * 0.3 +
                trad.get("token_f1", 0) * 0.7
            )
            score += trad_score * 0.5
            weight_used += 0.5

        # Normalize if not all metrics available
        if weight_used > 0:
            score = score / weight_used
        else:
            # Fallback to success rate
            total = metrics.get("queries_successful", 0) + metrics.get("queries_failed", 0)
            if total > 0:
                score = metrics.get("queries_successful", 0) / total

        return score


def create_test_search_space() -> SearchSpace:
    """Create search space for 10 configurations"""
    space = SearchSpace()

    # Balanced search space for meaningful comparison
    space.define_chunking_space(
        strategies=["fixed", "semantic"],
        sizes=[256, 512]
    )

    space.define_retrieval_space(
        methods=["dense", "hybrid"],
        top_k_values=[3, 5]
    )

    space.define_reranking_space(
        enabled_values=[False, True],
        models=["cross-encoder/ms-marco-MiniLM-L-6-v2"]
    )

    space.define_generation_space(
        temperatures=[0, 0.3],
        max_tokens=[200]  # Fixed to reduce variability
    )

    space.define_embedding_space(
        models=["text-embedding-ada-002"]
    )

    return space


def main():
    """Main function"""

    print("\n" + "="*60)
    print("10,000 DOCUMENT OPTIMIZATION TEST")
    print("="*60)

    # Configuration
    NUM_DOCS = 10000
    NUM_QUERIES = 30
    MAX_CONFIGS = 10
    BUDGET = 20.0
    USE_RAGAS = True

    # Cost estimate
    est_embedding = (NUM_DOCS * 100 / 1000) * 0.0001
    est_generation = (NUM_QUERIES * 500 / 1000) * 0.002
    est_per_config = est_embedding + est_generation
    est_total = est_per_config * MAX_CONFIGS

    print(f"\nConfiguration:")
    print(f"  Documents: {NUM_DOCS:,}")
    print(f"  Queries per config: {NUM_QUERIES}")
    print(f"  Configurations to test: {MAX_CONFIGS}")
    print(f"  Budget limit: ${BUDGET:.2f}")
    print(f"  RAGAS evaluation: {USE_RAGAS}")

    print(f"\nCost Estimate:")
    print(f"  Per configuration: ${est_per_config:.2f}")
    print(f"  Total estimated: ${est_total:.2f}")

    # Confirm
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return 0

    try:
        # Create search space
        print("\nCreating search space...")
        search_space = create_test_search_space()
        total_configs = search_space.calculate_total_combinations()
        print(f"  Total possible configurations: {total_configs}")

        # Initialize evaluator
        print(f"\nInitializing evaluator with {NUM_DOCS:,} documents...")
        evaluator = LargeDocumentEvaluator(
            num_docs=NUM_DOCS,
            num_queries=NUM_QUERIES,
            use_ragas=USE_RAGAS
        )

        # Create optimizer
        print("\nSetting up grid search optimizer...")
        checkpoint_dir = "checkpoints/10k_test"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        optimizer = GridSearchOptimizer(
            search_space=search_space,
            evaluator=evaluator,
            budget_limit=BUDGET,
            parallel_workers=1,
            early_stopping_threshold=0.15,
            checkpoint_dir=checkpoint_dir
        )

        # Run optimization
        print("\n" + "-"*40)
        print("STARTING OPTIMIZATION")
        print("-"*40)

        start_time = time.time()
        report = optimizer.search(max_configurations=MAX_CONFIGS)
        total_time = time.time() - start_time

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)

        results_file = results_dir / f"10k_test_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        summary = report["summary"]
        print(f"\nSummary:")
        print(f"  Configurations evaluated: {summary['configurations_evaluated']}")
        print(f"  Total cost: ${summary['total_cost']:.2f}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Best score: {summary['best_score']:.4f}")

        # Best configuration
        best = report["best_configuration"]
        print(f"\nBest Configuration ({best['config_id']}):")
        print(f"  Score: {best['score']:.4f}")
        for component, params in best["parameters"].items():
            print(f"  {component}: {params}")

        # Statistical comparison
        if len(optimizer.result_manager.results) >= 2:
            print("\n" + "-"*40)
            print("STATISTICAL ANALYSIS")
            print("-"*40)

            # Compare top 2 configurations
            top_2 = optimizer.result_manager.get_top_configurations(2)
            if len(top_2) == 2:
                comparison = optimizer.result_manager.compare_configurations(
                    top_2[0]["config_id"],
                    top_2[1]["config_id"]
                )
                print(f"\nTop 2 Comparison:")
                print(f"  Score difference: {comparison['score_difference']:.4f}")
                print(f"  Relative improvement: {comparison['relative_improvement']:.1f}%")
                print(f"  Cost difference: ${comparison['cost_difference']:.2f}")

            # Parameter impact
            impact = optimizer.result_manager.analyze_parameter_impact()
            if impact:
                print(f"\nParameter Impact:")
                for param, data in list(impact.items())[:5]:
                    print(f"  {param}: {data['variance_explained']:.3f}")

        # Save report
        report_file = results_dir / f"10k_report_{timestamp}.md"

        # Fix the result manager's directory path issue
        optimizer.result_manager.results_dir = results_dir
        optimizer.result_manager.generate_report(str(report_file))

        print(f"\nResults saved to:")
        print(f"  JSON: {results_file}")
        print(f"  Report: {report_file}")

        print("\nOptimization complete!")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved to checkpoint.")
        return 1
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())