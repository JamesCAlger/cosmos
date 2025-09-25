"""
Week 5 Real Optimization Test with 10,000 MS MARCO Documents

This script runs a real optimization test with:
- 10,000 MS MARCO documents
- Configurable number of test queries
- Multiple pipeline configurations
- Statistical comparison of results
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
import warnings
warnings.filterwarnings('ignore')

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
from autorag.datasets.enhanced_loader import EnhancedMSMARCOLoader
from autorag.evaluation.cost_tracker import CostTracker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LargeScaleEvaluator:
    """Evaluator for large-scale pipeline optimization"""

    def __init__(self, num_docs: int = 10000, num_queries: int = 50, use_ragas: bool = True):
        """
        Initialize large-scale evaluator

        Args:
            num_docs: Number of documents to index
            num_queries: Number of queries to evaluate per configuration
            use_ragas: Whether to use RAGAS evaluation
        """
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.use_ragas = use_ragas

        logger.info(f"Initializing evaluator with {num_docs:,} docs and {num_queries} queries")

        # Load data
        self._load_data()

        # Initialize evaluators
        if use_ragas:
            self.ragas_evaluator = RAGASEvaluator(use_semantic_metrics=False)

        # Cost tracker
        self.cost_tracker = CostTracker()

    def _load_data(self):
        """Load MS MARCO data"""
        try:
            # Try enhanced loader first for better data management
            loader = EnhancedMSMARCOLoader("data/msmarco")
            self.documents = loader.load_documents(max_docs=self.num_docs)
            self.queries = loader.load_queries(split="dev", max_queries=self.num_queries * 2)

            # Sample queries for evaluation
            if len(self.queries) > self.num_queries:
                import random
                random.seed(42)  # For reproducibility
                self.queries = random.sample(self.queries, self.num_queries)

        except Exception as e:
            logger.warning(f"Enhanced loader failed: {e}. Using basic loader.")

            # Fallback to basic loader
            loader = MSMARCOLoader()
            docs, queries = loader.load_subset(
                num_docs=self.num_docs,
                num_queries=self.num_queries,
                include_answers=False
            )
            self.documents = docs
            self.queries = queries

        logger.info(f"Loaded {len(self.documents):,} documents and {len(self.queries)} queries")

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
            logger.info(f"Parameters: {config['metadata']['parameters']}")

            # Track start time
            start_time = time.time()

            # Initialize pipeline
            pipeline = PipelineOrchestrator(config)

            # Index documents (one-time cost per configuration)
            logger.info(f"Indexing {len(self.documents):,} documents...")
            index_start = time.time()

            # Index in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i+batch_size]
                pipeline.index_documents(batch)
                if i % 5000 == 0:
                    logger.debug(f"Indexed {i:,} documents...")

            indexing_time = time.time() - index_start
            logger.info(f"Indexing completed in {indexing_time:.1f}s")

            # Estimate indexing cost
            # Embedding cost: ~$0.0001 per 1K tokens, assume 100 tokens per doc
            embedding_cost = (self.num_docs * 100 / 1000) * 0.0001
            self.cost_tracker.add_cost("embedding", embedding_cost)

            # Run queries
            results = []
            total_generation_tokens = 0
            failed_queries = 0

            logger.info(f"Running {len(self.queries)} queries...")
            for idx, query in enumerate(self.queries):
                try:
                    query_text = query.get("question", query) if isinstance(query, dict) else query

                    result = pipeline.query(query_text)

                    results.append({
                        "question": query_text,
                        "answer": result.get("answer", ""),
                        "contexts": result.get("contexts", [])
                    })

                    # Estimate tokens (rough)
                    answer_tokens = len(result.get("answer", "").split()) * 1.3
                    context_tokens = sum(len(c.split()) * 1.3 for c in result.get("contexts", []))
                    total_generation_tokens += (answer_tokens + context_tokens * 0.1)  # Context contributes less

                    if (idx + 1) % 10 == 0:
                        logger.debug(f"Processed {idx+1}/{len(self.queries)} queries")

                except Exception as e:
                    logger.warning(f"Query {idx} failed: {e}")
                    failed_queries += 1

            # Calculate metrics
            metrics = {
                "queries_processed": len(results),
                "queries_failed": failed_queries,
                "indexing_time": indexing_time,
                "avg_query_time": (time.time() - start_time - indexing_time) / max(1, len(results))
            }

            # RAGAS evaluation if enabled and results exist
            if self.use_ragas and results:
                try:
                    logger.info("Running RAGAS evaluation...")
                    ragas_metrics = self.ragas_evaluator.evaluate(results)
                    metrics["ragas_metrics"] = ragas_metrics.get("ragas_metrics", {})
                except Exception as e:
                    logger.warning(f"RAGAS evaluation failed: {e}")

            # Calculate total cost
            generation_cost = (total_generation_tokens / 1000) * 0.002  # GPT-3.5 pricing
            self.cost_tracker.add_cost("generation", generation_cost)
            total_cost = embedding_cost + generation_cost

            # Calculate composite score
            score = self._calculate_score(metrics)

            logger.info(f"Configuration {config_id} - Score: {score:.4f}, Cost: ${total_cost:.2f}")

            return {
                "metrics": metrics,
                "cost": total_cost,
                "score": score,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error evaluating configuration: {e}")
            import traceback
            traceback.print_exc()
            return {
                "metrics": {"error": str(e)},
                "cost": 0,
                "score": 0,
                "success": False
            }

    def _calculate_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite score from metrics"""
        score = 0.0
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.4,
            "context_relevance": 0.3
        }

        # RAGAS metrics contribution
        if "ragas_metrics" in metrics:
            ragas = metrics["ragas_metrics"]
            for metric, weight in weights.items():
                if metric in ragas:
                    score += ragas[metric] * weight

        # Fallback scoring if no RAGAS metrics
        if score == 0:
            # Use success rate as proxy
            total_queries = metrics.get("queries_processed", 0) + metrics.get("queries_failed", 0)
            if total_queries > 0:
                score = metrics.get("queries_processed", 0) / total_queries

        return score


def create_focused_search_space() -> SearchSpace:
    """Create a focused search space for optimization"""
    space = SearchSpace()

    # Chunking strategies (2 options)
    space.define_chunking_space(
        strategies=["fixed", "semantic"],
        sizes=[256, 512]
    )

    # Retrieval methods (3 options)
    space.define_retrieval_space(
        methods=["dense", "hybrid"],  # Skip sparse for now
        top_k_values=[3, 5]
    )

    # Reranking (2 options)
    space.define_reranking_space(
        enabled_values=[False, True],
        models=["cross-encoder/ms-marco-MiniLM-L-6-v2"]
    )

    # Generation parameters (2 options)
    space.define_generation_space(
        temperatures=[0, 0.3],
        max_tokens=[200]  # Fixed to reduce variability
    )

    # Embedding model (1 option - fixed)
    space.define_embedding_space(
        models=["text-embedding-ada-002"]
    )

    total = space.calculate_total_combinations()
    logger.info(f"Search space created with {total} total configurations")

    return space


def estimate_costs(num_docs: int, num_queries: int, num_configs: int) -> Dict[str, float]:
    """Estimate costs for the optimization run"""

    # Embedding costs (per configuration)
    tokens_per_doc = 100  # Average estimate
    embedding_cost_per_config = (num_docs * tokens_per_doc / 1000) * 0.0001

    # Generation costs (per query per configuration)
    tokens_per_query = 500  # Input + output estimate
    generation_cost_per_query = (tokens_per_query / 1000) * 0.002
    generation_cost_per_config = generation_cost_per_query * num_queries

    # Total costs
    cost_per_config = embedding_cost_per_config + generation_cost_per_config
    total_cost = cost_per_config * num_configs

    return {
        "embedding_per_config": embedding_cost_per_config,
        "generation_per_config": generation_cost_per_config,
        "total_per_config": cost_per_config,
        "total_estimated": total_cost,
        "configs": num_configs,
        "queries_per_config": num_queries
    }


def main():
    """Main optimization function"""
    parser = argparse.ArgumentParser(description="Week 5 Large-Scale Optimization")
    parser.add_argument("--num-docs", type=int, default=10000,
                       help="Number of documents to index (default: 10000)")
    parser.add_argument("--num-queries", type=int, default=50,
                       help="Number of queries per configuration (default: 50)")
    parser.add_argument("--max-configs", type=int, default=10,
                       help="Maximum configurations to evaluate (default: 10)")
    parser.add_argument("--budget", type=float, default=25.0,
                       help="Budget limit in dollars (default: 25.0)")
    parser.add_argument("--use-ragas", action="store_true",
                       help="Use RAGAS evaluation (increases cost)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Estimate costs without running")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("WEEK 5: LARGE-SCALE OPTIMIZATION TEST")
    print("="*60)

    # Estimate costs
    costs = estimate_costs(args.num_docs, args.num_queries, args.max_configs)

    print(f"\nConfiguration:")
    print(f"- Documents to index: {args.num_docs:,}")
    print(f"- Queries per config: {args.num_queries}")
    print(f"- Max configurations: {args.max_configs}")
    print(f"- Budget limit: ${args.budget:.2f}")
    print(f"- Use RAGAS: {args.use_ragas}")

    print(f"\nCost Estimates:")
    print(f"- Embedding cost per config: ${costs['embedding_per_config']:.2f}")
    print(f"- Generation cost per config: ${costs['generation_per_config']:.2f}")
    print(f"- Total per configuration: ${costs['total_per_config']:.2f}")
    print(f"- Estimated total cost: ${costs['total_estimated']:.2f}")

    if costs['total_estimated'] > args.budget:
        print(f"\n⚠️  Warning: Estimated cost exceeds budget!")
        print(f"   Consider reducing queries to {int(args.budget / costs['total_per_config'] * args.num_queries)}")

    if args.dry_run:
        print("\n✓ Dry run complete. Use --dry-run=false to execute.")
        return 0

    # Confirm before proceeding
    response = input("\nProceed with optimization? (yes/no): ")
    if response.lower() != "yes":
        print("Optimization cancelled.")
        return 0

    try:
        # Create search space
        print("\nCreating search space...")
        search_space = create_focused_search_space()

        # Create evaluator
        print(f"Initializing evaluator with {args.num_docs:,} documents...")
        evaluator = LargeScaleEvaluator(
            num_docs=args.num_docs,
            num_queries=args.num_queries,
            use_ragas=args.use_ragas
        )

        # Create optimizer
        checkpoint_dir = "checkpoints/week5_large_scale"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        optimizer = GridSearchOptimizer(
            search_space=search_space,
            evaluator=evaluator,
            budget_limit=args.budget,
            parallel_workers=1,  # Sequential for now
            early_stopping_threshold=0.2,  # Stop if score < 0.2
            checkpoint_dir=checkpoint_dir
        )

        # Resume if requested
        if args.resume:
            try:
                optimizer._load_checkpoint()
                print(f"✓ Resumed from checkpoint")
                print(f"  Configurations evaluated: {optimizer.configurations_evaluated}")
                print(f"  Total cost so far: ${optimizer.total_cost:.2f}")
            except:
                print("No checkpoint found, starting fresh")

        # Run optimization
        print("\n" + "-"*40)
        print("Starting optimization...")
        print("-"*40)

        report = optimizer.search(max_configurations=args.max_configs)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("optimization_results")
        results_dir.mkdir(exist_ok=True)

        # Save full report
        report_file = results_dir / f"week5_10k_docs_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Display results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)

        summary = report['summary']
        print(f"\nSummary:")
        print(f"- Configurations evaluated: {summary['configurations_evaluated']}")
        print(f"- Total cost: ${summary['total_cost']:.2f}")
        print(f"- Total time: {summary['total_time']:.1f}s")
        print(f"- Best score: {summary['best_score']:.4f}")

        if summary.get('improvement_over_baseline'):
            print(f"- Improvement over baseline: {summary['improvement_over_baseline']:.1f}%")

        # Best configuration
        best = report['best_configuration']
        print(f"\nBest Configuration (ID: {best['config_id']}):")
        print(f"- Score: {best['score']:.4f}")
        print(f"- Cost: ${best['cost']:.2f}")
        print(f"- Parameters:")
        for component, params in best['parameters'].items():
            print(f"  {component}: {params}")

        # Statistical analysis
        if len(optimizer.result_manager.results) >= 2:
            print("\n" + "-"*40)
            print("Statistical Analysis:")

            # Parameter impact
            impact = optimizer.result_manager.analyze_parameter_impact()
            if impact:
                print("\nParameter Impact (top 3):")
                for param, data in list(impact.items())[:3]:
                    print(f"- {param}: variance explained = {data['variance_explained']:.3f}")

            # Pareto optimal
            pareto = optimizer.result_manager.find_pareto_optimal()
            print(f"\nPareto Optimal Configurations: {len(pareto)}")
            for config in pareto[:3]:
                print(f"- {config['config_id']}: score={config['score']:.4f}, cost=${config['cost']:.2f}")

        # Generate markdown report
        report_md = results_dir / f"week5_10k_report_{timestamp}.md"
        optimizer.result_manager.generate_report(str(report_md))

        print(f"\n✓ Results saved to:")
        print(f"  - JSON: {report_file}")
        print(f"  - Report: {report_md}")

        print("\n✓ Optimization complete!")

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        print("Progress has been saved to checkpoint")
        return 1

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())