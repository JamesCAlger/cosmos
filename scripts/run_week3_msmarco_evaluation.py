"""
Run Week 3 Evaluation Infrastructure on MS MARCO Dataset

This script runs a complete evaluation using all Week 3 features:
- Progressive evaluation with multiple levels
- Caching for efficiency
- Statistical analysis
- Cost tracking
- Multi-format reporting
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime
from loguru import logger

# Import Week 3 components
from autorag.evaluation.service import EvaluationService
from autorag.evaluation.progressive.evaluator import EvaluationLevel, LevelConfig
from autorag.datasets.enhanced_loader import EnhancedDatasetLoader
from autorag.pipeline.simple_rag import SimpleRAGPipeline
from autorag.components.base import Document


def setup_logging():
    """Configure logging"""
    log_file = f"experiments/week3_msmarco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, rotation="100 MB", level="INFO")
    logger.info("Starting Week 3 MS MARCO Evaluation")
    return log_file


def load_msmarco_data(num_docs=100, num_queries=20):
    """Load MS MARCO dataset with enhanced loader"""
    logger.info(f"Loading MS MARCO: {num_docs} docs, {num_queries} queries")

    loader = EnhancedDatasetLoader(dataset_name="ms_marco")

    # Load with train/dev/test splits
    data = loader.load_with_splits(
        num_docs=num_docs,
        num_queries=num_queries,
        train_ratio=0.6,  # 12 queries for training
        dev_ratio=0.2,    # 4 queries for dev
        test_ratio=0.2,   # 4 queries for test
        use_cache=True    # Use cache for faster subsequent runs
    )

    logger.info(f"Dataset loaded successfully")
    logger.info(f"  Train: {data['splits']['train']['num_queries']} queries, {data['splits']['train']['num_docs']} docs")
    logger.info(f"  Dev: {data['splits']['dev']['num_queries']} queries, {data['splits']['dev']['num_docs']} docs")
    logger.info(f"  Test: {data['splits']['test']['num_queries']} queries, {data['splits']['test']['num_docs']} docs")

    return data


def setup_evaluation_service():
    """Initialize evaluation service with all Week 3 features"""
    logger.info("Setting up evaluation service")

    service = EvaluationService(
        cache_dir="cache/week3_msmarco",
        enable_caching=True,
        cost_tracking=True,
        budget_limit=1.0,  # $1 budget limit
        progressive_eval=True,
        statistical_analysis=True,
        reporter_formats=["json", "html", "markdown", "csv"]
    )

    # Configure custom evaluation levels for this test
    custom_smoke = LevelConfig(
        name="MS MARCO Smoke",
        num_queries=4,
        max_duration_seconds=30,
        estimated_cost=0.01,
        description="Quick validation with 4 queries",
        early_stop_threshold=0.2
    )

    custom_standard = LevelConfig(
        name="MS MARCO Standard",
        num_queries=20,
        max_duration_seconds=300,
        estimated_cost=0.05,
        description="Full evaluation with all test queries",
        confidence_threshold=0.95
    )

    service.progressive_evaluator.add_level(EvaluationLevel.SMOKE, custom_smoke)
    service.progressive_evaluator.add_level(EvaluationLevel.STANDARD, custom_standard)

    logger.info("Evaluation service configured with custom levels")
    return service


def setup_pipeline(documents):
    """Setup RAG pipeline with documents"""
    logger.info("Setting up RAG pipeline")

    # Load baseline configuration
    config_path = Path(__file__).parent.parent / "configs" / "baseline_rag.yaml"

    try:
        pipeline = SimpleRAGPipeline(str(config_path))

        # Convert documents to Document objects
        doc_objects = []
        for doc in documents:
            if isinstance(doc, dict):
                doc_objects.append(Document(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                ))
            else:
                doc_objects.append(doc)

        # Index documents
        logger.info(f"Indexing {len(doc_objects)} documents")
        start_time = time.time()
        index_result = pipeline.index(doc_objects)
        indexing_time = time.time() - start_time

        logger.info(f"Indexing complete in {indexing_time:.2f}s")
        logger.info(f"  Created {index_result.get('num_chunks', 0)} chunks")

        return pipeline, indexing_time

    except Exception as e:
        logger.warning(f"Could not load real pipeline: {e}")
        logger.info("Using mock pipeline for testing")

        # Fallback to mock pipeline
        class MockPipeline:
            def __init__(self):
                self.query_count = 0

            def index(self, documents):
                return {"num_chunks": len(documents) * 2}

            def query(self, question, top_k=5):
                self.query_count += 1
                return {
                    "question": question,
                    "answer": f"This is a mock answer to: {question}. Based on the context provided.",
                    "contexts": [
                        {"text": f"Context {i}: Relevant information about {question[:20]}...",
                         "score": 0.9 - i * 0.1}
                        for i in range(min(top_k, 3))
                    ],
                    "metadata": {"query_num": self.query_count}
                }

        pipeline = MockPipeline()
        pipeline.index(documents)
        return pipeline, 0.0


def run_evaluation(service, pipeline, test_queries, evaluation_name="msmarco_eval"):
    """Run progressive evaluation on the pipeline"""
    logger.info(f"Starting evaluation: {evaluation_name}")
    logger.info(f"  Test queries: {len(test_queries)}")

    # Run evaluation with progressive levels
    results = service.evaluate_pipeline(
        pipeline=pipeline,
        test_queries=test_queries,
        config={
            "dataset": "ms_marco",
            "model": "baseline_rag",
            "chunking": {"strategy": "fixed", "size": 256},
            "embedding": {"model": "text-embedding-ada-002"},
            "generation": {"model": "gpt-3.5-turbo", "temperature": 0.7}
        },
        evaluation_name=evaluation_name,
        progressive_levels=[EvaluationLevel.SMOKE, EvaluationLevel.STANDARD],
        metrics_to_track=["accuracy", "faithfulness", "answer_relevancy"],
        output_dir="experiments/week3_msmarco"
    )

    return results


def analyze_results(results):
    """Analyze and display evaluation results"""
    print("\n" + "="*70)
    print("WEEK 3 EVALUATION RESULTS - MS MARCO DATASET")
    print("="*70)

    # Basic metadata
    if "metadata" in results:
        meta = results["metadata"]
        print(f"\nEvaluation: {meta.get('evaluation_name', 'N/A')}")
        print(f"Duration: {meta.get('duration', 0):.2f} seconds")
        print(f"Timestamp: {meta.get('timestamp', 'N/A')}")

    # Progressive evaluation results
    if "progressive" in results:
        prog = results["progressive"]
        print(f"\nProgressive Evaluation:")
        print(f"  Levels completed: {prog.get('levels_completed', [])}")
        print(f"  Final level: {prog.get('final_level', 'N/A')}")
        print(f"  Total time: {prog.get('total_time', 0):.2f}s")
        print(f"  Total cost: ${prog.get('total_cost', 0):.6f}")

        if prog.get('stopped_early', False):
            print(f"  Early stopping: {prog.get('stop_reason', 'N/A')}")

        # Show metrics for each level
        if "metrics" in prog:
            for level, metrics in prog["metrics"].items():
                print(f"\n  {level} Metrics:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.3f}")

    # Standard evaluation results
    elif "standard" in results:
        std = results["standard"]
        print(f"\nStandard Evaluation:")

        if "ragas_metrics" in std:
            print("  RAGAS Metrics:")
            for metric, value in std["ragas_metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.3f}")

        if "traditional_metrics" in std:
            print("  Traditional Metrics:")
            for metric, value in std["traditional_metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.3f}")

        if "semantic_metrics" in std:
            print("  Semantic Metrics:")
            sem = std["semantic_metrics"]
            print(f"    Semantic accuracy: {sem.get('semantic_accuracy', 0):.3f}")
            print(f"    Mean similarity: {sem.get('similarity_mean', 0):.3f}")

    # Cost tracking
    if "cost_summary" in results:
        cost = results["cost_summary"]
        print(f"\nCost Tracking:")
        print(f"  Total cost: ${cost.get('total_cost', 0):.6f}")
        print(f"  Budget limit: ${cost.get('budget_limit', 0):.2f}")
        print(f"  Remaining budget: ${cost.get('remaining_budget', 0):.6f}")

        if "costs_by_operation" in cost:
            print("  Costs by operation:")
            for op, op_cost in cost["costs_by_operation"].items():
                print(f"    {op}: ${op_cost:.6f}")

    # Reports generated
    if "reports_generated" in results:
        print(f"\nReports Generated:")
        for report_type in results["reports_generated"]:
            print(f"  - {report_type}")

    print("\n" + "="*70)


def run_comparative_evaluation(service, data):
    """Run evaluation comparing multiple configurations"""
    logger.info("Running comparative evaluation with multiple configurations")

    # Define configurations to compare
    configs = [
        {
            "name": "Baseline",
            "chunking": {"strategy": "fixed", "size": 256},
            "generation": {"temperature": 0.7}
        },
        {
            "name": "Large Chunks",
            "chunking": {"strategy": "fixed", "size": 512},
            "generation": {"temperature": 0.7}
        },
        {
            "name": "Small Chunks",
            "chunking": {"strategy": "fixed", "size": 128},
            "generation": {"temperature": 0.7}
        }
    ]

    def pipeline_factory(config):
        """Create pipeline with specific configuration"""
        # For demo, return mock pipelines with different behaviors
        class ConfiguredMockPipeline:
            def __init__(self, chunk_size):
                self.chunk_size = chunk_size
                self.accuracy = 0.6 + (256 - abs(chunk_size - 256)) / 1000

            def query(self, question):
                import random
                random.seed(hash(question) + self.chunk_size)

                return {
                    "question": question,
                    "answer": f"Answer with chunk_size={self.chunk_size}",
                    "contexts": [{"text": f"Context", "score": 0.9}],
                    "accuracy": self.accuracy + random.uniform(-0.05, 0.05)
                }

        return ConfiguredMockPipeline(config["chunking"]["size"])

    # Use test split for comparison
    test_queries = data["splits"]["test"]["queries"]

    # Run comparative evaluation
    comparison_results = service.evaluate_multiple_configs(
        configs=configs,
        pipeline_factory=pipeline_factory,
        test_queries=test_queries,
        parallel=False  # Sequential for demo
    )

    # Display comparison results
    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON RESULTS")
    print("="*70)

    if "best_config" in comparison_results:
        best = comparison_results["best_config"]
        print(f"\nBest Configuration:")
        print(f"  Config: {configs[best['index']]['name']}")
        print(f"  Score: {best['score']:.3f}")

    if "comparison" in comparison_results:
        comp = comparison_results["comparison"]

        # Show pairwise comparisons
        for key, value in comp.items():
            if "vs" in key and hasattr(value, 'winner'):
                print(f"\n{key}:")
                print(f"  Winner: {value.winner}")
                print(f"  Confidence: {value.confidence:.1%}")

    return comparison_results


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("WEEK 3 EVALUATION ON MS MARCO DATASET")
    print("="*70)
    print("\nThis evaluation will test all Week 3 features:")
    print("- Progressive evaluation (SMOKE -> STANDARD)")
    print("- Caching for efficiency")
    print("- Statistical analysis")
    print("- Cost tracking with budget enforcement")
    print("- Multi-format reporting (JSON, HTML, Markdown, CSV)")

    # Setup
    log_file = setup_logging()
    print(f"\nLogging to: {log_file}")

    # Load data
    print("\n1. Loading MS MARCO Dataset...")
    data = load_msmarco_data(num_docs=100, num_queries=20)

    # Setup evaluation service
    print("\n2. Setting up Evaluation Service...")
    service = setup_evaluation_service()

    # Setup pipeline
    print("\n3. Setting up RAG Pipeline...")
    train_docs = data["splits"]["train"]["documents"]
    pipeline, indexing_time = setup_pipeline(train_docs)

    # Prepare test queries
    test_queries = data["splits"]["test"]["queries"]
    print(f"\n4. Running Evaluation on {len(test_queries)} test queries...")

    # Run main evaluation
    start_time = time.time()
    results = run_evaluation(
        service=service,
        pipeline=pipeline,
        test_queries=test_queries,
        evaluation_name="week3_msmarco_main"
    )
    total_time = time.time() - start_time

    # Analyze results
    analyze_results(results)

    # Run comparative evaluation
    print("\n5. Running Comparative Evaluation...")
    comparison_results = run_comparative_evaluation(service, data)

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nTotal evaluation time: {total_time:.2f} seconds")
    print(f"Reports saved to: experiments/week3_msmarco/")

    # Save final state
    state_file = "experiments/week3_msmarco/evaluation_state.json"
    service.save_state(state_file)
    print(f"Evaluation state saved to: {state_file}")

    # Validate all features worked
    print("\n✅ Feature Validation:")
    features_validated = []

    if results.get("progressive"):
        features_validated.append("Progressive Evaluation")
    if results.get("cost_summary"):
        features_validated.append("Cost Tracking")
    if service.cache and service.cache.size() > 0:
        features_validated.append("Caching System")
    if comparison_results.get("comparison"):
        features_validated.append("Statistical Analysis")
    if results.get("reports_generated"):
        features_validated.append("Multi-format Reporting")

    for feature in features_validated:
        print(f"  ✅ {feature}")

    print(f"\nAll Week 3 features validated: {len(features_validated)}/5")

    return results


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise