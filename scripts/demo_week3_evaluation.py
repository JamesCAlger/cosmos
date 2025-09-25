"""
Week 3 Evaluation Infrastructure Demo

This script demonstrates all the new features from Week 3:
1. File-based caching system (lowest overhead)
2. Progressive evaluation with configurable levels
3. Full statistical analysis suite
4. Token-based cost tracking
5. Enhanced dataset loader with stratified sampling
6. Configurable multi-format reporting
7. Standalone evaluation service
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime
from loguru import logger
import numpy as np

# Import Week 3 components
from autorag.evaluation.service import EvaluationService
from autorag.evaluation.progressive.evaluator import EvaluationLevel, LevelConfig
from autorag.evaluation.cost_tracker import CostTracker, ModelPricing
from autorag.datasets.enhanced_loader import EnhancedDatasetLoader
from autorag.pipeline.simple_rag import SimpleRAGPipeline


def setup_logging():
    """Configure logging for demo"""
    logger.add(
        "experiments/week3_demo_{time}.log",
        rotation="100 MB",
        level="INFO"
    )


def demonstrate_caching():
    """Demonstrate file-based caching system"""
    print("\n" + "="*60)
    print("DEMONSTRATING CACHING SYSTEM")
    print("="*60)

    from autorag.evaluation.cache.base import TieredCache, CacheKey

    # Initialize cache
    cache = TieredCache(cache_dir="cache/demo", memory_items=10)

    # Generate cache key
    config = {"model": "gpt-3.5-turbo", "temperature": 0.7}
    query = "What is machine learning?"
    context = "Machine learning is a subset of AI"
    answer = "Machine learning enables systems to learn from data"

    key = CacheKey.generate(config, query, context, answer)
    print(f"\nGenerated cache key: {key[:16]}...")

    # Store and retrieve
    result = {"answer": answer, "score": 0.95}
    cache.set(key, result)
    print(f"Stored result in cache")

    retrieved = cache.get(key)
    print(f"Retrieved from cache: {retrieved}")

    # Show cache size
    print(f"Cache size: {cache.size()} bytes")


def demonstrate_progressive_evaluation():
    """Demonstrate progressive evaluation system"""
    print("\n" + "="*60)
    print("DEMONSTRATING PROGRESSIVE EVALUATION")
    print("="*60)

    from autorag.evaluation.progressive.evaluator import ProgressiveEvaluator

    # Create evaluator with custom levels
    evaluator = ProgressiveEvaluator()

    # Add custom evaluation level
    custom_level = LevelConfig(
        name="Ultra Quick",
        num_queries=3,
        max_duration_seconds=15,
        estimated_cost=0.005,
        description="Bare minimum validation",
        early_stop_threshold=0.2
    )
    evaluator.add_level(EvaluationLevel.SMOKE, custom_level)

    # Show evaluation levels
    print(evaluator.summary())

    # Simulate evaluation
    def mock_pipeline(query):
        return {
            "answer": f"Answer to {query['question']}",
            "accuracy": np.random.uniform(0.6, 0.9)
        }

    def metrics_func(results):
        return {
            "mean_accuracy": np.mean([r.get("accuracy", 0) for r in results])
        }

    test_queries = [{"question": f"Question {i}"} for i in range(20)]

    print("\nRunning progressive evaluation...")
    results = evaluator.evaluate(
        mock_pipeline,
        test_queries,
        start_level=EvaluationLevel.SMOKE,
        target_level=EvaluationLevel.QUICK,
        metrics_func=metrics_func
    )

    print(f"Levels completed: {results['levels_completed']}")
    print(f"Total cost: ${results['total_cost']:.4f}")
    print(f"Total time: {results['total_time']:.2f}s")

    if results['stopped_early']:
        print(f"Stopped early: {results['stop_reason']}")


def demonstrate_statistical_analysis():
    """Demonstrate statistical analysis framework"""
    print("\n" + "="*60)
    print("DEMONSTRATING STATISTICAL ANALYSIS")
    print("="*60)

    from autorag.evaluation.statistics.analyzer import StatisticalAnalyzer

    analyzer = StatisticalAnalyzer(confidence_level=0.95)

    # Simulate results from two configurations
    np.random.seed(42)
    config_a_results = [{"accuracy": 0.75 + np.random.normal(0, 0.05)} for _ in range(30)]
    config_b_results = [{"accuracy": 0.65 + np.random.normal(0, 0.05)} for _ in range(30)]

    # Compare configurations
    comparison = analyzer.compare_configurations(
        config_a_results,
        config_b_results,
        "Enhanced RAG",
        "Baseline RAG"
    )

    print(f"\nStatistical Comparison:")
    print(f"Config A: {comparison.config_a}")
    print(f"Config B: {comparison.config_b}")
    print(f"Winner: {comparison.winner} (Confidence: {comparison.confidence:.1%})")

    # Show detailed metrics
    for metric, test in comparison.metrics.items():
        print(f"\n{metric}:")
        print(f"  P-value: {test.p_value:.4f}")
        print(f"  Significant: {test.significant}")
        print(f"  Effect size: {test.effect_size:.3f} ({analyzer.interpret_effect_size(test.effect_size)})")

    # Calculate required sample size
    required_n = analyzer.calculate_required_sample_size(effect_size=0.5, power=0.8)
    print(f"\nRequired sample size for effect=0.5, power=0.8: {required_n}")


def demonstrate_cost_tracking():
    """Demonstrate cost tracking system"""
    print("\n" + "="*60)
    print("DEMONSTRATING COST TRACKING")
    print("="*60)

    tracker = CostTracker(budget_limit=1.0)

    # Add custom model pricing
    tracker.pricing["custom-model"] = ModelPricing(
        model_name="custom-model",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002
    )

    # Simulate operations
    operations = [
        ("What is Python?", "Python is a programming language", "gpt-3.5-turbo"),
        ("Explain machine learning", "ML is a subset of AI that...", "gpt-4o-mini"),
        ("How does RAG work?", "RAG combines retrieval with generation...", "custom-model")
    ]

    for query, answer, model in operations:
        cost = tracker.estimate_cost(
            query,
            model=model,
            operation="generation",
            output_text=answer
        )
        print(f"Operation with {model}: ${cost:.6f}")

    # Show summary
    print("\n" + tracker.format_summary())

    # Estimate pipeline cost
    pipeline_config = {
        "retrieval": {"method": "dense"},
        "embedding": {"model": "text-embedding-3-small"},
        "generation": {"model": "gpt-3.5-turbo", "max_tokens": 300}
    }

    estimated_cost = tracker.estimate_pipeline_cost(
        pipeline_config,
        num_queries=100,
        avg_doc_length=500,
        avg_query_length=20
    )
    print(f"\nEstimated cost for 100 queries: ${estimated_cost:.4f}")


def demonstrate_enhanced_loader():
    """Demonstrate enhanced dataset loader"""
    print("\n" + "="*60)
    print("DEMONSTRATING ENHANCED DATASET LOADER")
    print("="*60)

    loader = EnhancedDatasetLoader(dataset_name="ms_marco")

    # Load with train/dev/test splits
    print("\nLoading dataset with configurable splits...")
    data = loader.load_with_splits(
        num_docs=100,
        num_queries=50,
        train_ratio=0.7,
        dev_ratio=0.15,
        test_ratio=0.15,
        use_cache=True
    )

    print(f"Dataset: {data['dataset_name']}")
    print(f"Total documents: {data['num_documents']}")
    print(f"Total queries: {data['num_queries']}")

    for split_name, split_data in data['splits'].items():
        print(f"\n{split_name.upper()} split:")
        print(f"  Documents: {split_data['num_docs']}")
        print(f"  Queries: {split_data['num_queries']}")

    # Show metadata
    if 'metadata' in data:
        print("\nDataset Statistics:")
        for key, stats in data['metadata'].items():
            print(f"\n{key}:")
            for stat_name, value in stats.items():
                if isinstance(value, float):
                    print(f"  {stat_name}: {value:.2f}")
                else:
                    print(f"  {stat_name}: {value}")


def demonstrate_reporters():
    """Demonstrate configurable reporting system"""
    print("\n" + "="*60)
    print("DEMONSTRATING CONFIGURABLE REPORTERS")
    print("="*60)

    from autorag.evaluation.reporters.base import CompositeReporter

    # Create sample results
    results = {
        "summary": {
            "total_queries": 100,
            "avg_accuracy": 0.756,
            "avg_latency": 1.23
        },
        "metrics": {
            "faithfulness": 0.823,
            "answer_relevancy": 0.791,
            "context_relevancy": 0.864
        },
        "cost_summary": {
            "total_cost": 0.1234,
            "total_input_tokens": 5000,
            "total_output_tokens": 2500
        },
        "statistical_analysis": {
            "p_value": 0.0234,
            "effect_size": 0.65,
            "confidence_interval": [0.72, 0.79]
        }
    }

    # Generate reports in multiple formats
    reporter = CompositeReporter()

    output_dir = Path("experiments") / "week3_demo_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = reporter.report(results, str(output_dir / "demo_report"))

    print("\nGenerated reports:")
    for format_name in reports.keys():
        print(f"  - {format_name}: {output_dir / f'demo_report.{format_name}'}")

    # Show sample of markdown report
    print("\nSample of Markdown report:")
    print("-" * 40)
    print(reports["markdown"][:500] + "...")


def demonstrate_evaluation_service():
    """Demonstrate integrated evaluation service"""
    print("\n" + "="*60)
    print("DEMONSTRATING EVALUATION SERVICE")
    print("="*60)

    # Initialize service with all Week 3 features
    service = EvaluationService(
        cache_dir="cache/service_demo",
        enable_caching=True,
        cost_tracking=True,
        budget_limit=0.50,
        progressive_eval=True,
        statistical_analysis=True,
        reporter_formats=["json", "html", "markdown"]
    )

    print("Evaluation service initialized with:")
    print("  ✓ File-based caching (lowest overhead)")
    print("  ✓ Progressive evaluation")
    print("  ✓ Statistical analysis")
    print("  ✓ Cost tracking with $0.50 budget")
    print("  ✓ Multi-format reporting")

    # Load pipeline
    try:
        config_path = Path(__file__).parent.parent / "configs" / "baseline_rag.yaml"
        pipeline = SimpleRAGPipeline(str(config_path))
        print(f"\nLoaded pipeline from: {config_path}")
    except Exception as e:
        print(f"\nUsing mock pipeline: {e}")

        class MockPipeline:
            def query(self, question):
                return {
                    "question": question,
                    "answer": f"Answer to: {question}",
                    "contexts": [{"text": f"Context {i}", "score": 0.9 - i*0.1} for i in range(3)]
                }

        pipeline = MockPipeline()

    # Prepare test data
    test_queries = [
        {
            "question": "What is machine learning?",
            "ground_truth_answer": "Machine learning is a subset of AI."
        },
        {
            "question": "How does Python work?",
            "ground_truth_answer": "Python is an interpreted language."
        },
        {
            "question": "What is RAG?",
            "ground_truth_answer": "RAG combines retrieval with generation."
        }
    ]

    # Run evaluation
    print("\nRunning evaluation with progressive levels...")
    results = service.evaluate_pipeline(
        pipeline,
        test_queries,
        config={"model": "demo", "temperature": 0.7},
        evaluation_name="week3_demo",
        progressive_levels=[EvaluationLevel.SMOKE],
        output_dir="experiments"
    )

    # Show results
    print("\nEvaluation Results:")
    print(f"  Duration: {results['metadata']['duration']:.2f}s")

    if 'cost_summary' in results:
        print(f"  Total cost: ${results['cost_summary']['total_cost']:.6f}")
        print(f"  Budget remaining: ${results['cost_summary']['remaining_budget']:.6f}")

    if 'progressive' in results:
        print(f"  Levels completed: {results['progressive']['levels_completed']}")

    print(f"  Reports generated: {results.get('reports_generated', [])}")

    # Compare multiple configurations
    print("\n" + "="*60)
    print("COMPARING MULTIPLE CONFIGURATIONS")
    print("="*60)

    configs = [
        {"temperature": 0.0, "name": "Deterministic"},
        {"temperature": 0.5, "name": "Balanced"},
        {"temperature": 1.0, "name": "Creative"}
    ]

    def pipeline_factory(config):
        class ConfiguredPipeline:
            def __init__(self, temp):
                self.temperature = temp

            def query(self, question):
                # Simulate different accuracy based on temperature
                accuracy = 0.8 - abs(0.3 - self.temperature)
                return {
                    "question": question,
                    "answer": f"Answer with temp={self.temperature}",
                    "contexts": [{"text": "context", "score": accuracy}],
                    "accuracy": accuracy
                }

        return ConfiguredPipeline(config["temperature"])

    print("\nComparing configurations:")
    for config in configs:
        print(f"  - {config['name']}: temperature={config['temperature']}")

    comparison_results = service.evaluate_multiple_configs(
        configs,
        pipeline_factory,
        test_queries,
        parallel=False  # Sequential for demo
    )

    if 'best_config' in comparison_results:
        best = comparison_results['best_config']
        print(f"\nBest configuration: {configs[best['index']]['name']}")
        print(f"Score: {best['score']:.3f}")

    # Save service state
    service.save_state("experiments/week3_service_state.json")
    print("\nService state saved for future analysis")


def main():
    """Run all Week 3 demonstrations"""
    setup_logging()

    print("\n" + "="*80)
    print(" " * 20 + "WEEK 3 EVALUATION INFRASTRUCTURE DEMO")
    print("="*80)
    print("\nThis demo showcases all the new features implemented in Week 3:")
    print("1. File-based caching (lowest overhead)")
    print("2. Progressive evaluation with configurable levels")
    print("3. Full statistical analysis suite")
    print("4. Token-based cost tracking")
    print("5. Enhanced dataset loader with stratified sampling")
    print("6. Multi-format configurable reporting")
    print("7. Standalone evaluation service")

    # Run demonstrations
    demonstrate_caching()
    demonstrate_progressive_evaluation()
    demonstrate_statistical_analysis()
    demonstrate_cost_tracking()
    demonstrate_enhanced_loader()
    demonstrate_reporters()
    demonstrate_evaluation_service()

    print("\n" + "="*80)
    print("WEEK 3 DEMO COMPLETE")
    print("="*80)
    print("\nKey Achievements:")
    print("✓ Implemented efficient file-based caching system")
    print("✓ Created configurable progressive evaluation levels")
    print("✓ Built comprehensive statistical analysis framework")
    print("✓ Added token-based cost tracking and budgeting")
    print("✓ Enhanced dataset loader with stratified sampling")
    print("✓ Developed multi-format reporting system")
    print("✓ Integrated everything into standalone evaluation service")
    print("\nThe evaluation infrastructure is now ready for Week 4 optimizations!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()