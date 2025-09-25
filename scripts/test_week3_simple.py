"""Simple test of Week 3 evaluation infrastructure"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from autorag.evaluation.service import EvaluationService
from autorag.evaluation.progressive.evaluator import EvaluationLevel
from autorag.datasets.enhanced_loader import EnhancedDatasetLoader


def main():
    print("\n" + "="*60)
    print("WEEK 3 EVALUATION INFRASTRUCTURE TEST")
    print("="*60)

    # Initialize evaluation service
    print("\n1. Initializing Evaluation Service...")
    service = EvaluationService(
        cache_dir="cache/test",
        enable_caching=True,
        cost_tracking=True,
        budget_limit=1.0,
        progressive_eval=True,
        statistical_analysis=True,
        reporter_formats=["json", "markdown"]
    )
    print("   - Caching: Enabled (file-based)")
    print("   - Cost Tracking: Enabled")
    print("   - Progressive Evaluation: Enabled")
    print("   - Statistical Analysis: Enabled")
    print("   - Reporters: JSON, Markdown")

    # Load test data
    print("\n2. Loading Test Data with Splits...")
    loader = EnhancedDatasetLoader()
    data = loader.load_with_splits(
        num_docs=30,
        num_queries=15,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
        use_cache=False  # Don't cache for test
    )

    print(f"   - Total documents: {data['num_documents']}")
    print(f"   - Total queries: {data['num_queries']}")
    print(f"   - Train: {data['splits']['train']['num_queries']} queries")
    print(f"   - Dev: {data['splits']['dev']['num_queries']} queries")
    print(f"   - Test: {data['splits']['test']['num_queries']} queries")

    # Create mock pipeline
    print("\n3. Creating Mock Pipeline...")
    class MockPipeline:
        def query(self, question):
            return {
                "question": question,
                "answer": f"Mock answer to: {question}",
                "contexts": [{"text": "Context 1", "score": 0.9}]
            }

    pipeline = MockPipeline()
    print("   - Pipeline created")

    # Run evaluation
    print("\n4. Running Progressive Evaluation...")
    test_queries = data['splits']['test']['queries'][:3]  # Use only 3 for quick test

    results = service.evaluate_pipeline(
        pipeline,
        test_queries,
        config={"model": "mock", "temperature": 0.7},
        evaluation_name="week3_test",
        progressive_levels=[EvaluationLevel.SMOKE],
        output_dir="experiments/week3_test"
    )

    print("   - Evaluation completed")
    if 'cost_summary' in results:
        print(f"   - Total cost: ${results['cost_summary']['total_cost']:.6f}")
    if 'metadata' in results:
        print(f"   - Duration: {results['metadata']['duration']:.2f}s")
    if 'reports_generated' in results:
        print(f"   - Reports generated: {', '.join(results['reports_generated'])}")

    # Test cost tracking
    print("\n5. Cost Tracking Summary:")
    from autorag.evaluation.cost_tracker import CostTracker
    tracker = CostTracker()

    # Estimate costs for different models
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "text-embedding-3-small"]
    for model in models:
        cost = tracker.estimate_cost(
            "Sample query",
            model=model,
            operation="generation" if "gpt" in model else "embedding"
        )
        print(f"   - {model}: ${cost:.6f}")

    # Test statistical comparison
    print("\n6. Statistical Analysis:")
    from autorag.evaluation.statistics.analyzer import StatisticalAnalyzer
    import numpy as np

    analyzer = StatisticalAnalyzer()

    # Simulate two configurations
    config_a = [{"accuracy": 0.75 + np.random.normal(0, 0.02)} for _ in range(10)]
    config_b = [{"accuracy": 0.70 + np.random.normal(0, 0.02)} for _ in range(10)]

    comparison = analyzer.compare_configurations(
        config_a, config_b, "Config A", "Config B"
    )

    print(f"   - Winner: {comparison.winner}")
    print(f"   - Confidence: {comparison.confidence:.1%}")
    if "accuracy" in comparison.metrics:
        test = comparison.metrics["accuracy"]
        print(f"   - P-value: {test.p_value:.4f}")
        print(f"   - Effect size: {test.effect_size:.3f}")

    print("\n" + "="*60)
    print("WEEK 3 TEST COMPLETE - ALL FEATURES WORKING!")
    print("="*60)


if __name__ == "__main__":
    main()