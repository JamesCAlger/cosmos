"""Integration tests for Week 3 evaluation infrastructure"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

# Import evaluation components
from autorag.evaluation.cache.base import FileCache, TieredCache, CacheKey
from autorag.evaluation.progressive.evaluator import ProgressiveEvaluator, EvaluationLevel, LevelConfig
from autorag.evaluation.statistics.analyzer import StatisticalAnalyzer
from autorag.evaluation.cost_tracker import CostTracker
from autorag.evaluation.reporters.base import JSONReporter, HTMLReporter, MarkdownReporter, CompositeReporter
from autorag.evaluation.service import EvaluationService
from autorag.datasets.enhanced_loader import EnhancedDatasetLoader


class MockPipeline:
    """Mock pipeline for testing"""

    def __init__(self, config: dict = None, accuracy: float = 0.75):
        self.config = config or {}
        self.accuracy = accuracy
        self.query_count = 0

    def query(self, question: str, top_k: int = 5) -> dict:
        """Mock query method"""
        self.query_count += 1

        # Simulate varying accuracy
        is_correct = np.random.random() < self.accuracy

        return {
            "question": question,
            "answer": f"Answer to: {question}" if is_correct else "Wrong answer",
            "contexts": [{"text": f"Context {i}", "score": 0.9 - i * 0.1} for i in range(top_k)],
            "accuracy": self.accuracy,
            "query_num": self.query_count
        }


class TestCacheSystem:
    """Test caching system"""

    def test_file_cache(self):
        """Test file-based cache"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=tmpdir)

            # Test set and get
            key = "test_key"
            value = {"result": "test_value", "score": 0.95}

            assert cache.set(key, value)
            retrieved = cache.get(key)
            assert retrieved == value

            # Test exists
            assert cache.exists(key)
            assert not cache.exists("nonexistent_key")

            # Test delete
            assert cache.delete(key)
            assert not cache.exists(key)

    def test_cache_key_generation(self):
        """Test cache key generation"""
        config = {"model": "gpt-3.5", "temperature": 0.7}
        query = "What is Python?"
        context = "Python is a programming language"
        answer = "Python is a high-level programming language"

        key1 = CacheKey.generate(config, query, context, answer)
        key2 = CacheKey.generate(config, query, context, answer)

        # Same inputs should generate same key
        assert key1 == key2

        # Different inputs should generate different keys
        key3 = CacheKey.generate(config, "Different query", context, answer)
        assert key1 != key3

    def test_tiered_cache(self):
        """Test tiered caching (memory + disk)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TieredCache(cache_dir=tmpdir, memory_items=2)

            # Add items
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.set("key3", "value3")  # Should evict from memory

            # All should be retrievable
            assert cache.get("key1") == "value1"
            assert cache.get("key2") == "value2"
            assert cache.get("key3") == "value3"


class TestProgressiveEvaluation:
    """Test progressive evaluation system"""

    def test_evaluation_levels(self):
        """Test configurable evaluation levels"""
        evaluator = ProgressiveEvaluator()

        # Add custom level
        custom_level = LevelConfig(
            name="Custom Test",
            num_queries=10,
            max_duration_seconds=60,
            estimated_cost=0.02,
            description="Custom test level"
        )
        evaluator.add_level(EvaluationLevel.SMOKE, custom_level)

        # Check level was added
        assert evaluator.levels[EvaluationLevel.SMOKE] == custom_level

    def test_progressive_evaluation(self):
        """Test progressive evaluation flow"""
        evaluator = ProgressiveEvaluator()
        pipeline = MockPipeline(accuracy=0.8)

        # Create test queries
        test_queries = [
            {"question": f"Question {i}", "ground_truth_answer": f"Answer {i}"}
            for i in range(20)
        ]

        # Define pipeline function
        def pipeline_func(query):
            return pipeline.query(query["question"])

        # Define metrics function
        def metrics_func(results):
            accuracies = [r.get("accuracy", 0) for r in results]
            return {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies)
            }

        # Run evaluation
        results = evaluator.evaluate(
            pipeline_func,
            test_queries,
            start_level=EvaluationLevel.SMOKE,
            target_level=EvaluationLevel.QUICK,
            metrics_func=metrics_func
        )

        # Check results
        assert "levels_completed" in results
        assert len(results["levels_completed"]) > 0
        assert "metrics" in results
        assert results["total_cost"] > 0

    def test_early_stopping(self):
        """Test early stopping on poor performance"""
        evaluator = ProgressiveEvaluator()
        pipeline = MockPipeline(accuracy=0.2)  # Poor accuracy

        test_queries = [
            {"question": f"Question {i}", "ground_truth_answer": f"Answer {i}"}
            for i in range(20)
        ]

        def pipeline_func(query):
            return pipeline.query(query["question"])

        def metrics_func(results):
            return {"accuracy": 0.2}  # Below threshold

        results = evaluator.evaluate(
            pipeline_func,
            test_queries,
            start_level=EvaluationLevel.SMOKE,
            metrics_func=metrics_func
        )

        # Should stop early
        assert results.get("stopped_early", False)
        assert results.get("stop_reason") is not None


class TestStatisticalAnalysis:
    """Test statistical analysis framework"""

    def test_cohens_d(self):
        """Test Cohen's d effect size calculation"""
        analyzer = StatisticalAnalyzer()

        group1 = np.array([0.7, 0.75, 0.8, 0.72, 0.78])
        group2 = np.array([0.6, 0.62, 0.65, 0.61, 0.63])

        d = analyzer.calculate_cohens_d(group1, group2)

        # Should show large effect size
        assert d > 0.8
        assert analyzer.interpret_effect_size(d) == "large"

    def test_configuration_comparison(self):
        """Test statistical comparison of configurations"""
        analyzer = StatisticalAnalyzer()

        # Simulate results from two configurations
        results_a = [{"accuracy": 0.7 + np.random.normal(0, 0.05)} for _ in range(20)]
        results_b = [{"accuracy": 0.6 + np.random.normal(0, 0.05)} for _ in range(20)]

        comparison = analyzer.compare_configurations(
            results_a,
            results_b,
            "Config A",
            "Config B"
        )

        # Check comparison structure
        assert comparison.config_a == "Config A"
        assert comparison.config_b == "Config B"
        assert "accuracy" in comparison.metrics
        assert comparison.winner is not None

    def test_anova(self):
        """Test ANOVA for multiple groups"""
        analyzer = StatisticalAnalyzer()

        groups = {
            "Config1": [0.7, 0.72, 0.71, 0.73, 0.69],
            "Config2": [0.65, 0.66, 0.64, 0.67, 0.63],
            "Config3": [0.75, 0.76, 0.74, 0.77, 0.73]
        }

        result = analyzer.run_anova(groups)

        assert "f_statistic" in result
        assert "p_value" in result
        assert "significant" in result


class TestCostTracking:
    """Test cost tracking system"""

    def test_token_counting(self):
        """Test token counting"""
        tracker = CostTracker()

        text = "This is a test sentence for counting tokens."
        tokens = tracker.count_tokens(text)

        # Should be reasonable number of tokens
        assert 5 < tokens < 20

    def test_cost_estimation(self):
        """Test cost estimation"""
        tracker = CostTracker()

        input_text = "What is machine learning?"
        output_text = "Machine learning is a subset of AI that enables systems to learn from data."

        cost = tracker.estimate_cost(
            input_text,
            model="gpt-3.5-turbo",
            operation="generation",
            output_text=output_text
        )

        # Should have non-zero cost
        assert cost > 0
        assert tracker.total_cost > 0

    def test_budget_enforcement(self):
        """Test budget limit enforcement"""
        tracker = CostTracker(budget_limit=0.01)

        # Simulate operations until budget exceeded
        for _ in range(100):
            tracker.track_cost(
                operation="generation",
                model="gpt-3.5-turbo",
                input_tokens=100,
                output_tokens=50,
                cost=0.001
            )

        # Check budget status
        assert tracker.total_cost > tracker.budget_limit
        remaining = tracker.get_remaining_budget()
        assert remaining == 0


class TestReporters:
    """Test reporting system"""

    def test_json_reporter(self):
        """Test JSON reporter"""
        reporter = JSONReporter()

        results = {
            "metrics": {"accuracy": 0.75, "f1": 0.72},
            "config": {"model": "gpt-3.5-turbo"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report = reporter.report(results, f.name)

            # Check file was created
            assert Path(f.name).exists()

            # Check content is valid JSON
            with open(f.name, 'r') as rf:
                loaded = json.load(rf)
                assert "results" in loaded

    def test_html_reporter(self):
        """Test HTML reporter"""
        reporter = HTMLReporter()

        results = {
            "summary": {"total_queries": 100, "avg_accuracy": 0.75},
            "metrics": {"accuracy": 0.75, "f1": 0.72}
        }

        html = reporter.format_results(results)

        # Check HTML structure
        assert "<html" in html
        assert "RAG Evaluation Report" in html
        assert "0.75" in html

    def test_composite_reporter(self):
        """Test composite reporter with multiple formats"""
        reporter = CompositeReporter()

        results = {
            "metrics": {"accuracy": 0.75},
            "summary": {"test": "value"}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report"
            reports = reporter.report(results, str(output_path))

            # Check all formats generated
            assert "json" in reports
            assert "html" in reports
            assert "markdown" in reports


class TestEnhancedDataLoader:
    """Test enhanced dataset loader"""

    def test_train_dev_test_splits(self):
        """Test dataset splitting"""
        loader = EnhancedDatasetLoader()

        # Use synthetic data for testing
        result = loader.load_with_splits(
            num_docs=100,
            num_queries=50,
            train_ratio=0.7,
            dev_ratio=0.15,
            test_ratio=0.15,
            use_cache=False
        )

        # Check splits
        assert "splits" in result
        assert "train" in result["splits"]
        assert "dev" in result["splits"]
        assert "test" in result["splits"]

        # Check ratios (approximately)
        total_queries = result["num_queries"]
        train_queries = result["splits"]["train"]["num_queries"]
        dev_queries = result["splits"]["dev"]["num_queries"]
        test_queries = result["splits"]["test"]["num_queries"]

        assert 0.6 < train_queries / total_queries < 0.8
        assert 0.1 < dev_queries / total_queries < 0.2
        assert 0.1 < test_queries / total_queries < 0.2


class TestEvaluationService:
    """Test integrated evaluation service"""

    def test_service_initialization(self):
        """Test service initialization"""
        service = EvaluationService(
            enable_caching=True,
            cost_tracking=True,
            progressive_eval=True,
            statistical_analysis=True
        )

        assert service.cache is not None
        assert service.cost_tracker is not None
        assert service.progressive_evaluator is not None
        assert service.statistical_analyzer is not None

    def test_pipeline_evaluation(self):
        """Test evaluating a single pipeline"""
        service = EvaluationService()
        pipeline = MockPipeline(accuracy=0.75)

        test_queries = [
            {"question": f"Question {i}", "ground_truth_answer": f"Answer {i}"}
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            results = service.evaluate_pipeline(
                pipeline,
                test_queries,
                config={"model": "test"},
                evaluation_name="test_eval",
                output_dir=tmpdir
            )

            # Check results structure
            assert "metadata" in results
            assert "reports_generated" in results
            assert results["metadata"]["evaluation_name"] == "test_eval"

    def test_multiple_config_evaluation(self):
        """Test evaluating multiple configurations"""
        service = EvaluationService()

        configs = [
            {"model": "gpt-3.5", "temperature": 0.0},
            {"model": "gpt-3.5", "temperature": 0.7},
            {"model": "gpt-4", "temperature": 0.3}
        ]

        def pipeline_factory(config):
            # Different configs have different accuracies
            accuracy = 0.7 + config["temperature"] * 0.1
            return MockPipeline(config, accuracy)

        test_queries = [
            {"question": f"Question {i}", "ground_truth_answer": f"Answer {i}"}
            for i in range(10)
        ]

        results = service.evaluate_multiple_configs(
            configs,
            pipeline_factory,
            test_queries,
            parallel=False  # Sequential for testing
        )

        # Check results
        assert "individual_results" in results
        assert len(results["individual_results"]) == len(configs)
        assert "best_config" in results
        assert results["best_config"]["config"] is not None


def test_integration():
    """Full integration test of Week 3 features"""
    # Initialize service with all features
    with tempfile.TemporaryDirectory() as tmpdir:
        service = EvaluationService(
            cache_dir=tmpdir,
            enable_caching=True,
            cost_tracking=True,
            budget_limit=1.0,
            progressive_eval=True,
            statistical_analysis=True,
            reporter_formats=["json", "html", "markdown"]
        )

        # Create mock pipeline
        pipeline = MockPipeline(accuracy=0.75)

        # Load test data
        loader = EnhancedDatasetLoader()
        data = loader.load_with_splits(
            num_docs=50,
            num_queries=20,
            use_cache=False
        )

        test_queries = data["splits"]["test"]["queries"]

        # Run evaluation
        results = service.evaluate_pipeline(
            pipeline,
            test_queries,
            config={"model": "test", "temperature": 0.7},
            evaluation_name="integration_test",
            progressive_levels=[EvaluationLevel.SMOKE],
            output_dir=tmpdir
        )

        # Verify all components worked
        assert results is not None
        assert "cost_summary" in results
        assert "reports_generated" in results
        assert len(results["reports_generated"]) == 3

        # Check reports were created
        report_path = Path(tmpdir) / "integration_test.json"
        assert report_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])