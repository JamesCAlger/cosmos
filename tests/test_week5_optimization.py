"""Comprehensive tests for Week 5 optimization modules"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

from autorag.optimization import (
    SearchSpace,
    ConfigurationGenerator,
    GridSearchOptimizer,
    ResultManager,
    StatisticalComparison
)


class TestSearchSpace:
    """Test search space definition"""

    def test_create_empty_search_space(self):
        """Test creating empty search space"""
        space = SearchSpace()
        assert len(space.components) == 0
        assert space.total_combinations == 0

    def test_add_component(self):
        """Test adding component to search space"""
        space = SearchSpace()
        component = space.add_component("chunking")
        assert "chunking" in space.components
        assert component.component_type == "chunking"

    def test_define_chunking_space(self):
        """Test defining chunking search space"""
        space = SearchSpace()
        space.define_chunking_space(
            strategies=["fixed", "semantic"],
            sizes=[256, 512]
        )
        assert "chunking" in space.components
        assert len(space.components["chunking"].parameters) == 2

    def test_define_retrieval_space(self):
        """Test defining retrieval search space"""
        space = SearchSpace()
        space.define_retrieval_space(
            methods=["dense", "sparse"],
            top_k_values=[3, 5]
        )
        assert "retrieval" in space.components
        # Should have method, top_k, and conditional hybrid_weight
        assert len(space.components["retrieval"].parameters) >= 2

    def test_calculate_total_combinations(self):
        """Test calculating total combinations"""
        space = SearchSpace()
        space.define_chunking_space(["fixed", "semantic"], [256, 512])
        space.define_retrieval_space(["dense", "sparse"], [3, 5])
        total = space.calculate_total_combinations()
        assert total > 0
        # 2 strategies * 2 sizes * 2 methods * 2 top_k = 16 base combinations
        assert total >= 16

    def test_enumerate_all_configurations(self):
        """Test enumerating all configurations"""
        space = SearchSpace()
        space.define_chunking_space(["fixed"], [256])
        space.define_retrieval_space(["dense"], [3])
        configs = space.enumerate_all()
        assert len(configs) == 1
        assert configs[0]["chunking"]["strategy"] == "fixed"
        assert configs[0]["chunking"]["size"] == 256

    def test_sample_configurations(self):
        """Test sampling configurations"""
        space = SearchSpace()
        space.define_chunking_space(["fixed", "semantic"], [256, 512])
        space.define_retrieval_space(["dense", "sparse"], [3, 5])

        # Random sampling
        random_sample = space.sample(5, method="random")
        assert len(random_sample) <= 5

        # Grid sampling
        grid_sample = space.sample(5, method="grid")
        assert len(grid_sample) <= 5

    def test_conditional_parameters(self):
        """Test conditional parameter dependencies"""
        space = SearchSpace()
        space.define_reranking_space([True, False], ["model1"])
        configs = space.enumerate_all()

        # Check that model only appears when enabled=True
        for config in configs:
            if config["reranking"]["enabled"]:
                assert "model" in config["reranking"]
            else:
                assert "model" not in config["reranking"] or config["reranking"]["model"] is None

    def test_save_and_load_search_space(self):
        """Test saving and loading search space"""
        space = SearchSpace()
        space.define_chunking_space(["fixed", "semantic"], [256, 512])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            space.save(f.name)

            # Load and verify
            loaded_space = SearchSpace.load(f.name)
            assert "chunking" in loaded_space.components
            assert len(loaded_space.components["chunking"].parameters) == 2

        Path(f.name).unlink()

    def test_create_default_search_space(self):
        """Test creating default Week 5 search space"""
        space = SearchSpace()
        space.create_default_search_space()

        assert "chunking" in space.components
        assert "retrieval" in space.components
        assert "reranking" in space.components
        assert "generation" in space.components
        assert "embedding" in space.components

        total = space.calculate_total_combinations()
        assert total > 100  # Should have many combinations


class TestConfigurationGenerator:
    """Test configuration generator"""

    def test_create_generator(self):
        """Test creating configuration generator"""
        space = SearchSpace()
        generator = ConfigurationGenerator(space)
        assert generator.search_space == space
        assert generator.base_config is None

    def test_generate_minimal_configuration(self):
        """Test generating minimal configuration"""
        space = SearchSpace()
        space.define_chunking_space(["fixed"], [256])
        generator = ConfigurationGenerator(space)

        params = {"chunking": {"strategy": "fixed", "size": 256}}
        config = generator.generate_configuration(params)

        assert "pipeline" in config
        assert "metadata" in config
        assert config["metadata"]["parameters"] == params

    def test_apply_chunking_params(self):
        """Test applying chunking parameters"""
        space = SearchSpace()
        generator = ConfigurationGenerator(space)

        config = generator._create_minimal_config()
        params = {"strategy": "semantic", "size": 512}
        config = generator._apply_chunking_params(config, params)

        # Find chunker node
        chunker_node = None
        for node in config["pipeline"]["nodes"]:
            if node["id"] == "chunker":
                chunker_node = node
                break

        assert chunker_node is not None
        assert chunker_node["component"] == "semantic"
        assert chunker_node["config"]["chunk_size"] == 512

    def test_apply_retrieval_params(self):
        """Test applying retrieval parameters"""
        space = SearchSpace()
        generator = ConfigurationGenerator(space)

        config = generator._create_minimal_config()

        # Test dense retrieval
        params = {"method": "dense", "top_k": 5}
        config = generator._apply_retrieval_params(config, params)
        assert any(n["id"] == "retriever" for n in config["pipeline"]["nodes"])

        # Test hybrid retrieval
        params = {"method": "hybrid", "top_k": 5, "hybrid_weight": 0.5}
        config = generator._apply_retrieval_params(config, params)
        assert any(n["id"] == "hybrid_retriever" for n in config["pipeline"]["nodes"])

    def test_apply_reranking_params(self):
        """Test applying reranking parameters"""
        space = SearchSpace()
        generator = ConfigurationGenerator(space)

        config = generator._create_minimal_config()

        # Test with reranking enabled
        params = {"enabled": True, "model": "test-model", "top_k_rerank": 20}
        config = generator._apply_reranking_params(config, params)
        assert any(n["id"] == "reranker" for n in config["pipeline"]["nodes"])

        # Test with reranking disabled
        params = {"enabled": False}
        config = generator._apply_reranking_params(config, params)
        assert not any(n.get("id") == "reranker" for n in config["pipeline"]["nodes"])

    def test_generate_all_configurations(self):
        """Test generating all configurations"""
        space = SearchSpace()
        space.define_chunking_space(["fixed"], [256, 512])
        generator = ConfigurationGenerator(space)

        configs = generator.generate_all_configurations()
        assert len(configs) == 2  # 1 strategy * 2 sizes

    def test_generate_subset(self):
        """Test generating subset of configurations"""
        space = SearchSpace()
        space.define_chunking_space(["fixed", "semantic"], [256, 512])
        generator = ConfigurationGenerator(space)

        configs = generator.generate_subset(2, method="random")
        assert len(configs) == 2

    def test_validate_configuration(self):
        """Test configuration validation"""
        space = SearchSpace()
        generator = ConfigurationGenerator(space)

        # Valid configuration
        valid_config = {
            "pipeline": {
                "nodes": [{"id": "test", "type": "test"}],
                "edges": []
            }
        }
        assert generator.validate_configuration(valid_config)

        # Invalid configuration (missing pipeline)
        invalid_config = {"metadata": {}}
        assert not generator.validate_configuration(invalid_config)

    def test_save_configurations(self):
        """Test saving configurations to files"""
        space = SearchSpace()
        space.define_chunking_space(["fixed"], [256])
        generator = ConfigurationGenerator(space)

        configs = generator.generate_all_configurations()

        with tempfile.TemporaryDirectory() as tmpdir:
            generator.save_configurations(configs, tmpdir)
            saved_files = list(Path(tmpdir).glob("*.yaml"))
            assert len(saved_files) == len(configs)


class TestResultManager:
    """Test result management"""

    def test_create_result_manager(self):
        """Test creating result manager"""
        manager = ResultManager()
        assert len(manager.results) == 0
        assert manager.best_result is None

    def test_add_result(self):
        """Test adding result"""
        manager = ResultManager()
        result = {
            "config_id": "test1",
            "score": 0.8,
            "cost": 0.1,
            "evaluation_time": 1.0,
            "parameters": {},
            "metrics": {}
        }
        manager.add_result(result)
        assert len(manager.results) == 1
        assert manager.best_result == result

    def test_update_best_result(self):
        """Test updating best result"""
        manager = ResultManager()
        result1 = {"config_id": "test1", "score": 0.8, "cost": 0.1,
                  "evaluation_time": 1.0, "parameters": {}, "metrics": {}}
        result2 = {"config_id": "test2", "score": 0.9, "cost": 0.1,
                  "evaluation_time": 1.0, "parameters": {}, "metrics": {}}

        manager.add_result(result1)
        assert manager.best_result["config_id"] == "test1"

        manager.add_result(result2)
        assert manager.best_result["config_id"] == "test2"

    def test_get_top_configurations(self):
        """Test getting top configurations"""
        manager = ResultManager()
        for i in range(10):
            manager.add_result({
                "config_id": f"test{i}",
                "score": i / 10,
                "cost": 0.1,
                "evaluation_time": 1.0,
                "parameters": {},
                "metrics": {}
            })

        top_5 = manager.get_top_configurations(5)
        assert len(top_5) == 5
        assert top_5[0]["score"] == 0.9
        assert top_5[4]["score"] == 0.5

    def test_compare_configurations(self):
        """Test comparing configurations"""
        manager = ResultManager()
        result1 = {"config_id": "test1", "score": 0.8, "cost": 0.1,
                  "evaluation_time": 1.0, "parameters": {}, "metrics": {}}
        result2 = {"config_id": "test2", "score": 0.9, "cost": 0.15,
                  "evaluation_time": 1.0, "parameters": {}, "metrics": {}}

        manager.add_result(result1)
        manager.add_result(result2)

        comparison = manager.compare_configurations("test1", "test2")
        assert comparison["score_difference"] == 0.1
        assert comparison["relative_improvement"] == 12.5
        assert comparison["cost_difference"] == 0.05

    def test_save_and_load_results(self):
        """Test saving and loading results"""
        manager = ResultManager()
        result = {"config_id": "test1", "score": 0.8, "cost": 0.1,
                 "evaluation_time": 1.0, "parameters": {}, "metrics": {}}
        manager.add_result(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager.results_dir = Path(tmpdir)
            manager.save_results("test_results.json")

            # Load into new manager
            new_manager = ResultManager(tmpdir)
            new_manager.load_results("test_results.json")
            assert len(new_manager.results) == 1
            assert new_manager.best_result["config_id"] == "test1"

    def test_export_to_dataframe(self):
        """Test exporting to DataFrame"""
        manager = ResultManager()
        for i in range(3):
            manager.add_result({
                "config_id": f"test{i}",
                "score": i / 10,
                "cost": 0.1,
                "evaluation_time": 1.0,
                "parameters": {"chunking": {"strategy": "fixed", "size": 256}},
                "metrics": {"accuracy": 0.8}
            })

        df = manager.export_to_dataframe()
        assert len(df) == 3
        assert "score" in df.columns
        assert "chunking_strategy" in df.columns
        assert "chunking_size" in df.columns

    def test_find_pareto_optimal(self):
        """Test finding Pareto optimal configurations"""
        manager = ResultManager()
        # Add configurations with different score/cost trade-offs
        manager.add_result({"config_id": "cheap_bad", "score": 0.6, "cost": 0.1,
                          "evaluation_time": 1.0, "parameters": {}, "metrics": {}})
        manager.add_result({"config_id": "expensive_good", "score": 0.9, "cost": 0.5,
                          "evaluation_time": 1.0, "parameters": {}, "metrics": {}})
        manager.add_result({"config_id": "balanced", "score": 0.8, "cost": 0.2,
                          "evaluation_time": 1.0, "parameters": {}, "metrics": {}})
        manager.add_result({"config_id": "dominated", "score": 0.7, "cost": 0.3,
                          "evaluation_time": 1.0, "parameters": {}, "metrics": {}})

        pareto = manager.find_pareto_optimal()
        pareto_ids = [p["config_id"] for p in pareto]

        # cheap_bad, expensive_good, and balanced should be on Pareto frontier
        assert "cheap_bad" in pareto_ids
        assert "expensive_good" in pareto_ids
        assert "balanced" in pareto_ids
        # dominated should not be on Pareto frontier (worse than balanced)
        assert "dominated" not in pareto_ids


class TestStatisticalComparison:
    """Test statistical comparison"""

    def test_create_statistical_comparison(self):
        """Test creating statistical comparison"""
        comp = StatisticalComparison(confidence_level=0.95)
        assert comp.confidence_level == 0.95
        assert comp.alpha == 0.05

    def test_paired_t_test(self):
        """Test paired t-test"""
        comp = StatisticalComparison()
        scores_a = [0.7, 0.8, 0.75, 0.82, 0.78]
        scores_b = [0.85, 0.88, 0.82, 0.90, 0.86]

        result = comp.paired_t_test(scores_a, scores_b)
        assert result.test_name == "Paired t-test"
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant
        assert result.effect_size > 0  # B is better than A

    def test_independent_t_test(self):
        """Test independent t-test"""
        comp = StatisticalComparison()
        scores_a = [0.7, 0.8, 0.75, 0.82, 0.78]
        scores_b = [0.85, 0.88, 0.82, 0.90, 0.86, 0.87]

        result = comp.independent_t_test(scores_a, scores_b)
        assert "Independent t-test" in result.test_name
        assert result.effect_size > 0  # B is better than A

    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test"""
        comp = StatisticalComparison()
        scores_a = [0.7, 0.8, 0.75, 0.82, 0.78]
        scores_b = [0.85, 0.88, 0.82, 0.90, 0.86]

        result = comp.mann_whitney_u_test(scores_a, scores_b)
        assert result.test_name == "Mann-Whitney U test"
        assert result.p_value is not None

    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction"""
        comp = StatisticalComparison()
        p_values = [0.01, 0.04, 0.03, 0.02, 0.06]

        # Bonferroni correction
        bonf_results = comp.multiple_comparison_correction(p_values, "bonferroni")
        assert bonf_results[0]  # 0.01 < 0.05/5 = 0.01
        assert not bonf_results[1]  # 0.04 > 0.01

        # Holm correction
        holm_results = comp.multiple_comparison_correction(p_values, "holm")
        assert sum(holm_results) >= sum(bonf_results)  # Holm is less conservative

    def test_calculate_cohens_d(self):
        """Test Cohen's d calculation"""
        comp = StatisticalComparison()
        group1 = [1, 2, 3, 4, 5]
        group2 = [3, 4, 5, 6, 7]

        d = comp.calculate_cohens_d(group1, group2)
        assert d > 0  # Group 2 has higher mean
        assert 0.5 < abs(d) < 1.5  # Reasonable effect size

    def test_calculate_required_sample_size(self):
        """Test sample size calculation"""
        comp = StatisticalComparison()
        n = comp.calculate_required_sample_size(effect_size=0.5, power=0.8)
        assert n > 0
        assert n < 1000  # Reasonable sample size

    def test_analyze_variance(self):
        """Test variance analysis"""
        import pandas as pd
        comp = StatisticalComparison()

        df = pd.DataFrame({
            "config_id": ["a", "a", "b", "b"],
            "score": [0.8, 0.82, 0.9, 0.92]
        })

        analysis = comp.analyze_variance(df, "score")
        assert "total_variance" in analysis
        assert "mean" in analysis
        assert "std" in analysis
        assert "cv" in analysis


class TestGridSearchOptimizer:
    """Test grid search optimizer"""

    def test_create_optimizer(self):
        """Test creating grid search optimizer"""
        space = SearchSpace()
        evaluator = Mock(return_value={"metrics": {}, "cost": 0.1})
        optimizer = GridSearchOptimizer(space, evaluator, budget_limit=5.0)
        assert optimizer.search_space == space
        assert optimizer.budget_limit == 5.0

    def test_evaluate_configuration(self):
        """Test evaluating single configuration"""
        space = SearchSpace()
        eval_result = {"metrics": {"accuracy": 0.8}, "cost": 0.1}
        evaluator = Mock(return_value=eval_result)
        optimizer = GridSearchOptimizer(space, evaluator)

        config = {
            "metadata": {"config_id": "test1", "parameters": {}},
            "pipeline": {}
        }
        result = optimizer._evaluate_configuration(config)

        assert result["config_id"] == "test1"
        assert result["cost"] == 0.1
        assert optimizer.total_cost == 0.1
        assert optimizer.configurations_evaluated == 1

    def test_budget_limit(self):
        """Test budget limit enforcement"""
        space = SearchSpace()
        space.define_chunking_space(["fixed", "semantic"], [256, 512])

        eval_result = {"metrics": {"accuracy": 0.8}, "cost": 2.0}
        evaluator = Mock(return_value=eval_result)

        optimizer = GridSearchOptimizer(space, evaluator, budget_limit=3.0)
        results = optimizer.search()

        # Should stop after 1 config due to budget
        assert optimizer.total_cost <= 3.0
        assert optimizer.configurations_evaluated <= 2

    def test_early_stopping(self):
        """Test early stopping on poor performance"""
        space = SearchSpace()
        space.define_chunking_space(["fixed"] * 10, [256])  # 10 configs

        # Return poor scores
        eval_result = {"metrics": {"accuracy": 0.1}, "cost": 0.1}
        evaluator = Mock(return_value=eval_result)

        optimizer = GridSearchOptimizer(
            space, evaluator,
            early_stopping_threshold=0.5,
            budget_limit=100
        )

        # Mock the score calculation to return low scores
        optimizer._calculate_score = Mock(return_value=0.1)

        results = optimizer.search()
        assert optimizer.is_stopped  # Should stop early

    def test_checkpoint_save_and_load(self):
        """Test checkpoint saving and loading"""
        space = SearchSpace()
        evaluator = Mock(return_value={"metrics": {}, "cost": 0.1})

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = GridSearchOptimizer(
                space, evaluator,
                checkpoint_dir=tmpdir
            )

            # Add some results
            optimizer.configurations_evaluated = 5
            optimizer.total_cost = 0.5
            optimizer._save_checkpoint()

            # Create new optimizer and load checkpoint
            new_optimizer = GridSearchOptimizer(
                space, evaluator,
                checkpoint_dir=tmpdir
            )
            new_optimizer._load_checkpoint()

            assert new_optimizer.configurations_evaluated == 5
            assert new_optimizer.total_cost == 0.5

    def test_parameter_importance_analysis(self):
        """Test parameter importance analysis"""
        space = SearchSpace()
        evaluator = Mock(return_value={"metrics": {}, "cost": 0.1})
        optimizer = GridSearchOptimizer(space, evaluator)

        # Add mock results with different parameters
        for i in range(10):
            result = {
                "config_id": f"test{i}",
                "score": 0.5 + (i * 0.05),  # Increasing scores
                "cost": 0.1,
                "parameters": {
                    "chunking": {"strategy": "fixed" if i < 5 else "semantic",
                               "size": 256 if i % 2 == 0 else 512}
                }
            }
            optimizer.result_manager.add_result(result)

        importance = optimizer._analyze_parameter_importance()
        assert len(importance) > 0
        # Strategy should have higher importance (changes at i=5 with score jump)
        assert "chunking.strategy" in importance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])