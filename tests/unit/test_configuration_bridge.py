"""Tests for configuration bridge"""

import pytest
from typing import Dict, Any

from autorag.optimization.configuration_bridge import ConfigurationBridge
from autorag.components.descriptors import (
    ComponentDescriptor,
    ParamSpec,
    ParamType,
    SelfDescribingComponent
)
from autorag.components.auto_register import auto_register
from autorag.components.base import Component, Chunker, Retriever, Generator, Reranker
from autorag.pipeline.registry import get_registry


class TestConfigurationBridge:
    """Test configuration bridge between optimization and pipeline"""

    def setup_method(self):
        """Setup for each test"""
        # Clear and setup registry
        registry = get_registry()
        registry.clear()

        # Register test components
        self._register_test_components()

        # Initialize bridge
        self.bridge = ConfigurationBridge()

    def _register_test_components(self):
        """Register test components"""

        @auto_register("chunker", "fixed")
        class FixedChunker(Chunker, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="FixedChunker",
                    type="chunker",
                    parameters={
                        "chunk_size": ParamSpec(
                            type=ParamType.INT,
                            default=256,
                            min_value=50,
                            max_value=1000,
                            tunable=True
                        ),
                        "overlap": ParamSpec(
                            type=ParamType.INT,
                            default=0,
                            min_value=0,
                            max_value=100,
                            tunable=True
                        )
                    },
                    inputs=["documents"],
                    outputs=["chunks"],
                    tunable_params=["chunk_size", "overlap"],
                    estimated_cost=0.0,
                    estimated_latency=0.1
                )

            def chunk(self, documents):
                return []

        @auto_register("retriever", "dense")
        class DenseRetriever(Retriever, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="DenseRetriever",
                    type="retriever",
                    parameters={
                        "top_k": ParamSpec(
                            type=ParamType.INT,
                            default=5,
                            min_value=1,
                            max_value=20,
                            tunable=True
                        ),
                        "model": ParamSpec(
                            type=ParamType.CHOICE,
                            default="ada-002",
                            choices=["ada-002", "e5-base"],
                            tunable=True
                        )
                    },
                    inputs=["query", "chunks"],
                    outputs=["retrieved_documents"],
                    tunable_params=["top_k", "model"],
                    estimated_cost=0.001,
                    estimated_latency=0.5
                )

            def retrieve(self, query, top_k=5):
                return []

            def process(self, *args, **kwargs):
                return self.retrieve(*args, **kwargs)

        @auto_register("generator", "openai")
        class OpenAIGenerator(Generator, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="OpenAIGenerator",
                    type="generator",
                    parameters={
                        "temperature": ParamSpec(
                            type=ParamType.FLOAT,
                            default=0.7,
                            min_value=0.0,
                            max_value=1.0,
                            tunable=True
                        ),
                        "max_tokens": ParamSpec(
                            type=ParamType.INT,
                            default=150,
                            min_value=50,
                            max_value=500,
                            tunable=True
                        ),
                        "model": ParamSpec(
                            type=ParamType.STRING,
                            default="gpt-3.5-turbo",
                            tunable=False
                        )
                    },
                    inputs=["query", "retrieved_documents"],
                    outputs=["answer"],
                    tunable_params=["temperature", "max_tokens"],
                    estimated_cost=0.01,
                    estimated_latency=1.0
                )

            def generate(self, query, context):
                return "answer"

            def process(self, *args, **kwargs):
                return self.generate(*args, **kwargs)

        @auto_register("reranker", "cross_encoder")
        class CrossEncoderReranker(Reranker, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="CrossEncoderReranker",
                    type="reranker",
                    parameters={
                        "top_k": ParamSpec(
                            type=ParamType.INT,
                            default=10,
                            min_value=5,
                            max_value=50,
                            tunable=True
                        ),
                        "enabled": ParamSpec(
                            type=ParamType.BOOL,
                            default=True,
                            tunable=True
                        )
                    },
                    inputs=["query", "retrieved_documents"],
                    outputs=["reranked_documents"],
                    tunable_params=["top_k", "enabled"],
                    requires=["retriever"],
                    estimated_cost=0.002,
                    estimated_latency=0.3
                )

            def rerank(self, query, documents, top_k=10):
                return documents[:top_k]

            def process(self, *args, **kwargs):
                return self.rerank(*args, **kwargs)

    def test_params_to_pipeline(self):
        """Test converting flat parameters to pipeline configuration"""
        params = {
            "chunker.fixed.chunk_size": 512,
            "chunker.fixed.overlap": 50,
            "retriever.dense.top_k": 10,
            "retriever.dense.model": "e5-base",
            "generator.openai.temperature": 0.5,
            "generator.openai.max_tokens": 200
        }

        pipeline_config = self.bridge.params_to_pipeline(params)

        # Check structure
        assert "pipeline" in pipeline_config
        assert "nodes" in pipeline_config["pipeline"]
        assert "edges" in pipeline_config["pipeline"]

        # Check nodes were created
        nodes = pipeline_config["pipeline"]["nodes"]
        assert len(nodes) == 3  # chunker, retriever, generator

        # Check parameters were applied
        chunker_node = next(n for n in nodes if n["type"] == "chunker")
        assert chunker_node["config"]["chunk_size"] == 512
        assert chunker_node["config"]["overlap"] == 50

        retriever_node = next(n for n in nodes if n["type"] == "retriever")
        assert retriever_node["config"]["top_k"] == 10
        assert retriever_node["config"]["model"] == "e5-base"

        generator_node = next(n for n in nodes if n["type"] == "generator")
        assert generator_node["config"]["temperature"] == 0.5
        assert generator_node["config"]["max_tokens"] == 200

    def test_pipeline_to_params(self):
        """Test extracting parameters from pipeline configuration"""
        pipeline_config = {
            "pipeline": {
                "nodes": [
                    {
                        "id": "chunker_1",
                        "type": "chunker",
                        "component": "fixed",
                        "config": {
                            "chunk_size": 512,
                            "overlap": 50,
                            "some_fixed_param": "value"  # Non-tunable
                        }
                    },
                    {
                        "id": "retriever_1",
                        "type": "retriever",
                        "component": "dense",
                        "config": {
                            "top_k": 10,
                            "model": "ada-002"
                        }
                    }
                ],
                "edges": []
            }
        }

        params = self.bridge.pipeline_to_params(pipeline_config)

        # Check extracted parameters
        assert "chunker.fixed.chunk_size" in params
        assert params["chunker.fixed.chunk_size"] == 512
        assert "chunker.fixed.overlap" in params
        assert params["chunker.fixed.overlap"] == 50

        assert "retriever.dense.top_k" in params
        assert params["retriever.dense.top_k"] == 10
        assert "retriever.dense.model" in params
        assert params["retriever.dense.model"] == "ada-002"

        # Non-tunable parameter should not be extracted
        assert "chunker.fixed.some_fixed_param" not in params

    def test_generate_search_space(self):
        """Test search space generation from component descriptors"""
        components = ["chunker/fixed", "retriever/dense", "generator/openai"]

        search_space = self.bridge.generate_search_space(components)

        # Check chunker parameters
        assert "chunker.fixed.chunk_size" in search_space
        assert search_space["chunker.fixed.chunk_size"]["type"] == "integer"
        assert search_space["chunker.fixed.chunk_size"]["min"] == 50
        assert search_space["chunker.fixed.chunk_size"]["max"] == 1000

        # Check retriever parameters
        assert "retriever.dense.top_k" in search_space
        assert search_space["retriever.dense.top_k"]["type"] == "integer"

        assert "retriever.dense.model" in search_space
        assert search_space["retriever.dense.model"]["type"] == "categorical"
        assert search_space["retriever.dense.model"]["choices"] == ["ada-002", "e5-base"]

        # Check generator parameters
        assert "generator.openai.temperature" in search_space
        assert search_space["generator.openai.temperature"]["type"] == "continuous"

        # Non-tunable parameter should not appear
        assert "generator.openai.model" not in search_space

    def test_conditional_components(self):
        """Test handling of conditional components"""
        params = {
            "chunker.fixed.chunk_size": 256,
            "retriever.dense.top_k": 5,
            "reranker.cross_encoder.enabled": True,
            "reranker.cross_encoder.top_k": 20,
            "generator.openai.temperature": 0.7
        }

        pipeline_config = self.bridge.params_to_pipeline(params)

        # Reranker should be included
        nodes = pipeline_config["pipeline"]["nodes"]
        assert any(n["type"] == "reranker" for n in nodes)

        # Test with reranker disabled
        params["reranker.cross_encoder.enabled"] = False
        pipeline_config = self.bridge.params_to_pipeline(params)

        # Reranker should not be included
        nodes = pipeline_config["pipeline"]["nodes"]
        assert not any(n["type"] == "reranker" for n in nodes)

    def test_validate_parameter_combination(self):
        """Test parameter validation"""
        # Valid parameters
        valid_params = {
            "chunker.fixed.chunk_size": 256,
            "retriever.dense.top_k": 5,
            "generator.openai.temperature": 0.7
        }

        is_valid, errors = self.bridge.validate_parameter_combination(valid_params)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid parameters (out of range)
        invalid_params = {
            "chunker.fixed.chunk_size": 2000,  # Max is 1000
            "retriever.dense.top_k": 50,  # Max is 20
            "generator.openai.temperature": 2.0  # Max is 1.0
        }

        is_valid, errors = self.bridge.validate_parameter_combination(invalid_params)
        assert is_valid is False
        assert len(errors) > 0
        assert any("chunker.fixed" in e for e in errors)
        assert any("retriever.dense" in e for e in errors)
        assert any("generator.openai" in e for e in errors)

    def test_component_order_inference(self):
        """Test automatic component ordering"""
        # Components in random order
        params = {
            "generator.openai.temperature": 0.7,
            "chunker.fixed.chunk_size": 256,
            "retriever.dense.top_k": 5,
            "reranker.cross_encoder.top_k": 10,
            "reranker.cross_encoder.enabled": True
        }

        pipeline_config = self.bridge.params_to_pipeline(params)

        # Check nodes are in sensible order
        nodes = pipeline_config["pipeline"]["nodes"]
        node_types = [n["type"] for n in nodes]

        # Chunker should come before retriever
        chunker_idx = node_types.index("chunker")
        retriever_idx = node_types.index("retriever")
        assert chunker_idx < retriever_idx

        # Retriever should come before reranker
        reranker_idx = node_types.index("reranker")
        assert retriever_idx < reranker_idx

        # Reranker should come before generator
        generator_idx = node_types.index("generator")
        assert reranker_idx < generator_idx

    def test_cost_and_latency_estimation(self):
        """Test cost and latency estimation"""
        params = {
            "chunker.fixed.chunk_size": 256,  # cost: 0.0, latency: 0.1
            "retriever.dense.top_k": 5,  # cost: 0.001, latency: 0.5
            "reranker.cross_encoder.enabled": True,  # cost: 0.002, latency: 0.3
            "reranker.cross_encoder.top_k": 10,
            "generator.openai.temperature": 0.7  # cost: 0.01, latency: 1.0
        }

        estimated_cost = self.bridge.estimate_configuration_cost(params)
        assert estimated_cost == pytest.approx(0.013)  # 0.0 + 0.001 + 0.002 + 0.01

        estimated_latency = self.bridge.estimate_configuration_latency(params)
        assert estimated_latency == pytest.approx(1.9)  # 0.1 + 0.5 + 0.3 + 1.0

    def test_round_trip_conversion(self):
        """Test that params -> pipeline -> params preserves information"""
        original_params = {
            "chunker.fixed.chunk_size": 512,
            "chunker.fixed.overlap": 25,
            "retriever.dense.top_k": 8,
            "retriever.dense.model": "e5-base",
            "generator.openai.temperature": 0.3,
            "generator.openai.max_tokens": 250
        }

        # Convert to pipeline
        pipeline_config = self.bridge.params_to_pipeline(original_params)

        # Convert back to params
        extracted_params = self.bridge.pipeline_to_params(pipeline_config)

        # Check that all original tunable parameters are preserved
        for param_name, param_value in original_params.items():
            assert param_name in extracted_params
            assert extracted_params[param_name] == param_value

    def test_search_space_with_conditionals(self):
        """Test search space generation with conditional parameters"""
        components = ["chunker/fixed", "retriever/dense", "reranker/cross_encoder"]

        search_space = self.bridge.generate_search_space(
            components,
            include_conditionals=True
        )

        # Should include enabled flag for reranker
        assert "reranker.cross_encoder.enabled" in search_space
        assert search_space["reranker.cross_encoder.enabled"]["type"] == "categorical"
        assert search_space["reranker.cross_encoder.enabled"]["choices"] == [True, False]

    def test_invalid_parameter_format(self):
        """Test handling of invalid parameter formats"""
        invalid_params = {
            "invalid_format": 123,  # No component/parameter structure
            "also.invalid": 456,  # Only two parts
            "chunker.fixed.chunk_size": 256  # This one is valid
        }

        # Should still process valid parameters
        pipeline_config = self.bridge.params_to_pipeline(invalid_params)

        nodes = pipeline_config["pipeline"]["nodes"]
        # Should have chunker from valid parameter
        assert any(n["type"] == "chunker" for n in nodes)

    def test_empty_parameters(self):
        """Test handling of empty parameters"""
        params = {}

        pipeline_config = self.bridge.params_to_pipeline(params)

        # Should return valid but empty pipeline config
        assert "pipeline" in pipeline_config
        assert pipeline_config["pipeline"]["nodes"] == []
        assert pipeline_config["pipeline"]["edges"] == []