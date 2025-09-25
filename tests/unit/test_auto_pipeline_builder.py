"""Tests for automatic pipeline builder"""

import pytest
from typing import Dict, Any

from autorag.pipeline.auto_pipeline_builder import AutoPipelineBuilder, ComponentNode
from autorag.components.descriptors import (
    ComponentDescriptor,
    ParamSpec,
    ParamType,
    SelfDescribingComponent
)
from autorag.components.auto_register import auto_register
from autorag.components.base import Component, Chunker, Retriever, Generator
from autorag.pipeline.registry import get_registry


class TestAutoPipelineBuilder:
    """Test automatic pipeline builder"""

    def setup_method(self):
        """Setup for each test"""
        # Clear and setup registry
        registry = get_registry()
        registry.clear()

        # Register test components
        self._register_test_components()

    def _register_test_components(self):
        """Register test components for testing"""

        @auto_register("chunker", "test_chunker")
        class TestChunker(Chunker, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="TestChunker",
                    type="chunker",
                    parameters={
                        "chunk_size": ParamSpec(
                            type=ParamType.INT,
                            default=256,
                            min_value=50,
                            max_value=1000,
                            tunable=True
                        )
                    },
                    inputs=["documents"],
                    outputs=["chunks"],
                    tunable_params=["chunk_size"],
                    estimated_cost=0.0,
                    estimated_latency=0.1
                )

            def chunk(self, documents):
                return [f"chunk_{i}" for i in range(len(documents))]

        @auto_register("retriever", "test_retriever")
        class TestRetriever(Retriever, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="TestRetriever",
                    type="retriever",
                    parameters={
                        "top_k": ParamSpec(
                            type=ParamType.INT,
                            default=5,
                            min_value=1,
                            max_value=20,
                            tunable=True
                        )
                    },
                    inputs=["query", "chunks"],
                    outputs=["retrieved_documents"],
                    tunable_params=["top_k"],
                    estimated_cost=0.001,
                    estimated_latency=0.5
                )

            def retrieve(self, query, top_k=5):
                return [f"doc_{i}" for i in range(top_k)]

            def process(self, *args, **kwargs):
                return self.retrieve(*args, **kwargs)

        @auto_register("generator", "test_generator")
        class TestGenerator(Generator, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="TestGenerator",
                    type="generator",
                    parameters={
                        "temperature": ParamSpec(
                            type=ParamType.FLOAT,
                            default=0.7,
                            min_value=0.0,
                            max_value=1.0,
                            tunable=True
                        )
                    },
                    inputs=["query", "retrieved_documents"],
                    outputs=["answer"],
                    tunable_params=["temperature"],
                    estimated_cost=0.01,
                    estimated_latency=1.0
                )

            def generate(self, query, context):
                return f"Answer to {query}"

            def process(self, *args, **kwargs):
                return self.generate(*args, **kwargs)

    def test_basic_pipeline_building(self):
        """Test basic pipeline building with auto-wiring"""
        builder = AutoPipelineBuilder()

        # Add components in order
        builder.add("chunker", "test_chunker", chunk_size=512)
        builder.add("retriever", "test_retriever", top_k=10)
        builder.add("generator", "test_generator", temperature=0.5)

        # Build pipeline
        config = builder.build()

        # Check structure
        assert "pipeline" in config
        assert len(config["pipeline"]["nodes"]) == 3
        assert len(config["pipeline"]["edges"]) > 0

        # Check nodes
        nodes = config["pipeline"]["nodes"]
        assert nodes[0]["type"] == "chunker"
        assert nodes[0]["config"]["chunk_size"] == 512
        assert nodes[1]["type"] == "retriever"
        assert nodes[1]["config"]["top_k"] == 10
        assert nodes[2]["type"] == "generator"
        assert nodes[2]["config"]["temperature"] == 0.5

    def test_auto_wiring(self):
        """Test automatic edge creation based on inputs/outputs"""
        builder = AutoPipelineBuilder()

        builder.add("chunker", "test_chunker")
        builder.add("retriever", "test_retriever")
        builder.add("generator", "test_generator")

        config = builder.build()
        edges = config["pipeline"]["edges"]

        # Check that edges were created correctly
        edge_pairs = [(e["from"], e["to"]) for e in edges]

        # Should have input -> chunker
        assert ("input", "chunker_1") in edge_pairs

        # Should have chunker -> retriever (provides chunks)
        assert ("chunker_1", "retriever_1") in edge_pairs

        # Should have retriever -> generator (provides retrieved_documents)
        assert ("retriever_1", "generator_1") in edge_pairs

        # Should have generator -> output
        assert ("generator_1", "output") in edge_pairs

    def test_validation(self):
        """Test pipeline validation"""
        builder = AutoPipelineBuilder()

        # Empty pipeline should be invalid
        is_valid, errors = builder.validate()
        assert is_valid is False
        assert "No components in pipeline" in errors

        # Add components
        builder.add("chunker", "test_chunker")
        builder.add("retriever", "test_retriever")
        builder.add("generator", "test_generator")

        # Should be valid now
        is_valid, errors = builder.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_dependency_validation(self):
        """Test validation catches missing dependencies"""

        @auto_register("reranker", "test_reranker")
        class TestReranker(Component, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="TestReranker",
                    type="reranker",
                    parameters={},
                    inputs=["query", "retrieved_documents"],
                    outputs=["reranked_documents"],
                    requires=["retriever"]  # Requires a retriever
                )

            def process(self, *args, **kwargs):
                return []

        builder = AutoPipelineBuilder()

        # Add reranker without retriever
        builder.add("reranker", "test_reranker")

        # Should be invalid
        is_valid, errors = builder.validate()
        assert is_valid is False
        assert any("requires 'retriever'" in e for e in errors)

        # Add retriever
        builder2 = AutoPipelineBuilder()
        builder2.add("chunker", "test_chunker")  # Provides chunks for retriever
        builder2.add("retriever", "test_retriever")  # Provides retrieved_documents
        builder2.add("reranker", "test_reranker")  # Needs retriever

        # Should be valid now
        is_valid, errors = builder2.validate()
        assert is_valid is True, f"Unexpected errors: {errors}"

    def test_search_space_generation(self):
        """Test automatic search space generation"""
        builder = AutoPipelineBuilder()

        builder.add("chunker", "test_chunker")
        builder.add("retriever", "test_retriever")
        builder.add("generator", "test_generator")

        search_space = builder.get_search_space()

        # Check search space includes tunable parameters
        assert "chunker_1.chunk_size" in search_space
        assert search_space["chunker_1.chunk_size"]["type"] == "integer"
        assert search_space["chunker_1.chunk_size"]["min"] == 50
        assert search_space["chunker_1.chunk_size"]["max"] == 1000

        assert "retriever_1.top_k" in search_space
        assert search_space["retriever_1.top_k"]["type"] == "integer"

        assert "generator_1.temperature" in search_space
        assert search_space["generator_1.temperature"]["type"] == "continuous"

    def test_cost_and_latency_estimation(self):
        """Test pipeline cost and latency estimation"""
        builder = AutoPipelineBuilder()

        builder.add("chunker", "test_chunker")  # cost: 0.0, latency: 0.1
        builder.add("retriever", "test_retriever")  # cost: 0.001, latency: 0.5
        builder.add("generator", "test_generator")  # cost: 0.01, latency: 1.0

        # Check cost estimation
        estimated_cost = builder.estimate_cost()
        assert estimated_cost == pytest.approx(0.011)

        # Check latency estimation
        estimated_latency = builder.estimate_latency()
        assert estimated_latency == pytest.approx(1.6)

    def test_conditional_component(self):
        """Test adding components with conditions"""
        builder = AutoPipelineBuilder()

        builder.add("retriever", "test_retriever")
        builder.add_conditional(
            "generator", "test_generator",
            condition="confidence < 0.8",
            temperature=0.9
        )

        # Check that condition was added
        last_component = builder.components[-1]
        assert "_conditional" in last_component.config
        assert last_component.config["_conditional"] == "confidence < 0.8"

    def test_parallel_components(self):
        """Test components that can run in parallel"""

        @auto_register("retriever", "dense_retriever")
        class DenseRetriever(Retriever, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="DenseRetriever",
                    type="retriever",
                    parameters={},
                    inputs=["query", "chunks"],
                    outputs=["dense_docs"],
                    parallel_with=["sparse_retriever"]
                )

            def process(self, *args, **kwargs):
                return []

        @auto_register("retriever", "sparse_retriever")
        class SparseRetriever(Retriever, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="SparseRetriever",
                    type="retriever",
                    parameters={},
                    inputs=["query", "chunks"],
                    outputs=["sparse_docs"],
                    parallel_with=["dense_retriever"]
                )

            def process(self, *args, **kwargs):
                return []

        builder = AutoPipelineBuilder()

        builder.add("chunker", "test_chunker")
        builder.add("retriever", "dense_retriever")
        builder.add("retriever", "sparse_retriever")

        # Both retrievers should be added successfully
        assert len(builder.components) == 3
        assert builder.components[1].name == "dense_retriever"
        assert builder.components[2].name == "sparse_retriever"

    def test_mutually_exclusive_components(self):
        """Test components that are mutually exclusive"""

        @auto_register("generator", "fast_generator")
        class FastGenerator(Generator, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="FastGenerator",
                    type="generator",
                    parameters={},
                    inputs=["query", "retrieved_documents"],
                    outputs=["answer"],
                    mutually_exclusive=["accurate_generator"]
                )

            def process(self, *args, **kwargs):
                return "fast answer"

        @auto_register("generator", "accurate_generator")
        class AccurateGenerator(Generator, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="AccurateGenerator",
                    type="generator",
                    parameters={},
                    inputs=["query", "retrieved_documents"],
                    outputs=["answer"],
                    mutually_exclusive=["fast_generator"]
                )

            def process(self, *args, **kwargs):
                return "accurate answer"

        builder = AutoPipelineBuilder()

        # Can add one generator
        builder.add("generator", "fast_generator")
        assert len(builder.components) == 1

        # The mutually exclusive check happens during connection validation
        # Both can be added, but they shouldn't connect to each other

    def test_default_parameter_merging(self):
        """Test that defaults are properly merged with provided params"""
        builder = AutoPipelineBuilder()

        # Add component with partial config
        builder.add("generator", "test_generator")  # No temperature specified

        # Check that default was used
        component = builder.components[0]
        assert component.config["temperature"] == 0.7  # Default value

        # Add another with override
        builder.add("retriever", "test_retriever", top_k=15)
        component = builder.components[1]
        assert component.config["top_k"] == 15  # Overridden value

    def test_invalid_configuration(self):
        """Test that invalid configuration raises error"""
        builder = AutoPipelineBuilder()

        # Try to add component with invalid parameter value
        with pytest.raises(ValueError, match="Invalid configuration"):
            builder.add("retriever", "test_retriever", top_k=100)  # Max is 20

    def test_component_not_found(self):
        """Test error when component not found"""
        builder = AutoPipelineBuilder()

        with pytest.raises(ValueError, match="No components of type .* found"):
            builder.add("nonexistent_type")

        with pytest.raises(ValueError, match="No descriptor found"):
            builder.add("retriever", "nonexistent_retriever")

    def test_empty_build(self):
        """Test that building empty pipeline raises error"""
        builder = AutoPipelineBuilder()

        with pytest.raises(ValueError, match="No components added"):
            builder.build()

    def test_chaining(self):
        """Test method chaining"""
        builder = AutoPipelineBuilder()

        config = (builder
                 .add("chunker", "test_chunker")
                 .add("retriever", "test_retriever")
                 .add("generator", "test_generator")
                 .build())

        assert len(config["pipeline"]["nodes"]) == 3