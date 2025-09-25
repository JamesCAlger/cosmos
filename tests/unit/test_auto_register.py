"""Tests for auto-registration system"""

import pytest
from typing import Dict, Any

from autorag.components.descriptors import (
    ComponentDescriptor,
    ParamSpec,
    ParamType,
    SelfDescribingComponent
)
from autorag.components.auto_register import auto_register, EnhancedComponentRegistry
from autorag.components.base import Component, Retriever
from autorag.pipeline.registry import get_registry


class TestAutoRegister:
    """Test auto-registration decorator"""

    def setup_method(self):
        """Setup for each test"""
        # Clear registry before each test
        registry = get_registry()
        registry.clear()

    def test_basic_auto_registration(self):
        """Test basic component auto-registration"""

        @auto_register("retriever", "test")
        class TestRetriever(Component, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="TestRetriever",
                    type="retriever",
                    version="1.0.0",
                    parameters={
                        "param1": ParamSpec(
                            type=ParamType.INT,
                            default=10
                        )
                    },
                    inputs=["query"],
                    outputs=["documents"]
                )

            def process(self, *args, **kwargs):
                return {"documents": []}

        # Check that component was registered
        registry = get_registry()
        component_class = registry.get_component("retriever", "test")
        assert component_class == TestRetriever

        # Check convenience attributes
        assert TestRetriever._component_type == "retriever"
        assert TestRetriever._component_name == "test"
        assert TestRetriever._registry_key == "retriever/test"

        # Check descriptor is stored
        assert hasattr(registry, '_descriptors')
        assert "retriever/test" in registry._descriptors
        descriptor = registry._descriptors["retriever/test"]
        assert descriptor.name == "TestRetriever"

    def test_auto_register_without_name(self):
        """Test auto-registration without explicit name"""

        @auto_register("chunker")
        class SemanticChunker(Component, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="SemanticChunker",
                    type="chunker",
                    parameters={}
                )

            def process(self, *args, **kwargs):
                return []

        # Name should be derived from class name
        assert SemanticChunker._component_name == "semantic"

    def test_auto_register_without_describe(self):
        """Test that components without describe() method raise error"""

        with pytest.raises(ValueError, match="must implement describe\\(\\) method"):
            @auto_register("retriever", "bad")
            class BadRetriever(Component):
                def process(self, *args, **kwargs):
                    return {}

    def test_auto_register_with_invalid_descriptor(self):
        """Test that invalid descriptor raises error"""

        with pytest.raises(ValueError, match="must return a ComponentDescriptor"):
            @auto_register("retriever", "invalid")
            class InvalidRetriever(Component):
                @classmethod
                def describe(cls):
                    return "not a descriptor"  # Wrong return type

                def process(self, *args, **kwargs):
                    return {}

    def test_helper_methods_added(self):
        """Test that helper methods are added to class"""

        @auto_register("generator", "test")
        class TestGenerator(Component, SelfDescribingComponent):
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
                            max_value=1.0
                        )
                    }
                )

            def process(self, *args, **kwargs):
                return {"answer": "test"}

        # Check helper methods
        assert TestGenerator.get_component_type() == "generator"
        assert TestGenerator.get_component_name() == "test"
        assert TestGenerator.get_descriptor().name == "TestGenerator"


class TestEnhancedComponentRegistry:
    """Test enhanced registry with descriptor support"""

    def test_register_with_descriptor(self):
        """Test registering component with descriptor"""
        registry = EnhancedComponentRegistry()

        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            parameters={
                "param": ParamSpec(type=ParamType.INT, default=10)
            }
        )

        class TestComponent(Component):
            def process(self, *args, **kwargs):
                return {}

        registry.register_with_descriptor(
            "processor",
            "test",
            TestComponent,
            descriptor
        )

        # Check registration
        assert "processor/test" in registry.components
        assert "processor/test" in registry.descriptors
        assert registry.get_descriptor("processor", "test") == descriptor

    def test_get_components_by_type(self):
        """Test getting all components of a type"""
        registry = EnhancedComponentRegistry()

        # Register multiple retrievers
        for i in range(3):
            class TestRetriever(Component):
                def process(self, *args, **kwargs):
                    return {}

            descriptor = ComponentDescriptor(
                name=f"Retriever{i}",
                type="retriever"
            )

            registry.register_with_descriptor(
                "retriever",
                f"test{i}",
                TestRetriever,
                descriptor
            )

        # Register a generator
        class TestGenerator(Component):
            def process(self, *args, **kwargs):
                return {}

        registry.register_with_descriptor(
            "generator",
            "test",
            TestGenerator,
            ComponentDescriptor(name="Generator", type="generator")
        )

        # Get retrievers
        retrievers = registry.get_components_by_type("retriever")
        assert len(retrievers) == 3
        assert "test0" in retrievers
        assert "test1" in retrievers
        assert "test2" in retrievers

        # Get generators
        generators = registry.get_components_by_type("generator")
        assert len(generators) == 1
        assert "test" in generators

    def test_discover_search_space(self):
        """Test search space discovery from descriptors"""
        registry = EnhancedComponentRegistry()

        # Register components with tunable parameters
        descriptor1 = ComponentDescriptor(
            name="Component1",
            type="processor",
            parameters={
                "int_param": ParamSpec(
                    type=ParamType.INT,
                    default=10,
                    min_value=1,
                    max_value=100,
                    tunable=True
                ),
                "fixed_param": ParamSpec(
                    type=ParamType.STRING,
                    default="fixed",
                    tunable=False
                )
            },
            tunable_params=["int_param"]
        )

        descriptor2 = ComponentDescriptor(
            name="Component2",
            type="retriever",
            parameters={
                "choice_param": ParamSpec(
                    type=ParamType.CHOICE,
                    default="a",
                    choices=["a", "b", "c"],
                    tunable=True
                )
            },
            tunable_params=["choice_param"]
        )

        registry.register_with_descriptor(
            "processor", "comp1", Component, descriptor1
        )
        registry.register_with_descriptor(
            "retriever", "comp2", Component, descriptor2
        )

        # Discover search space
        search_space = registry.discover_search_space([
            "processor/comp1",
            "retriever/comp2"
        ])

        # Check search space
        assert "processor/comp1.int_param" in search_space
        assert search_space["processor/comp1.int_param"]["type"] == "integer"
        assert search_space["processor/comp1.int_param"]["min"] == 1
        assert search_space["processor/comp1.int_param"]["max"] == 100

        assert "retriever/comp2.choice_param" in search_space
        assert search_space["retriever/comp2.choice_param"]["type"] == "categorical"
        assert search_space["retriever/comp2.choice_param"]["choices"] == ["a", "b", "c"]

        # Fixed parameter should not appear
        assert "processor/comp1.fixed_param" not in search_space

    def test_validate_component_config(self):
        """Test component configuration validation"""
        registry = EnhancedComponentRegistry()

        descriptor = ComponentDescriptor(
            name="TestComponent",
            type="processor",
            parameters={
                "required_param": ParamSpec(
                    type=ParamType.STRING,
                    default=None,
                    required=True
                ),
                "optional_param": ParamSpec(
                    type=ParamType.INT,
                    default=10,
                    min_value=0,
                    max_value=100
                )
            }
        )

        registry.register_with_descriptor(
            "processor", "test", Component, descriptor
        )

        # Valid config
        is_valid, errors = registry.validate_component_config(
            "processor", "test",
            {"required_param": "value", "optional_param": 50}
        )
        assert is_valid is True
        assert len(errors) == 0

        # Invalid config
        is_valid, errors = registry.validate_component_config(
            "processor", "test",
            {"optional_param": 200}  # Missing required, invalid value
        )
        assert is_valid is False
        assert len(errors) > 0

        # Non-existent component
        is_valid, errors = registry.validate_component_config(
            "processor", "nonexistent",
            {}
        )
        assert is_valid is False
        assert any("No descriptor found" in e for e in errors)


class TestIntegrationWithRegistry:
    """Test integration with existing registry system"""

    def setup_method(self):
        """Setup for each test"""
        registry = get_registry()
        registry.clear()

    def test_auto_register_with_existing_registry(self):
        """Test that auto-register works with existing registry"""

        @auto_register("retriever", "integrated")
        class IntegratedRetriever(Retriever, SelfDescribingComponent):
            @classmethod
            def describe(cls) -> ComponentDescriptor:
                return ComponentDescriptor(
                    name="IntegratedRetriever",
                    type="retriever",
                    parameters={
                        "top_k": ParamSpec(
                            type=ParamType.INT,
                            default=5,
                            min_value=1,
                            max_value=100,
                            tunable=True
                        )
                    },
                    inputs=["query"],
                    outputs=["documents"],
                    tunable_params=["top_k"]
                )

            def retrieve(self, query: str, top_k: int = 5):
                return [f"doc_{i}" for i in range(top_k)]

            def process(self, *args, **kwargs):
                return self.retrieve(*args, **kwargs)

        # Check registration through standard registry
        registry = get_registry()
        component = registry.create_component(
            "retriever", "integrated",
            {"top_k": 10}
        )

        assert isinstance(component, IntegratedRetriever)
        assert component.config["top_k"] == 10

        # Check descriptor is accessible
        if hasattr(registry, '_descriptors'):
            descriptor = registry._descriptors["retriever/integrated"]
            assert descriptor.name == "IntegratedRetriever"
            search_space = descriptor.get_search_space()
            assert "top_k" in search_space