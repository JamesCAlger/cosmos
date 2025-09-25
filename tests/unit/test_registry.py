"""Unit tests for component registry"""

import pytest
from autorag.pipeline.registry import ComponentRegistry, get_registry
from autorag.components.base import Component


class TestComponentRegistry:
    """Test component registry functionality"""

    def test_register_and_get_component(self):
        """Test registering and retrieving components"""
        registry = ComponentRegistry()

        # Create a mock component class
        class TestComponent(Component):
            def process(self, *args, **kwargs):
                return "test"

        # Register component
        registry.register("test_type", "test_component", TestComponent, "1.0.0")

        # Get component by name
        component_class = registry.get_component("test_type", "test_component")
        assert component_class == TestComponent

        # Get component with version
        component_class = registry.get_component("test_type", "test_component", "1.0.0")
        assert component_class == TestComponent

    def test_create_component(self):
        """Test creating component instances"""
        registry = ComponentRegistry()

        class TestComponent(Component):
            def process(self, *args, **kwargs):
                return self.config.get("value", "default")

        registry.register("test_type", "test_component", TestComponent)

        # Create component with config
        component = registry.create_component("test_type", "test_component",
                                             {"value": "custom"})
        assert component.process() == "custom"

    def test_list_components(self):
        """Test listing registered components"""
        registry = ComponentRegistry()

        class TestComponent1(Component):
            def process(self, *args, **kwargs):
                pass

        class TestComponent2(Component):
            def process(self, *args, **kwargs):
                pass

        registry.register("type1", "comp1", TestComponent1)
        registry.register("type1", "comp2", TestComponent2)
        registry.register("type2", "comp3", TestComponent1)

        # List all components
        all_components = registry.list_components()
        assert "type1" in all_components
        assert "type2" in all_components
        assert "comp1" in all_components["type1"]
        assert "comp2" in all_components["type1"]

        # List specific type
        type1_components = registry.list_components("type1")
        assert "type1" in type1_components
        assert len(type1_components) == 1

    def test_component_not_found(self):
        """Test error when component not found"""
        registry = ComponentRegistry()

        with pytest.raises(ValueError, match="Unknown component type"):
            registry.get_component("nonexistent_type", "name")

        with pytest.raises(ValueError, match="Component not found"):
            registry.register("test_type", "exists", Component)
            registry.get_component("test_type", "nonexistent")

    def test_clear_registry(self):
        """Test clearing the registry"""
        registry = ComponentRegistry()

        class TestComponent(Component):
            def process(self, *args, **kwargs):
                pass

        registry.register("test", "comp", TestComponent)
        assert registry.list_components()

        registry.clear()
        assert not registry.list_components()

    def test_global_registry(self):
        """Test global registry singleton"""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])