"""Auto-registration system for self-describing components"""

from functools import wraps
from typing import Type, Optional, Dict, Any
from loguru import logger

from .descriptors import ComponentDescriptor, SelfDescribingComponent
from ..pipeline.registry import get_registry


def auto_register(component_type: str, component_name: Optional[str] = None):
    """
    Decorator for automatic component registration

    Args:
        component_type: Type of component (retriever, generator, chunker, etc.)
        component_name: Optional name for the component (defaults to class name)

    Example:
        @auto_register("retriever", "graph")
        class GraphRetriever(Retriever, SelfDescribingComponent):
            ...
    """
    def decorator(cls: Type):
        # Extract name from class if not provided
        name = component_name or cls.__name__.lower().replace(component_type, "")

        # Verify class has describe method
        if not hasattr(cls, 'describe'):
            raise ValueError(
                f"{cls.__name__} must implement describe() method to use @auto_register. "
                f"Either inherit from SelfDescribingComponent or implement describe() classmethod."
            )

        # Get descriptor
        try:
            descriptor = cls.describe()
        except Exception as e:
            raise ValueError(f"Failed to get descriptor from {cls.__name__}: {e}")

        # Validate descriptor
        if not isinstance(descriptor, ComponentDescriptor):
            raise ValueError(
                f"{cls.__name__}.describe() must return a ComponentDescriptor instance"
            )

        # Auto-register with global registry
        registry = get_registry()

        # Create enhanced registry entry with descriptor
        registry.register(
            component_type=component_type,
            name=name,
            component_class=cls,
            version=descriptor.version
        )

        # Store descriptor separately for quick access
        if not hasattr(registry, '_descriptors'):
            registry._descriptors = {}

        registry_key = f"{component_type}/{name}"
        registry._descriptors[registry_key] = descriptor

        # Add convenience attributes to the class
        cls._descriptor = descriptor
        cls._component_type = component_type
        cls._component_name = name
        cls._registry_key = registry_key

        # Add helper methods
        cls.get_descriptor = classmethod(lambda cls: descriptor)
        cls.get_component_type = classmethod(lambda cls: component_type)
        cls.get_component_name = classmethod(lambda cls: name)

        logger.info(
            f"Auto-registered component: {component_type}/{name} "
            f"(class: {cls.__name__}, version: {descriptor.version})"
        )

        # Log registration details
        logger.debug(f"  Inputs: {descriptor.inputs}")
        logger.debug(f"  Outputs: {descriptor.outputs}")
        logger.debug(f"  Tunable params: {descriptor.tunable_params}")

        return cls

    return decorator


class EnhancedComponentRegistry:
    """Enhanced registry with descriptor support"""

    def __init__(self):
        self.components: Dict[str, Type] = {}
        self.descriptors: Dict[str, ComponentDescriptor] = {}

    def register_with_descriptor(self,
                                 component_type: str,
                                 component_name: str,
                                 component_class: Type,
                                 descriptor: ComponentDescriptor) -> None:
        """Register component with its descriptor"""
        key = f"{component_type}/{component_name}"

        self.components[key] = component_class
        self.descriptors[key] = descriptor

        logger.debug(f"Registered component with descriptor: {key}")

    def get_descriptor(self, component_type: str,
                      component_name: str) -> Optional[ComponentDescriptor]:
        """Get descriptor for a component"""
        key = f"{component_type}/{component_name}"
        return self.descriptors.get(key)

    def get_descriptor_by_key(self, key: str) -> Optional[ComponentDescriptor]:
        """Get descriptor by registry key"""
        return self.descriptors.get(key)

    def get_components_by_type(self, component_type: str) -> Dict[str, Type]:
        """Get all components of a specific type"""
        result = {}
        for key, component_class in self.components.items():
            if key.startswith(f"{component_type}/"):
                name = key.split("/", 1)[1]
                result[name] = component_class
        return result

    def discover_search_space(self, component_keys: list[str]) -> Dict[str, Any]:
        """
        Discover search space from component descriptors

        Args:
            component_keys: List of component registry keys

        Returns:
            Search space dictionary for optimization
        """
        search_space = {}

        for key in component_keys:
            descriptor = self.get_descriptor_by_key(key)
            if not descriptor:
                logger.warning(f"No descriptor found for component: {key}")
                continue

            # Get component's search space
            component_space = descriptor.get_search_space()

            # Prefix parameter names with component key
            for param_name, param_space in component_space.items():
                full_param_name = f"{key}.{param_name}"
                search_space[full_param_name] = param_space

        return search_space

    def validate_component_config(self,
                                  component_type: str,
                                  component_name: str,
                                  config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate component configuration using its descriptor

        Returns:
            Tuple of (is_valid, error_messages)
        """
        descriptor = self.get_descriptor(component_type, component_name)
        if not descriptor:
            return False, [f"No descriptor found for {component_type}/{component_name}"]

        return descriptor.validate_config(config)