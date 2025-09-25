"""Component registry for dynamic component loading"""

from typing import Type, Dict, Any, Optional, List
from loguru import logger
import importlib
import inspect
from ..components.base import Component


class ComponentRegistry:
    """Registry for managing and discovering pipeline components"""

    def __init__(self):
        self._components: Dict[str, Dict[str, Type[Component]]] = {}
        logger.info("Component registry initialized")

    def register(self, component_type: str, name: str,
                component_class: Type[Component], version: str = "1.0.0") -> None:
        """Register a component class"""
        if component_type not in self._components:
            self._components[component_type] = {}

        full_name = f"{name}:{version}"
        self._components[component_type][full_name] = component_class

        # Also register without version for convenience
        self._components[component_type][name] = component_class

        logger.info(f"Registered component: {component_type}/{name} (v{version})")

    def get_component(self, component_type: str, name: str,
                     version: Optional[str] = None) -> Type[Component]:
        """Get a registered component class"""
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")

        if version:
            full_name = f"{name}:{version}"
            if full_name in self._components[component_type]:
                return self._components[component_type][full_name]

        if name in self._components[component_type]:
            return self._components[component_type][name]

        raise ValueError(f"Component not found: {component_type}/{name}")

    def create_component(self, component_type: str, name: str,
                        config: Dict[str, Any] = None,
                        version: Optional[str] = None) -> Component:
        """Create an instance of a registered component"""
        component_class = self.get_component(component_type, name, version)
        return component_class(config or {})

    def list_components(self, component_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered components"""
        if component_type:
            if component_type in self._components:
                return {component_type: list(self._components[component_type].keys())}
            return {}

        return {
            comp_type: list(components.keys())
            for comp_type, components in self._components.items()
        }

    def discover_components(self, package_path: str) -> None:
        """Auto-discover components from a package"""
        try:
            module = importlib.import_module(package_path)

            # Scan module for Component subclasses
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Component) and obj != Component:
                    # Determine component type from base class
                    for base in obj.__bases__:
                        if base.__name__ in ["Chunker", "Embedder", "VectorStore",
                                            "Retriever", "Reranker", "Generator", "PostProcessor"]:
                            component_type = base.__name__.lower()
                            self.register(component_type, obj.__name__.lower(), obj)
                            break

            logger.info(f"Discovered components from {package_path}")

        except ImportError as e:
            logger.error(f"Failed to discover components from {package_path}: {e}")

    def clear(self) -> None:
        """Clear all registered components"""
        self._components.clear()
        logger.info("Component registry cleared")


# Global registry instance
_global_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry"""
    return _global_registry


def register_default_components():
    """Register all default components"""
    registry = get_registry()

    # Import component modules
    from ..components.chunkers.fixed_size import FixedSizeChunker
    from ..components.chunkers.mock import MockChunker
    from ..components.chunkers.semantic import SemanticChunker
    from ..components.chunkers.sliding_window import SlidingWindowChunker
    from ..components.chunkers.hierarchical import HierarchicalChunker
    from ..components.chunkers.document_aware import DocumentAwareChunker

    from ..components.embedders.openai import OpenAIEmbedder
    from ..components.embedders.mock import MockEmbedder

    from ..components.retrievers.faiss_store import FAISSVectorStore
    from ..components.retrievers.bm25 import BM25Retriever
    from ..components.retrievers.dense import DenseRetriever
    from ..components.retrievers.hybrid import HybridRetriever

    from ..components.rerankers.cross_encoder import CrossEncoderReranker

    from ..components.generators.openai import OpenAIGenerator
    from ..components.generators.mock import MockGenerator

    # Register chunkers
    registry.register("chunker", "fixed_size", FixedSizeChunker)
    registry.register("chunker", "mock", MockChunker)
    registry.register("chunker", "semantic", SemanticChunker)
    registry.register("chunker", "sliding_window", SlidingWindowChunker)
    registry.register("chunker", "hierarchical", HierarchicalChunker)
    registry.register("chunker", "document_aware", DocumentAwareChunker)

    # Register embedders
    registry.register("embedder", "openai", OpenAIEmbedder)
    registry.register("embedder", "mock", MockEmbedder)

    # Register vector stores
    registry.register("vectorstore", "faiss", FAISSVectorStore)

    # Register retrievers
    registry.register("retriever", "bm25", BM25Retriever)
    registry.register("retriever", "dense", DenseRetriever)
    registry.register("retriever", "hybrid", HybridRetriever)

    # Register rerankers
    registry.register("reranker", "cross_encoder", CrossEncoderReranker)

    # Register generators
    registry.register("generator", "openai", OpenAIGenerator)
    registry.register("generator", "mock", MockGenerator)

    logger.info("Default components registered")