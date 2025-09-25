"""Automatic pipeline builder with component auto-wiring"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from .graph import PipelineGraph, Node, Edge
from .registry import get_registry
from ..components.descriptors import ComponentDescriptor


@dataclass
class ComponentNode:
    """Enhanced node with descriptor information"""
    id: str
    type: str
    name: str
    descriptor: ComponentDescriptor
    config: Dict[str, Any]
    component_instance: Optional[Any] = None


class AutoPipelineBuilder:
    """Automatically builds and wires pipelines from components"""

    def __init__(self):
        """Initialize builder"""
        self.components: List[ComponentNode] = []
        self.registry = get_registry()
        self.graph = PipelineGraph()
        self.component_counter = defaultdict(int)

        # Track input/output types for auto-wiring
        self.producers: Dict[str, List[ComponentNode]] = defaultdict(list)
        self.consumers: Dict[str, List[ComponentNode]] = defaultdict(list)

        logger.info("AutoPipelineBuilder initialized")

    def add(self,
            component_type: str,
            component_name: Optional[str] = None,
            **params) -> 'AutoPipelineBuilder':
        """
        Add component and auto-wire connections

        Args:
            component_type: Type of component (retriever, generator, etc.)
            component_name: Optional specific component name
            **params: Component configuration parameters

        Returns:
            Self for chaining
        """
        # Find matching component in registry
        if component_name:
            registry_key = f"{component_type}/{component_name}"
        else:
            # Find first component of this type
            components = self._get_components_by_type(component_type)
            if not components:
                raise ValueError(f"No components of type '{component_type}' found in registry")
            component_name = list(components.keys())[0]
            registry_key = f"{component_type}/{component_name}"

        # Get descriptor
        descriptor = self._get_descriptor(registry_key)
        if not descriptor:
            raise ValueError(f"No descriptor found for {registry_key}")

        # Validate parameters
        merged_config = self._merge_with_defaults(descriptor, params)
        is_valid, errors = descriptor.validate_config(merged_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration for {registry_key}: {errors}")

        # Create component node
        self.component_counter[component_type] += 1
        node_id = f"{component_type}_{self.component_counter[component_type]}"

        node = ComponentNode(
            id=node_id,
            type=component_type,
            name=component_name,
            descriptor=descriptor,
            config=merged_config
        )

        # Auto-wire based on inputs/outputs
        self._auto_wire(node)

        # Track this component's inputs/outputs
        for output in descriptor.outputs:
            self.producers[output].append(node)
        for input_type in descriptor.inputs:
            self.consumers[input_type].append(node)

        self.components.append(node)

        logger.info(f"Added component: {node_id} ({component_type}/{component_name})")
        logger.debug(f"  Auto-wired connections based on inputs: {descriptor.inputs}, outputs: {descriptor.outputs}")

        return self

    def add_conditional(self,
                       component_type: str,
                       component_name: Optional[str] = None,
                       condition: Optional[str] = None,
                       **params) -> 'AutoPipelineBuilder':
        """
        Add component with conditional execution

        Args:
            component_type: Type of component
            component_name: Optional specific component name
            condition: Condition expression (e.g., "confidence < 0.8")
            **params: Component configuration parameters

        Returns:
            Self for chaining
        """
        # Add component normally
        self.add(component_type, component_name, **params)

        # Mark last component as conditional
        if condition and self.components:
            last_component = self.components[-1]
            last_component.config["_conditional"] = condition
            logger.debug(f"Added condition '{condition}' to {last_component.id}")

        return self

    def _auto_wire(self, new_node: ComponentNode) -> None:
        """Automatically connect components based on inputs/outputs"""
        descriptor = new_node.descriptor

        # Find components that produce what we need
        for input_type in descriptor.inputs:
            # Special case: 'query' typically comes from pipeline input
            if input_type == "query" and not self.producers[input_type]:
                # This will be connected to pipeline input
                continue

            # Find all producers of this input type
            producers = self.producers.get(input_type, [])
            for producer in producers:
                # Check if components can work together
                if self._can_connect(producer, new_node):
                    # Add edge in our tracking (actual graph edges created in build())
                    logger.debug(f"  Auto-wire: {producer.id} -> {new_node.id} (via {input_type})")

        # Check for parallel execution opportunities
        if descriptor.parallel_with:
            for parallel_type in descriptor.parallel_with:
                # Find components of this type to run in parallel with
                parallel_components = [
                    c for c in self.components
                    if c.type == parallel_type or c.name == parallel_type
                ]
                for comp in parallel_components:
                    logger.debug(f"  Can run in parallel with: {comp.id}")

    def _can_connect(self, producer: ComponentNode, consumer: ComponentNode) -> bool:
        """Check if two components can be connected"""
        # Check mutual exclusivity
        if producer.descriptor.mutually_exclusive:
            if consumer.type in producer.descriptor.mutually_exclusive:
                return False
            if consumer.name in producer.descriptor.mutually_exclusive:
                return False

        if consumer.descriptor.mutually_exclusive:
            if producer.type in consumer.descriptor.mutually_exclusive:
                return False
            if producer.name in consumer.descriptor.mutually_exclusive:
                return False

        # Check dependencies
        if consumer.descriptor.requires:
            # Consumer requires certain components to exist
            existing_types = {c.type for c in self.components}
            existing_names = {c.name for c in self.components}

            for requirement in consumer.descriptor.requires:
                if requirement not in existing_types and requirement not in existing_names:
                    return False

        return True

    def _find_producers(self, output_type: str) -> List[ComponentNode]:
        """Find components that produce this output"""
        return self.producers.get(output_type, [])

    def _find_consumers(self, input_type: str) -> List[ComponentNode]:
        """Find components that need this input"""
        return self.consumers.get(input_type, [])

    def _get_descriptor(self, registry_key: str) -> Optional[ComponentDescriptor]:
        """Get descriptor from registry"""
        if hasattr(self.registry, '_descriptors'):
            return self.registry._descriptors.get(registry_key)
        return None

    def _get_components_by_type(self, component_type: str) -> Dict[str, Any]:
        """Get all components of a specific type"""
        result = {}
        if hasattr(self.registry, '_components'):
            for key, component_class in self.registry._components.items():
                if key.startswith(f"{component_type}/"):
                    name = key.split("/", 1)[1]
                    result[name] = component_class
        return result

    def _merge_with_defaults(self,
                            descriptor: ComponentDescriptor,
                            params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided params with defaults from descriptor"""
        config = descriptor.get_default_config()
        config.update(params)
        return config

    def _create_pipeline_graph(self) -> Dict[str, Any]:
        """Create pipeline graph from components"""
        nodes = []
        edges = []
        edge_set = set()  # Track edges to avoid duplicates

        # Create nodes
        for component in self.components:
            nodes.append({
                "id": component.id,
                "type": component.type,
                "component": component.name,
                "config": component.config
            })

        # Create edges based on input/output matching
        for i, consumer in enumerate(self.components):
            consumer_inputs = set(consumer.descriptor.inputs)

            # Check which previous components can provide inputs
            for j, producer in enumerate(self.components[:i]):
                producer_outputs = set(producer.descriptor.outputs)

                # If there's an overlap, create edge
                if consumer_inputs & producer_outputs:
                    edge_key = (producer.id, consumer.id)
                    if edge_key not in edge_set:
                        edges.append({
                            "from": producer.id,
                            "to": consumer.id
                        })
                        edge_set.add(edge_key)

        # Handle pipeline input/output
        if self.components:
            # First component(s) that need 'query' or 'documents' get input
            first_components = []
            for comp in self.components:
                if "query" in comp.descriptor.inputs or "documents" in comp.descriptor.inputs:
                    if not any(
                        edge["to"] == comp.id
                        for edge in edges
                    ):
                        first_components.append(comp.id)

            for comp_id in first_components:
                edges.insert(0, {"from": "input", "to": comp_id})

            # Last component(s) that produce final outputs
            last_components = []
            for comp in reversed(self.components):
                if "answer" in comp.descriptor.outputs or "final_output" in comp.descriptor.outputs:
                    if not any(
                        edge["from"] == comp.id
                        for edge in edges
                    ):
                        last_components.append(comp.id)
                        break  # Usually just one final output

            for comp_id in last_components:
                edges.append({"from": comp_id, "to": "output"})

        return {
            "nodes": nodes,
            "edges": edges
        }

    def build(self) -> Dict[str, Any]:
        """
        Generate final pipeline configuration

        Returns:
            Pipeline configuration dictionary
        """
        if not self.components:
            raise ValueError("No components added to pipeline")

        pipeline_graph = self._create_pipeline_graph()

        config = {
            "pipeline": {
                "nodes": pipeline_graph["nodes"],
                "edges": pipeline_graph["edges"],
                "metadata": {
                    "auto_generated": True,
                    "component_count": len(self.components),
                    "builder_version": "1.0.0"
                }
            }
        }

        logger.info(f"Built pipeline with {len(self.components)} components and {len(pipeline_graph['edges'])} edges")

        return config

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the pipeline configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not self.components:
            errors.append("No components in pipeline")
            return False, errors

        # Check that all required inputs are satisfied
        for component in self.components:
            for input_type in component.descriptor.inputs:
                if input_type == "query":
                    continue  # Special case, comes from user

                # Special case for common inputs that typically come from pipeline input
                if input_type in ["documents", "text"]:
                    continue

                producers = self._find_producers(input_type)
                # Filter to only producers that come before this component
                component_index = self.components.index(component)
                valid_producers = [
                    p for p in producers
                    if self.components.index(p) < component_index
                ]

                if not valid_producers:
                    errors.append(
                        f"Component {component.id} requires input '{input_type}' "
                        f"but no prior component produces it"
                    )

        # Check dependency requirements
        for component in self.components:
            if component.descriptor.requires:
                # Get all components before this one (dependencies must come before)
                component_index = self.components.index(component)
                prior_components = self.components[:component_index]

                existing_types = {c.type for c in prior_components}
                existing_names = {c.name for c in prior_components}

                for requirement in component.descriptor.requires:
                    if requirement not in existing_types and requirement not in existing_names:
                        errors.append(
                            f"Component {component.id} requires '{requirement}' "
                            f"but it's not in the pipeline before this component"
                        )

        return len(errors) == 0, errors

    def get_search_space(self) -> Dict[str, Any]:
        """
        Generate search space for optimization from component descriptors

        Returns:
            Search space dictionary
        """
        search_space = {}

        for component in self.components:
            component_space = component.descriptor.get_search_space()

            for param_name, param_space in component_space.items():
                # Use component ID for unique parameter names
                full_param_name = f"{component.id}.{param_name}"
                search_space[full_param_name] = param_space

        return search_space

    def estimate_cost(self) -> float:
        """
        Estimate total pipeline cost per execution

        Returns:
            Estimated cost in dollars
        """
        total_cost = 0.0
        for component in self.components:
            total_cost += component.descriptor.estimated_cost

        return total_cost

    def estimate_latency(self) -> float:
        """
        Estimate total pipeline latency

        Returns:
            Estimated latency in seconds
        """
        # For now, assume sequential execution
        # More sophisticated analysis would consider parallelism
        total_latency = 0.0
        for component in self.components:
            total_latency += component.descriptor.estimated_latency

        return total_latency