"""Configuration bridge for translating between optimization parameters and pipeline configs"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from loguru import logger

from ..pipeline.auto_pipeline_builder import AutoPipelineBuilder
from ..pipeline.registry import get_registry
from ..components.descriptors import ComponentDescriptor


class ConfigurationBridge:
    """
    Bridges between optimization search parameters and executable pipeline configurations.

    This allows the optimization framework (Week 5) to work seamlessly with the
    DAG-based pipeline system (Week 2).
    """

    def __init__(self):
        """Initialize configuration bridge"""
        self.registry = get_registry()
        logger.info("ConfigurationBridge initialized")

    def params_to_pipeline(self, params: Dict[str, Any],
                          component_order: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert flat optimization parameters to executable pipeline configuration

        Args:
            params: Flat dictionary of parameters (e.g., {"chunker.size": 256, "retriever.top_k": 5})
            component_order: Optional list specifying component order

        Returns:
            Pipeline configuration dictionary with DAG structure

        Example:
            Input: {"chunker.fixed.chunk_size": 256, "retriever.dense.top_k": 5}
            Output: Full DAG configuration with nodes and edges
        """
        # Parse and group parameters by component
        component_configs = self._group_parameters_by_component(params)

        # Determine component order if not provided
        if not component_order:
            component_order = self._infer_component_order(component_configs)

        # Build pipeline using AutoPipelineBuilder
        builder = AutoPipelineBuilder()

        for component_key in component_order:
            if component_key not in component_configs:
                continue

            comp_type, comp_name = self._parse_component_key(component_key)
            comp_params = component_configs[component_key]

            # Handle conditional components
            if self._is_conditional_component(comp_params):
                if self._should_include_component(comp_params):
                    # Remove conditional markers before adding
                    clean_params = self._clean_conditional_params(comp_params)
                    builder.add(comp_type, comp_name, **clean_params)
            else:
                builder.add(comp_type, comp_name, **comp_params)

        # Handle empty pipeline case
        if not builder.components:
            # Return valid empty pipeline configuration
            pipeline_config = {
                "pipeline": {
                    "nodes": [],
                    "edges": [],
                    "metadata": {
                        "auto_generated": True,
                        "component_count": 0
                    }
                }
            }
        else:
            # Build and return pipeline configuration
            pipeline_config = builder.build()

        logger.info(f"Converted {len(params)} parameters to pipeline with "
                   f"{len(builder.components)} components")

        return pipeline_config

    def pipeline_to_params(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tunable parameters from pipeline configuration for optimization

        Args:
            pipeline_config: DAG-based pipeline configuration

        Returns:
            Flat dictionary of tunable parameters

        Example:
            Input: DAG config with nodes and edges
            Output: {"chunker.fixed.chunk_size": 256, "retriever.dense.top_k": 5}
        """
        params = {}

        nodes = pipeline_config.get("pipeline", {}).get("nodes", [])

        for node in nodes:
            node_type = node.get("type")
            node_component = node.get("component")
            node_config = node.get("config", {})
            node_id = node.get("id")

            # Get descriptor for this component
            registry_key = f"{node_type}/{node_component}"
            descriptor = self._get_descriptor(registry_key)

            if not descriptor:
                logger.warning(f"No descriptor found for {registry_key}, skipping")
                continue

            # Extract tunable parameters
            for param_name in descriptor.tunable_params:
                if param_name in node_config:
                    # Create flat parameter name
                    flat_param_name = f"{node_type}.{node_component}.{param_name}"
                    params[flat_param_name] = node_config[param_name]

        logger.info(f"Extracted {len(params)} tunable parameters from pipeline")

        return params

    def generate_search_space(self,
                             components: List[str],
                             include_conditionals: bool = True) -> Dict[str, Any]:
        """
        Auto-generate search space from component descriptors

        Args:
            components: List of component keys (e.g., ["chunker/fixed", "retriever/dense"])
            include_conditionals: Whether to include conditional parameters

        Returns:
            Search space dictionary for optimization
        """
        search_space = {}

        for component_key in components:
            descriptor = self._get_descriptor(component_key)

            if not descriptor:
                logger.warning(f"No descriptor found for {component_key}")
                continue

            # Get component search space
            component_space = descriptor.get_search_space()

            # Add to overall search space with prefixed names
            for param_name, param_space in component_space.items():
                full_param_name = f"{component_key.replace('/', '.')}.{param_name}"
                search_space[full_param_name] = param_space

        # Add conditional parameters if requested
        if include_conditionals:
            search_space = self._add_conditional_parameters(search_space, components)

        logger.info(f"Generated search space with {len(search_space)} parameters")

        return search_space

    def validate_parameter_combination(self,
                                      params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that a parameter combination is valid

        Args:
            params: Flat parameter dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Group parameters by component
        component_configs = self._group_parameters_by_component(params)

        for component_key, config in component_configs.items():
            comp_type, comp_name = self._parse_component_key(component_key)

            # Get descriptor
            registry_key = f"{comp_type}/{comp_name}"
            descriptor = self._get_descriptor(registry_key)

            if not descriptor:
                errors.append(f"Unknown component: {component_key}")
                continue

            # Validate configuration
            is_valid, comp_errors = descriptor.validate_config(config)
            if not is_valid:
                for error in comp_errors:
                    errors.append(f"{component_key}: {error}")

        # Check conditional parameter consistency
        cond_errors = self._validate_conditionals(params)
        errors.extend(cond_errors)

        return len(errors) == 0, errors

    def _group_parameters_by_component(self,
                                       params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Group flat parameters by component

        Example:
            Input: {"chunker.fixed.size": 256, "retriever.dense.top_k": 5}
            Output: {"chunker.fixed": {"size": 256}, "retriever.dense": {"top_k": 5}}
        """
        component_configs = defaultdict(dict)

        for param_name, param_value in params.items():
            parts = param_name.split(".", 2)

            if len(parts) < 3:
                logger.warning(f"Invalid parameter name format: {param_name}")
                continue

            comp_type, comp_name, param = parts
            component_key = f"{comp_type}.{comp_name}"
            component_configs[component_key][param] = param_value

        return dict(component_configs)

    def _parse_component_key(self, key: str) -> Tuple[str, str]:
        """
        Parse component key into type and name

        Example:
            Input: "chunker.fixed"
            Output: ("chunker", "fixed")
        """
        parts = key.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""

    def _infer_component_order(self,
                               component_configs: Dict[str, Dict]) -> List[str]:
        """
        Infer a sensible component order based on typical RAG pipeline structure
        """
        # Define typical order
        type_order = [
            "chunker",
            "preprocessor",
            "embedder",
            "retriever",
            "reranker",
            "generator",
            "postprocessor"
        ]

        ordered_components = []

        # Add components in type order
        for comp_type in type_order:
            for component_key in component_configs:
                if component_key.startswith(f"{comp_type}."):
                    ordered_components.append(component_key)

        # Add any remaining components not in standard order
        for component_key in component_configs:
            if component_key not in ordered_components:
                ordered_components.append(component_key)

        return ordered_components

    def _is_conditional_component(self, params: Dict[str, Any]) -> bool:
        """Check if component has conditional parameters"""
        return "enabled" in params or "_conditional" in params

    def _should_include_component(self, params: Dict[str, Any]) -> bool:
        """Check if conditional component should be included"""
        if "enabled" in params:
            return params["enabled"]
        return True  # Default to including

    def _clean_conditional_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove conditional markers from parameters"""
        clean = params.copy()
        conditional_keys = ["enabled", "_conditional", "_condition"]
        for key in conditional_keys:
            clean.pop(key, None)
        return clean

    def _add_conditional_parameters(self,
                                   search_space: Dict[str, Any],
                                   components: List[str]) -> Dict[str, Any]:
        """Add conditional parameters to search space"""
        # Add enabled flags for optional components
        optional_types = ["reranker", "postprocessor"]

        for component in components:
            comp_type = component.split("/")[0]
            if comp_type in optional_types:
                enabled_param = f"{component.replace('/', '.')}.enabled"
                if enabled_param not in search_space:
                    search_space[enabled_param] = {
                        "type": "categorical",
                        "choices": [True, False]
                    }

        return search_space

    def _validate_conditionals(self, params: Dict[str, Any]) -> List[str]:
        """Validate conditional parameter consistency"""
        errors = []

        # Group parameters by component
        component_configs = self._group_parameters_by_component(params)

        for component_key, config in component_configs.items():
            # Check if component is disabled but has other parameters set
            if "enabled" in config and not config["enabled"]:
                non_enabled_params = [k for k in config if k != "enabled"]
                if non_enabled_params:
                    logger.warning(
                        f"{component_key} is disabled but has parameters set: {non_enabled_params}"
                    )

        return errors

    def _get_descriptor(self, registry_key: str) -> Optional[ComponentDescriptor]:
        """Get descriptor from registry"""
        if hasattr(self.registry, '_descriptors'):
            return self.registry._descriptors.get(registry_key)
        return None

    def estimate_configuration_cost(self, params: Dict[str, Any]) -> float:
        """
        Estimate the cost of running a configuration

        Args:
            params: Flat parameter dictionary

        Returns:
            Estimated cost in dollars
        """
        total_cost = 0.0

        component_configs = self._group_parameters_by_component(params)

        for component_key in component_configs:
            comp_type, comp_name = self._parse_component_key(component_key)
            registry_key = f"{comp_type}/{comp_name}"
            descriptor = self._get_descriptor(registry_key)

            if descriptor:
                total_cost += descriptor.estimated_cost

        return total_cost

    def estimate_configuration_latency(self, params: Dict[str, Any]) -> float:
        """
        Estimate the latency of running a configuration

        Args:
            params: Flat parameter dictionary

        Returns:
            Estimated latency in seconds
        """
        total_latency = 0.0

        component_configs = self._group_parameters_by_component(params)

        for component_key in component_configs:
            comp_type, comp_name = self._parse_component_key(component_key)
            registry_key = f"{comp_type}/{comp_name}"
            descriptor = self._get_descriptor(registry_key)

            if descriptor:
                total_latency += descriptor.estimated_latency

        return total_latency