"""Search space definition for configuration optimization"""

from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field
from itertools import product
from loguru import logger
import yaml
import json
from pathlib import Path


@dataclass
class ParameterRange:
    """Define a parameter and its possible values"""
    name: str
    values: List[Any]
    parameter_type: str = "categorical"  # categorical, numerical, boolean
    conditions: Optional[Dict[str, Any]] = None  # Conditional dependencies

    def is_valid(self, context: Dict[str, Any]) -> bool:
        """Check if this parameter is valid given the context"""
        if not self.conditions:
            return True

        for key, required_value in self.conditions.items():
            if key not in context:
                return False
            if context[key] != required_value:
                return False
        return True


@dataclass
class ComponentSearchSpace:
    """Search space for a single component"""
    component_type: str
    parameters: List[ParameterRange] = field(default_factory=list)

    def add_parameter(self, name: str, values: List[Any],
                      parameter_type: str = "categorical",
                      conditions: Optional[Dict[str, Any]] = None):
        """Add a parameter to the search space"""
        param = ParameterRange(name, values, parameter_type, conditions)
        self.parameters.append(param)

    def get_combinations(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get all valid parameter combinations for this component"""
        # Separate conditional and non-conditional parameters
        base_params = []
        conditional_params = []

        for param in self.parameters:
            if param.conditions:
                conditional_params.append(param)
            else:
                base_params.append(param)

        # Get base combinations first
        if not base_params:
            base_combinations = [{}]
        else:
            base_names = [p.name for p in base_params]
            base_values = [p.values for p in base_params]
            base_combinations = []
            for values in product(*base_values):
                base_combinations.append(dict(zip(base_names, values)))

        # Now add conditional parameters to each base combination
        final_combinations = []
        for base_combo in base_combinations:
            # For each conditional parameter, check if it should be included
            combo_with_conditionals = base_combo.copy()

            for cond_param in conditional_params:
                if cond_param.is_valid(base_combo):
                    # Add all possible values for this conditional parameter
                    for value in cond_param.values:
                        full_combo = combo_with_conditionals.copy()
                        full_combo[cond_param.name] = value
                        final_combinations.append(full_combo)

            # If no conditional params were added, keep the base combination
            if not conditional_params or combo_with_conditionals == base_combo:
                final_combinations.append(base_combo)

        return final_combinations if final_combinations else [{}]


class SearchSpace:
    """Define the complete search space for RAG pipeline optimization"""

    def __init__(self):
        self.components: Dict[str, ComponentSearchSpace] = {}
        self.total_combinations = 0
        logger.info("SearchSpace initialized")

    def add_component(self, component_type: str) -> ComponentSearchSpace:
        """Add a component to the search space"""
        if component_type not in self.components:
            self.components[component_type] = ComponentSearchSpace(component_type)
            logger.info(f"Added component to search space: {component_type}")
        return self.components[component_type]

    def define_chunking_space(self, strategies: List[str] = None,
                             sizes: List[int] = None):
        """Define search space for chunking component"""
        chunking = self.add_component("chunking")

        if strategies is None:
            strategies = ["fixed", "semantic"]
        if sizes is None:
            sizes = [256, 512]

        chunking.add_parameter("strategy", strategies)
        chunking.add_parameter("size", sizes)

        logger.info(f"Defined chunking space: {len(strategies)} strategies, {len(sizes)} sizes")

    def define_retrieval_space(self, methods: List[str] = None,
                              top_k_values: List[int] = None):
        """Define search space for retrieval component"""
        retrieval = self.add_component("retrieval")

        if methods is None:
            methods = ["dense", "sparse", "hybrid"]
        if top_k_values is None:
            top_k_values = [3, 5]

        retrieval.add_parameter("method", methods)
        retrieval.add_parameter("top_k", top_k_values)

        # Add hybrid-specific parameters
        retrieval.add_parameter("hybrid_weight", [0.3, 0.5, 0.7],
                              conditions={"method": "hybrid"})

        logger.info(f"Defined retrieval space: {len(methods)} methods, {len(top_k_values)} top_k values")

    def define_reranking_space(self, enabled_values: List[bool] = None,
                              models: List[str] = None):
        """Define search space for reranking component"""
        reranking = self.add_component("reranking")

        if enabled_values is None:
            enabled_values = [True, False]
        if models is None:
            models = ["cross-encoder/ms-marco-MiniLM-L-6-v2"]

        reranking.add_parameter("enabled", enabled_values)
        reranking.add_parameter("model", models,
                              conditions={"enabled": True})
        reranking.add_parameter("top_k_rerank", [10, 20],
                              conditions={"enabled": True})

        logger.info(f"Defined reranking space: {len(enabled_values)} enabled options")

    def define_generation_space(self, temperatures: List[float] = None,
                               max_tokens: List[int] = None,
                               models: List[str] = None):
        """Define search space for generation component"""
        generation = self.add_component("generation")

        if temperatures is None:
            temperatures = [0, 0.3]
        if max_tokens is None:
            max_tokens = [150, 300]
        if models is None:
            models = ["gpt-3.5-turbo"]

        generation.add_parameter("temperature", temperatures)
        generation.add_parameter("max_tokens", max_tokens)
        generation.add_parameter("model", models)

        logger.info(f"Defined generation space: {len(temperatures)} temps, {len(max_tokens)} token limits")

    def define_embedding_space(self, models: List[str] = None):
        """Define search space for embedding component"""
        embedding = self.add_component("embedding")

        if models is None:
            models = ["text-embedding-ada-002"]

        embedding.add_parameter("model", models)

        logger.info(f"Defined embedding space: {len(models)} models")

    def calculate_total_combinations(self) -> int:
        """Calculate total number of possible configurations"""
        total = 1

        for component_name, component in self.components.items():
            # Get base combinations without conditions
            base_combos = []
            for param in component.parameters:
                if not param.conditions:
                    base_combos.append(len(param.values))

            component_total = 1 if not base_combos else 1
            for combo_count in base_combos:
                component_total *= combo_count

            # Estimate additional combinations from conditional parameters
            # This is a simplification; exact count requires enumeration
            conditional_multiplier = 1.5 if any(p.conditions for p in component.parameters) else 1
            component_total = int(component_total * conditional_multiplier)

            total *= component_total
            logger.debug(f"Component {component_name}: ~{component_total} combinations")

        self.total_combinations = total
        logger.info(f"Total search space: ~{total:,} configurations")
        return total

    def sample(self, n: int = 10, method: str = "random") -> List[Dict[str, Any]]:
        """Sample n configurations from the search space"""
        import random

        all_configs = list(self.enumerate_all())

        if method == "random":
            if n >= len(all_configs):
                return all_configs
            return random.sample(all_configs, n)
        elif method == "grid":
            # Return evenly spaced configurations
            step = max(1, len(all_configs) // n)
            return all_configs[::step][:n]
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def enumerate_all(self) -> List[Dict[str, Any]]:
        """Enumerate all possible configurations"""
        configurations = []

        # Get all component combinations
        component_combos = {}
        for comp_name, component in self.components.items():
            component_combos[comp_name] = component.get_combinations()

        # Create cartesian product of all component combinations
        comp_names = list(component_combos.keys())
        comp_values = [component_combos[name] for name in comp_names]

        for combo in product(*comp_values):
            config = {}
            for comp_name, comp_params in zip(comp_names, combo):
                config[comp_name] = comp_params

            # Check if configuration is valid (handle conditional dependencies)
            if self._is_valid_configuration(config):
                configurations.append(config)

        logger.info(f"Enumerated {len(configurations)} valid configurations")
        return configurations

    def _is_valid_configuration(self, config: Dict[str, Any]) -> bool:
        """Check if a configuration satisfies all constraints"""
        for comp_name, comp_params in config.items():
            if comp_name not in self.components:
                continue

            component = self.components[comp_name]
            for param in component.parameters:
                if param.conditions:
                    # Check if conditions are met
                    param_value = comp_params.get(param.name)
                    if param_value is not None:
                        # Build context from current component parameters
                        context = comp_params.copy()
                        if not param.is_valid(context):
                            return False
        return True

    def save(self, filepath: str):
        """Save search space definition to file"""
        path = Path(filepath)

        # Convert to serializable format
        space_dict = {
            "components": {}
        }

        for comp_name, component in self.components.items():
            space_dict["components"][comp_name] = {
                "parameters": []
            }
            for param in component.parameters:
                param_dict = {
                    "name": param.name,
                    "values": param.values,
                    "type": param.parameter_type
                }
                if param.conditions:
                    param_dict["conditions"] = param.conditions
                space_dict["components"][comp_name]["parameters"].append(param_dict)

        space_dict["total_combinations"] = self.calculate_total_combinations()

        # Save based on file extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(space_dict, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(space_dict, f, indent=2)

        logger.info(f"Saved search space to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SearchSpace":
        """Load search space definition from file"""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Search space file not found: {filepath}")

        # Load based on file extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                space_dict = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                space_dict = json.load(f)

        # Reconstruct search space
        search_space = cls()

        for comp_name, comp_data in space_dict.get("components", {}).items():
            component = search_space.add_component(comp_name)

            for param_data in comp_data.get("parameters", []):
                component.add_parameter(
                    name=param_data["name"],
                    values=param_data["values"],
                    parameter_type=param_data.get("type", "categorical"),
                    conditions=param_data.get("conditions")
                )

        logger.info(f"Loaded search space from {filepath}")
        return search_space

    def create_default_search_space(self) -> "SearchSpace":
        """Create the default Week 5 search space as specified in the implementation guide"""
        # As specified in the Week 5 requirements
        self.define_chunking_space(
            strategies=["fixed", "semantic"],
            sizes=[256, 512]
        )

        self.define_retrieval_space(
            methods=["dense", "sparse", "hybrid"],
            top_k_values=[3, 5]
        )

        self.define_reranking_space(
            enabled_values=[True, False],
            models=["cross-encoder/ms-marco-MiniLM-L-6-v2"]
        )

        self.define_generation_space(
            temperatures=[0, 0.3],
            max_tokens=[150, 300]
        )

        self.define_embedding_space(
            models=["text-embedding-ada-002"]
        )

        total = self.calculate_total_combinations()
        logger.info(f"Created default Week 5 search space with ~{total:,} configurations")

        return self