"""Configuration loader for pipeline configurations"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class ConfigLoader:
    """Load and validate pipeline configurations"""

    def __init__(self):
        self.base_configs: Dict[str, Dict[str, Any]] = {}
        logger.info("ConfigLoader initialized")

    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine format from extension
        if path.suffix in [".yaml", ".yml"]:
            config = self._load_yaml(path)
        elif path.suffix == ".json":
            config = self._load_json(path)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")

        # Process inheritance
        if "extends" in config:
            config = self._process_inheritance(config)

        # Validate configuration
        self._validate_config(config)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration"""
        with open(path, "r") as f:
            return json.load(f)

    def _process_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration inheritance"""
        extends = config.pop("extends")

        if isinstance(extends, str):
            # Load base configuration
            if extends in self.base_configs:
                base_config = self.base_configs[extends].copy()
            else:
                base_path = Path(extends)
                if base_path.exists():
                    base_config = self.load(extends)
                else:
                    raise ValueError(f"Base configuration not found: {extends}")

            # Merge configurations (child overrides parent)
            return self._merge_configs(base_config, config)

        return config

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configurations"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        # Check for required top-level keys
        if "pipeline" not in config:
            raise ValueError("Configuration must contain 'pipeline' section")

        pipeline = config["pipeline"]

        # For DAG-based pipeline
        if "nodes" in pipeline:
            self._validate_dag_pipeline(pipeline)
        # For linear pipeline (backward compatibility)
        elif "components" in pipeline:
            self._validate_linear_pipeline(pipeline)
        else:
            raise ValueError("Pipeline must contain either 'nodes' (DAG) or 'components' (linear)")

    def _validate_dag_pipeline(self, pipeline: Dict[str, Any]) -> None:
        """Validate DAG-based pipeline configuration"""
        nodes = pipeline.get("nodes", [])
        edges = pipeline.get("edges", [])

        if not nodes:
            raise ValueError("DAG pipeline must have at least one node")

        # Check node structure
        node_ids = set()
        for node in nodes:
            if "id" not in node:
                raise ValueError("Each node must have an 'id'")
            if "type" not in node:
                raise ValueError(f"Node {node['id']} must have a 'type'")
            node_ids.add(node["id"])

        # Check edge structure
        for edge in edges:
            if "from" not in edge or "to" not in edge:
                raise ValueError("Each edge must have 'from' and 'to'")

            # Validate edge references
            from_nodes = edge["from"] if isinstance(edge["from"], list) else [edge["from"]]
            to_nodes = edge["to"] if isinstance(edge["to"], list) else [edge["to"]]

            for node in from_nodes:
                if node != "input" and node not in node_ids:
                    raise ValueError(f"Edge references unknown node: {node}")

            for node in to_nodes:
                if node != "output" and node not in node_ids:
                    raise ValueError(f"Edge references unknown node: {node}")

    def _validate_linear_pipeline(self, pipeline: Dict[str, Any]) -> None:
        """Validate linear pipeline configuration"""
        components = pipeline.get("components", [])

        if not components:
            raise ValueError("Linear pipeline must have at least one component")

        for component in components:
            if "type" not in component:
                raise ValueError("Each component must have a 'type'")
            if "name" not in component:
                raise ValueError("Each component must have a 'name'")

    def register_base_config(self, name: str, config: Dict[str, Any]) -> None:
        """Register a base configuration for inheritance"""
        self.base_configs[name] = config
        logger.info(f"Registered base configuration: {name}")

    def save(self, config: Dict[str, Any], path: str) -> None:
        """Save configuration to file"""
        output_path = Path(path)

        # Determine format from extension
        if output_path.suffix in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        logger.info(f"Saved configuration to {path}")