"""Configuration generator for creating valid pipeline configurations from search space"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib
import json
import yaml
from loguru import logger
from .search_space import SearchSpace


class ConfigurationGenerator:
    """Generate valid pipeline configurations from search space"""

    def __init__(self, search_space: SearchSpace, base_config_path: Optional[str] = None):
        """
        Initialize configuration generator

        Args:
            search_space: The search space defining parameter ranges
            base_config_path: Optional path to base configuration template
        """
        self.search_space = search_space
        self.base_config = None

        if base_config_path:
            self.base_config = self._load_base_config(base_config_path)

        self.generated_configs = []
        logger.info("ConfigurationGenerator initialized")

    def _load_base_config(self, path: str) -> Dict[str, Any]:
        """Load base configuration template"""
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Base configuration not found: {path}")

        if config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            with open(config_path, "r") as f:
                return json.load(f)

    def generate_configuration(self, params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a complete pipeline configuration from parameters

        Args:
            params: Dictionary of component parameters from search space

        Returns:
            Complete pipeline configuration
        """
        # Start with base config or create minimal structure
        if self.base_config:
            config = self._deep_copy(self.base_config)
        else:
            config = self._create_minimal_config()

        # Apply parameters to configuration
        config = self._apply_chunking_params(config, params.get("chunking", {}))
        config = self._apply_retrieval_params(config, params.get("retrieval", {}))
        config = self._apply_reranking_params(config, params.get("reranking", {}))
        config = self._apply_generation_params(config, params.get("generation", {}))
        config = self._apply_embedding_params(config, params.get("embedding", {}))

        # Add metadata
        config["metadata"] = {
            "generated": True,
            "config_id": self._generate_config_id(params),
            "parameters": params
        }

        return config

    def _create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal pipeline configuration structure"""
        return {
            "pipeline": {
                "nodes": [],
                "edges": []
            },
            "metadata": {
                "name": "Generated Configuration",
                "version": "1.0.0"
            }
        }

    def _apply_chunking_params(self, config: Dict[str, Any],
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply chunking parameters to configuration"""
        if not params:
            return config

        strategy = params.get("strategy", "fixed")
        size = params.get("size", 256)

        # Map strategy names to component names
        strategy_map = {
            "fixed": "fixed_size",
            "semantic": "semantic",
            "sliding": "sliding_window",
            "hierarchical": "hierarchical"
        }

        component_name = strategy_map.get(strategy, "fixed_size")

        # Find or create chunker node
        chunker_node = self._find_or_create_node(
            config, "chunker", "chunker"
        )

        chunker_node["component"] = component_name
        chunker_node["config"] = {
            "chunk_size": size,
            "overlap": 0 if strategy == "fixed" else size // 4,
            "unit": "tokens"
        }

        return config

    def _apply_retrieval_params(self, config: Dict[str, Any],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply retrieval parameters to configuration"""
        if not params:
            return config

        method = params.get("method", "dense")
        top_k = params.get("top_k", 5)

        # Configure retriever based on method
        if method == "dense":
            retriever_node = self._find_or_create_node(
                config, "retriever", "retriever"
            )
            retriever_node["component"] = "dense"
            retriever_node["config"] = {"top_k": top_k}

        elif method == "sparse":
            retriever_node = self._find_or_create_node(
                config, "retriever", "retriever"
            )
            retriever_node["component"] = "bm25"
            retriever_node["config"] = {
                "top_k": top_k,
                "k1": 1.2,
                "b": 0.75
            }

        elif method == "hybrid":
            # Create both dense and sparse retrievers
            dense_node = self._find_or_create_node(
                config, "dense_retriever", "retriever"
            )
            dense_node["component"] = "dense"
            dense_node["config"] = {"top_k": top_k * 2}  # Get more for reranking

            sparse_node = self._find_or_create_node(
                config, "sparse_retriever", "retriever"
            )
            sparse_node["component"] = "bm25"
            sparse_node["config"] = {
                "top_k": top_k * 2,
                "k1": 1.2,
                "b": 0.75
            }

            # Create hybrid combiner
            hybrid_node = self._find_or_create_node(
                config, "hybrid_retriever", "retriever"
            )
            hybrid_node["component"] = "hybrid"
            hybrid_node["config"] = {
                "weight": params.get("hybrid_weight", 0.5),
                "top_k": top_k
            }

            # Update edges for hybrid flow
            self._update_hybrid_edges(config)

        return config

    def _apply_reranking_params(self, config: Dict[str, Any],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reranking parameters to configuration"""
        if not params or not params.get("enabled", False):
            # Remove reranker if it exists
            self._remove_node(config, "reranker")
            return config

        model = params.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_k_rerank = params.get("top_k_rerank", 20)

        reranker_node = self._find_or_create_node(
            config, "reranker", "reranker"
        )
        reranker_node["component"] = "cross_encoder"
        reranker_node["config"] = {
            "model_name": model,
            "top_k": top_k_rerank,
            "batch_size": 32
        }

        # Update edges to include reranker
        self._insert_reranker_in_pipeline(config)

        return config

    def _apply_generation_params(self, config: Dict[str, Any],
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply generation parameters to configuration"""
        if not params:
            return config

        generator_node = self._find_or_create_node(
            config, "generator", "generator"
        )

        generator_node["component"] = "openai"
        generator_node["config"] = {
            "model": params.get("model", "gpt-3.5-turbo"),
            "temperature": params.get("temperature", 0),
            "max_tokens": params.get("max_tokens", 300)
        }

        return config

    def _apply_embedding_params(self, config: Dict[str, Any],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply embedding parameters to configuration"""
        if not params:
            return config

        model = params.get("model", "text-embedding-ada-002")

        # Update document embedder
        doc_embedder = self._find_or_create_node(
            config, "embedder", "embedder"
        )
        doc_embedder["component"] = "openai"
        doc_embedder["config"] = {
            "model": model,
            "batch_size": 100
        }

        # Update query embedder
        query_embedder = self._find_or_create_node(
            config, "query_embedder", "embedder"
        )
        query_embedder["component"] = "openai"
        query_embedder["config"] = {
            "model": model
        }

        return config

    def _find_or_create_node(self, config: Dict[str, Any],
                            node_id: str, node_type: str) -> Dict[str, Any]:
        """Find existing node or create new one"""
        if "nodes" not in config["pipeline"]:
            config["pipeline"]["nodes"] = []

        for node in config["pipeline"]["nodes"]:
            if node.get("id") == node_id:
                return node

        # Create new node
        new_node = {
            "id": node_id,
            "type": node_type
        }
        config["pipeline"]["nodes"].append(new_node)
        return new_node

    def _remove_node(self, config: Dict[str, Any], node_id: str):
        """Remove node from configuration"""
        if "nodes" not in config["pipeline"]:
            return

        config["pipeline"]["nodes"] = [
            node for node in config["pipeline"]["nodes"]
            if node.get("id") != node_id
        ]

        # Also remove edges referencing this node
        if "edges" in config["pipeline"]:
            config["pipeline"]["edges"] = [
                edge for edge in config["pipeline"]["edges"]
                if edge.get("from") != node_id and edge.get("to") != node_id
            ]

    def _update_hybrid_edges(self, config: Dict[str, Any]):
        """Update edges for hybrid retrieval flow"""
        if "edges" not in config["pipeline"]:
            config["pipeline"]["edges"] = []

        # Ensure proper flow for hybrid retrieval
        edges_to_add = [
            {"from": "query_embedder", "to": "dense_retriever"},
            {"from": "query", "to": "sparse_retriever"},
            {"from": ["dense_retriever", "sparse_retriever"], "to": "hybrid_retriever"},
            {"from": "hybrid_retriever", "to": "generator"}
        ]

        for edge in edges_to_add:
            if edge not in config["pipeline"]["edges"]:
                config["pipeline"]["edges"].append(edge)

    def _insert_reranker_in_pipeline(self, config: Dict[str, Any]):
        """Insert reranker between retriever and generator"""
        if "edges" not in config["pipeline"]:
            return

        # Find edges from retriever to generator
        new_edges = []
        for edge in config["pipeline"]["edges"]:
            if edge.get("to") == "generator":
                # Check if from is a retriever
                from_node = edge.get("from")
                if isinstance(from_node, str) and "retriever" in from_node:
                    # Redirect through reranker
                    new_edges.append({"from": from_node, "to": "reranker"})
                    new_edges.append({"from": "reranker", "to": "generator"})
                else:
                    new_edges.append(edge)
            else:
                new_edges.append(edge)

        config["pipeline"]["edges"] = new_edges

    def _generate_config_id(self, params: Dict[str, Any]) -> str:
        """Generate unique ID for configuration"""
        # Create deterministic hash from parameters
        param_str = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.md5(param_str.encode())
        return hash_obj.hexdigest()[:8]

    def _deep_copy(self, obj: Any) -> Any:
        """Deep copy a configuration object"""
        import copy
        return copy.deepcopy(obj)

    def generate_all_configurations(self) -> List[Dict[str, Any]]:
        """Generate all possible configurations from search space"""
        all_params = self.search_space.enumerate_all()
        configurations = []

        for params in all_params:
            config = self.generate_configuration(params)
            configurations.append(config)

        self.generated_configs = configurations
        logger.info(f"Generated {len(configurations)} configurations")
        return configurations

    def generate_subset(self, n: int = 10, method: str = "random") -> List[Dict[str, Any]]:
        """Generate a subset of configurations"""
        sampled_params = self.search_space.sample(n, method)
        configurations = []

        for params in sampled_params:
            config = self.generate_configuration(params)
            configurations.append(config)

        logger.info(f"Generated {len(configurations)} configurations using {method} sampling")
        return configurations

    def save_configurations(self, configs: List[Dict[str, Any]],
                           output_dir: str):
        """Save generated configurations to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, config in enumerate(configs):
            config_id = config["metadata"].get("config_id", f"config_{i:04d}")
            filepath = output_path / f"{config_id}.yaml"

            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Saved {len(configs)} configurations to {output_dir}")

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate that a configuration is complete and valid"""
        try:
            # Check required structure
            if "pipeline" not in config:
                return False

            pipeline = config["pipeline"]

            # Check for nodes
            if "nodes" not in pipeline or not pipeline["nodes"]:
                return False

            # Check each node has required fields
            for node in pipeline["nodes"]:
                if "id" not in node or "type" not in node:
                    return False

            # If edges exist, validate them
            if "edges" in pipeline:
                node_ids = {node["id"] for node in pipeline["nodes"]}
                for edge in pipeline["edges"]:
                    if "from" not in edge or "to" not in edge:
                        return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False