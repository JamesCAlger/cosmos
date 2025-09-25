"""
Search space conversion utilities for Bayesian optimization.
Converts between auto-RAG SearchSpace and scikit-optimize formats.
"""

from typing import Dict, List, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class SearchSpaceConverter:
    """Convert between different search space representations"""

    @staticmethod
    def from_config_ranges(config_ranges: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
        """
        Convert configuration ranges to simplified search space.

        Args:
            config_ranges: Dict with component configurations and their ranges
                          e.g., {'chunker': {'chunk_size': [128, 256, 512]}}

        Returns:
            Simplified search space for Bayesian optimization
        """
        search_space = {}

        for component, params in config_ranges.items():
            for param_name, values in params.items():
                # Create flattened parameter name
                full_param_name = f"{component}.{param_name}"

                if isinstance(values, list):
                    if len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                        # Treat as range if exactly 2 numeric values
                        search_space[full_param_name] = tuple(values)
                    else:
                        # Categorical parameter
                        search_space[full_param_name] = values
                elif isinstance(values, tuple):
                    # Already a range
                    search_space[full_param_name] = values
                else:
                    # Single value - skip optimization
                    logger.debug(f"Skipping single-valued parameter: {full_param_name}")

        return search_space

    @staticmethod
    def to_pipeline_config(params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Convert flat parameters to pipeline configuration format.

        Args:
            params: Flat dictionary of parameters from optimizer
                   e.g., {'chunker.chunk_size': 256, 'retriever.top_k': 10}

        Returns:
            Nested configuration for pipeline
                   e.g., {'chunker': {'chunk_size': 256}, 'retriever': {'top_k': 10}}
        """
        config = {}

        for param_name, param_value in params.items():
            if '.' in param_name:
                component, param = param_name.split('.', 1)

                if component not in config:
                    config[component] = {}

                # Handle nested parameters (e.g., 'generator.params.temperature')
                if '.' in param:
                    # Further nesting needed
                    parts = param.split('.')
                    current = config[component]
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = param_value
                else:
                    config[component][param] = param_value
            else:
                # Top-level parameter
                config[param_name] = param_value

        return config

    @staticmethod
    def create_minimal_search_space() -> Dict[str, Any]:
        """
        Create a minimal search space for testing.

        Returns:
            Search space with key RAG parameters
        """
        return {
            # Chunking parameters
            'chunker.chunk_size': [128, 256, 512],
            'chunker.overlap': [0, 25, 50],

            # Retrieval parameters
            'retriever.top_k': (3, 10),
            'retriever.algorithm': ['bm25', 'dense', 'hybrid'],

            # Generation parameters
            'generator.temperature': (0.0, 1.0),
            'generator.max_tokens': [100, 150, 200]
        }

    @staticmethod
    def create_comprehensive_search_space() -> Dict[str, Any]:
        """
        Create a comprehensive search space for production use.

        Returns:
            Full search space covering all major RAG parameters
        """
        return {
            # Chunking strategies and parameters
            'chunker.type': ['fixed_size', 'sliding_window', 'semantic'],
            'chunker.chunk_size': (128, 1024),
            'chunker.overlap': (0, 100),

            # Embedding parameters
            'embedder.model': ['text-embedding-ada-002', 'text-embedding-3-small'],
            'embedder.batch_size': [16, 32, 64],

            # Retrieval parameters
            'retriever.type': ['bm25', 'dense', 'hybrid'],
            'retriever.top_k': (3, 20),
            'retriever.bm25_k1': (0.5, 2.0),  # BM25 specific
            'retriever.bm25_b': (0.0, 1.0),   # BM25 specific
            'retriever.dense_metric': ['cosine', 'euclidean', 'dot_product'],  # Dense specific

            # Reranking parameters (optional)
            'reranker.enabled': [True, False],
            'reranker.top_k': (3, 10),
            'reranker.model': ['cross-encoder/ms-marco-MiniLM-L-6-v2'],

            # Generation parameters
            'generator.model': ['gpt-3.5-turbo', 'gpt-4'],
            'generator.temperature': (0.0, 1.0),
            'generator.max_tokens': (100, 500),
            'generator.top_p': (0.5, 1.0),
            'generator.presence_penalty': (-2.0, 2.0),
            'generator.frequency_penalty': (-2.0, 2.0)
        }

    @staticmethod
    def filter_search_space(search_space: Dict[str, Any],
                           allowed_components: List[str] = None,
                           excluded_params: List[str] = None) -> Dict[str, Any]:
        """
        Filter search space to include/exclude specific components or parameters.

        Args:
            search_space: Original search space
            allowed_components: List of component names to include (None = all)
            excluded_params: List of parameter names to exclude

        Returns:
            Filtered search space
        """
        filtered = {}

        for param_name, param_spec in search_space.items():
            # Check component filter
            if allowed_components and '.' in param_name:
                component = param_name.split('.')[0]
                if component not in allowed_components:
                    continue

            # Check exclusion list
            if excluded_params and param_name in excluded_params:
                continue

            filtered[param_name] = param_spec

        return filtered

    @staticmethod
    def add_constraints(search_space: Dict[str, Any],
                       constraints: List[Tuple[str, str, Any]]) -> Dict[str, Any]:
        """
        Add constraints to search space.

        Args:
            search_space: Original search space
            constraints: List of (param_name, operator, value) tuples
                        e.g., [('chunker.chunk_size', '<=', 512)]

        Returns:
            Constrained search space
        """
        constrained = search_space.copy()

        for param_name, operator, value in constraints:
            if param_name not in constrained:
                logger.warning(f"Parameter {param_name} not in search space")
                continue

            param_spec = constrained[param_name]

            if isinstance(param_spec, tuple):
                # Range parameter
                min_val, max_val = param_spec
                if operator == '<=':
                    max_val = min(max_val, value)
                elif operator == '>=':
                    min_val = max(min_val, value)
                elif operator == '<':
                    max_val = min(max_val, value - 1)
                elif operator == '>':
                    min_val = max(min_val, value + 1)
                constrained[param_name] = (min_val, max_val)

            elif isinstance(param_spec, list):
                # Categorical parameter
                if operator == '==':
                    constrained[param_name] = [value] if value in param_spec else param_spec
                elif operator == '!=':
                    constrained[param_name] = [v for v in param_spec if v != value]
                elif operator == 'in':
                    constrained[param_name] = [v for v in param_spec if v in value]
                elif operator == 'not in':
                    constrained[param_name] = [v for v in param_spec if v not in value]

        return constrained


def validate_config_with_search_space(config: Dict[str, Any],
                                     search_space: Dict[str, Any]) -> bool:
    """
    Validate that a configuration is within the search space bounds.

    Args:
        config: Configuration to validate
        search_space: Search space definition

    Returns:
        True if configuration is valid
    """
    for param_name, param_value in config.items():
        if param_name not in search_space:
            logger.warning(f"Parameter {param_name} not in search space")
            continue

        param_spec = search_space[param_name]

        if isinstance(param_spec, tuple):
            # Range parameter
            min_val, max_val = param_spec
            if not min_val <= param_value <= max_val:
                logger.error(f"Parameter {param_name}={param_value} outside range [{min_val}, {max_val}]")
                return False

        elif isinstance(param_spec, list):
            # Categorical parameter
            if param_value not in param_spec:
                logger.error(f"Parameter {param_name}={param_value} not in allowed values {param_spec}")
                return False

    return True