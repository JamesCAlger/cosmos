"""
COSMOS Component Wrapper

Wraps existing RAG components to add component-intrinsic metrics collection
without modifying the original component implementation.
"""

import time
from typing import Any, Dict, List, Tuple, Optional
from loguru import logger
from autorag.cosmos.metrics import ComponentMetrics


class COSMOSComponent:
    """
    Wrapper that adds process_with_metrics() capability to existing components.

    This wrapper implements the COSMOS principle of component-intrinsic metrics
    by measuring performance during execution without requiring full pipeline context.

    Usage:
        # Wrap an existing component
        chunker = FixedSizeChunker({'chunk_size': 256})
        cosmos_chunker = COSMOSComponent(chunker, 'chunker', metric_collector)

        # Use with metrics
        chunks, metrics = cosmos_chunker.process_with_metrics(documents)

        # Access original component
        original = cosmos_chunker.base
    """

    def __init__(self,
                 base_component: Any,
                 component_type: str,
                 metric_collector: ComponentMetrics):
        """
        Initialize COSMOS component wrapper

        Args:
            base_component: Original component to wrap (Chunker, Retriever, Generator)
            component_type: Type of component ('chunker', 'retriever', 'generator')
            metric_collector: ComponentMetrics instance for computing metrics
        """
        self.base = base_component
        self.type = component_type
        self.metric_collector = metric_collector
        self.config = base_component.config if hasattr(base_component, 'config') else {}
        self.metrics_history: List[Dict[str, float]] = []

        logger.info(f"Wrapped {component_type} with COSMOS metrics: {base_component.__class__.__name__}")

    def process_with_metrics(self, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Execute component and collect intrinsic metrics

        This method delegates to the appropriate component-specific handler
        based on component type.

        Returns:
            Tuple of (output, metrics) where:
            - output: Same output as original component
            - metrics: Dictionary of component-intrinsic metrics

        Raises:
            ValueError: If component type is not supported
        """
        if self.type == 'chunker':
            return self._process_chunker(*args, **kwargs)
        elif self.type == 'retriever':
            return self._process_retriever(*args, **kwargs)
        elif self.type == 'generator':
            return self._process_generator(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported component type: {self.type}")

    def _process_chunker(self, documents: List, **kwargs) -> Tuple[List, Dict[str, float]]:
        """
        Process documents through chunker and collect metrics

        Args:
            documents: List of Document objects

        Returns:
            Tuple of (chunks, metrics)
        """
        # Time the chunking operation
        start_time = time.time()
        chunks = self.base.chunk(documents)
        latency = time.time() - start_time

        # Compute chunking metrics
        compute_coherence = kwargs.get('compute_coherence', True)
        metrics = self.metric_collector.compute_chunking_metrics(
            chunks,
            latency,
            compute_coherence=compute_coherence
        )

        # Store in history
        self.metrics_history.append(metrics)

        logger.debug(f"Chunker metrics: {len(chunks)} chunks in {latency:.3f}s")
        return chunks, metrics

    def _process_retriever(self, query: str, top_k: int = 5, **kwargs) -> Tuple[List, Dict[str, float]]:
        """
        Process query through retriever and collect metrics

        Args:
            query: Query string
            top_k: Number of results to retrieve

        Returns:
            Tuple of (results, metrics)
        """
        # Time the retrieval operation
        start_time = time.time()
        results = self.base.retrieve(query, top_k=top_k)
        latency = time.time() - start_time

        # Compute retrieval metrics
        ground_truth = kwargs.get('ground_truth', None)
        metrics = self.metric_collector.compute_retrieval_metrics(
            query,
            results,
            latency,
            ground_truth=ground_truth
        )

        # Store in history
        self.metrics_history.append(metrics)

        logger.debug(f"Retriever metrics: {len(results)} results in {latency:.3f}s")
        return results, metrics

    def _process_generator(self, query: str, context: List, **kwargs) -> Tuple[str, Dict[str, float]]:
        """
        Process query and context through generator and collect metrics

        Args:
            query: Query string
            context: List of context documents/results

        Returns:
            Tuple of (answer, metrics)
        """
        # Time the generation operation
        start_time = time.time()
        answer = self.base.generate(query, context)
        latency = time.time() - start_time

        # Compute generation metrics
        ground_truth_answer = kwargs.get('ground_truth_answer', None)
        metrics = self.metric_collector.compute_generation_metrics(
            query,
            answer,
            context,
            latency,
            ground_truth_answer=ground_truth_answer
        )

        # Store in history
        self.metrics_history.append(metrics)

        logger.debug(f"Generator metrics: {len(answer.split())} words in {latency:.3f}s")
        return answer, metrics

    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average metrics across all executions

        Returns:
            Dictionary with averaged metrics. Empty dict if no history.
        """
        if not self.metrics_history:
            return {}

        import numpy as np

        # Get all metric keys from first entry
        metric_keys = list(self.metrics_history[0].keys())

        # Compute average for each metric
        avg_metrics = {}
        for key in metric_keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                avg_metrics[key] = float(np.mean(values))

        return avg_metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of metrics history

        Returns:
            Dictionary with summary statistics:
            - count: Number of executions
            - mean: Average metrics
            - std: Standard deviation
            - min: Minimum values
            - max: Maximum values
        """
        if not self.metrics_history:
            return {'count': 0, 'mean': {}, 'std': {}, 'min': {}, 'max': {}}

        import numpy as np

        metric_keys = list(self.metrics_history[0].keys())

        summary = {
            'count': len(self.metrics_history),
            'mean': {},
            'std': {},
            'min': {},
            'max': {}
        }

        for key in metric_keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                summary['mean'][key] = float(np.mean(values))
                summary['std'][key] = float(np.std(values))
                summary['min'][key] = float(np.min(values))
                summary['max'][key] = float(np.max(values))

        return summary

    def clear_metrics(self):
        """Clear metrics history"""
        self.metrics_history = []
        logger.debug(f"Cleared metrics history for {self.type}")

    def get_quality_score(self) -> float:
        """
        Get quality score for latest execution

        Returns:
            Quality score [0, 1] or 0.0 if no history
        """
        if not self.metrics_history:
            return 0.0

        latest_metrics = self.metrics_history[-1]
        return self.metric_collector.compute_quality_score(self.type, latest_metrics)

    def get_average_quality_score(self) -> float:
        """
        Get average quality score across all executions

        Returns:
            Average quality score [0, 1] or 0.0 if no history
        """
        if not self.metrics_history:
            return 0.0

        scores = [
            self.metric_collector.compute_quality_score(self.type, m)
            for m in self.metrics_history
        ]

        import numpy as np
        return float(np.mean(scores))

    def __repr__(self) -> str:
        """String representation"""
        return f"COSMOSComponent({self.type}, base={self.base.__class__.__name__}, executions={len(self.metrics_history)})"

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to base component if not found in wrapper

        This allows the wrapper to be used as a drop-in replacement in most contexts.
        """
        return getattr(self.base, name)