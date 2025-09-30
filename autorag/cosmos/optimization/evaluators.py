"""
Component Evaluators for COSMOS

Evaluates individual components in isolation or with fixed upstream components.
This breaks the circular dependency problem by allowing component-by-component optimization.
"""

from typing import Dict, Any, List, Callable, Optional
from loguru import logger
import numpy as np

from autorag.cosmos.component_wrapper import COSMOSComponent
from autorag.cosmos.metrics import ComponentMetrics
from autorag.components.base import Document


def build_component(component_type: str, config: Dict[str, Any]) -> Any:
    """
    Build a component from configuration

    Args:
        component_type: Type of component ('chunker', 'retriever', 'generator')
        config: Configuration dictionary for the component

    Returns:
        Component instance

    Raises:
        ValueError: If component type is unknown
    """
    if component_type == 'chunker':
        chunking_strategy = config.get('chunking_strategy', 'fixed')

        if chunking_strategy == 'semantic':
            from autorag.components.chunkers.semantic import SemanticChunker
            return SemanticChunker({
                'chunk_size': config.get('chunk_size', 512),
                'threshold': config.get('threshold', 0.5)
            })
        else:  # fixed
            from autorag.components.chunkers.fixed_size import FixedSizeChunker
            return FixedSizeChunker({
                'chunk_size': config.get('chunk_size', 512),
                'overlap': config.get('overlap', 50),
                'unit': config.get('unit', 'tokens')
            })

    elif component_type == 'retriever':
        retrieval_method = config.get('retrieval_method', 'dense')

        if retrieval_method == 'sparse':
            from autorag.components.retrievers.bm25 import BM25Retriever
            return BM25Retriever({
                'k1': config.get('k1', 1.2),
                'b': config.get('b', 0.75)
            })
        elif retrieval_method == 'hybrid':
            from autorag.components.retrievers.hybrid import HybridRetriever
            from autorag.components.retrievers.dense import DenseRetriever
            from autorag.components.retrievers.bm25 import BM25Retriever
            from autorag.components.vector_stores.simple import SimpleVectorStore

            # Need to provide embedder - will be set later
            return HybridRetriever({
                'dense_weight': config.get('hybrid_weight', 0.5),
                'sparse_weight': 1.0 - config.get('hybrid_weight', 0.5)
            })
        else:  # dense
            from autorag.components.retrievers.dense import DenseRetriever
            from autorag.components.vector_stores.simple import SimpleVectorStore

            retriever = DenseRetriever({
                'metric': config.get('metric', 'cosine'),
                'top_k': config.get('retrieval_top_k', 5)
            })
            return retriever

    elif component_type == 'generator':
        use_real_api = config.get('use_real_api', False)

        if use_real_api:
            from autorag.components.generators.openai import OpenAIGenerator
            import os
            return OpenAIGenerator({
                'model': config.get('model', 'gpt-3.5-turbo'),
                'temperature': config.get('temperature', 0.3),
                'max_tokens': config.get('max_tokens', 150),
                'api_key': os.getenv('OPENAI_API_KEY')
            })
        else:
            from autorag.components.generators.mock import MockGenerator
            return MockGenerator({
                'temperature': config.get('temperature', 0.3)
            })

    else:
        raise ValueError(f"Unknown component type: {component_type}")


class ComponentEvaluator:
    """
    Evaluates components in isolation or with fixed upstream components

    This class solves the circular dependency problem by evaluating each component
    with FIXED upstream components from previous optimizations.
    """

    def __init__(self,
                 component_type: str,
                 test_data: Dict[str, Any],
                 metric_collector: ComponentMetrics,
                 upstream_components: Optional[Dict[str, Any]] = None):
        """
        Initialize component evaluator

        Args:
            component_type: Type of component to evaluate ('chunker', 'retriever', 'generator')
            test_data: Test data with 'documents' and 'queries' fields
            metric_collector: ComponentMetrics instance
            upstream_components: Dict of already-optimized upstream components
                                Keys: component types ('chunker', 'retriever')
                                Values: Component instances
        """
        self.component_type = component_type
        self.test_data = test_data
        self.metric_collector = metric_collector
        self.upstream_components = upstream_components or {}

        logger.info(f"ComponentEvaluator initialized for {component_type}")
        if upstream_components:
            logger.info(f"  Using upstream: {list(upstream_components.keys())}")

    def evaluate(self, config: Dict[str, Any]) -> float:
        """
        Evaluate a component configuration

        This is the main entry point called by optimization algorithms.

        Args:
            config: Configuration dictionary for the component

        Returns:
            Quality score [0, 1] where higher is better
        """
        if self.component_type == 'chunker':
            return self._evaluate_chunker(config)
        elif self.component_type == 'retriever':
            return self._evaluate_retriever(config)
        elif self.component_type == 'generator':
            return self._evaluate_generator(config)
        else:
            raise ValueError(f"Unknown component type: {self.component_type}")

    def _evaluate_chunker(self, config: Dict[str, Any]) -> float:
        """
        Evaluate chunker configuration in isolation

        Chunker has no upstream dependencies, so we can evaluate it directly
        using heuristic quality metrics.

        Args:
            config: Chunker configuration

        Returns:
            Quality score [0, 1]
        """
        try:
            # Build chunker from config
            chunker = build_component('chunker', config)

            # Wrap with COSMOS
            cosmos_chunker = COSMOSComponent(chunker, 'chunker', self.metric_collector)

            # Get test documents
            documents = self.test_data.get('documents', [])[:10]  # Limit for speed
            doc_objects = [
                Document(content=doc, doc_id=str(i))
                for i, doc in enumerate(documents)
            ]

            # Execute and collect metrics
            chunks, metrics = cosmos_chunker.process_with_metrics(
                doc_objects,
                compute_coherence=True
            )

            # Compute quality score
            quality = self.metric_collector.compute_quality_score('chunker', metrics)

            logger.debug(f"Chunker evaluation: {len(chunks)} chunks, quality={quality:.3f}")
            return float(quality)

        except Exception as e:
            logger.error(f"Chunker evaluation failed: {e}")
            return 0.0

    def _evaluate_retriever(self, config: Dict[str, Any]) -> float:
        """
        Evaluate retriever configuration with FIXED upstream chunker

        Uses the best chunker from previous optimization (stored in upstream_components).

        Args:
            config: Retriever configuration

        Returns:
            Quality score [0, 1]
        """
        try:
            # Check for upstream chunker
            if 'chunker' not in self.upstream_components:
                logger.warning("No upstream chunker provided, using default")
                # Use default chunker
                from autorag.components.chunkers.fixed_size import FixedSizeChunker
                chunker = FixedSizeChunker({'chunk_size': 256, 'overlap': 50})
            else:
                chunker = self.upstream_components['chunker']

            # Build retriever from config
            retriever = build_component('retriever', config)

            # Set up embedder for retriever
            use_real_api = config.get('use_real_api', False)
            if use_real_api:
                from autorag.components.embedders.openai import OpenAIEmbedder
                import os
                embedder = OpenAIEmbedder({
                    'model': 'text-embedding-ada-002',
                    'api_key': os.getenv('OPENAI_API_KEY')
                })
            else:
                from autorag.components.embedders.mock import MockEmbedder
                embedder = MockEmbedder({})

            # Set embedder and vector store for retriever
            from autorag.components.vector_stores.simple import SimpleVectorStore
            vector_store = SimpleVectorStore({})

            if hasattr(retriever, 'set_components'):
                retriever.set_components(embedder, vector_store)
            elif hasattr(retriever, 'embedder'):
                retriever.embedder = embedder
                retriever.vector_store = vector_store

            # Wrap with COSMOS
            cosmos_retriever = COSMOSComponent(retriever, 'retriever', self.metric_collector)

            # Chunk documents using upstream chunker
            documents = self.test_data.get('documents', [])[:10]
            doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
            chunks = chunker.chunk(doc_objects)

            # Index chunks
            retriever.index(chunks)

            # Evaluate on queries
            queries = self.test_data.get('queries', [])[:5]  # Limit for speed
            quality_scores = []

            for query_data in queries:
                if isinstance(query_data, dict):
                    query = query_data.get('query', query_data.get('question', ''))
                else:
                    query = str(query_data)

                if not query:
                    continue

                # Retrieve with metrics
                results, metrics = cosmos_retriever.process_with_metrics(
                    query,
                    top_k=config.get('retrieval_top_k', 5)
                )

                # Compute quality
                quality = self.metric_collector.compute_quality_score('retriever', metrics)
                quality_scores.append(quality)

            # Average quality across queries
            if quality_scores:
                avg_quality = float(np.mean(quality_scores))
            else:
                avg_quality = 0.0

            logger.debug(f"Retriever evaluation: {len(quality_scores)} queries, avg_quality={avg_quality:.3f}")
            return avg_quality

        except Exception as e:
            logger.error(f"Retriever evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0

    def _evaluate_generator(self, config: Dict[str, Any]) -> float:
        """
        Evaluate generator configuration with FIXED upstream components

        Uses best chunker and best retriever from previous optimizations.

        Args:
            config: Generator configuration

        Returns:
            Quality score [0, 1] (semantic similarity to ground truth)
        """
        try:
            # Check for upstream components
            if 'chunker' not in self.upstream_components:
                logger.warning("No upstream chunker, using default")
                from autorag.components.chunkers.fixed_size import FixedSizeChunker
                chunker = FixedSizeChunker({'chunk_size': 256})
            else:
                chunker = self.upstream_components['chunker']

            if 'retriever' not in self.upstream_components:
                logger.warning("No upstream retriever, using default")
                from autorag.components.retrievers.dense import DenseRetriever
                from autorag.components.embedders.mock import MockEmbedder
                from autorag.components.vector_stores.simple import SimpleVectorStore

                retriever = DenseRetriever({'metric': 'cosine', 'top_k': 5})
                retriever.set_components(MockEmbedder({}), SimpleVectorStore({}))
            else:
                retriever = self.upstream_components['retriever']

            # Build generator from config
            generator = build_component('generator', config)

            # Wrap with COSMOS
            cosmos_generator = COSMOSComponent(generator, 'generator', self.metric_collector)

            # Prepare documents and index
            documents = self.test_data.get('documents', [])[:10]
            doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
            chunks = chunker.chunk(doc_objects)
            retriever.index(chunks)

            # Evaluate on queries with ground truth
            queries = self.test_data.get('queries', [])[:5]
            quality_scores = []

            for query_data in queries:
                if isinstance(query_data, dict):
                    query = query_data.get('query', query_data.get('question', ''))
                    ground_truth = query_data.get('answer', query_data.get('expected_answer', ''))
                else:
                    query = str(query_data)
                    ground_truth = None

                if not query:
                    continue

                # Retrieve context
                context = retriever.retrieve(query, top_k=5)

                # Generate answer with metrics
                answer, metrics = cosmos_generator.process_with_metrics(
                    query,
                    context,
                    ground_truth_answer=ground_truth
                )

                # Compute quality (uses accuracy if ground truth available)
                quality = self.metric_collector.compute_quality_score('generator', metrics)
                quality_scores.append(quality)

            # Average quality across queries
            if quality_scores:
                avg_quality = float(np.mean(quality_scores))
            else:
                avg_quality = 0.0

            logger.debug(f"Generator evaluation: {len(quality_scores)} queries, avg_quality={avg_quality:.3f}")
            return avg_quality

        except Exception as e:
            logger.error(f"Generator evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0