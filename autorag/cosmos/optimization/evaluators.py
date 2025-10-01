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
                 upstream_components: Optional[Dict[str, Any]] = None,
                 max_queries: int = 10,
                 cache_manager: Optional[Any] = None,
                 dataset_name: Optional[str] = None):
        """
        Initialize component evaluator

        Args:
            component_type: Type of component to evaluate ('chunker', 'retriever', 'generator')
            test_data: Test data with 'documents' and 'queries' fields
            metric_collector: ComponentMetrics instance
            upstream_components: Dict of already-optimized upstream components
                                Keys: component types ('chunker', 'retriever')
                                Values: Component instances
            max_queries: Maximum number of queries to use for evaluation (default: 10)
            cache_manager: Optional EmbeddingCacheManager for caching embeddings
            dataset_name: Name of dataset being used (e.g., 'marco', 'beir/scifact')
        """
        self.component_type = component_type
        self.test_data = test_data
        self.metric_collector = metric_collector
        self.upstream_components = upstream_components or {}
        self.max_queries = max_queries
        self.cache_manager = cache_manager
        self.dataset_name = dataset_name or test_data.get('dataset_name', 'unknown')
        self.dataset_size = len(test_data.get('documents', []))

        logger.info(f"ComponentEvaluator initialized for {component_type}")
        if upstream_components:
            logger.info(f"  Using upstream: {list(upstream_components.keys())}")
        logger.debug(f"  Max queries for evaluation: {max_queries}")
        if cache_manager:
            logger.info(f"  Cache enabled: dataset={self.dataset_name}, size={self.dataset_size}")

    def _get_cache_config(self, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert COSMOS component config to cache manager config format

        Args:
            component_config: COSMOS component configuration

        Returns:
            Cache manager compatible configuration
        """
        # Map COSMOS chunking config to cache format
        cache_config = {
            'strategy': component_config.get('chunking_strategy', 'fixed'),
            'chunk_size': component_config.get('chunk_size', 512),
            'overlap': component_config.get('overlap', 0),
        }

        # Add semantic threshold if applicable
        if cache_config['strategy'] == 'semantic':
            cache_config['threshold'] = component_config.get('threshold', 0.5)

        return cache_config

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
            # Default to real API if available, otherwise use mock
            use_real_api = config.get('use_real_api', True)  # Changed default to True

            if use_real_api:
                import os
                api_key = os.getenv('OPENAI_API_KEY')

                if api_key:
                    from autorag.components.embedders.openai import OpenAIEmbedder
                    from autorag.components.embedders.cached import CachedEmbedder

                    # Create real embedder
                    real_embedder = OpenAIEmbedder({
                        'model': 'text-embedding-ada-002',
                        'api_key': api_key
                    })

                    # Wrap with caching to avoid redundant API calls during optimization
                    embedder = CachedEmbedder(real_embedder)
                    logger.info("Using OpenAI embeddings with caching")
                else:
                    # Fallback to mock if API key not available
                    from autorag.components.embedders.mock import MockEmbedder
                    embedder = MockEmbedder({})
                    logger.warning("OpenAI API key not found, falling back to mock embeddings")
            else:
                from autorag.components.embedders.mock import MockEmbedder
                embedder = MockEmbedder({})
                logger.debug("Using mock embeddings (explicitly disabled real API)")

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

            # Get all documents (not just first 10 - we want full dataset for caching)
            documents = self.test_data.get('documents', [])

            # Use cache manager if available
            if self.cache_manager and config.get('retrieval_method') == 'dense':
                logger.debug("Using cache manager for chunking and embedding")

                # Extract chunker configuration for caching
                from autorag.components.chunkers.semantic import SemanticChunker
                chunker_config = {
                    'strategy': 'semantic' if isinstance(chunker, SemanticChunker) else 'fixed',
                    'chunk_size': getattr(chunker, 'chunk_size', 512),
                    'overlap': getattr(chunker, 'overlap', 0),
                }
                if isinstance(chunker, SemanticChunker):
                    chunker_config['threshold'] = getattr(chunker, 'threshold', 0.5)

                # Get cached or compute embeddings
                chunks, embeddings = self.cache_manager.get_or_compute_embeddings(
                    documents,
                    chunker_config,
                    embedder,
                    chunker,
                    dataset_name=self.dataset_name,
                    dataset_size=self.dataset_size
                )

                # Index with pre-computed embeddings
                retriever.index_with_embeddings(chunks, embeddings)
                logger.debug(f"Indexed {len(chunks)} chunks using cached embeddings")
            else:
                # Original flow without caching
                doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
                chunks = chunker.chunk(doc_objects)

                # Index chunks (will compute embeddings internally)
                retriever.index(chunks)
                logger.debug(f"Indexed {len(chunks)} chunks without caching")

            # Build mapping from doc_id to chunk_ids for ground truth
            doc_to_chunks = {}
            for chunk in chunks:
                doc_id = chunk.doc_id if hasattr(chunk, 'doc_id') else str(chunk)
                chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else str(chunk)
                if doc_id not in doc_to_chunks:
                    doc_to_chunks[doc_id] = []
                doc_to_chunks[doc_id].append(chunk_id)

            # Index chunks
            retriever.index(chunks)

            # Evaluate on queries (limit for speed and cost)
            queries = self.test_data.get('queries', [])[:self.max_queries]
            quality_scores = []

            for query_data in queries:
                if isinstance(query_data, dict):
                    query = query_data.get('query', query_data.get('question', ''))
                    # Extract relevant document IDs if available (for precision calculation)
                    relevant_doc_ids = query_data.get('relevant_doc_ids', [])
                else:
                    query = str(query_data)
                    relevant_doc_ids = []

                if not query:
                    continue

                # Map relevant doc IDs to chunk IDs
                relevant_chunk_ids = []
                for doc_id in relevant_doc_ids:
                    doc_id_str = str(doc_id)
                    if doc_id_str in doc_to_chunks:
                        relevant_chunk_ids.extend(doc_to_chunks[doc_id_str])

                # Prepare ground truth for metrics
                ground_truth = {'relevant_chunks': relevant_chunk_ids} if relevant_chunk_ids else None

                # Retrieve with metrics
                results, metrics = cosmos_retriever.process_with_metrics(
                    query,
                    top_k=config.get('retrieval_top_k', 5),
                    ground_truth=ground_truth
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

                # FIX: Upstream retriever needs embedder and vector store configured
                # The retriever component doesn't persist these dependencies, so we must set them up
                if hasattr(retriever, 'set_components') or hasattr(retriever, 'embedder'):
                    import os
                    api_key = os.getenv('OPENAI_API_KEY')

                    if api_key:
                        from autorag.components.embedders.openai import OpenAIEmbedder
                        from autorag.components.embedders.cached import CachedEmbedder

                        # Create real embedder with caching
                        real_embedder = OpenAIEmbedder({
                            'model': 'text-embedding-ada-002',
                            'api_key': api_key
                        })
                        embedder = CachedEmbedder(real_embedder)
                        logger.info("Generator eval: Using OpenAI embeddings with caching for upstream retriever")
                    else:
                        from autorag.components.embedders.mock import MockEmbedder
                        embedder = MockEmbedder({})
                        logger.warning("Generator eval: OpenAI API key not found, using mock embeddings")

                    from autorag.components.vector_stores.simple import SimpleVectorStore
                    vector_store = SimpleVectorStore({})

                    # Set components on retriever
                    if hasattr(retriever, 'set_components'):
                        retriever.set_components(embedder, vector_store)
                    elif hasattr(retriever, 'embedder'):
                        retriever.embedder = embedder
                        retriever.vector_store = vector_store

            # Build generator from config
            generator = build_component('generator', config)

            # Wrap with COSMOS
            cosmos_generator = COSMOSComponent(generator, 'generator', self.metric_collector)

            # Prepare documents and index
            documents = self.test_data.get('documents', [])[:10]
            doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
            chunks = chunker.chunk(doc_objects)
            retriever.index(chunks)

            # Evaluate on queries with ground truth (limit for speed and cost)
            queries = self.test_data.get('queries', [])[:self.max_queries]
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