"""
Simplified Bayesian Optimization for RAG Pipeline (With Caching)
Optimized implementation with embedding cache for faster evaluations
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
import argparse
import numpy as np
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from datasets import load_dataset
from loguru import logger

# Import optimization components
from autorag.optimization.bayesian_search import SimpleBayesianOptimizer
from autorag.optimization.cache_manager import EmbeddingCacheManager
from autorag.evaluation.external_metrics import ExternalMetricsCollector

# Import pipeline components
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.components.chunkers.semantic import SemanticChunker
from autorag.components.retrievers.bm25 import BM25Retriever
from autorag.components.retrievers.dense import DenseRetriever
from autorag.components.retrievers.hybrid import HybridRetriever
from autorag.components.generators.openai import OpenAIGenerator
from autorag.components.generators.mock import MockGenerator
from autorag.components.embedders.openai import OpenAIEmbedder
from autorag.components.embedders.mock import MockEmbedder
from autorag.components.embedders.cached import CachedEmbedder
from autorag.components.rerankers.cross_encoder import CrossEncoderReranker
from autorag.components.base import Document

# Load environment variables
load_dotenv()

# Setup logging
logger.add("bayesian_with_cache.log", rotation="10 MB", level="INFO")


class CacheStats:
    """Track cache performance statistics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_time_saved = 0
        self.embeddings_cached = 0

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    def to_dict(self):
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'embeddings_cached': self.embeddings_cached,
            'time_saved': round(self.total_time_saved, 2),
            'cost_saved': round(self.embeddings_cached * 0.00002, 4)  # Approx cost per embedding
        }


def build_pipeline(config: Dict[str, Any], use_real_api: bool = False,
                  cache_manager: Optional[EmbeddingCacheManager] = None):
    """Build RAG pipeline from configuration with optional caching"""

    # Select chunker
    if config['chunking_strategy'] == 'fixed':
        chunker_config = {
            'chunk_size': config['chunk_size'],
            'overlap': config.get('chunk_overlap', 0)
        }
        chunker = FixedSizeChunker(chunker_config)
    else:  # semantic
        chunker_config = {
            'max_chunk_size': config['chunk_size'],
            'similarity_threshold': config.get('similarity_threshold', 0.5)
        }
        chunker = SemanticChunker(chunker_config)

    # Select embedder (with caching if available)
    if use_real_api and os.getenv('OPENAI_API_KEY'):
        embedder_config = {
            'model': 'text-embedding-ada-002',
            'api_key': os.getenv('OPENAI_API_KEY')
        }
        base_embedder = OpenAIEmbedder(embedder_config)
    else:
        base_embedder = MockEmbedder()

    # Wrap with cache if available
    if cache_manager:
        embedder = CachedEmbedder(
            embedder=base_embedder,
            cache_manager=cache_manager
        )
    else:
        embedder = base_embedder

    # Select retriever
    if config['retrieval_method'] == 'bm25':
        retriever = BM25Retriever()
    elif config['retrieval_method'] == 'dense':
        # DenseRetriever needs proper initialization
        from autorag.components.vector_stores.simple import SimpleVectorStore
        retriever = DenseRetriever()
        vector_store = SimpleVectorStore()
        retriever.set_components(embedder=embedder, vector_store=vector_store)
    else:  # hybrid
        # HybridRetriever needs both dense and sparse retrievers
        from autorag.components.vector_stores.simple import SimpleVectorStore

        # Create dense retriever
        dense_retriever = DenseRetriever()
        vector_store = SimpleVectorStore()
        dense_retriever.set_components(embedder=embedder, vector_store=vector_store)

        # Create sparse retriever
        sparse_retriever = BM25Retriever()

        # Create hybrid retriever with config
        hybrid_config = {
            'dense_weight': config.get('hybrid_weight', 0.5),
            'sparse_weight': 1.0 - config.get('hybrid_weight', 0.5),
            'fusion_method': 'weighted_sum'
        }
        retriever = HybridRetriever(hybrid_config)
        retriever.set_retrievers(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever)

    # Select generator
    if use_real_api and os.getenv('OPENAI_API_KEY'):
        # OpenAIGenerator expects a config dict
        generator_config = {
            'model': 'gpt-3.5-turbo',
            'api_key': os.getenv('OPENAI_API_KEY'),
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 500)
        }
        generator = OpenAIGenerator(generator_config)
    else:
        generator = MockGenerator()

    # Optional reranker
    reranker = None
    if config.get('reranking_enabled', False):
        reranker_config = {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'normalize_scores': True
        }
        reranker = CrossEncoderReranker(reranker_config)

    # Create a simple wrapper instead of using RAGPipeline
    from autorag.components.base import Document, Chunk, QueryResult

    class SimplePipeline:
        def __init__(self, chunker, embedder, retriever, generator, reranker=None):
            self.chunker = chunker
            self.embedder = embedder
            self.retriever = retriever
            self.generator = generator
            self.reranker = reranker
            self.documents_indexed = False

        def add_documents(self, documents):
            # Convert strings to Document objects
            doc_objects = []
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    doc_objects.append(Document(content=doc, doc_id=str(i)))
                else:
                    doc_objects.append(doc)

            # Chunk documents using the chunker
            if hasattr(self.chunker, 'chunk'):
                # Chunker expects Document objects
                chunks = self.chunker.chunk(doc_objects)
            else:
                # Fallback - create chunks manually
                chunks = []
                for doc in doc_objects:
                    text = doc.content
                    for j, i in enumerate(range(0, len(text), 400)):
                        chunk_text = text[i:i+500]
                        chunk = Chunk(
                            content=chunk_text,
                            metadata={'doc_id': doc.doc_id},
                            doc_id=doc.doc_id,
                            chunk_id=f"{doc.doc_id}_{j}",
                            start_char=i,
                            end_char=min(i+500, len(text))
                        )
                        chunks.append(chunk)

            self.chunks = chunks

            # Index chunks - this depends on the retriever type
            if hasattr(self.retriever, 'add_documents'):
                self.retriever.add_documents(chunks)
            elif hasattr(self.retriever, 'index'):
                self.retriever.index(chunks)
            else:
                # For BM25 and others that may need text indexing
                chunk_texts = [chunk.content for chunk in chunks]
                if hasattr(self.retriever, 'fit'):
                    self.retriever.fit(chunk_texts)
                self.retriever._documents = chunk_texts  # Store for later retrieval

            self.documents_indexed = True

        def query(self, query_text, top_k=5):
            if not self.documents_indexed:
                return "No documents indexed yet."

            # Retrieve relevant content
            try:
                if hasattr(self.retriever, 'retrieve'):
                    results = self.retriever.retrieve(query_text, top_k)
                    if results and isinstance(results[0], QueryResult):
                        context = ' '.join([r.chunk.content for r in results[:top_k]])
                    else:
                        context = ' '.join([str(r) for r in results[:top_k]])
                elif hasattr(self.retriever, 'search'):
                    results = self.retriever.search(query_text, top_k)
                    context = ' '.join([str(r) for r in results[:top_k]])
                else:
                    # Fallback
                    import random
                    sample_chunks = random.sample(self.chunks, min(top_k, len(self.chunks)))
                    context = ' '.join([c.content for c in sample_chunks])
                    results = sample_chunks

                # Rerank if available
                if self.reranker and hasattr(self.reranker, 'rerank'):
                    results = self.reranker.rerank(query_text, results, top_k)
                    if results and isinstance(results[0], QueryResult):
                        context = ' '.join([r.chunk.content for r in results[:top_k]])

                # Generate answer
                if hasattr(self.generator, 'generate'):
                    answer = self.generator.generate(query_text, context)
                else:
                    answer = f"Based on the context, here is a response to '{query_text}'."

                return answer if answer else "Could not generate an answer."

            except Exception as e:
                logger.warning(f"Error in query: {e}")
                return f"Based on the available information, here is a response to '{query_text}'."

    pipeline = SimplePipeline(
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        reranker=reranker
    )

    return pipeline


def evaluate_configuration(config: Dict[str, Any], dataset, use_real_api: bool = False,
                         cache_manager: Optional[EmbeddingCacheManager] = None,
                         cache_stats: Optional[CacheStats] = None, num_queries: int = 5) -> float:
    """Evaluate a single configuration with caching support"""

    try:
        # Build pipeline
        pipeline = build_pipeline(config, use_real_api, cache_manager)

        # Track cache performance
        cache_start_hits = cache_manager.stats['hits'] if cache_manager else 0

        # Process documents
        documents = [item['document'] for item in dataset]
        pipeline.add_documents(documents)

        # Evaluate on queries
        total_score = 0
        queries_to_eval = min(num_queries, len(dataset))

        for i in range(queries_to_eval):
            query = dataset[i]['query']
            expected = dataset[i]['answers'][0] if dataset[i]['answers'] else ""

            # Get answer from pipeline
            answer = pipeline.query(query, top_k=config.get('retrieval_top_k', 5))

            # Simple scoring based on answer similarity
            from autorag.evaluation.semantic_metrics import SemanticMetrics
            evaluator = SemanticMetrics()  # Now uses gte-large-en-v1.5 by default
            score = evaluator.similarity_score(answer, expected)
            total_score += score

        # Update cache stats if available
        if cache_manager and cache_stats:
            new_hits = cache_manager.stats['hits'] - cache_start_hits
            cache_stats.hits += new_hits
            cache_stats.misses += cache_manager.stats['misses']
            cache_stats.embeddings_cached += new_hits
            # Estimate time saved (0.5s per embedding)
            cache_stats.total_time_saved += new_hits * 0.5

        final_score = total_score / queries_to_eval
        return final_score

    except Exception as e:
        logger.error(f"Error evaluating config: {e}")
        return 0.0


def run_optimization(n_calls: int = 20, use_real_api: bool = False, num_docs: int = 20,
                    num_queries: int = 5, cache_dir: str = '.embedding_cache',
                    cache_memory_limit: int = 256, clear_cache: bool = False):
    """Run Bayesian optimization with caching"""

    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION (WITH CACHE)")
    print("="*60)

    # Initialize cache manager
    cache_manager = EmbeddingCacheManager(
        cache_dir=cache_dir,
        max_memory_mb=cache_memory_limit
    )

    if clear_cache:
        print(f"Clearing cache at {cache_dir}...")
        cache_manager.clear_cache()

    cache_stats = cache_manager.get_cache_stats()
    print(f"\nCache status:")
    print(f"  Directory: {cache_dir}")
    print(f"  Entries: {len(cache_manager.cache_index)}")
    print(f"  Memory limit: {cache_memory_limit} MB")

    # Load dataset
    print(f"\nLoading MS MARCO dataset ({num_docs} documents)...")
    dataset = load_dataset('ms_marco', 'v2.1', split='train', streaming=True)
    dataset_items = []
    for i, item in enumerate(dataset):
        if i >= num_docs:
            break
        dataset_items.append({
            'document': item['passages']['passage_text'][0],
            'query': item['query'],
            'answers': item['answers'] if 'answers' in item else []
        })

    print(f"Loaded {len(dataset_items)} documents")

    # Define search space
    search_space = {
        'chunking_strategy': ['fixed', 'semantic'],
        'chunk_size': [128, 256, 512],
        'retrieval_method': ['bm25', 'dense', 'hybrid'],
        'retrieval_top_k': [3, 5, 10],
        'reranking_enabled': [False, True],
        'temperature': [0.3, 0.7, 1.0],
        'max_tokens': [256, 512]
    }

    # Calculate total combinations
    total_combinations = 1
    for param, values in search_space.items():
        total_combinations *= len(values)

    print(f"\nSearch space: {total_combinations} total combinations")
    print(f"Evaluations: {n_calls} ({n_calls/total_combinations*100:.1f}% of space)")

    # Initialize cache statistics
    cache_stats = CacheStats()

    # Create evaluator function
    eval_count = [0]

    def evaluator(params):
        eval_count[0] += 1
        print(f"\n[Eval {eval_count[0]}/{n_calls}] Testing configuration...")

        # Convert to proper types
        config = {
            'chunking_strategy': params['chunking_strategy'],
            'chunk_size': int(params['chunk_size']),
            'retrieval_method': params['retrieval_method'],
            'retrieval_top_k': int(params['retrieval_top_k']),
            'reranking_enabled': params['reranking_enabled'],
            'temperature': float(params['temperature']),
            'max_tokens': int(params['max_tokens'])
        }

        # Add dependent parameters
        if config['retrieval_method'] == 'hybrid':
            config['hybrid_weight'] = 0.5
        if config['reranking_enabled']:
            config['top_k_rerank'] = 3

        # Evaluate
        start_time = time.time()
        score = evaluate_configuration(config, dataset_items, use_real_api, cache_manager, cache_stats, num_queries)
        eval_time = time.time() - start_time

        # Show cache hit rate
        cache_hit_rate = cache_stats.hit_rate
        print(f"  Score: {score:.4f} | Time: {eval_time:.2f}s | Cache hit: {cache_hit_rate:.1%}")

        # Return metrics dict for the optimizer
        return {
            'accuracy': score,
            'eval_time': eval_time
        }

    # Run optimization
    print("\nStarting optimization with caching...")
    start_time = time.time()

    optimizer = SimpleBayesianOptimizer(
        search_space=search_space,
        evaluator=evaluator,
        n_calls=n_calls,
        n_initial_points=5,
        objective='accuracy'
    )
    result = optimizer.optimize()

    total_time = time.time() - start_time

    # Get final cache stats
    final_cache_stats = cache_manager.get_cache_stats()

    # Save results
    results = {
        'optimization_type': 'with_cache',
        'best_params': result.best_config,
        'best_score': result.best_score,
        'n_calls': result.n_evaluations,
        'total_time': round(total_time, 2),
        'avg_time_per_eval': round(total_time / result.n_evaluations, 2) if result.n_evaluations > 0 else 0,
        'all_scores': result.all_scores,
        'cache_stats': cache_stats.to_dict(),
        'final_cache_stats': final_cache_stats
    }

    output_file = 'bayesian_with_cache_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best score: {result.best_score:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per eval: {total_time/result.n_evaluations:.2f}s" if result.n_evaluations > 0 else "N/A")

    print(f"\nCache Performance:")
    print(f"  Hit rate: {cache_stats.hit_rate:.1%}")
    print(f"  API calls saved: {cache_stats.embeddings_cached}")
    print(f"  Time saved: {cache_stats.total_time_saved:.1f}s")
    print(f"  Cost saved: ${cache_stats.to_dict()['cost_saved']}")

    print(f"\nBest configuration:")
    for key, value in result.best_config.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayesian optimization with caching')
    parser.add_argument('--n-calls', type=int, default=20,
                       help='Number of evaluations (default: 20)')
    parser.add_argument('--num-docs', type=int, default=20,
                       help='Number of documents to use (default: 20)')
    parser.add_argument('--num-queries', type=int, default=5,
                       help='Number of queries to evaluate per configuration (default: 5)')
    parser.add_argument('--real-api', action='store_true',
                       help='Use real OpenAI API instead of mock')
    parser.add_argument('--cache-dir', type=str, default='.embedding_cache',
                       help='Directory for cache storage (default: .embedding_cache)')
    parser.add_argument('--cache-memory-limit', type=int, default=256,
                       help='Maximum memory for cache in MB (default: 256)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cache before starting')

    args = parser.parse_args()

    run_optimization(
        n_calls=args.n_calls,
        use_real_api=args.real_api,
        num_docs=args.num_docs,
        num_queries=args.num_queries,
        cache_dir=args.cache_dir,
        cache_memory_limit=args.cache_memory_limit,
        clear_cache=args.clear_cache
    )