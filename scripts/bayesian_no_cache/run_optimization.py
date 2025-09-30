"""
Simplified Bayesian Optimization for RAG Pipeline (No Caching)
Clean implementation without embedding cache for baseline comparison
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
from typing import Dict, Any, List
from dotenv import load_dotenv
from datasets import load_dataset
from loguru import logger

# Import optimization components
from autorag.optimization.bayesian_search import SimpleBayesianOptimizer
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
from autorag.components.rerankers.cross_encoder import CrossEncoderReranker
from autorag.components.base import Document

# Load environment variables
load_dotenv()

# Setup logging
logger.add("bayesian_no_cache.log", rotation="10 MB", level="INFO")


def build_pipeline(config: Dict[str, Any], use_real_api: bool = False):
    """Build RAG pipeline from configuration"""

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

    # Select embedder (no caching)
    if use_real_api and os.getenv('OPENAI_API_KEY'):
        embedder_config = {
            'model': 'text-embedding-ada-002',
            'api_key': os.getenv('OPENAI_API_KEY')
        }
        embedder = OpenAIEmbedder(embedder_config)
    else:
        embedder = MockEmbedder()

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
                         num_queries: int = 5) -> float:
    """Evaluate a single configuration"""

    try:
        # Build pipeline
        pipeline = build_pipeline(config, use_real_api)

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

        final_score = total_score / queries_to_eval
        return final_score

    except Exception as e:
        logger.error(f"Error evaluating config: {e}")
        return 0.0


def run_optimization(n_calls: int = 20, use_real_api: bool = False, num_docs: int = 20,
                    num_queries: int = 5):
    """Run Bayesian optimization without caching"""

    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION (NO CACHE)")
    print("="*60)

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
        score = evaluate_configuration(config, dataset_items, use_real_api, num_queries)
        eval_time = time.time() - start_time

        print(f"  Score: {score:.4f} | Time: {eval_time:.2f}s")

        # Return metrics dict for the optimizer
        return {
            'accuracy': score,
            'eval_time': eval_time
        }

    # Run optimization
    print("\nStarting optimization...")
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

    # Save results
    results = {
        'optimization_type': 'no_cache',
        'best_params': result.best_config,
        'best_score': result.best_score,
        'n_calls': result.n_evaluations,
        'total_time': round(total_time, 2),
        'avg_time_per_eval': round(total_time / result.n_evaluations, 2) if result.n_evaluations > 0 else 0,
        'all_scores': result.all_scores
    }

    output_file = 'bayesian_no_cache_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best score: {result.best_score:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per eval: {total_time/result.n_evaluations:.2f}s" if result.n_evaluations > 0 else "N/A")
    print(f"\nBest configuration:")
    for key, value in result.best_config.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayesian optimization without caching')
    parser.add_argument('--n-calls', type=int, default=20,
                       help='Number of evaluations (default: 20)')
    parser.add_argument('--num-docs', type=int, default=20,
                       help='Number of documents to use (default: 20)')
    parser.add_argument('--num-queries', type=int, default=5,
                       help='Number of queries to evaluate per configuration (default: 5)')
    parser.add_argument('--real-api', action='store_true',
                       help='Use real OpenAI API instead of mock')

    args = parser.parse_args()

    run_optimization(
        n_calls=args.n_calls,
        use_real_api=args.real_api,
        num_docs=args.num_docs,
        num_queries=args.num_queries
    )