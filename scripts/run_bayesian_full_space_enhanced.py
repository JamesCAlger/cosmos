"""
Enhanced Bayesian Optimization for Full RAG Configuration Space
- Shows interim stage metrics (chunking coherence, retrieval precision, etc.)
- Uses MS MARCO dataset with real answers for accuracy evaluation
- Displays detailed metrics for each evaluation
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Import optimization components
from autorag.optimization.bayesian_search import SimpleBayesianOptimizer
from autorag.evaluation.external_metrics import ExternalMetricsCollector
from autorag.evaluation.semantic_metrics import SemanticMetrics

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

# Setup logging with detailed format
logger.add("bayesian_enhanced.log", rotation="10 MB", level="DEBUG")


class EnhancedMetricsCollector:
    """Enhanced metrics collector that shows interim stage metrics"""

    def __init__(self, show_details: bool = True):
        """
        Initialize enhanced metrics collector

        Args:
            show_details: Whether to print detailed interim metrics
        """
        self.show_details = show_details
        self.semantic_evaluator = SemanticMetrics(
            model_name='all-MiniLM-L6-v2',
            similarity_threshold=0.75
        )

    def evaluate_with_detailed_metrics(self, pipeline, query: str, documents: List[str],
                                      ground_truth: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluate pipeline with detailed stage metrics

        Returns:
            answer: Generated answer
            metrics: Detailed metrics including stage-wise measurements
        """
        metrics = {}

        # 1. CHUNKING METRICS
        print("\n  [CHUNKING]", end="")
        chunk_start = time.time()

        # Convert to Document objects
        doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]
        chunks = pipeline.chunker.chunk(doc_objects)

        # Extract text from chunks
        chunk_texts = [c.content if hasattr(c, 'content') else str(c) for c in chunks]

        # Calculate chunking metrics
        chunk_sizes = [len(c.split()) for c in chunk_texts]

        # Semantic coherence: How similar are chunks to each other (should be moderate)
        if len(chunk_texts) > 1:
            chunk_embeddings = self.semantic_evaluator.model.encode(chunk_texts[:10])  # Limit for speed
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(chunk_embeddings)
            coherence = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        else:
            coherence = 0.5

        chunk_time = time.time() - chunk_start

        metrics['chunking'] = {
            'time': chunk_time,
            'chunks_created': len(chunks),
            'avg_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0,
            'size_variance': np.std(chunk_sizes) if chunk_sizes else 0,
            'semantic_coherence': coherence
        }

        if self.show_details:
            print(f" {len(chunks)} chunks, avg_size={np.mean(chunk_sizes):.0f}, coherence={coherence:.3f}")

        # 2. RETRIEVAL METRICS
        print("  [RETRIEVAL]", end="")
        retrieval_start = time.time()

        # Index chunks
        pipeline.retriever.index(chunks)

        # Retrieve documents
        top_k = getattr(pipeline, 'top_k', 5)
        retrieved = pipeline.retriever.retrieve(query, top_k=top_k)

        # Extract retrieved texts
        retrieved_texts = [r.content if hasattr(r, 'content') else str(r) for r in retrieved]

        # Calculate retrieval quality
        query_embedding = self.semantic_evaluator.model.encode([query])
        retrieved_embeddings = self.semantic_evaluator.model.encode(retrieved_texts)

        from sklearn.metrics.pairwise import cosine_similarity
        relevance_scores = cosine_similarity(query_embedding, retrieved_embeddings)[0]

        # Retrieval precision: How many retrieved docs are actually relevant
        if 'relevant_chunks' in ground_truth:
            relevant_indices = set(ground_truth['relevant_chunks'])
            retrieved_indices = set([r.doc_id if hasattr(r, 'doc_id') else i
                                    for i, r in enumerate(retrieved)])
            precision = len(relevant_indices & retrieved_indices) / len(retrieved_indices) if retrieved_indices else 0
        else:
            # Use semantic similarity as proxy
            precision = np.mean(relevance_scores > 0.7)

        retrieval_time = time.time() - retrieval_start

        metrics['retrieval'] = {
            'time': retrieval_time,
            'docs_retrieved': len(retrieved),
            'avg_relevance': float(np.mean(relevance_scores)),
            'max_relevance': float(np.max(relevance_scores)),
            'precision': precision,
            'score_spread': float(np.max(relevance_scores) - np.min(relevance_scores))
        }

        if self.show_details:
            print(f" {len(retrieved)} docs, relevance={np.mean(relevance_scores):.3f}, precision={precision:.3f}")

        # 3. RERANKING METRICS (if enabled)
        if hasattr(pipeline, 'reranker') and pipeline.reranker:
            print("  [RERANKING]", end="")
            rerank_start = time.time()

            # Rerank
            reranked = pipeline.reranker.rerank(query, retrieved)
            reranked_texts = [r.content if hasattr(r, 'content') else str(r) for r in reranked]

            # Check if reranking improved relevance
            reranked_embeddings = self.semantic_evaluator.model.encode(reranked_texts[:5])
            reranked_scores = cosine_similarity(query_embedding, reranked_embeddings)[0]

            rerank_time = time.time() - rerank_start

            metrics['reranking'] = {
                'time': rerank_time,
                'docs_reranked': len(reranked),
                'relevance_improvement': float(np.mean(reranked_scores) - np.mean(relevance_scores[:5])),
                'new_avg_relevance': float(np.mean(reranked_scores))
            }

            if self.show_details:
                improvement = np.mean(reranked_scores) - np.mean(relevance_scores[:5])
                print(f" improvement={improvement:+.3f}")

            # Use reranked for generation
            context_docs = reranked_texts[:5]
        else:
            context_docs = retrieved_texts[:5]

        # 4. GENERATION METRICS
        print("  [GENERATION]", end="")
        gen_start = time.time()

        # Generate answer
        answer = pipeline.generator.generate(query, context_docs)

        gen_time = time.time() - gen_start

        # Calculate answer quality metrics
        answer_relevance = self.semantic_evaluator.compute_similarity(
            self.semantic_evaluator.model.encode(answer),
            self.semantic_evaluator.model.encode(query)
        )

        # Context utilization: How much of the context is reflected in the answer
        context_text = ' '.join(context_docs)
        answer_words = set(answer.lower().split())
        context_words = set(context_text.lower().split())
        context_utilization = len(answer_words & context_words) / max(len(answer_words), 1)

        metrics['generation'] = {
            'time': gen_time,
            'answer_length': len(answer.split()),
            'answer_relevance': answer_relevance,
            'context_utilization': context_utilization,
            'answer_generated': bool(answer and len(answer) > 10)
        }

        if self.show_details:
            print(f" relevance={answer_relevance:.3f}, utilization={context_utilization:.3f}")

        # 5. OVERALL ACCURACY (using MS MARCO ground truth)
        if 'answer' in ground_truth:
            # Calculate semantic similarity between answer and ground truth
            answer_emb = self.semantic_evaluator.model.encode(answer)
            truth_emb = self.semantic_evaluator.model.encode(ground_truth['answer'])
            accuracy = self.semantic_evaluator.compute_similarity(answer_emb, truth_emb)

            # Also calculate ROUGE-L score for better evaluation
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(ground_truth['answer'], answer)
            rouge_l = rouge_scores['rougeL'].fmeasure
        else:
            accuracy = answer_relevance  # Fallback to query relevance
            rouge_l = 0.0

        metrics['overall'] = {
            'accuracy': accuracy,
            'rouge_l': rouge_l,
            'total_time': chunk_time + retrieval_time + gen_time,
            'estimated_cost': self._estimate_cost(pipeline, len(chunks), len(retrieved), answer)
        }

        if self.show_details:
            print(f"  [ACCURACY: {accuracy:.3f}]")

        return answer, metrics

    def _estimate_cost(self, pipeline, n_chunks: int, n_retrieved: int, answer: str) -> float:
        """Estimate API cost"""
        cost = 0.0

        # Embedding cost (if using OpenAI)
        if isinstance(pipeline.embedder, OpenAIEmbedder):
            cost += (n_chunks + 1) * 0.0001  # Chunks + query

        # Generation cost (if using OpenAI)
        if isinstance(pipeline.generator, OpenAIGenerator):
            # Rough token estimation
            input_tokens = (n_retrieved * 100) + 50  # Context + prompt
            output_tokens = len(answer.split()) * 1.3
            cost += (input_tokens / 1000) * 0.001  # Input tokens
            cost += (output_tokens / 1000) * 0.002  # Output tokens

        return cost


class MSMARCODataLoader:
    """Load MS MARCO dataset with answers"""

    def __init__(self, num_docs: int = 20, num_queries: int = 5):
        """
        Initialize MS MARCO data loader

        Args:
            num_docs: Number of documents to load
            num_queries: Number of queries with answers to load
        """
        self.num_docs = num_docs
        self.num_queries = num_queries

    def load_data(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load MS MARCO documents and queries with answers

        Returns:
            documents: List of document texts
            queries: List of query/answer pairs
        """
        try:
            logger.info("Loading MS MARCO dataset...")

            # Try to load MS MARCO
            dataset = load_dataset(
                "microsoft/ms_marco",
                "v2.1",
                split="train",
                streaming=True  # Use streaming to avoid downloading entire dataset
            )

            documents = []
            queries = []

            # Get documents and queries
            for i, item in enumerate(dataset):
                if i >= max(self.num_docs, self.num_queries):
                    break

                # Extract passages
                if len(documents) < self.num_docs:
                    for passage in item.get('passages', {}).get('passage_text', []):
                        if passage and len(passage) > 50:
                            documents.append(passage)
                            if len(documents) >= self.num_docs:
                                break

                # Extract queries with answers
                if len(queries) < self.num_queries:
                    if item.get('query') and item.get('answers'):
                        answer_list = item['answers']
                        if isinstance(answer_list, list) and answer_list:
                            queries.append({
                                'query': item['query'],
                                'answer': answer_list[0],
                                'query_id': item.get('query_id', f'q{i}')
                            })

            logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries from MS MARCO")
            return documents, queries

        except Exception as e:
            logger.warning(f"Could not load MS MARCO: {e}. Using fallback data.")

            # Fallback data with comprehensive answers
            documents = [
                "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
                "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
                "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward.",
                "Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.",
                "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.",
                "Unsupervised learning is a type of algorithm that learns patterns from untagged data without any supervision.",
                "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
                "Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function."
            ]

            queries = [
                {
                    'query': "What is machine learning?",
                    'answer': "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming.",
                    'query_id': 'q1'
                },
                {
                    'query': "How does deep learning work?",
                    'answer': "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input.",
                    'query_id': 'q2'
                },
                {
                    'query': "What is NLP used for?",
                    'answer': "NLP is used for interactions between computers and human language, enabling machines to understand, interpret, and generate human language.",
                    'query_id': 'q3'
                },
                {
                    'query': "What is reinforcement learning?",
                    'answer': "Reinforcement learning is a machine learning method where agents learn to make decisions by taking actions in an environment to maximize cumulative reward.",
                    'query_id': 'q4'
                },
                {
                    'query': "What is transfer learning?",
                    'answer': "Transfer learning is a technique that applies knowledge gained from solving one problem to solve different but related problems.",
                    'query_id': 'q5'
                }
            ]

            return documents[:self.num_docs], queries[:self.num_queries]


class EnhancedPipelineBuilder:
    """Build pipelines with detailed configuration"""

    def __init__(self, use_real_api: bool = False):
        self.use_real_api = use_real_api and bool(os.getenv('OPENAI_API_KEY'))

    def build_pipeline(self, config: Dict[str, Any]) -> Any:
        """Build pipeline from configuration"""
        pipeline = type('Pipeline', (), {})()

        # Chunking
        chunking_strategy = config.get('chunking_strategy', 'fixed')
        chunk_size = int(config.get('chunk_size', 512))

        if chunking_strategy == 'semantic':
            pipeline.chunker = SemanticChunker({
                'chunk_size': chunk_size,
                'threshold': 0.5
            })
        else:
            pipeline.chunker = FixedSizeChunker({
                'chunk_size': chunk_size,
                'overlap': 50
            })

        # Embedder
        if self.use_real_api:
            pipeline.embedder = OpenAIEmbedder({
                'model': 'text-embedding-ada-002',
                'api_key': os.getenv('OPENAI_API_KEY')
            })
        else:
            pipeline.embedder = MockEmbedder({})

        # Retriever
        retrieval_method = config.get('retrieval_method', 'dense')
        top_k = int(config.get('retrieval_top_k', 5))

        if retrieval_method == 'sparse':
            pipeline.retriever = BM25Retriever({
                'k1': 1.2,
                'b': 0.75
            })
        elif retrieval_method == 'hybrid':
            # Create both retrievers for hybrid approach
            from autorag.components.vector_stores.simple import SimpleVectorStore

            dense_retriever = DenseRetriever({
                'embedder': pipeline.embedder,
                'metric': 'cosine',
                'top_k': top_k
            })
            # Set embedder and vector store for dense retriever
            vector_store = SimpleVectorStore({})
            dense_retriever.set_components(pipeline.embedder, vector_store)

            sparse_retriever = BM25Retriever({
                'k1': 1.2,
                'b': 0.75
            })

            hybrid_weight = float(config.get('hybrid_weight', 0.5))
            pipeline.retriever = HybridRetriever({
                'dense_weight': hybrid_weight,
                'sparse_weight': 1.0 - hybrid_weight,
                'embedder': pipeline.embedder
            })
            # Set the sub-retrievers
            pipeline.retriever.dense_retriever = dense_retriever
            pipeline.retriever.sparse_retriever = sparse_retriever
        else:
            pipeline.retriever = DenseRetriever({
                'embedder': pipeline.embedder,
                'metric': 'cosine',
                'top_k': top_k
            })
            # Set embedder and vector store for dense retriever
            from autorag.components.vector_stores.simple import SimpleVectorStore
            vector_store = SimpleVectorStore({})
            pipeline.retriever.set_components(pipeline.embedder, vector_store)

        # Reranker
        if config.get('reranking_enabled', False):
            pipeline.reranker = CrossEncoderReranker({
                'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'top_k': int(config.get('top_k_rerank', 10))
            })
        else:
            pipeline.reranker = None

        # Generator
        temperature = float(config.get('temperature', 0.3))
        max_tokens = int(config.get('max_tokens', 150))

        if self.use_real_api:
            pipeline.generator = OpenAIGenerator({
                'model': 'gpt-3.5-turbo',
                'temperature': temperature,
                'max_tokens': max_tokens,
                'api_key': os.getenv('OPENAI_API_KEY')
            })
        else:
            pipeline.generator = MockGenerator({
                'temperature': temperature
            })

        pipeline.config = config
        pipeline.top_k = top_k

        return pipeline


def run_enhanced_bayesian_optimization(n_calls: int = 30, use_real_api: bool = False,
                                      num_docs: int = 20, num_queries: int = 5):
    """
    Run enhanced Bayesian optimization with detailed metrics

    Args:
        n_calls: Number of configurations to evaluate
        use_real_api: Whether to use real OpenAI API
    """
    print("\n" + "=" * 80)
    print("ENHANCED BAYESIAN OPTIMIZATION WITH MS MARCO")
    print("Shows interim stage metrics and uses MS MARCO answers for accuracy")
    print("=" * 80)

    # Load MS MARCO data
    loader = MSMARCODataLoader(num_docs=num_docs, num_queries=num_queries)
    documents, queries = loader.load_data()

    print(f"\nDataset:")
    print(f"  Documents: {len(documents)}")
    print(f"  Queries with answers: {len(queries)}")

    # Sample queries
    print(f"\nSample Queries:")
    for i, q in enumerate(queries[:3], 1):
        print(f"  {i}. Q: {q['query'][:60]}...")
        print(f"     A: {q['answer'][:60]}...")

    # Search space (full 432+ configurations)
    search_space = {
        'chunking_strategy': ['fixed', 'semantic'],
        'chunk_size': [256, 512],
        'retrieval_method': ['dense', 'sparse', 'hybrid'],
        'retrieval_top_k': [3, 5],
        'hybrid_weight': (0.3, 0.7),
        'reranking_enabled': [True, False],
        'top_k_rerank': [10, 20],
        'temperature': (0.0, 0.3),
        'max_tokens': (150, 300)
    }

    print(f"\nSearch Space: ~432 discrete combinations")
    print(f"Bayesian Optimization: {n_calls} evaluations")
    print(f"Reduction: {(1 - n_calls/432)*100:.1f}%")

    # Create evaluator
    builder = EnhancedPipelineBuilder(use_real_api)
    metrics_collector = EnhancedMetricsCollector(show_details=True)

    eval_count = [0]  # Use list to allow modification in nested function

    def evaluator(config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate configuration with detailed metrics"""
        eval_count[0] += 1

        print(f"\n{'='*60}")
        print(f"Evaluation #{eval_count[0]}")
        print(f"Config: chunk={config.get('chunking_strategy')}/{config.get('chunk_size')}, "
              f"retrieval={config.get('retrieval_method')}, "
              f"rerank={config.get('reranking_enabled')}")

        try:
            pipeline = builder.build_pipeline(config)

            # Evaluate on all queries
            all_accuracies = []
            all_rouge = []
            all_times = []
            total_cost = 0
            query_count = 0

            for query_data in queries:
                query_count += 1
                answer, metrics = metrics_collector.evaluate_with_detailed_metrics(
                    pipeline,
                    query_data['query'],
                    documents,
                    query_data
                )

                accuracy = metrics['overall']['accuracy']
                rouge = metrics['overall']['rouge_l']
                all_accuracies.append(accuracy)
                all_rouge.append(rouge)
                all_times.append(metrics['overall']['total_time'])
                total_cost += metrics['overall']['estimated_cost']

                # Print running mean and median after each query
                running_mean_accuracy = np.mean(all_accuracies)
                running_median_accuracy = np.median(all_accuracies)
                running_mean_rouge = np.mean(all_rouge)
                print(f"  [Query {query_count}/{len(queries)}] Mean Acc: {running_mean_accuracy:.3f}, Median Acc: {running_median_accuracy:.3f}, ROUGE-L: {running_mean_rouge:.3f}")

            # Calculate both mean and median metrics
            mean_accuracy = np.mean(all_accuracies) if all_accuracies else 0
            median_accuracy = np.median(all_accuracies) if all_accuracies else 0
            mean_rouge = np.mean(all_rouge) if all_rouge else 0
            median_rouge = np.median(all_rouge) if all_rouge else 0
            mean_time = np.mean(all_times) if all_times else 0

            print(f"\nFINAL METRICS (using MEDIAN for optimization):")
            print(f"  Median Accuracy (semantic): {median_accuracy:.3f} [OPTIMIZATION TARGET]")
            print(f"  Mean Accuracy (semantic): {mean_accuracy:.3f}")
            print(f"  Median ROUGE-L: {median_rouge:.3f}")
            print(f"  Mean ROUGE-L: {mean_rouge:.3f}")
            print(f"  Mean Latency: {mean_time:.3f}s")
            print(f"  Total Cost: ${total_cost:.4f}")

            # Debug output
            if median_accuracy == 0:
                logger.warning(f"Zero median accuracy detected! All accuracies: {all_accuracies}")

            # Use MEDIAN accuracy as the optimization score
            composite_score = median_accuracy  # Using median for robustness

            return {
                'metrics': {
                    'median_accuracy': median_accuracy,
                    'mean_accuracy': mean_accuracy,
                    'accuracy': median_accuracy,  # For backward compatibility
                    'rouge_l': median_rouge,
                    'composite_score': composite_score,
                    'latency': mean_time,
                    'cost': total_cost
                }
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'metrics': {'accuracy': 0.0, 'composite_score': 0.0}}

    # Create optimizer optimizing for accuracy
    optimizer = SimpleBayesianOptimizer(
        search_space=search_space,
        evaluator=evaluator,
        n_calls=n_calls,
        n_initial_points=5,
        objective='accuracy',  # Optimize 100% for accuracy
        minimize=False,
        random_state=42,
        save_results=True,
        results_dir='bayesian_enhanced_results'
    )

    print(f"\n{'='*80}")
    print("Starting Optimization (optimizing for 100% answer accuracy)...")
    print(f"{'='*80}")

    # Run optimization
    start_time = time.time()
    result = optimizer.optimize()
    total_time = time.time() - start_time

    # Display results
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

    print(f"\nBest Configuration Found:")
    print("-" * 40)
    for param, value in result.best_config.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.3f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nBest Performance:")
    print(f"  Median Accuracy Score: {result.best_score:.3f} (optimization target)")
    print(f"  Configurations Tested: {result.n_evaluations}")
    print(f"  Total Time: {total_time:.1f}s")

    print(f"\nOptimization Metric:")
    print(f"  The optimizer maximized: MEDIAN semantic similarity (not mean)")
    print(f"  This measures the typical performance across queries, ignoring outliers")
    print(f"  Median is more robust than mean for handling query difficulty variations")

    # Save results
    # Convert numpy types to JSON-serializable types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj

    with open('bayesian_enhanced_results.json', 'w') as f:
        json.dump({
            'best_config': make_serializable(result.best_config),
            'best_score': float(result.best_score),
            'n_evaluations': result.n_evaluations,
            'total_time': total_time,
            'optimization_metric': '100% answer_accuracy',
            'dataset': 'MS MARCO',
            'num_queries': len(queries),
            'num_docs': len(documents)
        }, f, indent=2)

    print(f"\nResults saved to: bayesian_enhanced_results.json")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Bayesian optimization with detailed metrics')
    parser.add_argument('--n-calls', type=int, default=30,
                       help='Number of configurations to evaluate')
    parser.add_argument('--real-api', action='store_true',
                       help='Use real OpenAI API')
    parser.add_argument('--num-docs', type=int, default=20,
                       help='Number of documents to load (default: 20)')
    parser.add_argument('--num-queries', type=int, default=5,
                       help='Number of queries to evaluate per configuration (default: 5)')

    args = parser.parse_args()

    # Install rouge-score if needed
    try:
        import rouge_score
    except ImportError:
        print("Installing rouge-score for evaluation...")
        os.system("pip install rouge-score")

    result = run_enhanced_bayesian_optimization(
        n_calls=args.n_calls,
        use_real_api=args.real_api,
        num_docs=args.num_docs,
        num_queries=args.num_queries
    )

    print("\n" + "=" * 80)
    print("ENHANCED BAYESIAN OPTIMIZATION COMPLETE!")
    print("=" * 80)