"""
Bayesian optimization with MS MARCO dataset and real OpenAI API calls.
Equivalent to the 3x3x3 grid search but using Bayesian optimization for efficiency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import json
import time
import numpy as np
from typing import Dict, Any, List
from loguru import logger
import openai
from datasets import load_dataset
import traceback

# Setup logging
logger.add("bayesian_msmarco.log", rotation="10 MB")

# Import our Bayesian optimization components
from autorag.optimization.bayesian_search import SimpleBayesianOptimizer
from autorag.optimization.search_space_converter import SearchSpaceConverter
from autorag.evaluation.external_metrics import ExternalMetricsCollector

# Import components from the grid search script
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.components.retrievers.bm25 import BM25Retriever
from autorag.components.generators.openai import OpenAIGenerator
from autorag.components.generators.mock import MockGenerator
from autorag.components.base import Document

# For semantic evaluation
from autorag.evaluation.semantic_metrics import SemanticMetrics


class MSMARCOBayesianOptimizer:
    """Bayesian optimization for RAG pipeline using MS MARCO dataset"""

    def __init__(self, num_docs: int = 10, num_queries: int = 3,
                 api_delay: float = 0.15, use_real_api: bool = True,
                 evaluation_method: str = "semantic_fixed",
                 semantic_threshold: float = 0.75):
        """
        Initialize Bayesian optimizer for MS MARCO.

        Args:
            num_docs: Number of documents to use
            num_queries: Number of queries to evaluate
            api_delay: Delay between API calls (0.15s for free tier safety)
            use_real_api: Whether to use real OpenAI API or mock
            evaluation_method: How to evaluate ("semantic_fixed" or "keyword")
            semantic_threshold: Threshold for semantic similarity
        """
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.api_delay = api_delay
        self.use_real_api = use_real_api and bool(os.getenv('OPENAI_API_KEY'))
        self.evaluation_method = evaluation_method
        self.semantic_threshold = semantic_threshold

        # Track API usage
        self.api_call_count = 0
        self.total_cost = 0.0

        # Initialize semantic evaluator if needed
        self.semantic_evaluator = None
        if evaluation_method == "semantic_fixed":
            self.semantic_evaluator = SemanticMetrics(
                model_name='all-MiniLM-L6-v2',
                similarity_threshold=semantic_threshold
            )
            logger.info(f"Using semantic evaluation with threshold {semantic_threshold}")

        if not self.use_real_api:
            logger.warning("No OpenAI API key found or mock mode selected. Using mock generator.")

    def load_minimal_data(self):
        """Load minimal MS MARCO dataset"""
        logger.info(f"Loading MS MARCO dataset with {self.num_docs} docs and {self.num_queries} queries...")

        # For quick testing, use fallback data immediately
        # Comment this out to use real MS MARCO data
        use_fallback = True

        if use_fallback:
            logger.info("Using fallback test data for quick testing")
            documents = [
                "Python is a high-level programming language known for its simplicity and readability.",
                "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
                "The speed of light in vacuum is approximately 299,792 kilometers per second.",
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "Natural language processing helps computers understand, interpret, and generate human language.",
                "Deep learning uses neural networks with multiple layers to learn from large amounts of data.",
                "Computer vision enables machines to interpret and understand visual information from the world.",
                "The Great Wall of China is one of the most famous landmarks in the world.",
                "Quantum computing uses quantum mechanics principles to process information.",
                "Climate change refers to long-term shifts in global temperatures and weather patterns."
            ]
            queries = [
                {'query': "What is Python programming?", 'ground_truth': "Python is a high-level programming language"},
                {'query': "Where is the Eiffel Tower located?", 'ground_truth': "The Eiffel Tower is in Paris, France"},
                {'query': "What is the speed of light?", 'ground_truth': "The speed of light is 299,792 km/s"},
                {'query': "What is machine learning?", 'ground_truth': "Machine learning is a subset of AI"},
                {'query': "What is NLP?", 'ground_truth': "Natural language processing helps computers understand human language"}
            ]
            return documents[:self.num_docs], queries[:self.num_queries]

        try:
            # Load MS MARCO dataset (slower, real data)
            dataset = load_dataset(
                "microsoft/ms_marco",
                "v2.1",
                split="train",
                trust_remote_code=False  # Changed to False
            )

            # Extract documents and queries
            documents = []
            queries = []

            # Get unique passages
            seen_passages = set()
            for item in dataset:
                for passage in item.get('passages', {}).get('passage_text', []):
                    if passage and passage not in seen_passages:
                        documents.append(passage)
                        seen_passages.add(passage)
                        if len(documents) >= self.num_docs:
                            break
                if len(documents) >= self.num_docs:
                    break

            # Get queries with answers
            for item in dataset:
                if item.get('query') and item.get('answers'):
                    query_text = item['query']
                    answer = item['answers'][0] if isinstance(item['answers'], list) else item['answers']
                    queries.append({
                        'query': query_text,
                        'ground_truth': answer
                    })
                    if len(queries) >= self.num_queries:
                        break

            logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
            return documents, queries

        except Exception as e:
            logger.error(f"Error loading MS MARCO: {e}")
            # Fallback to simple test data
            logger.warning("Using fallback test data")
            documents = [
                "Python is a high-level programming language known for its simplicity.",
                "The Eiffel Tower is located in Paris, France.",
                "The speed of light is approximately 299,792 kilometers per second.",
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing helps computers understand human language."
            ]
            queries = [
                {'query': "What is Python?", 'ground_truth': "Python is a programming language"},
                {'query': "Where is the Eiffel Tower?", 'ground_truth': "Paris, France"},
                {'query': "What is the speed of light?", 'ground_truth': "299,792 km/s"}
            ]
            return documents[:self.num_docs], queries[:self.num_queries]

    def create_pipeline(self, config: Dict[str, Any]):
        """Create RAG pipeline from configuration"""
        pipeline = type('Pipeline', (), {})()

        # Create chunker
        chunk_size = int(config.get('chunk_size', 512))
        pipeline.chunker = FixedSizeChunker({
            'chunk_size': chunk_size,
            'overlap': 50  # Fixed overlap
        })

        # Create retriever
        top_k = int(config.get('top_k', 5))
        pipeline.retriever = BM25Retriever({
            'k1': 1.2,  # Fixed BM25 parameters
            'b': 0.75
        })

        # Create generator
        temperature = float(config.get('temperature', 0.3))
        if self.use_real_api:
            pipeline.generator = OpenAIGenerator({
                'model': 'gpt-3.5-turbo',
                'temperature': temperature,
                'max_tokens': 100,  # Limit to save costs
                'api_key': os.getenv('OPENAI_API_KEY')
            })
        else:
            pipeline.generator = MockGenerator({
                'temperature': temperature
            })

        # Store config for retrieval
        pipeline.top_k = top_k

        return pipeline

    def evaluate_configuration(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single configuration"""
        try:
            # Load data
            documents, queries = self.load_minimal_data()

            # Create pipeline
            pipeline = self.create_pipeline(config)

            # Create metrics collector
            collector = ExternalMetricsCollector(pipeline)

            # Evaluate on queries
            scores = []
            total_time = 0
            answers = []
            ground_truths = []

            for i, query_data in enumerate(queries):
                query = query_data['query']
                ground_truth = query_data['ground_truth']

                # Add delay to avoid rate limiting
                if self.use_real_api and i > 0:
                    time.sleep(self.api_delay)

                # Run pipeline with metrics
                start = time.time()

                # Convert documents to Document objects
                doc_objects = [Document(content=doc, doc_id=str(i)) for i, doc in enumerate(documents)]

                # Chunk documents
                chunks = pipeline.chunker.chunk(doc_objects)

                # Index for retrieval
                pipeline.retriever.index(chunks)

                # Retrieve
                retrieved = pipeline.retriever.retrieve(query, top_k=pipeline.top_k)

                # Generate answer
                contexts = [r.content if hasattr(r, 'content') else str(r) for r in retrieved]
                answer = pipeline.generator.generate(query, contexts)

                total_time += time.time() - start

                # Track for evaluation
                answers.append(answer)
                ground_truths.append(ground_truth)

                # Update API usage
                if self.use_real_api:
                    self.api_call_count += 1
                    self.total_cost += 0.0005  # Rough estimate per API call

                logger.debug(f"Query: {query[:50]}...")
                logger.debug(f"Answer: {answer[:100]}...")

            # Calculate scores
            if self.evaluation_method == "semantic_fixed" and self.semantic_evaluator:
                # Semantic similarity evaluation
                similarities = self.semantic_evaluator.semantic_similarity_batch(answers, ground_truths)
                scores = [1 if sim >= self.semantic_threshold else 0 for sim in similarities]
                avg_similarity = np.mean(similarities)
                logger.info(f"Average semantic similarity: {avg_similarity:.3f}")
            else:
                # Keyword matching evaluation
                for answer, ground_truth in zip(answers, ground_truths):
                    answer_lower = answer.lower()
                    truth_lower = ground_truth.lower()
                    keywords = truth_lower.split()
                    matches = sum(1 for kw in keywords if kw in answer_lower)
                    score = matches / max(len(keywords), 1)
                    scores.append(score)

            # Return metrics
            accuracy = np.mean(scores)
            avg_latency = total_time / len(queries)

            logger.info(f"Config evaluation - Chunk: {config.get('chunk_size')}, "
                       f"TopK: {config.get('top_k')}, Temp: {config.get('temperature'):.1f} "
                       f"-> Accuracy: {accuracy:.3f}, Latency: {avg_latency:.2f}s")

            return {
                'accuracy': accuracy,
                'latency': avg_latency,
                'cost': self.total_cost,
                'api_calls': self.api_call_count
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'accuracy': 0.0,
                'latency': 999.0,
                'cost': 0.0,
                'api_calls': self.api_call_count
            }

    def run_optimization(self, n_calls: int = 15):
        """
        Run Bayesian optimization.

        Args:
            n_calls: Number of configurations to evaluate (default 15 vs 27 for grid search)
        """
        print("\n" + "=" * 80)
        print("BAYESIAN OPTIMIZATION WITH MS MARCO")
        print("=" * 80)
        print(f"Documents: {self.num_docs}")
        print(f"Queries: {self.num_queries}")
        print(f"Configurations to test: {n_calls} (vs 27 for grid search)")
        print(f"API Mode: {'REAL' if self.use_real_api else 'MOCK'}")
        print(f"Evaluation: {self.evaluation_method}")
        print("=" * 80)

        # Define search space (same as grid search but continuous)
        search_space = {
            'chunk_size': (256, 1024),      # Was [256, 512, 1024]
            'top_k': (3, 10),               # Was [3, 5, 10]
            'temperature': (0.0, 0.7)       # Was [0.0, 0.3, 0.7]
        }

        print("\nSearch Space:")
        for param, range_vals in search_space.items():
            print(f"  {param}: {range_vals}")

        # Create evaluator wrapper
        def evaluator(config):
            metrics = self.evaluate_configuration(config)
            # Return in format expected by optimizer
            return {'metrics': metrics}

        # Create and run optimizer
        optimizer = SimpleBayesianOptimizer(
            search_space=search_space,
            evaluator=evaluator,
            n_calls=n_calls,
            n_initial_points=5,  # Random exploration first
            objective='accuracy',
            minimize=False,
            random_state=42,
            save_results=True,
            results_dir='bayesian_msmarco_results'
        )

        print("\nStarting optimization...")
        print("-" * 60)
        print("Iter | Chunk | TopK | Temp | Accuracy | Latency | Status")
        print("-" * 60)

        # Run optimization
        start_time = time.time()
        result = optimizer.optimize()
        total_time = time.time() - start_time

        print("=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)

        # Display results
        print(f"\nBest Configuration Found:")
        print("-" * 40)
        for param, value in result.best_config.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.3f}")
            else:
                print(f"  {param}: {value}")

        print(f"\nPerformance:")
        print(f"  Best Accuracy: {result.best_score:.3f}")
        print(f"  Total Configurations Tested: {result.n_evaluations}")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Time per Config: {total_time/result.n_evaluations:.1f}s")

        # Compare with grid search
        print(f"\nComparison with Grid Search:")
        print(f"  Grid Search Configs: 27")
        print(f"  Bayesian Configs: {result.n_evaluations}")
        print(f"  Reduction: {(1 - result.n_evaluations/27)*100:.1f}%")

        if self.use_real_api:
            print(f"\nAPI Usage:")
            print(f"  Total API Calls: {self.api_call_count}")
            print(f"  Estimated Cost: ${self.total_cost:.4f}")
            print(f"  Cost Savings vs Grid: ${(27-n_calls)*self.num_queries*0.0005:.4f}")

        # Show convergence
        print(f"\nOptimization Convergence:")
        convergence_points = [1, 5, 10, min(15, len(result.convergence_history))]
        for point in convergence_points:
            if point <= len(result.convergence_history):
                print(f"  After {point:2d} evals: {result.convergence_history[point-1]:.3f}")

        # Save results (handle numpy types)
        def make_json_safe(obj):
            """Convert numpy types to Python types"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_file = "bayesian_msmarco_results.json"
        with open(results_file, "w") as f:
            json.dump(make_json_safe({
                'best_config': result.best_config,
                'best_score': float(result.best_score),
                'n_evaluations': result.n_evaluations,
                'total_time': total_time,
                'convergence': [float(s) for s in result.convergence_history],
                'search_space': {k: str(v) for k, v in search_space.items()},
                'settings': {
                    'num_docs': self.num_docs,
                    'num_queries': self.num_queries,
                    'use_real_api': self.use_real_api,
                    'evaluation_method': self.evaluation_method
                }
            }), f, indent=2)

        print(f"\nResults saved to: {results_file}")

        return result


def main():
    """Main function to run Bayesian optimization"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Bayesian optimization on MS MARCO')
    parser.add_argument('--docs', type=int, default=10, help='Number of documents')
    parser.add_argument('--queries', type=int, default=3, help='Number of queries')
    parser.add_argument('--n-calls', type=int, default=15, help='Number of configurations to test')
    parser.add_argument('--mock', action='store_true', help='Use mock API instead of real')
    parser.add_argument('--eval-method', choices=['semantic_fixed', 'keyword'],
                       default='semantic_fixed', help='Evaluation method')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Semantic similarity threshold')

    args = parser.parse_args()

    # Check for API key
    if not args.mock and not os.getenv('OPENAI_API_KEY'):
        print("\nWarning: No OpenAI API key found. Using mock mode.")
        print("To use real API, set OPENAI_API_KEY in your .env file")
        use_real = False
    else:
        use_real = not args.mock

    # Create optimizer
    optimizer = MSMARCOBayesianOptimizer(
        num_docs=args.docs,
        num_queries=args.queries,
        use_real_api=use_real,
        evaluation_method=args.eval_method,
        semantic_threshold=args.threshold
    )

    # Run optimization
    result = optimizer.run_optimization(n_calls=args.n_calls)

    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION COMPLETE")
    print("=" * 80)

    return result


if __name__ == "__main__":
    main()