"""
Run Bayesian optimization for RAG pipeline configuration.
This script integrates Phase 1 components for end-to-end optimization.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autorag.optimization.bayesian_search import SimpleBayesianOptimizer
from autorag.optimization.search_space_converter import SearchSpaceConverter
from autorag.evaluation.external_metrics import ExternalMetricsCollector
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.components.chunkers.sliding_window import SlidingWindowChunker
from autorag.components.retrievers.bm25 import BM25Retriever
from autorag.components.retrievers.dense import DenseRetriever
from autorag.components.generators.mock import MockGenerator
from autorag.components.embedders.mock import MockEmbedder

# For real evaluation (if API key available)
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_OPENAI = bool(os.getenv('OPENAI_API_KEY'))
except:
    HAS_OPENAI = False


class PipelineFactory:
    """Factory for creating pipeline instances from configurations"""

    @staticmethod
    def create_pipeline(config: Dict[str, Any]):
        """Create a pipeline instance from configuration"""
        pipeline = type('Pipeline', (), {})()

        # Create chunker
        chunker_config = config.get('chunker', {})
        chunker_type = chunker_config.get('type', 'fixed_size')

        if chunker_type == 'fixed_size':
            pipeline.chunker = FixedSizeChunker({
                'chunk_size': chunker_config.get('chunk_size', 256),
                'overlap': chunker_config.get('overlap', 50)
            })
        elif chunker_type == 'sliding_window':
            pipeline.chunker = SlidingWindowChunker({
                'window_size': chunker_config.get('chunk_size', 256),
                'step_size': chunker_config.get('chunk_size', 256) - chunker_config.get('overlap', 50)
            })
        else:
            pipeline.chunker = FixedSizeChunker({'chunk_size': 256})

        # Create embedder (mock for now)
        pipeline.embedder = MockEmbedder({})

        # Create retriever
        retriever_config = config.get('retriever', {})
        retriever_type = retriever_config.get('type', 'bm25')

        if retriever_type == 'bm25':
            pipeline.retriever = BM25Retriever({
                'k1': retriever_config.get('k1', 1.2),
                'b': retriever_config.get('b', 0.75)
            })
        elif retriever_type == 'dense':
            pipeline.retriever = DenseRetriever({
                'embedder': pipeline.embedder,
                'metric': retriever_config.get('metric', 'cosine'),
                'top_k': retriever_config.get('top_k', 5)
            })
        else:
            pipeline.retriever = BM25Retriever({'k1': 1.2, 'b': 0.75})

        # Create generator
        generator_config = config.get('generator', {})
        if HAS_OPENAI and generator_config.get('use_openai', False):
            from autorag.components.generators.openai import OpenAIGenerator
            pipeline.generator = OpenAIGenerator({
                'model': generator_config.get('model', 'gpt-3.5-turbo'),
                'temperature': generator_config.get('temperature', 0.7),
                'max_tokens': generator_config.get('max_tokens', 150)
            })
        else:
            pipeline.generator = MockGenerator({})

        return pipeline


def create_evaluator(test_queries: List[Dict[str, Any]], documents: List[str]):
    """
    Create an evaluator function for Bayesian optimization.

    Args:
        test_queries: List of test queries with ground truth
        documents: List of documents to process

    Returns:
        Evaluator function that takes config and returns metrics
    """
    def evaluate_config(flat_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single configuration"""

        # Convert flat config to nested format
        converter = SearchSpaceConverter()
        config = converter.to_pipeline_config(flat_config)

        # Create pipeline
        try:
            pipeline = PipelineFactory.create_pipeline(config)
        except Exception as e:
            print(f"Failed to create pipeline: {e}")
            return {'metrics': {'accuracy': 0.0, 'error': str(e)}}

        # Create metrics collector
        collector = ExternalMetricsCollector(pipeline)

        # Evaluate on test queries
        results = []
        total_time = 0
        total_cost = 0

        for query_data in test_queries:
            query = query_data['query']
            ground_truth = query_data.get('ground_truth', {})

            try:
                answer, metrics = collector.evaluate_with_metrics(
                    query=query,
                    documents=documents,
                    ground_truth=ground_truth
                )

                # Calculate accuracy (simple overlap for now)
                if 'expected_answer' in ground_truth:
                    expected = ground_truth['expected_answer'].lower()
                    actual = answer.lower()
                    common_words = set(expected.split()) & set(actual.split())
                    accuracy = len(common_words) / max(len(expected.split()), 1)
                else:
                    accuracy = metrics.get('quality', {}).get('answer_relevance_to_query', 0.5)

                results.append({
                    'accuracy': accuracy,
                    'time': metrics['total']['pipeline_time'],
                    'cost': metrics['total']['estimated_cost'],
                    'chunks': metrics['chunking']['chunks_count'],
                    'retrieved': metrics['retrieval']['docs_retrieved']
                })

                total_time += metrics['total']['pipeline_time']
                total_cost += metrics['total']['estimated_cost']

            except Exception as e:
                print(f"Evaluation error for query '{query}': {e}")
                results.append({'accuracy': 0.0, 'time': 0, 'cost': 0})

        # Aggregate metrics
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_time = np.mean([r['time'] for r in results])

        return {
            'metrics': {
                'accuracy': avg_accuracy,
                'latency': avg_time,
                'cost': total_cost,
                'avg_chunks': np.mean([r.get('chunks', 0) for r in results]),
                'avg_retrieved': np.mean([r.get('retrieved', 0) for r in results])
            }
        }

    return evaluate_config


def run_bayesian_optimization():
    """Run Bayesian optimization on RAG pipeline"""
    print("=" * 60)
    print("Starting Bayesian Optimization for RAG Pipeline")
    print("=" * 60)

    # Prepare test data
    test_queries = [
        {
            'query': "What is machine learning?",
            'ground_truth': {
                'expected_answer': "Machine learning is a type of artificial intelligence that enables computers to learn from data.",
                'relevant_docs': [0, 1]
            }
        },
        {
            'query': "What is deep learning?",
            'ground_truth': {
                'expected_answer': "Deep learning is a subset of machine learning that uses neural networks.",
                'relevant_docs': [1]
            }
        },
        {
            'query': "What are neural networks?",
            'ground_truth': {
                'expected_answer': "Neural networks are computing systems inspired by biological neural networks.",
                'relevant_docs': [1, 2]
            }
        }
    ]

    documents = [
        "Machine learning is a type of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to progressively extract features.",
        "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
        "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
        "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world."
    ]

    # Define search space (simplified for testing)
    search_space = {
        'chunker.type': ['fixed_size', 'sliding_window'],
        'chunker.chunk_size': (128, 512),
        'chunker.overlap': (0, 100),
        'retriever.type': ['bm25'],  # Only BM25 for mock testing
        'retriever.top_k': (3, 10),
        'retriever.k1': (0.5, 2.0),
        'retriever.b': (0.0, 1.0),
        'generator.temperature': (0.0, 1.0),
        'generator.max_tokens': (50, 200)
    }

    print(f"\nSearch Space:")
    print(f"  Dimensions: {len(search_space)}")
    for param, spec in search_space.items():
        print(f"    {param}: {spec}")

    # Create evaluator
    evaluator = create_evaluator(test_queries, documents)

    # Initialize Bayesian optimizer
    optimizer = SimpleBayesianOptimizer(
        search_space=search_space,
        evaluator=evaluator,
        n_calls=30,  # Fewer calls than grid search
        n_initial_points=10,
        objective='accuracy',
        minimize=False,
        random_state=42,
        save_results=True,
        results_dir='bayesian_results'
    )

    # Run optimization
    print(f"\nStarting optimization with {optimizer.n_calls} evaluations...")
    print("(Grid search would require ~960 evaluations)")
    print("-" * 40)

    start_time = time.time()
    result = optimizer.optimize()
    total_time = time.time() - start_time

    # Display results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)

    print(f"\nBest Configuration Found:")
    for param, value in result.best_config.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.3f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nPerformance Metrics:")
    print(f"  Best Accuracy: {result.best_score:.4f}")
    print(f"  Total Evaluations: {result.n_evaluations}")
    print(f"  Total Time: {total_time:.1f} seconds")
    print(f"  Time per Evaluation: {total_time/result.n_evaluations:.2f} seconds")

    # Compare with grid search estimate
    grid_search_configs = 1
    for param, spec in search_space.items():
        if isinstance(spec, list):
            grid_search_configs *= len(spec)
        elif isinstance(spec, tuple):
            # Assume 5 points for continuous parameters
            grid_search_configs *= 5

    print(f"\nComparison with Grid Search:")
    print(f"  Grid Search Configurations: ~{grid_search_configs}")
    print(f"  Bayesian Configurations: {result.n_evaluations}")
    print(f"  Reduction: {(1 - result.n_evaluations/grid_search_configs)*100:.1f}%")

    # Show convergence
    print(f"\nConvergence History (last 10 iterations):")
    for i, score in enumerate(result.convergence_history[-10:], start=len(result.convergence_history)-9):
        print(f"  Iteration {i}: {score:.4f}")

    # Save detailed results
    results_file = Path('bayesian_results') / 'detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'best_config': result.best_config,
            'best_score': float(result.best_score),
            'n_evaluations': result.n_evaluations,
            'total_time': total_time,
            'search_space': {k: str(v) for k, v in search_space.items()},
            'convergence': [float(s) for s in result.convergence_history],
            'comparison': {
                'grid_search_configs': grid_search_configs,
                'bayesian_configs': result.n_evaluations,
                'reduction_percent': (1 - result.n_evaluations/grid_search_configs)*100
            }
        }, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    return result


def compare_with_baseline():
    """Compare Bayesian optimization with random baseline"""
    print("\n" + "=" * 60)
    print("Comparing with Random Search Baseline")
    print("=" * 60)

    # Run random search for comparison
    from autorag.optimization.bayesian_search import SimpleBayesianOptimizer

    search_space = {
        'chunker.chunk_size': (128, 512),
        'chunker.overlap': (0, 100),
        'retriever.top_k': (3, 10),
        'retriever.k1': (0.5, 2.0),
        'retriever.b': (0.0, 1.0)
    }

    # Simple test data
    test_queries = [{'query': "What is AI?", 'ground_truth': {}}]
    documents = ["Artificial intelligence is the simulation of human intelligence."]

    evaluator = create_evaluator(test_queries, documents)

    # Random search (all initial points)
    random_optimizer = SimpleBayesianOptimizer(
        search_space=search_space,
        evaluator=evaluator,
        n_calls=20,
        n_initial_points=20,  # All random
        objective='accuracy',
        save_results=False
    )

    # Bayesian search
    bayesian_optimizer = SimpleBayesianOptimizer(
        search_space=search_space,
        evaluator=evaluator,
        n_calls=20,
        n_initial_points=5,  # Only 5 random, rest guided
        objective='accuracy',
        save_results=False
    )

    print("\nRunning Random Search (20 evaluations)...")
    random_result = random_optimizer.optimize()

    print("Running Bayesian Search (20 evaluations)...")
    bayesian_result = bayesian_optimizer.optimize()

    print("\nComparison Results:")
    print(f"  Random Search Best: {random_result.best_score:.4f}")
    print(f"  Bayesian Search Best: {bayesian_result.best_score:.4f}")
    print(f"  Improvement: {(bayesian_result.best_score - random_result.best_score):.4f}")

    return random_result, bayesian_result


if __name__ == "__main__":
    # Check for scikit-optimize
    try:
        import skopt
        print("[OK] scikit-optimize is installed")
    except ImportError:
        print("ERROR: scikit-optimize not installed")
        print("Please install with: pip install scikit-optimize")
        sys.exit(1)

    # Run main optimization
    result = run_bayesian_optimization()

    # Optional: Compare with baseline
    print("\n" + "=" * 60)
    response = input("Compare with random search baseline? (y/n): ")
    if response.lower() == 'y':
        compare_with_baseline()

    print("\n" + "=" * 60)
    print("Bayesian Optimization Complete!")
    print("=" * 60)