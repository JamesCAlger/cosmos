"""
COSMOS Framework Demo Script

Demonstrates compositional RAG optimization using the COSMOS framework.
Optimizes chunker, retriever, and generator sequentially with context passing.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import argparse
from typing import Dict, Any, List
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# COSMOS imports
from autorag.cosmos.optimization import CompositionalOptimizerBuilder
from autorag.cosmos.metrics import ComponentMetrics
from autorag.evaluation.semantic_metrics import SemanticMetrics


def load_test_data(num_docs: int = 10, num_queries: int = 3) -> Dict[str, Any]:
    """
    Load test data for optimization

    Args:
        num_docs: Number of documents to load
        num_queries: Number of queries to use

    Returns:
        Dict with 'documents' and 'queries' fields
    """
    logger.info(f"Loading test data: {num_docs} docs, {num_queries} queries")

    # Fallback data (can be replaced with MS MARCO loader)
    documents = [
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. " * 3,
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. The networks can be supervised, semi-supervised or unsupervised. " * 3,
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. " * 3,
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. " * 3,
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. It differs from supervised learning in that correct input-output pairs need not be presented. " * 3,
        "Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. " * 3,
        "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. " * 3,
        "Unsupervised learning is a type of algorithm that learns patterns from untagged data. The hope is that through mimicry, the machine is forced to build a compact internal representation of its world and then generate imaginative content. " * 3,
        "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. They are based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. " * 3,
        "Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient of the function at the current point. " * 3
    ]

    queries = [
        {
            'query': "What is machine learning?",
            'answer': "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming."
        },
        {
            'query': "How does deep learning work?",
            'answer': "Deep learning uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input data."
        },
        {
            'query': "What is NLP used for?",
            'answer': "NLP is used for interactions between computers and human language, enabling machines to process and analyze large amounts of natural language data."
        },
        {
            'query': "What is reinforcement learning?",
            'answer': "Reinforcement learning is a machine learning method where agents learn to make decisions by taking actions in an environment to maximize cumulative reward."
        },
        {
            'query': "What is transfer learning?",
            'answer': "Transfer learning applies knowledge gained from solving one problem to solve different but related problems, like using car recognition to help recognize trucks."
        }
    ]

    return {
        'documents': documents[:num_docs],
        'queries': queries[:num_queries]
    }


def define_search_spaces() -> Dict[str, Dict[str, Any]]:
    """
    Define search spaces for each component

    Returns:
        Dict mapping component_id -> search_space
    """
    search_spaces = {
        'chunker': {
            'chunking_strategy': ['fixed', 'semantic'],
            'chunk_size': [128, 256, 512],
            'overlap': [0, 25, 50]
        },
        'retriever': {
            'retrieval_method': ['sparse', 'dense'],  # Use sparse for speed
            'retrieval_top_k': [3, 5, 7]
        },
        'generator': {
            'use_real_api': [False],  # Mock generator for demo
            'temperature': [0.3, 0.5, 0.7]
        }
    }

    return search_spaces


def run_cosmos_optimization(
    components: List[str],
    strategy: str = 'random',
    total_budget: int = 15,
    num_docs: int = 10,
    num_queries: int = 3,
    n_initial_points: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run COSMOS compositional optimization

    Args:
        components: List of components to optimize (e.g., ['chunker', 'retriever'])
        strategy: Optimization strategy ('bayesian' or 'random')
        total_budget: Total evaluation budget
        num_docs: Number of test documents
        num_queries: Number of test queries
        n_initial_points: Initial random points for Bayesian (ignored for random)
        random_state: Random seed

    Returns:
        Dictionary with optimization results
    """
    print("\n" + "=" * 80)
    print("COSMOS COMPOSITIONAL OPTIMIZATION DEMO")
    print("=" * 80)
    print(f"Strategy: {strategy.upper()}")
    print(f"Components: {components}")
    print(f"Total Budget: {total_budget}")
    print(f"Test Data: {num_docs} docs, {num_queries} queries")
    print("=" * 80)

    # Load test data
    print("\n[1/5] Loading test data...")
    test_data = load_test_data(num_docs=num_docs, num_queries=num_queries)
    print(f"  [OK] Loaded {len(test_data['documents'])} documents")
    print(f"  [OK] Loaded {len(test_data['queries'])} queries")

    # Initialize metric collector
    print("\n[2/5] Initializing metric collector...")
    semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
    metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)
    print("  [OK] ComponentMetrics initialized with all-MiniLM-L6-v2")

    # Define search spaces
    print("\n[3/5] Defining search spaces...")
    all_search_spaces = define_search_spaces()
    search_spaces = {comp: all_search_spaces[comp] for comp in components}
    for comp_id, space in search_spaces.items():
        print(f"  [OK] {comp_id}: {list(space.keys())}")

    # Create optimizer
    print(f"\n[4/5] Creating compositional optimizer ({strategy})...")
    if strategy.lower() == 'bayesian':
        optimizer = CompositionalOptimizerBuilder.create_with_bayesian(
            n_initial_points=n_initial_points,
            random_state=random_state
        )
    else:
        optimizer = CompositionalOptimizerBuilder.create_with_random(
            random_state=random_state
        )
    print(f"  [OK] Optimizer created: {optimizer.strategy.get_name()}")

    # Run optimization
    print("\n[5/5] Running optimization...")
    print("-" * 80)

    start_time = time.time()
    results = optimizer.optimize(
        components=components,
        search_spaces=search_spaces,
        test_data=test_data,
        metric_collector=metric_collector,
        total_budget=total_budget
    )
    total_time = time.time() - start_time

    print("-" * 80)
    print(f"\n[DONE] Optimization complete in {total_time:.2f}s")

    # Get results
    pipeline_config = optimizer.get_best_pipeline_config()
    summary = optimizer.get_optimization_summary()

    # Display results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print("\nBest Configuration Per Component:")
    print("-" * 80)
    for comp_id in components:
        if comp_id in results:
            print(f"\n{comp_id.upper()}:")
            print(f"  Score: {results[comp_id].best_score:.4f}")
            print(f"  Evaluations: {results[comp_id].n_evaluations}")
            print(f"  Config:")
            for param, value in results[comp_id].best_config.items():
                print(f"    {param}: {value}")

    print("\n" + "-" * 80)
    print("Overall Statistics:")
    print(f"  Total evaluations: {summary['total_evaluations']}")
    print(f"  Average score: {summary['average_score']:.4f}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per evaluation: {total_time / summary['total_evaluations']:.2f}s")

    # Prepare return data
    return {
        'components': components,
        'strategy': strategy,
        'total_budget': total_budget,
        'total_time': total_time,
        'results': {
            comp_id: {
                'best_score': result.best_score,
                'best_config': result.best_config,
                'n_evaluations': result.n_evaluations
            }
            for comp_id, result in results.items()
        },
        'summary': summary
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='COSMOS Compositional RAG Optimization Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize chunker only with random search
  python scripts/run_cosmos_optimization.py --components chunker --budget 10

  # Optimize chunker and retriever with Bayesian
  python scripts/run_cosmos_optimization.py --components chunker retriever --strategy bayesian --budget 20

  # Optimize full pipeline
  python scripts/run_cosmos_optimization.py --components chunker retriever generator --strategy bayesian --budget 30

  # Use more test data
  python scripts/run_cosmos_optimization.py --components chunker retriever --budget 20 --num-docs 20 --num-queries 5
        """
    )

    parser.add_argument('--components', nargs='+',
                       choices=['chunker', 'retriever', 'generator'],
                       default=['chunker', 'retriever'],
                       help='Components to optimize (default: chunker retriever)')

    parser.add_argument('--strategy', choices=['bayesian', 'random'],
                       default='random',
                       help='Optimization strategy (default: random)')

    parser.add_argument('--budget', type=int, default=15,
                       help='Total evaluation budget (default: 15)')

    parser.add_argument('--num-docs', type=int, default=10,
                       help='Number of test documents (default: 10)')

    parser.add_argument('--num-queries', type=int, default=3,
                       help='Number of test queries (default: 3)')

    parser.add_argument('--n-initial', type=int, default=3,
                       help='Bayesian: initial random points (default: 3)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    parser.add_argument('--output', type=str, default='cosmos_results.json',
                       help='Output file for results (default: cosmos_results.json)')

    args = parser.parse_args()

    # Run optimization
    results = run_cosmos_optimization(
        components=args.components,
        strategy=args.strategy,
        total_budget=args.budget,
        num_docs=args.num_docs,
        num_queries=args.num_queries,
        n_initial_points=args.n_initial,
        random_state=args.seed
    )

    # Save results
    print(f"\n[SAVING] Writing results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] Results saved")

    print("\n" + "=" * 80)
    print("COSMOS OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output}")
    print("\nTry different configurations:")
    print("  - Different components: --components chunker retriever generator")
    print("  - Different strategy: --strategy bayesian")
    print("  - More budget: --budget 30")
    print("  - More data: --num-docs 20 --num-queries 5")


if __name__ == "__main__":
    main()