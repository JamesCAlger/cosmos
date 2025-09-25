"""
Simple test of Bayesian optimization implementation.
Verifies Phase 1 is working correctly.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from autorag.optimization.bayesian_search import SimpleBayesianOptimizer
from autorag.optimization.search_space_converter import SearchSpaceConverter


def test_simple_function():
    """Test Bayesian optimization on a simple function"""
    print("Testing Bayesian Optimization on Simple Function")
    print("-" * 50)

    # Define a simple test function (negative quadratic with noise)
    def test_function(config):
        x = config.get('x', 0)
        y = config.get('y', 0)

        # Simple quadratic with maximum at (5, 5)
        score = -(x - 5)**2 - (y - 5)**2 + 100 + np.random.normal(0, 1)

        return {'metrics': {'accuracy': score}}

    # Define search space
    search_space = {
        'x': (0, 10),
        'y': (0, 10)
    }

    # Create optimizer
    optimizer = SimpleBayesianOptimizer(
        search_space=search_space,
        evaluator=test_function,
        n_calls=20,
        n_initial_points=5,
        objective='accuracy',
        minimize=False,
        save_results=False
    )

    # Run optimization
    result = optimizer.optimize()

    print(f"\nResults:")
    print(f"  Best configuration: x={result.best_config['x']:.2f}, y={result.best_config['y']:.2f}")
    print(f"  Best score: {result.best_score:.2f}")
    print(f"  Total evaluations: {result.n_evaluations}")
    print(f"  Expected optimum: x=5.00, y=5.00, score~100.00")

    # Check if we found something close to the optimum
    distance = ((result.best_config['x'] - 5)**2 + (result.best_config['y'] - 5)**2)**0.5
    print(f"  Distance from optimum: {distance:.2f}")

    success = distance < 2.0  # Within 2 units of optimum
    print(f"  Test {'PASSED' if success else 'FAILED'}")

    return success


def test_rag_config():
    """Test Bayesian optimization with RAG-like configuration"""
    print("\n" + "=" * 50)
    print("Testing with RAG-like Configuration")
    print("-" * 50)

    # Mock evaluator that prefers certain configurations
    def mock_rag_evaluator(config):
        score = 0.5  # Base score

        # Prefer medium chunk sizes
        if 'chunker.chunk_size' in config:
            chunk_size = config['chunker.chunk_size']
            if 200 <= chunk_size <= 300:
                score += 0.2

        # Prefer moderate overlap
        if 'chunker.overlap' in config:
            overlap = config['chunker.overlap']
            if 20 <= overlap <= 50:
                score += 0.1

        # Prefer moderate retrieval
        if 'retriever.top_k' in config:
            top_k = config['retriever.top_k']
            if 5 <= top_k <= 7:
                score += 0.15

        # Add some noise
        score += np.random.normal(0, 0.05)

        return {'metrics': {'accuracy': score}}

    # Define search space
    search_space = {
        'chunker.chunk_size': (128, 512),
        'chunker.overlap': (0, 100),
        'retriever.top_k': (3, 10)
    }

    # Create optimizer
    optimizer = SimpleBayesianOptimizer(
        search_space=search_space,
        evaluator=mock_rag_evaluator,
        n_calls=15,
        n_initial_points=5,
        objective='accuracy',
        minimize=False,
        save_results=True,
        results_dir='test_bayesian_results'
    )

    # Run optimization
    result = optimizer.optimize()

    print(f"\nResults:")
    print(f"  Best configuration:")
    for param, value in result.best_config.items():
        if isinstance(value, float):
            print(f"    {param}: {value:.1f}")
        else:
            print(f"    {param}: {value}")
    print(f"  Best score: {result.best_score:.3f}")
    print(f"  Total evaluations: {result.n_evaluations}")

    # Show convergence
    print(f"\nConvergence (last 5 iterations):")
    for i, score in enumerate(result.convergence_history[-5:],
                             start=len(result.convergence_history)-4):
        print(f"    Iteration {i}: {score:.3f}")

    success = result.best_score > 0.7  # Should achieve decent score
    print(f"\nTest {'PASSED' if success else 'FAILED'}")

    # Check if results were saved
    results_file = Path('test_bayesian_results') / 'bayesian_optimization_results.json'
    if results_file.exists():
        print(f"Results saved to: {results_file}")
    else:
        print("Warning: Results file not found")

    return success


def test_search_space_converter():
    """Test the search space converter"""
    print("\n" + "=" * 50)
    print("Testing Search Space Converter")
    print("-" * 50)

    converter = SearchSpaceConverter()

    # Test config ranges conversion
    config_ranges = {
        'chunker': {
            'chunk_size': [128, 256, 512],
            'overlap': [0, 50]
        },
        'retriever': {
            'top_k': [5, 10]
        }
    }

    search_space = converter.from_config_ranges(config_ranges)
    print(f"Converted search space: {search_space}")

    # Test pipeline config conversion
    flat_params = {
        'chunker.chunk_size': 256,
        'chunker.overlap': 25,
        'retriever.top_k': 7
    }

    pipeline_config = converter.to_pipeline_config(flat_params)
    print(f"\nConverted pipeline config: {pipeline_config}")

    success = (
        'chunker.chunk_size' in search_space and
        'chunker' in pipeline_config and
        pipeline_config['chunker']['chunk_size'] == 256
    )

    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    return success


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Bayesian Optimization Tests")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed = test_simple_function() and all_passed
    all_passed = test_rag_config() and all_passed
    all_passed = test_search_space_converter() and all_passed

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("Phase 1 Bayesian optimization is working correctly.")
    else:
        print("Some tests failed. Please review the output above.")
    print("=" * 60)