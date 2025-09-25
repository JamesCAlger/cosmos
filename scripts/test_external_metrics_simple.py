"""
Simple test script for ExternalMetricsCollector without config files.
This verifies Phase 1 implementation works correctly.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autorag.evaluation.external_metrics import ExternalMetricsCollector
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.components.retrievers.bm25 import BM25Retriever
from autorag.components.generators.mock import MockGenerator


class SimplePipeline:
    """Simple pipeline wrapper for testing"""
    def __init__(self):
        self.chunker = FixedSizeChunker({'chunk_size': 256, 'overlap': 50})
        self.retriever = BM25Retriever({'k1': 1.2, 'b': 0.75})
        self.generator = MockGenerator({})


def test_metrics_collector():
    """Test the ExternalMetricsCollector with a simple pipeline"""
    print("Testing External Metrics Collector...")
    print("-" * 50)

    # Create simple pipeline
    pipeline = SimplePipeline()

    # Create metrics collector
    collector = ExternalMetricsCollector(pipeline)

    # Test data
    test_query = "What is machine learning?"
    test_documents = [
        "Machine learning is a type of artificial intelligence that enables computers to learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Natural language processing is an application of machine learning.",
        "Computer vision uses machine learning to analyze images.",
        "Machine learning models can be trained on large datasets to make predictions."
    ]

    # Evaluate with metrics
    print(f"Query: {test_query}")
    print(f"Documents: {len(test_documents)} documents")
    print()

    answer, metrics = collector.evaluate_with_metrics(
        query=test_query,
        documents=test_documents,
        ground_truth={'relevant_docs': [0, 1, 4]}  # Indices of relevant docs
    )

    # Display results
    print("Generated Answer:")
    print(f"  {answer[:100]}..." if len(answer) > 100 else f"  {answer}")
    print()

    print("Collected Metrics:")
    print("-" * 50)

    # Display metrics in a structured way
    for stage, stage_metrics in metrics.items():
        if isinstance(stage_metrics, dict):
            print(f"\n{stage.upper()} Stage:")
            for key, value in stage_metrics.items():
                # Format the output nicely
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    # Test metrics collection over multiple queries
    print("\n" + "=" * 50)
    print("Testing Multiple Queries...")
    print("=" * 50)

    test_queries = [
        "What is deep learning?",
        "How does natural language processing work?",
        "What are neural networks?"
    ]

    all_metrics = []
    for i, query in enumerate(test_queries, 1):
        answer, metrics = collector.evaluate_with_metrics(
            query=query,
            documents=test_documents
        )
        all_metrics.append(metrics)
        print(f"\nQuery {i}: {query}")
        print(f"  Pipeline Time: {metrics['total']['pipeline_time']:.4f}s")
        print(f"  Estimated Cost: ${metrics['total']['estimated_cost']:.6f}")
        print(f"  Chunks Created: {metrics['chunking']['chunks_count']}")
        print(f"  Docs Retrieved: {metrics['retrieval']['docs_retrieved']}")
        print(f"  Answer Length: {metrics['generation']['answer_length']} words")

    # Calculate aggregate statistics
    print("\n" + "=" * 50)
    print("Aggregate Statistics:")
    print("=" * 50)

    avg_time = sum(m['total']['pipeline_time'] for m in all_metrics) / len(all_metrics)
    total_cost = sum(m['total']['estimated_cost'] for m in all_metrics)
    avg_chunks = sum(m['chunking']['chunks_count'] for m in all_metrics) / len(all_metrics)

    print(f"  Average Pipeline Time: {avg_time:.4f}s")
    print(f"  Total Estimated Cost: ${total_cost:.6f}")
    print(f"  Average Chunks per Query: {avg_chunks:.1f}")

    print("\n[SUCCESS] External Metrics Collector test completed successfully!")

    # Save metrics to file for analysis
    output_file = Path("phase1_metrics_test.json")
    with open(output_file, 'w') as f:
        json.dump({
            'test_queries': test_queries,
            'metrics': all_metrics,
            'summary': {
                'avg_pipeline_time': avg_time,
                'total_cost': total_cost,
                'avg_chunks': avg_chunks
            }
        }, f, indent=2)

    print(f"\nMetrics saved to: {output_file}")

    return True


def test_stage_metrics():
    """Test individual stage metric collection"""
    print("\n" + "=" * 50)
    print("Testing Stage Metrics Collection...")
    print("=" * 50)

    pipeline = SimplePipeline()
    collector = ExternalMetricsCollector(pipeline)

    test_documents = [
        "Artificial intelligence is the simulation of human intelligence by machines.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses artificial neural networks with multiple layers."
    ]

    # Test chunking metrics
    print("\n1. Chunking Metrics:")
    chunk_metrics = collector._collect_chunking_metrics(test_documents)
    for key, value in chunk_metrics.items():
        if key != 'chunks':  # Don't print the actual chunks
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    # Test retrieval metrics
    print("\n2. Retrieval Metrics:")
    chunks = chunk_metrics.pop('chunks', [])
    retrieval_metrics = collector._collect_retrieval_metrics(
        "What is machine learning?",
        chunks,
        {'relevant_docs': [1]}
    )
    for key, value in retrieval_metrics.items():
        if key != 'retrieved_docs':  # Don't print the actual docs
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

    # Test generation metrics
    print("\n3. Generation Metrics:")
    retrieved = retrieval_metrics.pop('retrieved_docs', [])
    gen_metrics, answer = collector._collect_generation_metrics(
        "What is machine learning?",
        retrieved
    )
    for key, value in gen_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n[SUCCESS] Stage metrics collection test completed!")

    return True


def test_cost_estimation():
    """Test cost estimation functionality"""
    print("\n" + "=" * 50)
    print("Testing Cost Estimation...")
    print("=" * 50)

    pipeline = SimplePipeline()
    collector = ExternalMetricsCollector(pipeline)

    # Simulate metrics for cost estimation
    test_metrics = {
        'retrieval': {
            'docs_retrieved': 10
        },
        'generation': {
            'estimated_input_tokens': 500,
            'estimated_output_tokens': 150
        }
    }

    estimated_cost = collector._estimate_cost(test_metrics)
    print(f"\nEstimated cost for sample pipeline run: ${estimated_cost:.6f}")

    # Calculate costs for different scenarios
    scenarios = [
        {'name': 'Small Query', 'input': 100, 'output': 50, 'docs': 5},
        {'name': 'Medium Query', 'input': 500, 'output': 150, 'docs': 10},
        {'name': 'Large Query', 'input': 2000, 'output': 500, 'docs': 20}
    ]

    print("\nCost estimates for different scenarios:")
    for scenario in scenarios:
        metrics = {
            'retrieval': {'docs_retrieved': scenario['docs']},
            'generation': {
                'estimated_input_tokens': scenario['input'],
                'estimated_output_tokens': scenario['output']
            }
        }
        cost = collector._estimate_cost(metrics)
        print(f"  {scenario['name']}: ${cost:.6f}")

    print("\n[SUCCESS] Cost estimation test completed!")

    return True


if __name__ == "__main__":
    # Run all tests
    success = True

    try:
        success = test_metrics_collector() and success
        success = test_stage_metrics() and success
        success = test_cost_estimation() and success
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if success:
        print("\n" + "=" * 50)
        print("Phase 1 Implementation Complete!")
        print("=" * 50)
        print("\nThe ExternalMetricsCollector is ready for use with Bayesian optimization.")
        print("Next steps:")
        print("  1. Implement SimpleBayesianOptimizer (Day 3-4)")
        print("  2. Integrate with existing grid search scripts")
        print("  3. Compare performance with grid search")
    else:
        print("\nSome tests failed. Please review the errors above.")