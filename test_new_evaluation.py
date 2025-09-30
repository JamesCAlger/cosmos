#!/usr/bin/env python
"""Test the new evaluation model in action"""

import time
from autorag.evaluation.semantic_metrics import SemanticMetrics

print("Testing new evaluation model...")
print("="*60)

# Initialize evaluator (will use new default model)
print("\n1. Initializing SemanticMetrics...")
start = time.time()
evaluator = SemanticMetrics()
init_time = time.time() - start

print(f"   Model: {evaluator.model}")
print(f"   Continuous scores: {evaluator.use_continuous_scores}")
print(f"   Initialization time: {init_time:.2f}s")

# Test data simulating RAG outputs
test_cases = [
    {
        "query": "What is the capital of France?",
        "generated": "Paris is the capital of France.",
        "ground_truth": "The capital of France is Paris."
    },
    {
        "query": "What is machine learning?",
        "generated": "Machine learning is a type of artificial intelligence that enables computers to learn from data.",
        "ground_truth": "Machine learning is a subset of AI that allows systems to learn and improve from experience."
    },
    {
        "query": "Who invented the telephone?",
        "generated": "Alexander Graham Bell invented the telephone.",
        "ground_truth": "The telephone was invented by Alexander Graham Bell in 1876."
    }
]

print(f"\n2. Running evaluation on {len(test_cases)} test cases...")
print("-"*60)

total_score = 0
for i, case in enumerate(test_cases, 1):
    # Get continuous similarity score
    score = evaluator.similarity_score(case["generated"], case["ground_truth"])
    total_score += score

    print(f"\nCase {i}:")
    print(f"  Query: {case['query']}")
    print(f"  Generated: {case['generated'][:50]}...")
    print(f"  Ground Truth: {case['ground_truth'][:50]}...")
    print(f"  Similarity Score: {score:.4f} (continuous)")

# Calculate mean score (as used in Bayesian optimization)
mean_score = total_score / len(test_cases)

print("\n" + "="*60)
print("RESULTS:")
print(f"  Mean similarity score: {mean_score:.4f}")
print(f"  Model used: all-mpnet-base-v2 (768 dims)")
print(f"  Scoring type: Continuous [0, 1]")
print("\nThis continuous score is what Bayesian optimization uses as its objective.")
print("Higher scores indicate better semantic similarity between generated and expected answers.")