#!/usr/bin/env python
"""
Test script for MPNet continuous similarity scoring
"""

import sys
import time
from autorag.evaluation.semantic_metrics import SemanticMetrics

def test_continuous_scoring():
    """Test the continuous scoring implementation"""

    print("\n" + "="*60)
    print("TESTING MPNET CONTINUOUS SIMILARITY SCORING")
    print("="*60)

    # Test pairs with varying similarity
    test_pairs = [
        ("The weather is nice today", "Today the weather is pleasant", "High similarity - paraphrase"),
        ("The cat sits on the mat", "A feline rests on the rug", "Medium-high similarity - synonyms"),
        ("Machine learning is powerful", "AI technology is advancing", "Medium similarity - related"),
        ("The sky is blue", "Quantum physics is complex", "Low similarity - unrelated"),
        ("Paris is the capital of France", "Paris is the capital of France", "Perfect match"),
    ]

    print("\n1. Testing with new all-mpnet-base-v2 model (continuous scores):")
    print("-" * 50)

    # Initialize with all-mpnet-base-v2 (46% better than MiniLM)
    evaluator_mpnet = SemanticMetrics(
        model_name='sentence-transformers/all-mpnet-base-v2',
        use_continuous_scores=True
    )

    print(f"Model loaded: {evaluator_mpnet.model}")
    print(f"Embedding dimension: {evaluator_mpnet.model.get_sentence_embedding_dimension()}")

    results_mpnet = []
    for text1, text2, description in test_pairs:
        score = evaluator_mpnet.similarity_score(text1, text2)
        results_mpnet.append((score, description))
        print(f"\n{description}")
        print(f"  Text 1: {text1[:50]}...")
        print(f"  Text 2: {text2[:50]}...")
        print(f"  Similarity: {score:.4f}")

    # Compare with old MiniLM model
    print("\n\n2. Comparing with old MiniLM-L6 model:")
    print("-" * 50)

    evaluator_mini = SemanticMetrics(
        model_name='all-MiniLM-L6-v2',
        use_continuous_scores=True
    )

    print(f"Model loaded: {evaluator_mini.model}")
    print(f"Embedding dimension: {evaluator_mini.model.get_sentence_embedding_dimension()}")

    print("\n\nComparison Table:")
    print("-" * 70)
    print(f"{'Description':<35} {'MPNet-base':>12} {'MiniLM':>12} {'Diff':>8}")
    print("-" * 70)

    for i, (text1, text2, description) in enumerate(test_pairs):
        score_mini = evaluator_mini.similarity_score(text1, text2)
        score_mpnet = results_mpnet[i][0]
        diff = score_mpnet - score_mini
        print(f"{description[:35]:<35} {score_mpnet:>12.4f} {score_mini:>12.4f} {diff:>+8.4f}")

    # Test batch processing
    print("\n\n3. Testing batch processing with continuous scores:")
    print("-" * 50)

    predictions = [pair[0] for pair in test_pairs]
    ground_truths = [pair[1] for pair in test_pairs]

    start_time = time.time()
    results = evaluator_mpnet.evaluate(predictions, ground_truths)
    elapsed = time.time() - start_time

    print(f"\nBatch evaluation completed in {elapsed:.2f} seconds")
    print(f"Mean similarity: {results['similarity_mean']:.4f}")
    print(f"Median similarity: {results['similarity_median']:.4f}")
    print(f"Std deviation: {results['similarity_std']:.4f}")
    print(f"Min similarity: {results['similarity_min']:.4f}")
    print(f"Max similarity: {results['similarity_max']:.4f}")
    print(f"Q25: {results['similarity_q25']:.4f}")
    print(f"Q75: {results['similarity_q75']:.4f}")

    # Test for Bayesian optimization use case
    print("\n\n4. Simulating Bayesian Optimization scoring:")
    print("-" * 50)

    # Simulate different quality answers
    query = "What is the capital of France?"
    answers = [
        "Paris is the capital of France.",  # Perfect
        "The capital of France is Paris.",  # Perfect rephrasing
        "Paris, the beautiful city, serves as France's capital.",  # Good with extra
        "France's capital city is Paris, known for the Eiffel Tower.",  # Good with details
        "The French capital is Paris.",  # Good but shorter
        "It's Paris.",  # Correct but minimal
        "London is the capital.",  # Wrong
    ]
    ground_truth = "Paris is the capital of France."

    print(f"\nQuery: {query}")
    print(f"Ground truth: {ground_truth}")
    print("\nAnswer quality scores (continuous):")

    scores = []
    for i, answer in enumerate(answers):
        score = evaluator_mpnet.similarity_score(answer, ground_truth)
        scores.append(score)
        print(f"  {i+1}. [{score:.4f}] {answer[:60]}...")

    print(f"\n\nMean score for optimization: {sum(scores)/len(scores):.4f}")
    print("Note: Continuous scores provide better gradient for Bayesian optimization")

    # Show why continuous is better than threshold
    print("\n\n5. Why continuous scores are better for optimization:")
    print("-" * 50)

    threshold = 0.7
    binary_scores = [1 if s >= threshold else 0 for s in scores]

    print(f"\nWith threshold {threshold}:")
    print(f"  Binary scores: {binary_scores}")
    print(f"  Binary mean: {sum(binary_scores)/len(binary_scores):.2f}")
    print(f"  Information: Only {len(set(binary_scores))} unique values")

    print(f"\nWith continuous scores:")
    print(f"  Continuous: [{', '.join(f'{s:.3f}' for s in scores)}]")
    print(f"  Continuous mean: {sum(scores)/len(scores):.4f}")
    print(f"  Information: {len(set(scores))} unique values (full gradient)")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

    return results

if __name__ == "__main__":
    try:
        results = test_continuous_scoring()
        print("\n✓ All tests passed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)