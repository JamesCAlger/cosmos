#!/usr/bin/env python
"""Simple test for continuous similarity scoring"""

from autorag.evaluation.semantic_metrics import SemanticMetrics

# Test continuous scoring
print("Testing continuous similarity scoring...")
print("-" * 50)

# Create evaluator with default model (all-mpnet-base-v2)
evaluator = SemanticMetrics()

# Test pairs
test_pairs = [
    ("Paris is the capital of France", "Paris is the capital of France"),
    ("The cat sits on the mat", "A feline rests on the rug"),
    ("Machine learning is powerful", "AI technology is advancing"),
]

print(f"\nUsing model: {evaluator.model}")
print(f"Continuous scores enabled: {evaluator.use_continuous_scores}")
print(f"Threshold: {evaluator.similarity_threshold}")

print("\nSimilarity scores (continuous):")
for text1, text2 in test_pairs:
    score = evaluator.similarity_score(text1, text2)
    print(f"  {score:.4f} : '{text1[:30]}...' vs '{text2[:30]}...'")

# Show that we're returning raw scores, not thresholded values
scores = [evaluator.similarity_score(t1, t2) for t1, t2 in test_pairs]
print(f"\nMean score: {sum(scores)/len(scores):.4f}")
print("All scores are continuous values in [0, 1] - perfect for Bayesian optimization!")