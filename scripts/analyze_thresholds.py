"""Analyze semantic similarity at different thresholds"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from autorag.evaluation.semantic_metrics import SemanticMetrics


def analyze_thresholds():
    """Analyze how different thresholds affect metrics"""

    # Load the latest evaluation results
    experiments_dir = Path("experiments")
    latest_file = sorted(experiments_dir.glob("baseline_*.json"))[-1]

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # Extract generated answers and ground truths
    results = data.get("sample_results", [])
    if not results:
        print("No sample results found in evaluation file")
        return

    # Get the answers (assuming ground truths are in the data)
    predictions = [r["answer"] for r in results[:20]]

    # For this analysis, we'll need to manually set ground truths
    # or extract from queries if available
    print(f"Loaded {len(predictions)} predictions")

    # Initialize semantic metrics evaluator
    evaluator = SemanticMetrics()

    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # Get similarities from the semantic metrics
    if "evaluation" in data and "semantic_metrics" in data["evaluation"]:
        similarities = data["evaluation"]["semantic_metrics"].get("per_sample_similarities", [])
        if not similarities:
            print("No per-sample similarities found. Running new evaluation...")
            return

        print(f"\nSimilarity Distribution:")
        print(f"  Min: {min(similarities):.3f}")
        print(f"  Max: {max(similarities):.3f}")
        print(f"  Mean: {np.mean(similarities):.3f}")
        print(f"  Median: {np.median(similarities):.3f}")
        print(f"  Std: {np.std(similarities):.3f}")

        print(f"\nThreshold Analysis:")
        print(f"{'Threshold':<12} {'Matches':<10} {'Accuracy':<10} {'F1':<10}")
        print("-" * 42)

        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            matches = sum(1 for s in similarities if s >= threshold)
            accuracy = matches / len(similarities)

            # Simple F1 calculation (assuming all ground truths are positive)
            precision = 1.0 if matches > 0 else 0.0
            recall = matches / len(similarities)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{threshold:<12.2f} {matches:<10} {accuracy:<10.2%} {f1:<10.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"\nOptimal threshold: {best_threshold:.2f} (F1: {best_f1:.3f})")

        # Distribution visualization
        print("\nSimilarity Distribution:")
        bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(similarities, bins=bins)

        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            bar = "█" * int(hist[i] * 50 / max(hist))
            print(f"{low:.1f}-{high:.1f}: {bar} ({hist[i]})")


if __name__ == "__main__":
    analyze_thresholds()