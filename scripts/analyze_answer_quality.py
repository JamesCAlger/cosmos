"""Analyze answer quality by showing generated vs ground truth with similarity scores"""

import sys
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from autorag.evaluation.semantic_metrics import SemanticMetrics
from autorag.datasets.msmarco_loader import MSMARCOLoader


def load_latest_results():
    """Load the latest evaluation results"""
    experiments_dir = Path("experiments")
    latest_file = sorted(experiments_dir.glob("baseline_*.json"))[-1]

    with open(latest_file, 'r') as f:
        return json.load(f), latest_file.name


def get_ground_truths():
    """Load ground truth answers from MS MARCO"""
    loader = MSMARCOLoader()
    _, queries = loader.load_subset(num_docs=100, num_queries=20, include_answers=True)
    return queries


def analyze_answer_quality(show_all=False, threshold=0.7):
    """
    Analyze and display answer quality with semantic similarity scores

    Args:
        show_all: If True, show all answers. If False, show only examples
        threshold: Similarity threshold for "good" answers
    """
    print("Loading evaluation results and ground truths...")
    data, filename = load_latest_results()
    queries_with_truth = get_ground_truths()

    # Extract generated answers
    results = data.get("sample_results", [])
    if not results:
        print("No sample results found")
        return

    # Initialize semantic evaluator
    print("Initializing semantic similarity evaluator...")
    evaluator = SemanticMetrics(similarity_threshold=threshold)

    # Prepare data
    generated_answers = []
    ground_truths = []
    questions = []

    for i, result in enumerate(results[:20]):
        generated_answers.append(result["answer"])
        questions.append(result["question"])

        # Find matching ground truth
        for query in queries_with_truth:
            if query["question"] == result["question"]:
                ground_truths.append(query.get("ground_truth_answer", "N/A"))
                break
        else:
            ground_truths.append("N/A")

    # Calculate similarities
    print("Calculating semantic similarities...\n")
    similarities = evaluator.semantic_similarity_batch(generated_answers, ground_truths)

    # Sort by similarity score
    indexed_results = list(enumerate(zip(questions, generated_answers, ground_truths, similarities)))
    indexed_results.sort(key=lambda x: x[1][3], reverse=True)

    # Display header
    print("=" * 100)
    print("ANSWER QUALITY ANALYSIS")
    print("=" * 100)
    print(f"Threshold for 'Good' Answer: {threshold}")
    print(f"File: {filename}\n")

    # Statistics
    similarities_array = np.array(similarities)
    print("OVERALL STATISTICS:")
    print(f"  Mean Similarity: {similarities_array.mean():.3f}")
    print(f"  Median Similarity: {np.median(similarities_array):.3f}")
    print(f"  Max Similarity: {similarities_array.max():.3f}")
    print(f"  Min Similarity: {similarities_array.min():.3f}")
    print(f"  Answers Above Threshold ({threshold}): {sum(s >= threshold for s in similarities)}/20")
    print()

    # Categories
    excellent = []
    good = []
    moderate = []
    poor = []

    for i, (q, g, t, s) in indexed_results:
        if s >= 0.8:
            excellent.append((i, q, g, t, s))
        elif s >= 0.7:
            good.append((i, q, g, t, s))
        elif s >= 0.5:
            moderate.append((i, q, g, t, s))
        else:
            poor.append((i, q, g, t, s))

    def print_answer_comparison(category_name, answers, limit=None):
        """Print formatted answer comparisons"""
        print(f"\n{'='*100}")
        print(f"{category_name} (Showing {min(len(answers), limit) if limit else len(answers)}/{len(answers)})")
        print("="*100)

        for idx, (original_idx, question, generated, truth, score) in enumerate(answers[:limit]):
            print(f"\n[Query {original_idx + 1}] Similarity Score: {score:.3f} {'✓' if score >= threshold else '✗'}")
            print(f"Question: {question[:100]}...")
            print(f"\nGround Truth: {truth[:200]}..." if len(truth) > 200 else f"\nGround Truth: {truth}")
            print(f"\nGenerated:    {generated[:200]}..." if len(generated) > 200 else f"\nGenerated:    {generated}")

            # Analyze why it got this score
            if score >= 0.8:
                print("→ Analysis: Excellent match - semantically equivalent")
            elif score >= 0.7:
                print("→ Analysis: Good match - core information preserved")
            elif score >= 0.5:
                print("→ Analysis: Moderate match - partially correct")
            else:
                print("→ Analysis: Poor match - different information or no answer")

            print("-" * 100)

    # Show examples from each category
    if not show_all:
        print("\n" + "="*100)
        print("SHOWING EXAMPLES FROM EACH CATEGORY")
        print("="*100)

        if excellent:
            print_answer_comparison(f"🟢 EXCELLENT ANSWERS (≥0.8 similarity)", excellent, limit=2)

        if good:
            print_answer_comparison(f"🟡 GOOD ANSWERS (0.7-0.8 similarity)", good, limit=2)

        if moderate:
            print_answer_comparison(f"🟠 MODERATE ANSWERS (0.5-0.7 similarity)", moderate, limit=2)

        if poor:
            print_answer_comparison(f"🔴 POOR ANSWERS (<0.5 similarity)", poor, limit=2)

    else:
        # Show all answers
        print_answer_comparison("ALL ANSWERS (Sorted by Similarity)", indexed_results)

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Excellent (≥0.8): {len(excellent)} answers")
    print(f"Good (0.7-0.8):   {len(good)} answers")
    print(f"Moderate (0.5-0.7): {len(moderate)} answers")
    print(f"Poor (<0.5):      {len(poor)} answers")

    # Distribution graph
    print("\nSCORE DISTRIBUTION:")
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(similarities, bins=bins)

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        bar = "█" * int(hist[i] * 10) if hist[i] > 0 else ""
        label = f"{low:.1f}-{high:.1f}"
        print(f"  {label:8} [{hist[i]:2}] {bar}")

    return similarities, indexed_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze answer quality with semantic scores")
    parser.add_argument("--all", action="store_true", help="Show all answers (default: show examples)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold (default: 0.7)")
    args = parser.parse_args()

    analyze_answer_quality(show_all=args.all, threshold=args.threshold)