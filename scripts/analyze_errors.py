"""Analyze incorrect and low-scoring answers from evaluation results"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))


def load_evaluation_results(filename: str = None) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    experiments_dir = Path("experiments")

    if filename:
        filepath = experiments_dir / filename
    else:
        # Get the most recent file
        files = sorted(experiments_dir.glob("baseline_*.json"))
        if not files:
            raise FileNotFoundError("No evaluation files found")
        filepath = files[-1]

    with open(filepath, 'r') as f:
        return json.load(f), filepath.name


def analyze_errors(threshold: float = 0.7, show_correct: bool = False):
    """
    Analyze incorrect answers and patterns

    Args:
        threshold: Similarity threshold for "correct" answers
        show_correct: If True, also show correct answers for comparison
    """
    data, filename = load_evaluation_results()

    print(f"Analyzing errors in: {filename}")
    print("=" * 100)

    # Check if we have detailed answers
    if "detailed_answers" not in data:
        print("No detailed answer data found. Please run evaluation with --with-ground-truth")
        return

    detailed_answers = data["detailed_answers"]

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(detailed_answers)

    # Filter out answers without ground truth or similarity scores
    df_with_scores = df[df['semantic_similarity'].notna()].copy()

    if len(df_with_scores) == 0:
        print("No answers with similarity scores found.")
        return

    # Categorize answers
    df_with_scores['category'] = pd.cut(
        df_with_scores['semantic_similarity'],
        bins=[-1, 0.3, 0.5, 0.7, 0.8, 1.01],
        labels=['Very Poor', 'Poor', 'Moderate', 'Good', 'Excellent']
    )

    # Statistics
    print("\nOVERALL STATISTICS:")
    print(f"  Total answers analyzed: {len(df_with_scores)}")
    print(f"  Mean similarity: {df_with_scores['semantic_similarity'].mean():.3f}")
    print(f"  Median similarity: {df_with_scores['semantic_similarity'].median():.3f}")
    print(f"  Std deviation: {df_with_scores['semantic_similarity'].std():.3f}")

    print("\nDISTRIBUTION BY CATEGORY:")
    category_counts = df_with_scores['category'].value_counts()
    for cat in ['Excellent', 'Good', 'Moderate', 'Poor', 'Very Poor']:
        if cat in category_counts.index:
            count = category_counts[cat]
            pct = count / len(df_with_scores) * 100
            print(f"  {cat:12} ({df_with_scores[df_with_scores['category'] == cat]['semantic_similarity'].min():.2f}-"
                  f"{df_with_scores[df_with_scores['category'] == cat]['semantic_similarity'].max():.2f}): "
                  f"{count:3d} ({pct:5.1f}%)")

    # Incorrect answers (below threshold)
    incorrect = df_with_scores[df_with_scores['semantic_similarity'] < threshold].sort_values('semantic_similarity')

    if len(incorrect) > 0:
        print(f"\n{'='*100}")
        print(f"INCORRECT ANSWERS (similarity < {threshold}):")
        print(f"{'='*100}")

        for idx, row in incorrect.iterrows():
            print(f"\n[Question {row['question_id'] + 1}] Similarity: {row['semantic_similarity']:.3f}")
            print(f"Question: {row['question'][:150]}...")
            print(f"\nGround Truth: {row['ground_truth'][:200]}..." if len(str(row['ground_truth'])) > 200 else f"\nGround Truth: {row['ground_truth']}")
            print(f"\nGenerated: {row['generated_answer'][:200]}..." if len(row['generated_answer']) > 200 else f"\nGenerated: {row['generated_answer']}")
            print(f"\nRetrieval Quality: Top context score = {row['top_context_score']:.3f}, Contexts used = {row['contexts_used']}")

            # Analyze error type
            if row['semantic_similarity'] < 0.3:
                if "cannot find" in row['generated_answer'].lower():
                    print(">> Error Type: No information found (retrieval failure)")
                else:
                    print(">> Error Type: Completely wrong answer (generation failure)")
            elif row['semantic_similarity'] < 0.5:
                print(">> Error Type: Partially incorrect or different focus")
            else:
                print(">> Error Type: Missing key details or incorrect phrasing")
            print("-" * 100)

    # Show some correct answers for comparison if requested
    if show_correct:
        correct = df_with_scores[df_with_scores['semantic_similarity'] >= threshold].sort_values('semantic_similarity', ascending=False)

        if len(correct) > 0:
            print(f"\n{'='*100}")
            print(f"CORRECT ANSWERS (similarity >= {threshold}) - Top 3:")
            print(f"{'='*100}")

            for idx, row in correct.head(3).iterrows():
                print(f"\n[Question {row['question_id'] + 1}] Similarity: {row['semantic_similarity']:.3f}")
                print(f"Question: {row['question'][:150]}...")
                print(f"\nGround Truth: {row['ground_truth'][:200]}..." if len(str(row['ground_truth'])) > 200 else f"\nGround Truth: {row['ground_truth']}")
                print(f"\nGenerated: {row['generated_answer'][:200]}..." if len(row['generated_answer']) > 200 else f"\nGenerated: {row['generated_answer']}")
                print(f"\nRetrieval Quality: Top context score = {row['top_context_score']:.3f}")
                print("-" * 100)

    # Pattern analysis
    print(f"\n{'='*100}")
    print("ERROR PATTERNS:")
    print(f"{'='*100}")

    # Check for "cannot find" responses
    cannot_find = df_with_scores[df_with_scores['generated_answer'].str.contains("cannot find", case=False, na=False)]
    print(f"\n'Cannot find answer' responses: {len(cannot_find)} ({len(cannot_find)/len(df_with_scores)*100:.1f}%)")

    # Check retrieval quality correlation
    low_retrieval = df_with_scores[df_with_scores['top_context_score'] < 0.7]
    print(f"Low retrieval score (<0.7): {len(low_retrieval)} ({len(low_retrieval)/len(df_with_scores)*100:.1f}%)")

    # Correlation between retrieval and answer quality
    if len(df_with_scores) > 5:
        corr = df_with_scores['top_context_score'].corr(df_with_scores['semantic_similarity'])
        print(f"Correlation between retrieval score and answer quality: {corr:.3f}")

    # Save error analysis to CSV for further investigation
    output_csv = f"experiments/error_analysis_{Path(filename).stem}.csv"
    incorrect.to_csv(output_csv, index=False)
    print(f"\nError details saved to: {output_csv}")

    return df_with_scores


def export_for_manual_review(output_file: str = "experiments/answers_for_review.xlsx"):
    """Export answers in a format suitable for manual review"""
    data, filename = load_evaluation_results()

    if "detailed_answers" not in data:
        print("No detailed answer data found.")
        return

    df = pd.DataFrame(data["detailed_answers"])

    # Reorder columns for better readability
    columns_order = ['question_id', 'question', 'ground_truth', 'generated_answer',
                    'semantic_similarity', 'is_correct', 'top_context_score', 'contexts_used']

    # Filter to existing columns
    columns_order = [col for col in columns_order if col in df.columns]
    df = df[columns_order]

    # Sort by similarity score (worst first)
    if 'semantic_similarity' in df.columns:
        df = df.sort_values('semantic_similarity', na_position='last')

    # Save to Excel with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Answers', index=False)

        # Get the worksheet
        worksheet = writer.sheets['Answers']

        # Adjust column widths
        worksheet.column_dimensions['A'].width = 10  # question_id
        worksheet.column_dimensions['B'].width = 50  # question
        worksheet.column_dimensions['C'].width = 50  # ground_truth
        worksheet.column_dimensions['D'].width = 50  # generated_answer
        worksheet.column_dimensions['E'].width = 15  # similarity

    print(f"Answers exported to: {output_file}")
    print(f"Total answers: {len(df)}")
    if 'semantic_similarity' in df.columns:
        print(f"Answers with scores: {df['semantic_similarity'].notna().sum()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze incorrect answers from evaluation")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold for correct answers (default: 0.7)")
    parser.add_argument("--show-correct", action="store_true",
                       help="Also show some correct answers for comparison")
    parser.add_argument("--export", action="store_true",
                       help="Export to Excel for manual review")
    parser.add_argument("--file", help="Specific evaluation file to analyze")

    args = parser.parse_args()

    if args.export:
        export_for_manual_review()
    else:
        analyze_errors(threshold=args.threshold, show_correct=args.show_correct)