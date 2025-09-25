"""View baseline evaluation metrics"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def view_metrics(filename=None):
    """Display metrics from evaluation results"""

    experiments_dir = Path("experiments")

    if filename:
        filepath = experiments_dir / filename
    else:
        # Get the most recent baseline file
        baseline_files = sorted(experiments_dir.glob("baseline_*.json"))
        if not baseline_files:
            print("No baseline evaluation files found in experiments/")
            return
        filepath = baseline_files[-1]
        print(f"Loading most recent file: {filepath.name}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    print("\n" + "="*60)
    print("BASELINE EVALUATION METRICS")
    print("="*60)

    # Timestamp and configuration
    print(f"\nEvaluation Date: {data['timestamp'][:19]}")
    print(f"Configuration: {data['configuration']['generation']['model']}, "
          f"chunk_size={data['configuration']['chunking']['size']}")

    # Dataset info
    print(f"\nDataset:")
    print(f"  Documents: {data['dataset']['num_documents']}")
    print(f"  Queries: {data['dataset']['num_queries']}")

    # Performance metrics
    print(f"\nPerformance:")
    perf = data['performance']
    print(f"  Indexing time: {perf['indexing_time_seconds']:.2f} seconds")
    print(f"  Avg query time: {perf['avg_query_time_seconds']:.2f} seconds")
    print(f"  Total time: {perf['total_time_seconds']:.2f} seconds")

    # RAGAS metrics
    if 'ragas_metrics' in data['evaluation']:
        print(f"\nRAGAS Metrics (Quality):")
        ragas = data['evaluation']['ragas_metrics']
        if 'faithfulness' in ragas:
            print(f"  Faithfulness: {ragas['faithfulness']:.3f}")
        if 'answer_relevancy' in ragas:
            val = ragas['answer_relevancy']
            if isinstance(val, (int, float)) and not (val != val):  # Check for NaN
                print(f"  Answer Relevancy: {val:.3f}")
            else:
                print(f"  Answer Relevancy: N/A")
        if 'nv_context_relevance' in ragas:
            print(f"  Context Relevance: {ragas['nv_context_relevance']:.3f}")

    # Traditional metrics
    if 'traditional_metrics' in data['evaluation']:
        print(f"\nTraditional Metrics (Accuracy):")
        trad = data['evaluation']['traditional_metrics']
        print(f"  Exact Match Accuracy: {trad['exact_match_accuracy']:.3f} ({trad['exact_match_accuracy']*100:.1f}%)")
        print(f"  Token Precision: {trad['token_precision']:.3f} ({trad['token_precision']*100:.1f}%)")
        print(f"  Token Recall: {trad['token_recall']:.3f} ({trad['token_recall']*100:.1f}%)")
        print(f"  Token F1: {trad['token_f1']:.3f} ({trad['token_f1']*100:.1f}%)")

    print("\n" + "="*60)

    # Cost estimation
    embedding_tokens = data['dataset']['num_documents'] * 50  # Rough estimate
    query_tokens = data['dataset']['num_queries'] * 500  # Rough estimate
    embedding_cost = (embedding_tokens / 1000) * 0.0001  # Ada pricing
    generation_cost = (query_tokens / 1000) * 0.0015  # GPT-3.5 pricing
    print(f"\nEstimated Cost:")
    print(f"  Embedding: ~${embedding_cost:.4f}")
    print(f"  Generation: ~${generation_cost:.4f}")
    print(f"  Total: ~${embedding_cost + generation_cost:.4f}")

    print("\nFile location:", filepath.absolute())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="View evaluation metrics")
    parser.add_argument("--file", help="Specific JSON file to load (default: most recent)")
    args = parser.parse_args()

    view_metrics(args.file)