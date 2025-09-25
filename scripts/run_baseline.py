"""Run baseline RAG evaluation for Week 1"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from autorag.core.rag_pipeline import RAGPipeline
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.evaluation.ragas_evaluator import RAGASEvaluator


def run_baseline_evaluation(include_ground_truth=False):
    """Run the Week 1 baseline evaluation"""

    # Configure logging
    logger.add("experiments/baseline_{time}.log", rotation="100 MB")
    logger.info(f"Starting Week 1 baseline RAG evaluation (ground_truth={include_ground_truth})")

    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)

    # Load dataset
    logger.info("Loading MS MARCO subset")
    loader = MSMARCOLoader()
    documents, queries = loader.load_subset(num_docs=100, num_queries=20,
                                           include_answers=include_ground_truth)

    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline")
    pipeline = RAGPipeline()

    # Index documents
    logger.info("Indexing documents")
    start_time = time.time()
    pipeline.index(documents)
    indexing_time = time.time() - start_time
    logger.info(f"Indexing completed in {indexing_time:.2f} seconds")

    # Process queries
    logger.info("Processing queries")
    results = []
    query_times = []
    ground_truths = [] if include_ground_truth else None

    for i, query_data in enumerate(queries):
        question = query_data["question"]
        logger.info(f"Processing query {i+1}/{len(queries)}: {question[:50]}...")

        start_time = time.time()
        try:
            result = pipeline.query(question, top_k=5)
            query_time = time.time() - start_time
            query_times.append(query_time)
            results.append(result)

            # Collect ground truth if available
            if include_ground_truth and "ground_truth_answer" in query_data:
                ground_truths.append(query_data["ground_truth_answer"])

            logger.debug(f"Query processed in {query_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            continue

    # Evaluate with RAGAS (and traditional metrics if ground truth available)
    logger.info(f"Running evaluation{' with ground truth' if ground_truths else ''}")
    evaluator = RAGASEvaluator()
    eval_results = evaluator.evaluate(results, ground_truths=ground_truths)

    # Calculate performance metrics
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    total_time = indexing_time + sum(query_times)

    # Compile final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "chunking": {"strategy": "fixed", "size": 256, "overlap": 0},
            "embedding": {"model": "text-embedding-ada-002"},
            "retrieval": {"method": "dense", "top_k": 5},
            "generation": {"model": "gpt-3.5-turbo", "temperature": 0, "max_tokens": 300}
        },
        "dataset": {
            "num_documents": len(documents),
            "num_queries": len(queries)
        },
        "performance": {
            "indexing_time_seconds": indexing_time,
            "avg_query_time_seconds": avg_query_time,
            "total_time_seconds": total_time,
            "queries_processed": len(results)
        },
        "evaluation": eval_results
    }

    # Add detailed answer analysis if ground truth available
    if ground_truths and "semantic_metrics" in eval_results:
        per_sample_similarities = eval_results["semantic_metrics"].get("per_sample_similarities", [])
        per_sample_matches = eval_results["semantic_metrics"].get("per_sample_matches", [])

        # Create detailed answer records
        detailed_answers = []
        for i, result in enumerate(results):
            answer_detail = {
                "question_id": i,
                "question": result["question"],
                "generated_answer": result["answer"],
                "ground_truth": ground_truths[i] if i < len(ground_truths) else None,
                "semantic_similarity": per_sample_similarities[i] if i < len(per_sample_similarities) else None,
                "is_correct": per_sample_matches[i] if i < len(per_sample_matches) else None,
                "contexts_used": len(result["contexts"]),
                "top_context_score": result["contexts"][0]["score"] if result["contexts"] else 0
            }
            detailed_answers.append(answer_detail)

        final_results["detailed_answers"] = detailed_answers
    else:
        # If no ground truth, still save basic answer details
        final_results["detailed_answers"] = [
            {
                "question_id": i,
                "question": result["question"],
                "generated_answer": result["answer"],
                "contexts_used": len(result["contexts"]),
                "top_context_score": result["contexts"][0]["score"] if result["contexts"] else 0
            }
            for i, result in enumerate(results)
        ]

    # Save results
    output_file = f"experiments/baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Print summary to console
    print("\n" + "="*60)
    print("WEEK 1 BASELINE EVALUATION RESULTS")
    print("="*60)
    print(f"\nDataset:")
    print(f"  Documents indexed: {len(documents)}")
    print(f"  Queries processed: {len(results)}/{len(queries)}")

    print(f"\nPerformance:")
    print(f"  Indexing time: {indexing_time:.2f} seconds")
    print(f"  Avg query time: {avg_query_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")

    if "ragas_metrics" in eval_results:
        print(f"\nRAGAS Metrics:")
        for metric, value in eval_results["ragas_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.3f}")

    if "traditional_metrics" in eval_results:
        print(f"\nTraditional Metrics (Token-level):")
        trad_metrics = eval_results["traditional_metrics"]
        print(f"  Exact Match Accuracy: {trad_metrics.get('exact_match_accuracy', 0):.3f}")
        print(f"  Token Precision: {trad_metrics.get('token_precision', 0):.3f}")
        print(f"  Token Recall: {trad_metrics.get('token_recall', 0):.3f}")
        print(f"  Token F1: {trad_metrics.get('token_f1', 0):.3f}")

    if "semantic_metrics" in eval_results:
        print(f"\nSemantic Metrics (Meaning-based):")
        sem_metrics = eval_results["semantic_metrics"]
        print(f"  Semantic Accuracy: {sem_metrics.get('semantic_accuracy', 0):.3f}")
        print(f"  Semantic Precision: {sem_metrics.get('semantic_precision', 0):.3f}")
        print(f"  Semantic Recall: {sem_metrics.get('semantic_recall', 0):.3f}")
        print(f"  Semantic F1: {sem_metrics.get('semantic_f1', 0):.3f}")
        print(f"  Mean Similarity: {sem_metrics.get('similarity_mean', 0):.3f}")
        print(f"  Threshold Used: {sem_metrics.get('similarity_threshold', 0.7):.2f}")

    print(f"\nResults saved to: {output_file}")
    print("="*60)

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline RAG evaluation")
    parser.add_argument("--with-ground-truth", action="store_true",
                       help="Include ground truth answers for traditional metrics (accuracy, F1, etc.)")
    args = parser.parse_args()

    try:
        results = run_baseline_evaluation(include_ground_truth=args.with_ground_truth)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise