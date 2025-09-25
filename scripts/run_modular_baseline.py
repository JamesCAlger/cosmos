"""Run baseline evaluation using Week 2 modular architecture"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from autorag.pipeline.simple_rag import SimpleRAGPipeline
from autorag.components.base import Document
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.evaluation.ragas_evaluator import RAGASEvaluator


def run_modular_baseline(num_queries: int = 20, include_ground_truth: bool = True):
    """Run baseline evaluation with modular architecture"""

    # Configure logging
    logger.add("experiments/modular_{time}.log", rotation="100 MB")
    logger.info(f"Starting modular baseline evaluation with {num_queries} queries")

    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)

    # Load dataset
    logger.info("Loading MS MARCO dataset")
    loader = MSMARCOLoader()
    documents, queries = loader.load_subset(
        num_docs=100,
        num_queries=num_queries,
        include_answers=include_ground_truth
    )

    # Convert to Document objects
    doc_objects = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_objects.append(Document(
                content=doc.get("content", doc.get("text", "")),
                metadata=doc.get("metadata", {}),
                doc_id=doc.get("id")
            ))
        elif hasattr(doc, 'content'):
            doc_objects.append(doc)
        else:
            doc_objects.append(Document(content=str(doc), metadata={}))

    # Initialize pipeline with modular architecture
    logger.info("Initializing modular RAG pipeline")
    config_path = Path(__file__).parent.parent / "configs" / "baseline_rag.yaml"
    pipeline = SimpleRAGPipeline(str(config_path))

    # Show components
    components = pipeline.get_components()
    logger.info("Pipeline components:")
    for comp_id, info in components.items():
        logger.info(f"  - {comp_id}: {info['component']} ({info['type']})")

    # Index documents
    logger.info(f"Indexing {len(doc_objects)} documents")
    start_time = time.time()
    index_result = pipeline.index(doc_objects)
    indexing_time = time.time() - start_time
    logger.info(f"Indexing completed in {indexing_time:.2f}s - {index_result['num_chunks']} chunks created")

    # Process queries
    logger.info(f"Processing {len(queries)} queries")
    results = []
    query_times = []
    ground_truths = [] if include_ground_truth else None

    for i, query_data in enumerate(queries):
        question = query_data["question"]
        logger.info(f"Query {i+1}/{len(queries)}: {question[:50]}...")

        start_time = time.time()
        try:
            result = pipeline.query(question, top_k=5)
            query_time = time.time() - start_time
            query_times.append(query_time)
            results.append(result)

            # Collect ground truth if available
            if include_ground_truth and "ground_truth_answer" in query_data:
                ground_truths.append(query_data["ground_truth_answer"])

            logger.debug(f"Query processed in {query_time:.2f}s")

        except Exception as e:
            logger.error(f"Error processing query {i+1}: {e}")
            continue

    # Evaluate with RAGAS
    logger.info("Running RAGAS evaluation")
    evaluator = RAGASEvaluator()
    eval_results = evaluator.evaluate(results, ground_truths=ground_truths)

    # Calculate performance metrics
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    total_time = indexing_time + sum(query_times)

    # Compile results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "architecture": "modular_week2",
        "configuration": {
            "config_file": str(config_path),
            "components": components
        },
        "dataset": {
            "num_documents": len(doc_objects),
            "num_queries": len(queries),
            "num_chunks": index_result['num_chunks']
        },
        "performance": {
            "indexing_time_seconds": indexing_time,
            "avg_query_time_seconds": avg_query_time,
            "total_time_seconds": total_time,
            "queries_processed": len(results)
        },
        "evaluation": eval_results
    }

    # Add detailed answers if ground truth available
    if ground_truths and "semantic_metrics" in eval_results:
        per_sample_similarities = eval_results["semantic_metrics"].get("per_sample_similarities", [])
        per_sample_matches = eval_results["semantic_metrics"].get("per_sample_matches", [])

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

    # Save results
    output_file = f"experiments/modular_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("MODULAR ARCHITECTURE BASELINE EVALUATION")
    print("="*60)

    print(f"\nArchitecture: Week 2 Modular")
    print(f"Configuration: {config_path.name}")

    print(f"\nDataset:")
    print(f"  Documents indexed: {len(doc_objects)}")
    print(f"  Chunks created: {index_result['num_chunks']}")
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
        print(f"\nTraditional Metrics:")
        trad = eval_results["traditional_metrics"]
        print(f"  Exact Match: {trad.get('exact_match_accuracy', 0):.3f}")
        print(f"  Token F1: {trad.get('token_f1', 0):.3f}")

    if "semantic_metrics" in eval_results:
        print(f"\nSemantic Metrics:")
        sem = eval_results["semantic_metrics"]
        print(f"  Semantic Accuracy: {sem.get('semantic_accuracy', 0):.3f}")
        print(f"  Semantic F1: {sem.get('semantic_f1', 0):.3f}")
        print(f"  Mean Similarity: {sem.get('similarity_mean', 0):.3f}")

    print(f"\nResults saved to: {output_file}")

    # Compare with Week 1 if available
    compare_with_week1(final_results)

    print("="*60)

    return final_results


def compare_with_week1(modular_results):
    """Compare modular results with Week 1 baseline"""
    # Find most recent Week 1 results
    experiments_dir = Path("experiments")
    week1_files = list(experiments_dir.glob("baseline_2025*.json"))

    if not week1_files:
        return

    latest_week1 = max(week1_files, key=lambda f: f.stat().st_mtime)

    with open(latest_week1) as f:
        week1_data = json.load(f)

    print("\n" + "="*60)
    print("COMPARISON WITH WEEK 1 BASELINE")
    print("="*60)

    # Performance comparison
    w1_perf = week1_data["performance"]
    w2_perf = modular_results["performance"]

    print("\nPerformance Comparison:")
    print(f"  Indexing time:")
    print(f"    Week 1: {w1_perf['indexing_time_seconds']:.2f}s")
    print(f"    Week 2: {w2_perf['indexing_time_seconds']:.2f}s")
    diff = ((w2_perf['indexing_time_seconds'] - w1_perf['indexing_time_seconds']) /
            w1_perf['indexing_time_seconds'] * 100)
    print(f"    Difference: {diff:+.1f}%")

    print(f"  Average query time:")
    print(f"    Week 1: {w1_perf['avg_query_time_seconds']:.2f}s")
    print(f"    Week 2: {w2_perf['avg_query_time_seconds']:.2f}s")
    diff = ((w2_perf['avg_query_time_seconds'] - w1_perf['avg_query_time_seconds']) /
            w1_perf['avg_query_time_seconds'] * 100)
    print(f"    Difference: {diff:+.1f}%")

    # Metrics comparison if available
    if "evaluation" in week1_data and "evaluation" in modular_results:
        w1_eval = week1_data["evaluation"]
        w2_eval = modular_results["evaluation"]

        if "semantic_metrics" in w1_eval and "semantic_metrics" in w2_eval:
            print("\nSemantic Metrics Comparison:")
            w1_sem = w1_eval["semantic_metrics"]
            w2_sem = w2_eval["semantic_metrics"]

            print(f"  Semantic Accuracy:")
            print(f"    Week 1: {w1_sem.get('semantic_accuracy', 0):.3f}")
            print(f"    Week 2: {w2_sem.get('semantic_accuracy', 0):.3f}")

            print(f"  Semantic F1:")
            print(f"    Week 1: {w1_sem.get('semantic_f1', 0):.3f}")
            print(f"    Week 2: {w2_sem.get('semantic_f1', 0):.3f}")

    # Validation summary
    print("\nValidation Summary:")
    perf_similar = abs(diff) < 20  # Within 20% is acceptable
    if perf_similar:
        print("  ✓ Performance is comparable to Week 1")
        print("  ✓ Modular architecture validated successfully")
    else:
        print("  ⚠ Performance difference detected")
        print("  Further optimization may be needed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run modular baseline evaluation")
    parser.add_argument("--queries", type=int, default=20,
                       help="Number of queries to process (default: 20)")
    parser.add_argument("--no-ground-truth", action="store_true",
                       help="Skip ground truth evaluation")

    args = parser.parse_args()

    try:
        results = run_modular_baseline(
            num_queries=args.queries,
            include_ground_truth=not args.no_ground_truth
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise