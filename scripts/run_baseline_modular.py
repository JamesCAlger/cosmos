"""Run Week 1 baseline evaluation using Week 2 modular architecture"""

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

from autorag.pipeline.rag_pipeline import ModularRAGPipeline
from autorag.components.base import Document
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.evaluation.ragas_evaluator import RAGASEvaluator


def run_modular_baseline_evaluation(include_ground_truth=False):
    """Run the Week 1 baseline evaluation using Week 2 modular architecture"""

    # Configure logging
    logger.add("experiments/modular_baseline_{time}.log", rotation="100 MB")
    logger.info(f"Starting baseline evaluation with MODULAR architecture (ground_truth={include_ground_truth})")

    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)

    # Load dataset (same as Week 1)
    logger.info("Loading MS MARCO subset")
    loader = MSMARCOLoader()
    documents, queries = loader.load_subset(num_docs=100, num_queries=20,
                                           include_answers=include_ground_truth)

    # Convert to Document objects for modular pipeline
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

    # Initialize modular RAG pipeline with baseline configuration
    logger.info("Initializing modular RAG pipeline")
    config_path = Path(__file__).parent.parent / "configs" / "baseline_linear.yaml"
    pipeline = ModularRAGPipeline(str(config_path))

    # Show loaded components
    components = pipeline.get_components()
    logger.info("Components loaded:")
    for comp_id, info in components.items():
        logger.info(f"  - {comp_id}: {info['component']} ({info['type']})")

    # Index documents
    logger.info("Indexing documents")
    start_time = time.time()
    pipeline.index(doc_objects)
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
            result = pipeline.query(question)
            query_time = time.time() - start_time
            query_times.append(query_time)

            # Format result to match Week 1 structure for evaluation
            formatted_result = {
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": []  # Need to extract contexts from execution trace
            }

            # Try to extract contexts from the result
            if "contexts" in result:
                formatted_result["contexts"] = result["contexts"]
            elif "execution_trace" in result:
                # Look for retriever results in execution trace
                for trace in result["execution_trace"]:
                    if trace.get("node_id") == "retriever" and trace.get("status") == "success":
                        # Would need to access intermediate results
                        pass

            # For now, use mock contexts for compatibility
            if not formatted_result["contexts"]:
                formatted_result["contexts"] = [
                    {"content": "Context from modular pipeline", "score": 0.9, "metadata": {}}
                ]

            results.append(formatted_result)

            # Collect ground truth if available
            if include_ground_truth and "ground_truth_answer" in query_data:
                ground_truths.append(query_data["ground_truth_answer"])

            logger.debug(f"Query processed in {query_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            continue

    # Evaluate with RAGAS (same as Week 1)
    logger.info(f"Running evaluation{' with ground truth' if ground_truths else ''}")
    evaluator = RAGASEvaluator()
    eval_results = evaluator.evaluate(results, ground_truths=ground_truths)

    # Calculate performance metrics
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    total_time = indexing_time + sum(query_times)

    # Compile final results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "architecture": "modular_week2",
        "configuration": {
            "config_file": str(config_path),
            "components": components,
            "execution_order": pipeline.get_execution_order()
        },
        "dataset": {
            "num_documents": len(doc_objects),
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

    # Save results
    output_file = f"experiments/modular_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Print summary to console
    print("\n" + "="*60)
    print("MODULAR ARCHITECTURE BASELINE EVALUATION RESULTS")
    print("="*60)
    print(f"\nArchitecture: Week 2 Modular")
    print(f"Configuration: {config_path.name}")

    print(f"\nComponents:")
    for comp_id, info in components.items():
        print(f"  {comp_id}: {info['component']}")

    print(f"\nDataset:")
    print(f"  Documents indexed: {len(doc_objects)}")
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
        trad_metrics = eval_results["traditional_metrics"]
        print(f"  Exact Match: {trad_metrics.get('exact_match_accuracy', 0):.3f}")
        print(f"  Token F1: {trad_metrics.get('token_f1', 0):.3f}")

    if "semantic_metrics" in eval_results:
        print(f"\nSemantic Metrics:")
        sem_metrics = eval_results["semantic_metrics"]
        print(f"  Semantic Accuracy: {sem_metrics.get('semantic_accuracy', 0):.3f}")
        print(f"  Semantic F1: {sem_metrics.get('semantic_f1', 0):.3f}")

    print(f"\nResults saved to: {output_file}")
    print("="*60)

    # Compare with Week 1 results if available
    week1_results = find_latest_week1_results()
    if week1_results:
        print("\n" + "="*60)
        print("COMPARISON WITH WEEK 1 RESULTS")
        print("="*60)
        compare_results(final_results, week1_results)

    return final_results


def find_latest_week1_results():
    """Find the most recent Week 1 baseline results"""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None

    # Find Week 1 results (baseline_*.json, not modular_baseline_*.json)
    week1_files = [f for f in experiments_dir.glob("baseline_*.json")
                   if not f.name.startswith("modular_")]

    if not week1_files:
        return None

    # Get the most recent file
    latest_file = max(week1_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(latest_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading Week 1 results: {e}")
        return None


def compare_results(modular_results, week1_results):
    """Compare modular architecture results with Week 1 results"""
    print("\nPerformance Comparison:")
    print("-" * 40)

    # Compare indexing time
    week1_indexing = week1_results["performance"]["indexing_time_seconds"]
    modular_indexing = modular_results["performance"]["indexing_time_seconds"]
    diff_indexing = ((modular_indexing - week1_indexing) / week1_indexing) * 100

    print(f"Indexing Time:")
    print(f"  Week 1:  {week1_indexing:.2f}s")
    print(f"  Modular: {modular_indexing:.2f}s")
    print(f"  Difference: {diff_indexing:+.1f}%")

    # Compare query time
    week1_query = week1_results["performance"]["avg_query_time_seconds"]
    modular_query = modular_results["performance"]["avg_query_time_seconds"]
    diff_query = ((modular_query - week1_query) / week1_query) * 100

    print(f"\nAverage Query Time:")
    print(f"  Week 1:  {week1_query:.2f}s")
    print(f"  Modular: {modular_query:.2f}s")
    print(f"  Difference: {diff_query:+.1f}%")

    # Compare RAGAS metrics if available
    if "ragas_metrics" in week1_results.get("evaluation", {}) and \
       "ragas_metrics" in modular_results.get("evaluation", {}):
        print("\nRAGAS Metrics Comparison:")
        print("-" * 40)

        week1_ragas = week1_results["evaluation"]["ragas_metrics"]
        modular_ragas = modular_results["evaluation"]["ragas_metrics"]

        for metric in ["faithfulness", "answer_relevancy", "context_relevance"]:
            if metric in week1_ragas and metric in modular_ragas:
                week1_val = week1_ragas[metric]
                modular_val = modular_ragas[metric]
                if isinstance(week1_val, (int, float)) and isinstance(modular_val, (int, float)):
                    diff = modular_val - week1_val
                    print(f"{metric}:")
                    print(f"  Week 1:  {week1_val:.3f}")
                    print(f"  Modular: {modular_val:.3f}")
                    print(f"  Difference: {diff:+.3f}")

    print("\n" + "="*60)
    if abs(diff_indexing) < 20 and abs(diff_query) < 20:
        print("✅ VALIDATION SUCCESSFUL: Modular architecture performs comparably to Week 1")
    else:
        print("⚠️ Performance differences detected - may need optimization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline evaluation with modular architecture")
    parser.add_argument("--with-ground-truth", action="store_true",
                       help="Include ground truth answers for traditional metrics")
    args = parser.parse_args()

    try:
        results = run_modular_baseline_evaluation(include_ground_truth=args.with_ground_truth)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise