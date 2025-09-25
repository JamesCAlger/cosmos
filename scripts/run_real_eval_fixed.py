"""Run real evaluation with OpenAI models - Fixed version"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import json
from datetime import datetime
from loguru import logger

from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.pipeline.simple_rag import SimpleRAGPipeline
from autorag.components.base import Document
from autorag.evaluation.cost_tracker import CostTracker
from autorag.evaluation.reporters.base import CompositeReporter


def evaluate_with_openai():
    """Run evaluation with real OpenAI models and generate reports"""

    print("\n" + "="*70)
    print("REAL OPENAI EVALUATION WITH METRICS")
    print("="*70)

    # 1. Load data
    print("\n1. Loading MS MARCO dataset...")
    loader = MSMARCOLoader()
    documents, queries = loader.load_subset(
        num_docs=30,  # Smaller for quick test
        num_queries=5,
        include_answers=True
    )

    # Convert to Document objects
    doc_objects = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_objects.append(Document(
                content=doc.get("content", doc.get("text", "")),
                metadata=doc.get("metadata", {})
            ))
        else:
            doc_objects.append(Document(content=str(doc), metadata={}))

    print(f"   Loaded {len(doc_objects)} documents and {len(queries)} queries")

    # 2. Initialize pipeline with OpenAI
    print("\n2. Setting up OpenAI RAG pipeline...")
    config_path = Path(__file__).parent.parent / "configs" / "baseline_rag.yaml"
    pipeline = SimpleRAGPipeline(str(config_path))

    # 3. Index documents
    print("\n3. Indexing documents with OpenAI embeddings...")
    start_time = time.time()
    index_result = pipeline.index(doc_objects)
    indexing_time = time.time() - start_time
    print(f"   Indexed {index_result['num_chunks']} chunks in {indexing_time:.2f}s")

    # 4. Initialize cost tracker
    cost_tracker = CostTracker()

    # 5. Run queries and collect results
    print("\n4. Running queries through pipeline...")
    results = []
    ground_truths = []

    for i, query_data in enumerate(queries):
        question = query_data["question"]
        ground_truth = query_data.get("answers", [""])[0] if "answers" in query_data else ""

        print(f"   Query {i+1}/{len(queries)}: {question[:50]}...")

        # Run query
        start = time.time()
        result = pipeline.query(question)
        query_time = time.time() - start

        # Track costs
        cost_tracker.estimate_cost(
            question,
            model="gpt-3.5-turbo",
            operation="generation",
            output_text=result.get("answer", "")
        )

        # Store results
        results.append({
            "question": question,
            "answer": result.get("answer", ""),
            "contexts": result.get("contexts", []),
            "query_time": query_time
        })
        ground_truths.append(ground_truth)

        print(f"      Answer: {result.get('answer', '')[:80]}...")

    # 6. Calculate metrics
    print("\n5. Calculating evaluation metrics...")
    metrics = calculate_metrics(results, ground_truths)

    # 7. Prepare evaluation report
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "OpenAI RAG (text-embedding-ada-002 + gpt-3.5-turbo)",
        "dataset": {
            "source": "MS MARCO",
            "num_documents": len(doc_objects),
            "num_queries": len(queries),
            "num_chunks": index_result['num_chunks']
        },
        "performance": {
            "indexing_time": indexing_time,
            "avg_query_time": sum(r["query_time"] for r in results) / len(results),
            "total_time": indexing_time + sum(r["query_time"] for r in results)
        },
        "metrics": metrics,
        "cost_summary": cost_tracker.get_summary(),
        "sample_results": results[:3]  # Include first 3 for inspection
    }

    # 8. Generate reports
    print("\n6. Generating evaluation reports...")
    output_dir = Path("experiments/real_openai_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"   JSON report: {json_path}")

    # Generate multiple format reports
    reporter = CompositeReporter()
    report_path = output_dir / "evaluation_report"
    reports = reporter.report(evaluation_results, str(report_path))

    for format_name in reports.keys():
        print(f"   {format_name.upper()} report: {report_path}.{format_name}")

    # 9. Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS WITH REAL OPENAI MODELS")
    print("="*70)

    print("\n📊 PERFORMANCE METRICS:")
    print(f"   Indexing time: {evaluation_results['performance']['indexing_time']:.2f}s")
    print(f"   Avg query time: {evaluation_results['performance']['avg_query_time']:.2f}s")
    print(f"   Total time: {evaluation_results['performance']['total_time']:.2f}s")

    print("\n📈 QUALITY METRICS:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {metric}: {value:.3f}")

    print("\n💰 COST TRACKING:")
    cost_summary = evaluation_results['cost_summary']
    print(f"   Total cost: ${cost_summary['total_cost']:.6f}")
    print(f"   Input tokens: {cost_summary.get('total_input_tokens', 0)}")
    print(f"   Output tokens: {cost_summary.get('total_output_tokens', 0)}")

    print("\n📝 SAMPLE Q&A:")
    for i, result in enumerate(results[:2]):
        print(f"\n   Q{i+1}: {result['question'][:60]}...")
        print(f"   A{i+1}: {result['answer'][:100]}...")

    print("\n" + "="*70)
    print("REPORTS LOCATION:")
    print(f"   {output_dir.absolute()}")
    print("="*70)

    return evaluation_results


def calculate_metrics(results, ground_truths):
    """Calculate various evaluation metrics"""
    from autorag.evaluation.semantic_metrics import SemanticMetrics

    metrics = {}

    # Extract answers
    generated_answers = [r["answer"] for r in results]

    # Basic metrics
    metrics["num_queries"] = len(results)
    metrics["avg_answer_length"] = sum(len(a) for a in generated_answers) / len(generated_answers)

    # Semantic similarity if we have ground truth
    if ground_truths and ground_truths[0]:  # Check if we have actual ground truth
        try:
            semantic_evaluator = SemanticMetrics()
            semantic_results = semantic_evaluator.evaluate(generated_answers, ground_truths)

            metrics["semantic_accuracy"] = semantic_results.get("semantic_accuracy", 0)
            metrics["semantic_f1"] = semantic_results.get("semantic_f1", 0)
            metrics["mean_similarity"] = semantic_results.get("similarity_mean", 0)
            metrics["similarity_threshold"] = semantic_results.get("similarity_threshold", 0.7)
        except Exception as e:
            print(f"   Warning: Could not calculate semantic metrics: {e}")

    # Context usage metrics
    contexts_per_query = [len(r.get("contexts", [])) for r in results]
    metrics["avg_contexts_used"] = sum(contexts_per_query) / len(contexts_per_query)

    # Response time metrics
    query_times = [r["query_time"] for r in results]
    metrics["min_query_time"] = min(query_times)
    metrics["max_query_time"] = max(query_times)
    metrics["median_query_time"] = sorted(query_times)[len(query_times)//2]

    return metrics


def interpret_metrics():
    """Provide interpretation guide for the metrics"""

    print("\n" + "="*70)
    print("METRICS INTERPRETATION GUIDE")
    print("="*70)

    interpretations = {
        "semantic_accuracy": {
            "description": "Percentage of answers semantically similar to ground truth",
            "good": ">0.7", "acceptable": "0.5-0.7", "poor": "<0.5",
            "meaning": "Higher is better. Shows how well the RAG understands and answers correctly."
        },
        "semantic_f1": {
            "description": "Harmonic mean of semantic precision and recall",
            "good": ">0.7", "acceptable": "0.5-0.7", "poor": "<0.5",
            "meaning": "Balanced measure of answer quality. Considers both relevance and completeness."
        },
        "mean_similarity": {
            "description": "Average cosine similarity between generated and expected answers",
            "good": ">0.8", "acceptable": "0.6-0.8", "poor": "<0.6",
            "meaning": "Direct measure of how similar answers are to ground truth (0=different, 1=identical)."
        },
        "avg_query_time": {
            "description": "Average time to answer a query in seconds",
            "good": "<2s", "acceptable": "2-5s", "poor": ">5s",
            "meaning": "Lower is better. Indicates system responsiveness."
        },
        "avg_contexts_used": {
            "description": "Average number of retrieved documents used per query",
            "good": "3-5", "acceptable": "2-3 or 5-7", "poor": "<2 or >7",
            "meaning": "Should be balanced - too few may miss info, too many may add noise."
        },
        "total_cost": {
            "description": "Total cost in USD for the evaluation",
            "good": "<$0.10", "acceptable": "$0.10-$0.50", "poor": ">$0.50",
            "meaning": "Lower is better. Depends on query complexity and model choice."
        }
    }

    print("\n📊 KEY METRICS EXPLAINED:\n")
    for metric, info in interpretations.items():
        print(f"📈 {metric.upper()}:")
        print(f"   What: {info['description']}")
        print(f"   Range: Good({info['good']}) | OK({info['acceptable']}) | Poor({info['poor']})")
        print(f"   Why it matters: {info['meaning']}")
        print()

    print("💡 OVERALL INTERPRETATION TIPS:")
    print("   1. Semantic metrics (accuracy, F1, similarity) measure answer quality")
    print("   2. Performance metrics (query time) measure system speed")
    print("   3. Cost metrics track API usage and expenses")
    print("   4. Context metrics show retrieval effectiveness")
    print("\n   A good RAG system balances quality (>0.7 semantic scores)")
    print("   with performance (<2s response) and reasonable cost (<$0.01/query)")


if __name__ == "__main__":
    # Run evaluation
    results = evaluate_with_openai()

    # Show interpretation guide
    interpret_metrics()

    print("\n✅ Evaluation complete! Check the reports in:")
    print(f"   experiments/real_openai_evaluation/")