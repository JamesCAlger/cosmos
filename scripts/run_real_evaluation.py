"""Run real Week 3 evaluation with working OpenAI pipeline"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
from loguru import logger

from autorag.evaluation.service import EvaluationService
from autorag.evaluation.progressive.evaluator import EvaluationLevel
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.pipeline.simple_rag import SimpleRAGPipeline
from autorag.components.base import Document


def main():
    print("\n" + "="*70)
    print("REAL EVALUATION WITH OPENAI MODELS")
    print("="*70)
    print("\nThis will use:")
    print("- OpenAI text-embedding-ada-002 for embeddings")
    print("- OpenAI gpt-3.5-turbo for generation")
    print("- Real MS MARCO data")
    print("- Week 3 evaluation infrastructure")

    # 1. Load MS MARCO data
    print("\n1. Loading MS MARCO dataset...")
    loader = MSMARCOLoader()
    documents, queries = loader.load_subset(
        num_docs=50,  # Small subset for testing
        num_queries=10,
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

    # 2. Initialize pipeline
    print("\n2. Initializing RAG pipeline with OpenAI models...")
    config_path = Path(__file__).parent.parent / "configs" / "baseline_rag.yaml"

    try:
        pipeline = SimpleRAGPipeline(str(config_path))
        print("   [OK] Pipeline initialized with OpenAI components")

        # 3. Index documents
        print("\n3. Indexing documents...")
        start_time = time.time()
        result = pipeline.index(doc_objects)
        print(f"   [OK] Indexed {result['num_chunks']} chunks in {time.time()-start_time:.2f}s")

        # 4. Test a query first
        print("\n4. Testing pipeline with sample query...")
        test_query = queries[0]["question"] if queries else "What is Python?"
        test_result = pipeline.query(test_query)
        print(f"   Query: {test_query[:50]}...")
        print(f"   Answer: {test_result['answer'][:100]}...")
        print(f"   [OK] Pipeline is working!")

        # 5. Run evaluation
        print("\n5. Running Week 3 evaluation...")
        service = EvaluationService(
            enable_caching=True,
            cost_tracking=True,
            budget_limit=0.50,  # $0.50 budget
            progressive_eval=True,
            statistical_analysis=True,
            reporter_formats=["json", "html", "markdown"]
        )

        # Run evaluation
        eval_results = service.evaluate_pipeline(
            pipeline=pipeline,
            test_queries=queries[:5],  # Use first 5 queries
            config={
                "model": "openai",
                "embedding": "text-embedding-ada-002",
                "generation": "gpt-3.5-turbo"
            },
            evaluation_name="real_openai_eval",
            progressive_levels=[EvaluationLevel.SMOKE],
            output_dir="experiments/real_evaluation"
        )

        # 6. Display results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        if "metadata" in eval_results:
            print(f"\nDuration: {eval_results['metadata']['duration']:.2f}s")

        if "cost_summary" in eval_results:
            cost = eval_results["cost_summary"]
            print(f"\nCost Tracking:")
            print(f"  Total cost: ${cost.get('total_cost', 0):.6f}")
            print(f"  Remaining budget: ${cost.get('remaining_budget', 0):.6f}")

        if "progressive" in eval_results:
            prog = eval_results["progressive"]
            if "metrics" in prog:
                print(f"\nMetrics:")
                for level, metrics in prog["metrics"].items():
                    print(f"\n  {level}:")
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)) and key != "num_samples":
                                print(f"    {key}: {value:.3f}")

        print(f"\nReports saved to: experiments/real_evaluation/")
        print("\n[SUCCESS] Real evaluation completed with OpenAI models!")

        return True

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your API key in .env is valid")
        print("2. Check that you have API credits available")
        print("3. Check rate limits on your OpenAI account")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)