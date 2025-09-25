"""Simple script to compare generated vs ground truth answers with similarity scores"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from autorag.core.rag_pipeline import RAGPipeline
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.evaluation.semantic_metrics import SemanticMetrics
from autorag.core.document_processor import Document


def compare_answers():
    """Generate answers and compare with ground truth"""

    print("Loading MS MARCO data with ground truth...")
    loader = MSMARCOLoader()
    documents, queries = loader.load_subset(num_docs=100, num_queries=20, include_answers=True)

    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    print("Indexing documents...")
    pipeline.index(documents)

    print("Initializing semantic evaluator...")
    evaluator = SemanticMetrics(similarity_threshold=0.7)

    print("\nGenerating answers and calculating similarities...\n")
    print("=" * 120)

    results = []
    for i, query_data in enumerate(queries[:10]):  # Process first 10 for demonstration
        question = query_data["question"]
        ground_truth = query_data.get("ground_truth_answer", "N/A")

        # Generate answer
        result = pipeline.query(question, top_k=5)
        generated = result["answer"]

        # Calculate similarity
        similarities = evaluator.semantic_similarity_batch([generated], [ground_truth])
        similarity = similarities[0]

        results.append((question, ground_truth, generated, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[3], reverse=True)

    # Show examples
    print("\n[HIGH SIMILARITY] GOOD ANSWERS (>= 0.6):")
    print("=" * 120)

    high_sim = [r for r in results if r[3] >= 0.6]
    for q, truth, gen, sim in high_sim[:3]:
        print(f"\nSimilarity Score: {sim:.3f} [GOOD]")
        print(f"Question: {q[:100]}...")
        print(f"Ground Truth: {truth[:150]}...")
        print(f"Generated:    {gen[:150]}...")
        if sim >= 0.7:
            print(">> Analysis: Good semantic match - core information preserved")
        else:
            print(">> Analysis: Moderate match - partially correct information")
        print("-" * 120)

    print("\n[LOW SIMILARITY] POOR ANSWERS (< 0.5):")
    print("=" * 120)

    low_sim = [r for r in results if r[3] < 0.5]
    for q, truth, gen, sim in low_sim[:3]:
        print(f"\nSimilarity Score: {sim:.3f} [POOR]")
        print(f"Question: {q[:100]}...")
        print(f"Ground Truth: {truth[:150]}...")
        print(f"Generated:    {gen[:150]}...")
        print(">> Analysis: Poor match - different information or incorrect answer")
        print("-" * 120)

    # Show statistics
    all_sims = [r[3] for r in results]
    print(f"\nOVERALL STATISTICS FOR {len(results)} ANSWERS:")
    print(f"  Mean Similarity: {sum(all_sims)/len(all_sims):.3f}")
    print(f"  Max Similarity: {max(all_sims):.3f}")
    print(f"  Min Similarity: {min(all_sims):.3f}")
    print(f"  Above 0.7 (Good): {sum(1 for s in all_sims if s >= 0.7)}/{len(all_sims)}")
    print(f"  Above 0.5 (Moderate): {sum(1 for s in all_sims if s >= 0.5)}/{len(all_sims)}")

    # Show what different similarity scores mean
    print("\n[SCORE INTERPRETATION]:")
    print("  0.8-1.0: Excellent - Nearly identical meaning")
    print("  0.7-0.8: Good - Same core information, different phrasing")
    print("  0.5-0.7: Moderate - Partially correct, missing some details")
    print("  0.3-0.5: Poor - Different focus or incorrect")
    print("  0.0-0.3: Very Poor - Completely different or no answer")


if __name__ == "__main__":
    compare_answers()