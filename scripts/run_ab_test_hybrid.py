"""A/B test comparing BM25 vs Hybrid retrieval"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger
import numpy as np
from scipy import stats

# Import components
from autorag.components.base import Document
from autorag.components.chunkers import FixedSizeChunker
from autorag.components.retrievers import BM25Retriever, HybridRetriever, DenseRetriever
from autorag.components.embedders.mock import MockEmbedder
from autorag.components.retrievers.faiss_store import FAISSVectorStore
from autorag.components.generators.mock import MockGenerator
from autorag.pipeline.registry import register_default_components


def create_test_documents() -> List[Document]:
    """Create more realistic test documents"""
    documents = [
        Document(
            content="Machine learning is a subset of artificial intelligence that focuses on "
                   "building systems that learn from data. Instead of being explicitly programmed, "
                   "these systems identify patterns and make decisions based on data analysis.",
            metadata={"topic": "ML basics"}
        ),
        Document(
            content="Deep learning is a specialized form of machine learning that uses neural "
                   "networks with multiple layers. These deep neural networks can automatically "
                   "learn hierarchical representations from raw data.",
            metadata={"topic": "Deep learning"}
        ),
        Document(
            content="Natural language processing enables computers to understand, interpret, and "
                   "generate human language. Modern NLP systems use transformer architectures like "
                   "BERT and GPT to achieve state-of-the-art performance.",
            metadata={"topic": "NLP"}
        ),
        Document(
            content="Computer vision is the field of AI that trains computers to interpret and "
                   "understand visual information from the world. Applications include image "
                   "classification, object detection, and facial recognition.",
            metadata={"topic": "Computer vision"}
        ),
        Document(
            content="Reinforcement learning is a type of machine learning where agents learn to "
                   "make decisions by interacting with an environment. The agent receives rewards "
                   "or penalties based on its actions and learns to maximize cumulative reward.",
            metadata={"topic": "RL"}
        ),
        Document(
            content="Supervised learning requires labeled training data where each example has an "
                   "input and a corresponding output. Common algorithms include linear regression, "
                   "decision trees, and support vector machines.",
            metadata={"topic": "Supervised learning"}
        ),
        Document(
            content="Unsupervised learning finds patterns in data without labeled examples. "
                   "Clustering algorithms like K-means and dimensionality reduction techniques "
                   "like PCA are common unsupervised methods.",
            metadata={"topic": "Unsupervised learning"}
        ),
        Document(
            content="Transfer learning leverages knowledge from pre-trained models to solve new "
                   "but related problems. This approach significantly reduces training time and "
                   "data requirements for new tasks.",
            metadata={"topic": "Transfer learning"}
        ),
        Document(
            content="Neural networks are computing systems inspired by biological neural networks. "
                   "They consist of interconnected nodes or neurons that process information using "
                   "connectionist approaches to computation.",
            metadata={"topic": "Neural networks"}
        ),
        Document(
            content="Gradient descent is an optimization algorithm used to minimize loss functions "
                   "in machine learning. It iteratively adjusts parameters in the direction of "
                   "steepest descent to find optimal values.",
            metadata={"topic": "Optimization"}
        )
    ]
    return documents


def run_experiment(config: Dict[str, Any], documents: List[Document],
                  queries: List[str]) -> tuple:
    """Run experiment with given configuration"""

    # Initialize chunker
    chunker = FixedSizeChunker({
        "chunk_size": 100,
        "overlap": 20,
        "unit": "tokens"
    })

    # Initialize retriever based on config
    if config["retriever"] == "bm25":
        retriever = BM25Retriever({
            "k1": 1.2,
            "b": 0.75,
            "tokenizer": "simple"
        })
    else:  # hybrid
        # Create components for hybrid
        dense = DenseRetriever()
        embedder = MockEmbedder({"dimension": 384})
        vector_store = FAISSVectorStore({"dimension": 384})
        dense.set_components(embedder, vector_store)

        sparse = BM25Retriever({"k1": 1.2, "b": 0.75})

        retriever = HybridRetriever({
            "fusion_method": "rrf",
            "rrf_k": 60,
            "normalization": "min_max"
        })
        retriever.set_retrievers(dense, sparse)

    # Initialize generator
    generator = MockGenerator({"response_type": "relevant"})

    # Index documents
    chunks = chunker.chunk(documents)
    retriever.index(chunks)

    # Run queries
    results = []
    scores = []
    times = []

    for query in queries:
        start = time.time()
        retrieved = retriever.retrieve(query, top_k=5)
        answer = generator.generate(query, retrieved)
        elapsed = time.time() - start

        results.append({
            "query": query,
            "answer": answer,
            "num_results": len(retrieved),
            "avg_score": np.mean([r.score for r in retrieved]) if retrieved else 0
        })

        scores.append(results[-1]["avg_score"])
        times.append(elapsed)

    return results, scores, times


def main():
    logger.info("=" * 60)
    logger.info("A/B Test: BM25 vs Hybrid Retrieval")
    logger.info("=" * 60)

    # Register components
    register_default_components()

    # Configurations
    config_a = {"name": "BM25_Only", "retriever": "bm25"}
    config_b = {"name": "Hybrid_RRF", "retriever": "hybrid"}

    # Create documents and queries
    documents = create_test_documents()
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain supervised learning",
        "What is reinforcement learning?",
        "How does transfer learning help?",
        "What are neural networks?",
        "Describe unsupervised learning methods",
        "What is gradient descent?",
        "How does computer vision work?",
        "What is natural language processing?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the applications of deep learning?",
        "How do transformers work in NLP?",
        "What is the role of optimization in ML?",
        "Describe clustering algorithms"
    ]

    logger.info(f"\n📚 Test Setup:")
    logger.info(f"   Documents: {len(documents)}")
    logger.info(f"   Queries: {len(queries)}")

    # Run experiments
    logger.info(f"\n🔬 Running Configuration A: {config_a['name']}")
    results_a, scores_a, times_a = run_experiment(config_a, documents, queries)

    logger.info(f"\n🔬 Running Configuration B: {config_b['name']}")
    results_b, scores_b, times_b = run_experiment(config_b, documents, queries)

    # Statistical analysis
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    cohens_d = (np.mean(scores_b) - np.mean(scores_a)) / np.std(np.array(scores_b) - np.array(scores_a))

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    logger.info(f"\n📊 Configuration A: {config_a['name']}")
    logger.info(f"   Mean Score: {np.mean(scores_a):.4f} ± {np.std(scores_a):.4f}")
    logger.info(f"   Mean Time: {np.mean(times_a)*1000:.2f}ms")

    logger.info(f"\n📊 Configuration B: {config_b['name']}")
    logger.info(f"   Mean Score: {np.mean(scores_b):.4f} ± {np.std(scores_b):.4f}")
    logger.info(f"   Mean Time: {np.mean(times_b)*1000:.2f}ms")

    # Comparison
    score_improvement = ((np.mean(scores_b) - np.mean(scores_a)) / np.mean(scores_a)) * 100
    time_difference = ((np.mean(times_b) - np.mean(times_a)) / np.mean(times_a)) * 100

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    logger.info(f"📈 Score Improvement: {score_improvement:+.1f}%")
    logger.info(f"⏱️ Time Difference: {time_difference:+.1f}%")
    logger.info(f"📊 t-statistic: {t_stat:.4f}")
    logger.info(f"📊 p-value: {p_value:.4f}")
    logger.info(f"📊 Cohen's d: {cohens_d:.4f}")

    # Significance
    if p_value < 0.05:
        winner = "Hybrid" if np.mean(scores_b) > np.mean(scores_a) else "BM25"
        logger.info(f"\n✅ {winner} is SIGNIFICANTLY BETTER (p < 0.05)")
    else:
        logger.info("\n➖ No significant difference (p ≥ 0.05)")

    # Query-by-query comparison
    logger.info("\n" + "=" * 60)
    logger.info("QUERY-BY-QUERY COMPARISON")
    logger.info("=" * 60)

    wins_a = 0
    wins_b = 0
    ties = 0

    for i, query in enumerate(queries[:5]):  # Show first 5
        score_a = scores_a[i]
        score_b = scores_b[i]

        if abs(score_a - score_b) < 0.001:
            result = "TIE"
            ties += 1
        elif score_b > score_a:
            result = "Hybrid +"
            wins_b += 1
        else:
            result = "BM25 +"
            wins_a += 1

        logger.info(f"{query[:40]:40} | BM25: {score_a:.3f} | Hybrid: {score_b:.3f} | {result}")

    logger.info(f"\nTotal: BM25 wins: {wins_a}, Hybrid wins: {wins_b}, Ties: {ties}")

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    results = {
        "experiment": "BM25 vs Hybrid",
        "config_a": config_a,
        "config_b": config_b,
        "metrics": {
            "mean_score_a": float(np.mean(scores_a)),
            "mean_score_b": float(np.mean(scores_b)),
            "std_score_a": float(np.std(scores_a)),
            "std_score_b": float(np.std(scores_b)),
            "score_improvement": float(score_improvement),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d)
        },
        "num_queries": len(queries),
        "num_documents": len(documents)
    }

    output_file = output_dir / "ab_test_bm25_vs_hybrid.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to {output_file}")
    logger.info("\n✅ A/B test complete!")


if __name__ == "__main__":
    main()