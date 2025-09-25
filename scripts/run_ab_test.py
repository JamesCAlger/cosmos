"""Simple A/B test comparing two RAG architectures"""

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
from autorag.components.chunkers import FixedSizeChunker, SemanticChunker
from autorag.components.retrievers import BM25Retriever, HybridRetriever, DenseRetriever
from autorag.components.embedders.mock import MockEmbedder
from autorag.components.retrievers.faiss_store import FAISSVectorStore
from autorag.components.generators.mock import MockGenerator
from autorag.pipeline.registry import register_default_components
from autorag.datasets.msmarco_loader import MSMARCOLoader


class SimpleRAGPipeline:
    """Simple RAG pipeline for A/B testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unnamed")
        self._initialize_components()

    def _initialize_components(self):
        """Initialize pipeline components based on config"""
        # Initialize chunker
        chunker_type = self.config.get("chunker", "fixed_size")
        if chunker_type == "fixed_size":
            self.chunker = FixedSizeChunker({
                "chunk_size": 256,
                "overlap": 50,
                "unit": "tokens"
            })
        elif chunker_type == "semantic":
            self.chunker = SemanticChunker({
                "chunk_size": 256,
                "respect_sentence_boundary": True,
                "overlap_sentences": 1
            })
        else:
            self.chunker = FixedSizeChunker()

        # Initialize retriever
        retriever_type = self.config.get("retriever", "bm25")
        if retriever_type == "bm25":
            self.retriever = BM25Retriever({
                "k1": 1.2,
                "b": 0.75,
                "tokenizer": "simple"
            })
        elif retriever_type == "hybrid":
            # Create hybrid with mock components for speed
            dense = DenseRetriever()
            embedder = MockEmbedder({"dimension": 384})
            vector_store = FAISSVectorStore({"dimension": 384})
            dense.set_components(embedder, vector_store)

            sparse = BM25Retriever({"k1": 1.2, "b": 0.75})

            self.retriever = HybridRetriever({
                "fusion_method": "rrf",
                "rrf_k": 60
            })
            self.retriever.set_retrievers(dense, sparse)
        else:
            self.retriever = BM25Retriever()

        # Use mock generator for speed
        self.generator = MockGenerator({
            "response_type": "relevant",
            "include_context": True
        })

        self.chunks = []

    def index_documents(self, documents: List[Document]):
        """Index documents for retrieval"""
        # Chunk documents
        self.chunks = self.chunker.chunk(documents)
        logger.info(f"[{self.name}] Created {len(self.chunks)} chunks")

        # Index chunks
        if hasattr(self.retriever, 'index'):
            self.retriever.index(self.chunks)

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query through the pipeline"""
        # Retrieve relevant chunks
        results = self.retriever.retrieve(question, top_k=top_k)

        # Generate answer
        answer = self.generator.generate(question, results)

        return {
            "question": question,
            "answer": answer,
            "contexts": [{"text": r.chunk.content[:100], "score": r.score} for r in results[:3]]
        }


def calculate_metrics(pipeline_results: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Calculate simple evaluation metrics"""
    metrics = {
        "retrieval_success": 0,
        "avg_retrieval_score": 0,
        "answer_generated": 0,
        "avg_context_length": 0
    }

    total_score = 0
    total_contexts = 0

    for result in pipeline_results:
        # Check if retrieval succeeded
        if result.get("contexts"):
            metrics["retrieval_success"] += 1

            # Calculate average score
            scores = [c["score"] for c in result["contexts"] if "score" in c]
            if scores:
                total_score += np.mean(scores)

            # Calculate context length
            for ctx in result["contexts"]:
                total_contexts += len(ctx.get("text", ""))

        # Check if answer was generated
        if result.get("answer"):
            metrics["answer_generated"] += 1

    # Calculate averages
    n = len(pipeline_results)
    if n > 0:
        metrics["retrieval_success"] /= n
        metrics["answer_generated"] /= n
        metrics["avg_retrieval_score"] = total_score / n if metrics["retrieval_success"] > 0 else 0
        metrics["avg_context_length"] = total_contexts / (n * 3) if total_contexts > 0 else 0

    return metrics


def run_statistical_test(results_a: List[float], results_b: List[float]) -> Dict[str, Any]:
    """Run statistical significance test between two result sets"""
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(results_a, results_b)

    # Calculate effect size (Cohen's d)
    diff = np.array(results_a) - np.array(results_b)
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

    # Determine significance
    is_significant = p_value < 0.05

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "is_significant": is_significant,
        "mean_a": np.mean(results_a),
        "mean_b": np.mean(results_b),
        "std_a": np.std(results_a),
        "std_b": np.std(results_b)
    }


def main():
    """Run A/B test comparing two pipeline configurations"""
    logger.info("=" * 60)
    logger.info("A/B Test: RAG Architecture Comparison")
    logger.info("=" * 60)

    # Register components
    register_default_components()

    # Configuration A: Baseline with fixed-size chunking and BM25
    config_a = {
        "name": "A_baseline",
        "chunker": "fixed_size",
        "retriever": "bm25"
    }

    # Configuration B: Changed to semantic chunking (single component swap)
    config_b = {
        "name": "B_semantic_chunking",
        "chunker": "semantic",  # <-- Changed component
        "retriever": "bm25"
    }

    logger.info(f"\n📊 Configuration A: {config_a}")
    logger.info(f"📊 Configuration B: {config_b}")

    # Load sample data
    logger.info("\n📚 Loading MS MARCO sample data...")
    loader = MSMARCOLoader()

    try:
        # Try to load cached data first
        sample_data = loader.load_sample(num_queries=20)
        logger.info(f"Loaded {len(sample_data['queries'])} queries")
    except Exception as e:
        logger.warning(f"Could not load MS MARCO data: {e}")
        logger.info("Creating synthetic test data...")

        # Create synthetic data for testing
        sample_data = {
            "queries": [
                {"qid": f"q{i}", "query": f"What is {topic}?"}
                for i, topic in enumerate([
                    "machine learning", "artificial intelligence", "deep learning",
                    "neural networks", "computer vision", "natural language processing",
                    "reinforcement learning", "supervised learning", "unsupervised learning",
                    "transfer learning", "federated learning", "meta learning",
                    "active learning", "online learning", "batch learning",
                    "gradient descent", "backpropagation", "convolutional networks",
                    "recurrent networks", "transformer models"
                ])
            ],
            "passages": [
                {
                    "pid": f"p{i}",
                    "passage": f"{topic} is a fundamental concept in artificial intelligence. "
                              f"It involves training models to learn patterns from data. "
                              f"The process includes collecting data, preprocessing, training, and evaluation. "
                              f"{topic} has many applications in real-world scenarios."
                }
                for i, topic in enumerate([
                    "Machine learning", "Artificial intelligence", "Deep learning",
                    "Neural networks", "Computer vision", "Natural language processing",
                    "Reinforcement learning", "Supervised learning", "Unsupervised learning",
                    "Transfer learning"
                ])
            ]
        }

    # Prepare documents
    documents = []
    for passage in sample_data.get("passages", []):
        if isinstance(passage, dict):
            content = passage.get("passage", passage.get("passage_text", ""))
        else:
            content = str(passage)

        doc = Document(
            content=content,
            metadata={"passage_id": passage.get("pid", "")} if isinstance(passage, dict) else {}
        )
        documents.append(doc)

    logger.info(f"Prepared {len(documents)} documents for indexing")

    # Initialize pipelines
    logger.info("\n🔧 Initializing pipelines...")
    pipeline_a = SimpleRAGPipeline(config_a)
    pipeline_b = SimpleRAGPipeline(config_b)

    # Index documents
    logger.info("\n📝 Indexing documents...")
    start_time = time.time()
    pipeline_a.index_documents(documents)
    time_a_index = time.time() - start_time

    start_time = time.time()
    pipeline_b.index_documents(documents)
    time_b_index = time.time() - start_time

    # Run queries
    logger.info("\n🔍 Running queries...")
    results_a = []
    results_b = []
    scores_a = []
    scores_b = []

    queries = sample_data.get("queries", [])[:20]  # Limit to 20 queries

    for query_data in queries:
        if isinstance(query_data, dict):
            question = query_data.get("query", query_data.get("question", ""))
        else:
            question = str(query_data)

        if not question:
            continue

        # Query pipeline A
        start_time = time.time()
        result_a = pipeline_a.query(question)
        time_a = time.time() - start_time
        result_a["time"] = time_a
        results_a.append(result_a)

        # Extract score for statistical test
        if result_a.get("contexts"):
            avg_score = np.mean([c.get("score", 0) for c in result_a["contexts"]])
            scores_a.append(avg_score)
        else:
            scores_a.append(0)

        # Query pipeline B
        start_time = time.time()
        result_b = pipeline_b.query(question)
        time_b = time.time() - start_time
        result_b["time"] = time_b
        results_b.append(result_b)

        # Extract score for statistical test
        if result_b.get("contexts"):
            avg_score = np.mean([c.get("score", 0) for c in result_b["contexts"]])
            scores_b.append(avg_score)
        else:
            scores_b.append(0)

    # Calculate metrics
    logger.info("\n📈 Calculating metrics...")
    metrics_a = calculate_metrics(results_a, queries)
    metrics_b = calculate_metrics(results_b, queries)

    # Add timing metrics
    metrics_a["avg_query_time"] = np.mean([r["time"] for r in results_a])
    metrics_b["avg_query_time"] = np.mean([r["time"] for r in results_b])
    metrics_a["indexing_time"] = time_a_index
    metrics_b["indexing_time"] = time_b_index

    # Statistical significance test
    logger.info("\n📊 Statistical Analysis...")
    stat_results = run_statistical_test(scores_a, scores_b)

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    logger.info(f"\n🅰️  Configuration A ({config_a['name']}):")
    logger.info(f"   Chunker: {config_a['chunker']}")
    logger.info(f"   Retriever: {config_a['retriever']}")
    logger.info(f"   Metrics:")
    for metric, value in metrics_a.items():
        logger.info(f"     {metric}: {value:.4f}")

    logger.info(f"\n🅱️  Configuration B ({config_b['name']}):")
    logger.info(f"   Chunker: {config_b['chunker']}")
    logger.info(f"   Retriever: {config_b['retriever']}")
    logger.info(f"   Metrics:")
    for metric, value in metrics_b.items():
        logger.info(f"     {metric}: {value:.4f}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    for metric in metrics_a.keys():
        if metric in metrics_b:
            diff = metrics_b[metric] - metrics_a[metric]
            pct_change = (diff / metrics_a[metric] * 100) if metrics_a[metric] != 0 else 0

            if abs(pct_change) > 5:  # Significant change
                emoji = "✅" if diff > 0 else "❌"
            else:
                emoji = "➖"

            logger.info(f"{emoji} {metric}: {pct_change:+.1f}% "
                       f"(A: {metrics_a[metric]:.4f} → B: {metrics_b[metric]:.4f})")

    # Statistical significance
    logger.info("\n" + "=" * 60)
    logger.info("STATISTICAL SIGNIFICANCE")
    logger.info("=" * 60)

    logger.info(f"📊 t-statistic: {stat_results['t_statistic']:.4f}")
    logger.info(f"📊 p-value: {stat_results['p_value']:.4f}")
    logger.info(f"📊 Cohen's d: {stat_results['cohens_d']:.4f}")
    logger.info(f"📊 Mean retrieval score A: {stat_results['mean_a']:.4f} ± {stat_results['std_a']:.4f}")
    logger.info(f"📊 Mean retrieval score B: {stat_results['mean_b']:.4f} ± {stat_results['std_b']:.4f}")

    if stat_results['is_significant']:
        if stat_results['mean_b'] > stat_results['mean_a']:
            logger.info("✅ Configuration B is SIGNIFICANTLY BETTER than A (p < 0.05)")
        else:
            logger.info("❌ Configuration A is SIGNIFICANTLY BETTER than B (p < 0.05)")
    else:
        logger.info("➖ No significant difference between configurations (p ≥ 0.05)")

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Convert numpy values to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    results = {
        "config_a": config_a,
        "config_b": config_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "statistical_test": convert_to_json_serializable(stat_results),
        "num_queries": len(queries),
        "num_documents": len(documents),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    output_file = output_dir / "ab_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to {output_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    winner = None
    if stat_results['is_significant']:
        winner = "B" if stat_results['mean_b'] > stat_results['mean_a'] else "A"
        logger.info(f"🏆 Winner: Configuration {winner}")
        logger.info(f"   Effect size (Cohen's d): {abs(stat_results['cohens_d']):.4f}")

        if abs(stat_results['cohens_d']) < 0.2:
            logger.info("   Effect size: Small")
        elif abs(stat_results['cohens_d']) < 0.5:
            logger.info("   Effect size: Medium")
        else:
            logger.info("   Effect size: Large")
    else:
        logger.info("🤝 No clear winner - configurations perform similarly")

    logger.info("\n✅ A/B test complete!")


if __name__ == "__main__":
    main()