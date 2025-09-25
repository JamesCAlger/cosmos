"""A/B test with real MS MARCO data and OpenAI APIs"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger
import numpy as np
from scipy import stats
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
from autorag.components.base import Document
from autorag.components.chunkers import FixedSizeChunker, SemanticChunker
from autorag.components.retrievers import BM25Retriever, HybridRetriever, DenseRetriever
from autorag.components.embedders.openai import OpenAIEmbedder
from autorag.components.retrievers.faiss_store import FAISSVectorStore
from autorag.components.generators.openai import OpenAIGenerator
from autorag.components.rerankers import CrossEncoderReranker
from autorag.pipeline.registry import register_default_components
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.evaluation.ragas_evaluator import RAGASEvaluator


class RealRAGPipeline:
    """RAG pipeline with real OpenAI components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unnamed")
        self._initialize_components()
        self.chunks = []

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
        elif retriever_type == "dense":
            self.retriever = self._create_dense_retriever()
        elif retriever_type == "hybrid":
            self.retriever = self._create_hybrid_retriever()
        else:
            self.retriever = BM25Retriever()

        # Initialize reranker (optional)
        use_reranker = self.config.get("use_reranker", False)
        if use_reranker:
            try:
                self.reranker = CrossEncoderReranker({
                    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "batch_size": 32,
                    "normalize_scores": True
                })
                logger.info(f"[{self.name}] Reranker initialized")
            except Exception as e:
                logger.warning(f"[{self.name}] Could not initialize reranker: {e}")
                self.reranker = None
        else:
            self.reranker = None

        # Initialize generator
        self.generator = OpenAIGenerator({
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 150
        })

    def _create_dense_retriever(self) -> DenseRetriever:
        """Create dense retriever with OpenAI embeddings"""
        dense = DenseRetriever()
        embedder = OpenAIEmbedder({
            "model": "text-embedding-ada-002"
        })
        vector_store = FAISSVectorStore({
            "dimension": 1536  # OpenAI ada-002 dimension
        })
        dense.set_components(embedder, vector_store)
        return dense

    def _create_hybrid_retriever(self) -> HybridRetriever:
        """Create hybrid retriever combining dense and sparse"""
        # Create dense retriever
        dense = self._create_dense_retriever()

        # Create sparse retriever
        sparse = BM25Retriever({
            "k1": 1.2,
            "b": 0.75,
            "tokenizer": "simple"
        })

        # Create hybrid retriever
        fusion_method = self.config.get("fusion_method", "rrf")
        hybrid = HybridRetriever({
            "fusion_method": fusion_method,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "rrf_k": 60,
            "normalization": "min_max"
        })

        hybrid.set_retrievers(dense, sparse)
        return hybrid

    def index_documents(self, documents: List[Document]):
        """Index documents for retrieval"""
        # Chunk documents
        logger.info(f"[{self.name}] Chunking {len(documents)} documents...")
        self.chunks = self.chunker.chunk(documents)
        logger.info(f"[{self.name}] Created {len(self.chunks)} chunks")

        # Index chunks
        logger.info(f"[{self.name}] Indexing chunks...")
        if hasattr(self.retriever, 'index'):
            self.retriever.index(self.chunks)
        logger.info(f"[{self.name}] Indexing complete")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query through the pipeline"""
        # Retrieve relevant chunks
        if self.reranker:
            # Retrieve more candidates for reranking
            results = self.retriever.retrieve(question, top_k=top_k * 3)
            # Rerank
            results = self.reranker.rerank(question, results, top_k=top_k)
        else:
            results = self.retriever.retrieve(question, top_k=top_k)

        # Generate answer
        answer = self.generator.generate(question, results)

        return {
            "question": question,
            "answer": answer,
            "contexts": [{"text": r.chunk.content, "score": r.score} for r in results]
        }


def load_msmarco_data(num_queries: int = 20) -> Tuple[List[Dict], List[Document]]:
    """Load MS MARCO data"""
    logger.info("Loading MS MARCO data...")
    loader = MSMARCOLoader()

    # Try to load from cache first
    cache_file = Path("data/msmarco_cache.json")
    if cache_file.exists():
        logger.info("Loading from cache...")
        with open(cache_file, "r") as f:
            data = json.load(f)
            queries = data["queries"][:num_queries]
            documents = [Document(content=d["content"], metadata=d.get("metadata", {}))
                        for d in data["documents"]]
            logger.info(f"Loaded {len(queries)} queries and {len(documents)} documents from cache")
            return queries, documents

    # Load fresh data
    try:
        # Load queries
        queries_df = loader.load_queries()
        queries = []
        for idx, row in queries_df.head(num_queries).iterrows():
            queries.append({
                "qid": row.get("qid", str(idx)),
                "query": row["query"]
            })

        # Load passages (documents)
        passages_df = loader.load_collection(num_passages=100)
        documents = []
        for idx, row in passages_df.iterrows():
            doc = Document(
                content=row["passage"],
                metadata={"pid": row.get("pid", str(idx))}
            )
            documents.append(doc)

        # Cache the data
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({
                "queries": queries,
                "documents": [{"content": d.content, "metadata": d.metadata} for d in documents]
            }, f)

        logger.info(f"Loaded {len(queries)} queries and {len(documents)} documents")
        return queries, documents

    except Exception as e:
        logger.error(f"Failed to load MS MARCO data: {e}")
        logger.info("Creating fallback synthetic data...")

        # Create synthetic fallback data
        queries = [
            {"qid": f"q{i}", "query": q}
            for i, q in enumerate([
                "What is machine learning?",
                "How does deep learning work?",
                "What are neural networks?",
                "Explain natural language processing",
                "What is computer vision?",
                "How does reinforcement learning work?",
                "What is supervised learning?",
                "Describe unsupervised learning",
                "What is transfer learning?",
                "How does gradient descent work?"
            ])
        ][:num_queries]

        documents = [
            Document(
                content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                metadata={"topic": "ML"}
            ),
            Document(
                content="Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
                metadata={"topic": "DL"}
            ),
            Document(
                content="Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
                metadata={"topic": "NN"}
            ),
            Document(
                content="Natural language processing is a branch of AI that helps computers understand, interpret and manipulate human language.",
                metadata={"topic": "NLP"}
            ),
            Document(
                content="Computer vision enables machines to interpret and understand visual information from the world.",
                metadata={"topic": "CV"}
            )
        ]

        return queries, documents


def evaluate_with_ragas(pipeline_results: List[Dict], queries: List[Dict]) -> Dict[str, float]:
    """Evaluate results using RAGAS metrics"""
    try:
        evaluator = RAGASEvaluator()

        # Prepare data for RAGAS
        questions = [r["question"] for r in pipeline_results]
        answers = [r["answer"] for r in pipeline_results]
        contexts = [[c["text"] for c in r["contexts"]] for r in pipeline_results]

        # Calculate metrics
        metrics = evaluator.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts
        )

        return metrics
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")

        # Fallback to simple metrics
        metrics = {
            "retrieval_success": sum(1 for r in pipeline_results if r.get("contexts")) / len(pipeline_results),
            "avg_contexts": np.mean([len(r.get("contexts", [])) for r in pipeline_results]),
            "answer_generated": sum(1 for r in pipeline_results if r.get("answer")) / len(pipeline_results)
        }
        return metrics


def main():
    """Run real A/B test with MS MARCO and OpenAI"""
    logger.info("=" * 60)
    logger.info("A/B Test with Real MS MARCO Data and OpenAI APIs")
    logger.info("=" * 60)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables!")
        logger.info("Please set your OpenAI API key in a .env file or environment variable")
        return

    # Register components
    register_default_components()

    # Configuration A: BM25 only
    config_a = {
        "name": "A_BM25",
        "chunker": "fixed_size",
        "retriever": "bm25",
        "use_reranker": False
    }

    # Configuration B: Hybrid with reranking
    config_b = {
        "name": "B_Hybrid_Reranked",
        "chunker": "semantic",
        "retriever": "hybrid",
        "fusion_method": "rrf",
        "use_reranker": True
    }

    logger.info(f"\n📊 Configuration A: {config_a}")
    logger.info(f"📊 Configuration B: {config_b}")

    # Load MS MARCO data
    logger.info("\n📚 Loading MS MARCO data...")
    queries, documents = load_msmarco_data(num_queries=10)  # Start with 10 queries to control costs

    if not queries or not documents:
        logger.error("Failed to load data")
        return

    logger.info(f"Loaded {len(queries)} queries and {len(documents)} documents")

    # Initialize pipelines
    logger.info("\n🔧 Initializing pipelines...")
    pipeline_a = RealRAGPipeline(config_a)
    pipeline_b = RealRAGPipeline(config_b)

    # Index documents
    logger.info("\n📝 Indexing documents...")

    start_time = time.time()
    pipeline_a.index_documents(documents)
    time_a_index = time.time() - start_time
    logger.info(f"Pipeline A indexing time: {time_a_index:.2f}s")

    start_time = time.time()
    pipeline_b.index_documents(documents)
    time_b_index = time.time() - start_time
    logger.info(f"Pipeline B indexing time: {time_b_index:.2f}s")

    # Run queries
    logger.info("\n🔍 Running queries (this will use OpenAI API credits)...")
    results_a = []
    results_b = []
    times_a = []
    times_b = []

    for i, query_data in enumerate(queries):
        query_text = query_data["query"]
        logger.info(f"\nQuery {i+1}/{len(queries)}: {query_text[:50]}...")

        try:
            # Query pipeline A
            start_time = time.time()
            result_a = pipeline_a.query(query_text, top_k=5)
            time_a = time.time() - start_time
            results_a.append(result_a)
            times_a.append(time_a)
            logger.info(f"  Pipeline A: {time_a:.2f}s")

            # Query pipeline B
            start_time = time.time()
            result_b = pipeline_b.query(query_text, top_k=5)
            time_b = time.time() - start_time
            results_b.append(result_b)
            times_b.append(time_b)
            logger.info(f"  Pipeline B: {time_b:.2f}s")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            continue

    if not results_a or not results_b:
        logger.error("No results to compare")
        return

    # Evaluate with RAGAS
    logger.info("\n📈 Evaluating with RAGAS metrics...")
    metrics_a = evaluate_with_ragas(results_a, queries)
    metrics_b = evaluate_with_ragas(results_b, queries)

    # Add timing metrics
    metrics_a["avg_query_time"] = np.mean(times_a)
    metrics_b["avg_query_time"] = np.mean(times_b)
    metrics_a["indexing_time"] = time_a_index
    metrics_b["indexing_time"] = time_b_index

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    logger.info(f"\n🅰️ Configuration A ({config_a['name']}):")
    for metric, value in metrics_a.items():
        logger.info(f"   {metric}: {value:.4f}")

    logger.info(f"\n🅱️ Configuration B ({config_b['name']}):")
    for metric, value in metrics_b.items():
        logger.info(f"   {metric}: {value:.4f}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    for metric in metrics_a.keys():
        if metric in metrics_b:
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]

            if value_a != 0:
                pct_change = ((value_b - value_a) / value_a) * 100
            else:
                pct_change = 0

            if abs(pct_change) > 5:
                emoji = "✅" if pct_change > 0 else "❌"
            else:
                emoji = "➖"

            logger.info(f"{emoji} {metric}: {pct_change:+.1f}% "
                       f"(A: {value_a:.4f} → B: {value_b:.4f})")

    # Statistical test on retrieval scores
    if results_a and results_b:
        scores_a = [np.mean([c["score"] for c in r["contexts"]]) if r["contexts"] else 0
                   for r in results_a]
        scores_b = [np.mean([c["score"] for c in r["contexts"]]) if r["contexts"] else 0
                   for r in results_b]

        if len(scores_a) > 1 and len(scores_b) > 1:
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

            logger.info("\n" + "=" * 60)
            logger.info("STATISTICAL SIGNIFICANCE")
            logger.info("=" * 60)
            logger.info(f"📊 t-statistic: {t_stat:.4f}")
            logger.info(f"📊 p-value: {p_value:.4f}")

            if p_value < 0.05:
                winner = "B" if np.mean(scores_b) > np.mean(scores_a) else "A"
                logger.info(f"✅ Configuration {winner} is SIGNIFICANTLY BETTER (p < 0.05)")
            else:
                logger.info("➖ No significant difference (p ≥ 0.05)")

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    results = {
        "experiment": "Real MS MARCO with OpenAI",
        "config_a": config_a,
        "config_b": config_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "num_queries": len(queries),
        "num_documents": len(documents),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    output_file = output_dir / "real_ab_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n💾 Results saved to {output_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Determine winner based on key metrics
    key_metrics = ["answer_relevancy", "faithfulness", "context_relevancy"]
    wins_a = 0
    wins_b = 0

    for metric in key_metrics:
        if metric in metrics_a and metric in metrics_b:
            if metrics_b[metric] > metrics_a[metric]:
                wins_b += 1
                logger.info(f"✅ B wins on {metric}")
            elif metrics_a[metric] > metrics_b[metric]:
                wins_a += 1
                logger.info(f"✅ A wins on {metric}")

    if wins_b > wins_a:
        logger.info(f"\n🏆 Overall Winner: Configuration B (Hybrid + Reranking)")
    elif wins_a > wins_b:
        logger.info(f"\n🏆 Overall Winner: Configuration A (BM25)")
    else:
        logger.info(f"\n🤝 Tie - both configurations perform similarly")

    # Cost estimate
    total_queries = len(results_a) + len(results_b)
    embeddings_b = len(pipeline_b.chunks) if hasattr(pipeline_b, 'chunks') else 0

    logger.info("\n💰 Estimated API Usage:")
    logger.info(f"   Queries processed: {total_queries}")
    logger.info(f"   Embeddings created (B): ~{embeddings_b}")
    logger.info(f"   Estimated cost: ~${(total_queries * 0.002 + embeddings_b * 0.0001):.4f}")

    logger.info("\n✅ Real A/B test complete!")


if __name__ == "__main__":
    main()