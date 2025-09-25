"""Evaluate Week 4 components on MS MARCO dataset"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger
import time

# Import evaluation components
from autorag.evaluation.service import EvaluationService
from autorag.evaluation.progressive.evaluator import ProgressiveEvaluator, EvaluationLevel
from autorag.evaluation.cost_tracker import CostTracker
from autorag.evaluation.reporters.base import JSONReporter
from autorag.datasets.enhanced_loader import EnhancedDatasetLoader

# Import pipeline components
from autorag.pipeline.registry import get_registry, register_default_components
from autorag.components.base import Document
from autorag.components.chunkers import SemanticChunker, SlidingWindowChunker
from autorag.components.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from autorag.components.embedders.openai import OpenAIEmbedder
from autorag.components.retrievers.faiss_store import FAISSVectorStore
from autorag.components.generators.openai import OpenAIGenerator
from autorag.components.rerankers import CrossEncoderReranker


class Week4Pipeline:
    """Pipeline with Week 4 components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunker = None
        self.retriever = None
        self.reranker = None
        self.generator = None
        self.documents = []

        self._initialize_components()

    def _initialize_components(self):
        """Initialize pipeline components based on config"""
        # Initialize chunker
        chunker_type = self.config.get("chunker", "semantic")
        if chunker_type == "semantic":
            self.chunker = SemanticChunker({
                "chunk_size": 256,
                "respect_sentence_boundary": True,
                "overlap_sentences": 1
            })
        elif chunker_type == "sliding_window":
            self.chunker = SlidingWindowChunker({
                "window_size": 256,
                "step_size": 128,
                "unit": "tokens"
            })
        else:
            # Default to semantic
            self.chunker = SemanticChunker()

        # Initialize retriever
        retriever_type = self.config.get("retriever", "hybrid")
        if retriever_type == "bm25":
            self.retriever = BM25Retriever({
                "k1": 1.2,
                "b": 0.75
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
                logger.info("Reranker initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize reranker: {e}")
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
            "dimension": 1536  # OpenAI embedding dimension
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
            "b": 0.75
        })

        # Create hybrid retriever
        hybrid = HybridRetriever({
            "fusion_method": self.config.get("fusion_method", "rrf"),
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "rrf_k": 60,
            "normalization": "min_max"
        })

        hybrid.set_retrievers(dense, sparse)
        return hybrid

    def index_documents(self, documents: List[Document]):
        """Index documents for retrieval"""
        self.documents = documents

        # Chunk documents
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = self.chunker.chunk(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Index chunks
        logger.info("Indexing chunks for retrieval...")
        if hasattr(self.retriever, 'index'):
            self.retriever.index(chunks)
        else:
            logger.warning("Retriever does not support indexing")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query through the pipeline"""
        # Retrieve relevant chunks
        results = self.retriever.retrieve(question, top_k=top_k * 2 if self.reranker else top_k)

        # Rerank if configured
        if self.reranker and results:
            results = self.reranker.rerank(question, results, top_k=top_k)

        # Generate answer
        answer = self.generator.generate(question, results)

        return {
            "question": question,
            "answer": answer,
            "contexts": [{"text": r.chunk.content, "score": r.score} for r in results]
        }


def evaluate_configuration(config: Dict[str, Any], dataset: List[Dict],
                          evaluator: ProgressiveEvaluator) -> Dict[str, Any]:
    """Evaluate a specific configuration"""
    logger.info(f"\nEvaluating configuration: {config}")

    # Create pipeline
    pipeline = Week4Pipeline(config)

    # Prepare documents from dataset
    documents = []
    for item in dataset:
        # Use passages as documents
        for passage in item.get("passages", []):
            doc = Document(
                content=passage["passage_text"],
                metadata={"passage_id": passage.get("passage_id", "")}
            )
            documents.append(doc)

    # Index documents
    pipeline.index_documents(documents)

    # Evaluate using progressive evaluator
    evaluation_results = evaluator.evaluate(
        pipeline=pipeline,
        dataset=dataset[:20],  # Limit to 20 queries
        level=EvaluationLevel.QUICK
    )

    return evaluation_results


def main():
    """Run Week 4 component evaluation"""
    logger.info("=" * 60)
    logger.info("Week 4 Component Evaluation on MS MARCO")
    logger.info("=" * 60)

    # Register components
    register_default_components()

    # Load dataset
    logger.info("\nLoading MS MARCO dataset...")
    loader = EnhancedDatasetLoader()
    dataset = loader.load_dataset("msmarco", split="dev", sample_size=100)
    logger.info(f"Loaded {len(dataset)} queries")

    # Initialize evaluation components
    logger.info("\nInitializing evaluation infrastructure...")
    evaluator = ProgressiveEvaluator()
    cost_tracker = CostTracker()
    reporter = JSONReporter(output_dir="evaluation_results")

    # Define configurations to test
    configurations = [
        {
            "name": "baseline_bm25",
            "chunker": "semantic",
            "retriever": "bm25",
            "use_reranker": False
        },
        {
            "name": "dense_only",
            "chunker": "semantic",
            "retriever": "dense",
            "use_reranker": False
        },
        {
            "name": "hybrid_rrf",
            "chunker": "semantic",
            "retriever": "hybrid",
            "fusion_method": "rrf",
            "use_reranker": False
        },
        {
            "name": "hybrid_weighted",
            "chunker": "semantic",
            "retriever": "hybrid",
            "fusion_method": "weighted_sum",
            "use_reranker": False
        },
        {
            "name": "hybrid_rrf_reranked",
            "chunker": "semantic",
            "retriever": "hybrid",
            "fusion_method": "rrf",
            "use_reranker": True
        },
        {
            "name": "sliding_window_hybrid",
            "chunker": "sliding_window",
            "retriever": "hybrid",
            "fusion_method": "rrf",
            "use_reranker": False
        }
    ]

    # Evaluate each configuration
    results = {}
    for config in configurations:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"{'=' * 40}")

        try:
            start_time = time.time()
            eval_results = evaluate_configuration(config, dataset, evaluator)
            elapsed_time = time.time() - start_time

            results[config["name"]] = {
                "config": config,
                "metrics": eval_results.get("metrics", {}),
                "time": elapsed_time
            }

            # Log results
            logger.info(f"\nResults for {config['name']}:")
            for metric, value in eval_results.get("metrics", {}).items():
                logger.info(f"  {metric}: {value:.3f}")
            logger.info(f"  Time: {elapsed_time:.2f}s")

        except Exception as e:
            logger.error(f"Error evaluating {config['name']}: {e}")
            results[config["name"]] = {
                "config": config,
                "error": str(e)
            }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    # Find best configuration
    best_config = None
    best_score = -1
    metric_key = "answer_relevancy"  # Primary metric

    for name, result in results.items():
        if "error" not in result:
            score = result.get("metrics", {}).get(metric_key, 0)
            logger.info(f"{name}: {metric_key}={score:.3f}")
            if score > best_score:
                best_score = score
                best_config = name

    if best_config:
        logger.info(f"\nBest configuration: {best_config} ({metric_key}={best_score:.3f})")

        # Calculate improvements
        baseline_score = results.get("baseline_bm25", {}).get("metrics", {}).get(metric_key, 0)
        if baseline_score > 0:
            improvement = ((best_score - baseline_score) / baseline_score) * 100
            logger.info(f"Improvement over baseline: {improvement:.1f}%")

    # Save results
    output_file = Path("evaluation_results") / "week4_evaluation.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")

    # Week 4 specific validations
    logger.info("\n" + "=" * 60)
    logger.info("Week 4 Success Criteria Validation")
    logger.info("=" * 60)

    # Check hybrid vs single method
    bm25_score = results.get("baseline_bm25", {}).get("metrics", {}).get(metric_key, 0)
    dense_score = results.get("dense_only", {}).get("metrics", {}).get(metric_key, 0)
    hybrid_score = results.get("hybrid_rrf", {}).get("metrics", {}).get(metric_key, 0)

    if bm25_score > 0 and dense_score > 0:
        hybrid_improvement_bm25 = ((hybrid_score - bm25_score) / bm25_score) * 100
        hybrid_improvement_dense = ((hybrid_score - dense_score) / dense_score) * 100
        logger.info(f"✓ Hybrid retrieval improvement over BM25: {hybrid_improvement_bm25:.1f}%")
        logger.info(f"✓ Hybrid retrieval improvement over Dense: {hybrid_improvement_dense:.1f}%")

    # Check reranker improvement
    base_hybrid = results.get("hybrid_rrf", {}).get("metrics", {}).get(metric_key, 0)
    reranked_hybrid = results.get("hybrid_rrf_reranked", {}).get("metrics", {}).get(metric_key, 0)
    if base_hybrid > 0:
        reranker_improvement = ((reranked_hybrid - base_hybrid) / base_hybrid) * 100
        logger.info(f"✓ Reranker improvement: {reranker_improvement:.1f}%")

    logger.info("\n✅ Week 4 evaluation complete!")


if __name__ == "__main__":
    main()