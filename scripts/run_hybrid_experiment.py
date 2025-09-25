"""Run experiment with hybrid retrieval and reranking"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from autorag.components.base import Document
from autorag.components.chunkers import SemanticChunker
from autorag.components.retrievers import HybridRetriever, BM25Retriever, DenseRetriever
from autorag.components.embedders.openai import OpenAIEmbedder
from autorag.components.retrievers.faiss_store import FAISSVectorStore
from autorag.components.generators.openai import OpenAIGenerator
from autorag.components.rerankers import CrossEncoderReranker
from autorag.pipeline.registry import register_default_components
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.evaluation.ragas_evaluator import RAGASEvaluator
from autorag.evaluation.traditional_metrics import TraditionalMetrics
from autorag.evaluation.semantic_metrics import SemanticMetrics


class HybridRAGPipeline:
    """Pipeline with hybrid retrieval and reranking"""
    
    def __init__(self):
        logger.info("Initializing Hybrid RAG Pipeline with Reranking")
        
        # Initialize chunker
        self.chunker = SemanticChunker({
            "chunk_size": 256,
            "respect_sentence_boundary": True,
            "overlap_sentences": 1
        })
        
        # Initialize hybrid retriever
        self.retriever = self._create_hybrid_retriever()
        
        # Initialize reranker
        self.reranker = CrossEncoderReranker({
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "batch_size": 32,
            "normalize_scores": True
        })
        
        # Initialize generator
        self.generator = OpenAIGenerator({
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 150
        })
        
        self.chunks = []
    
    def _create_hybrid_retriever(self) -> HybridRetriever:
        """Create hybrid retriever combining dense and sparse"""
        # Create dense retriever
        dense = DenseRetriever()
        embedder = OpenAIEmbedder({
            "model": "text-embedding-ada-002"
        })
        vector_store = FAISSVectorStore({
            "dimension": 1536
        })
        dense.set_components(embedder, vector_store)
        
        # Create sparse retriever
        sparse = BM25Retriever({
            "k1": 1.2,
            "b": 0.75
        })
        
        # Create hybrid retriever with RRF
        hybrid = HybridRetriever({
            "fusion_method": "rrf",
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "rrf_k": 60,
            "normalization": "min_max"
        })
        
        hybrid.set_retrievers(dense, sparse)
        return hybrid
    
    def index(self, documents: list):
        """Index documents"""
        # Chunk documents
        logger.info(f"Chunking {len(documents)} documents...")
        self.chunks = self.chunker.chunk(documents)
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # Index chunks
        logger.info("Indexing chunks...")
        self.retriever.index(self.chunks)
        logger.info("Indexing complete")
    
    def query(self, question: str, top_k: int = 5) -> dict:
        """Process a query"""
        # Retrieve with more candidates for reranking
        results = self.retriever.retrieve(question, top_k=top_k * 3)
        
        # Rerank
        results = self.reranker.rerank(question, results, top_k=top_k)
        
        # Generate answer
        answer = self.generator.generate(question, results)
        
        return {
            "question": question,
            "answer": answer,
            "contexts": [r.chunk.content for r in results],
            "scores": [r.score for r in results]
        }


def main():
    """Run hybrid experiment"""
    logger.info("=" * 60)
    logger.info("Hybrid Retrieval with Reranking Experiment")
    logger.info("=" * 60)
    
    # Register components
    register_default_components()
    
    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)
    
    # Load dataset
    logger.info("Loading MS MARCO dataset...")
    loader = MSMARCOLoader()
    documents, queries = loader.load_subset(
        num_docs=100,
        num_queries=20,
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
            doc_objects.append(doc)
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = HybridRAGPipeline()
    
    # Index documents
    start_time = time.time()
    pipeline.index(doc_objects)
    indexing_time = time.time() - start_time
    logger.info(f"Indexing completed in {indexing_time:.2f}s")
    
    # Process queries
    logger.info("Processing queries...")
    results = []
    query_times = []
    
    for i, query_data in enumerate(queries[:20]):
        query_text = query_data.get("query", query_data) if isinstance(query_data, dict) else query_data
        logger.info(f"Query {i+1}/20: {query_text[:50]}...")
        
        start_time = time.time()
        result = pipeline.query(query_text)
        query_time = time.time() - start_time
        
        results.append(result)
        query_times.append(query_time)
        logger.debug(f"Query processed in {query_time:.2f}s")
    
    # Evaluate
    logger.info("Running evaluation...")
    
    # Traditional metrics
    trad_evaluator = TraditionalMetrics()
    predictions = [r["answer"] for r in results]
    ground_truths = [q.get("answer", "") for q in queries[:20]] if isinstance(queries[0], dict) else [""] * 20
    
    if any(ground_truths):
        trad_metrics = trad_evaluator.evaluate(predictions, ground_truths)
    else:
        trad_metrics = {"note": "No ground truth available"}
    
    # Semantic metrics
    sem_evaluator = SemanticMetrics()
    sem_metrics = sem_evaluator.evaluate(predictions, ground_truths) if any(ground_truths) else {}
    
    # Try RAGAS
    try:
        ragas_evaluator = RAGASEvaluator()
        questions = [r["question"] for r in results]
        answers = [r["answer"] for r in results]
        contexts = [[c] for c in [r["contexts"][0] if r["contexts"] else "" for r in results]]
        ragas_metrics = ragas_evaluator.evaluate(questions, answers, contexts)
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        ragas_metrics = {}
    
    # Compile results
    experiment_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "architecture": "hybrid_with_reranking",
            "chunker": "semantic",
            "retriever": "hybrid_rrf",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "generator": "gpt-3.5-turbo"
        },
        "dataset": {
            "num_documents": len(documents),
            "num_queries": 20,
            "num_chunks": len(pipeline.chunks)
        },
        "performance": {
            "indexing_time_seconds": indexing_time,
            "avg_query_time_seconds": sum(query_times) / len(query_times),
            "total_time_seconds": indexing_time + sum(query_times)
        },
        "evaluation": {
            "traditional_metrics": trad_metrics,
            "semantic_metrics": sem_metrics,
            "ragas_metrics": ragas_metrics
        },
        "sample_results": results[:3]
    }
    
    # Save results
    output_file = Path("experiments") / f"hybrid_reranked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")
    
    # Display summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Architecture: Hybrid with Reranking")
    logger.info(f"Documents: {len(documents)}")
    logger.info(f"Chunks: {len(pipeline.chunks)}")
    logger.info(f"Queries: 20")
    logger.info(f"Avg Query Time: {sum(query_times)/len(query_times):.2f}s")
    
    if trad_metrics and "token_f1" in trad_metrics:
        logger.info(f"Token F1: {trad_metrics['token_f1']:.3f}")
    if sem_metrics and "semantic_f1" in sem_metrics:
        logger.info(f"Semantic F1: {sem_metrics['semantic_f1']:.3f}")
    if ragas_metrics:
        for metric, value in ragas_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric}: {value:.3f}")
    
    logger.info("\n✅ Hybrid experiment complete!")
    return experiment_results


if __name__ == "__main__":
    main()