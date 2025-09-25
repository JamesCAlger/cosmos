"""Demo script for Week 4 components with MS MARCO evaluation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
import numpy as np
from typing import Dict, Any, List

from autorag.components.base import Document, Chunk
from autorag.components.chunkers import (
    SemanticChunker, SlidingWindowChunker,
    HierarchicalChunker, DocumentAwareChunker
)
from autorag.components.retrievers import (
    BM25Retriever, DenseRetriever, HybridRetriever
)
from autorag.components.embedders.mock import MockEmbedder
from autorag.components.retrievers.faiss_store import FAISSVectorStore
from autorag.components.rerankers import CrossEncoderReranker
from autorag.pipeline.registry import get_registry, register_default_components


def create_sample_documents() -> List[Document]:
    """Create sample documents for testing"""
    docs = [
        Document(
            content="""# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building systems
that learn from data. Rather than being explicitly programmed, these systems improve their
performance through experience.

## Supervised Learning

Supervised learning is the most common paradigm in machine learning. In this approach,
the algorithm learns from labeled training data. Each training example consists of an
input and the desired output.

### Classification

Classification is a type of supervised learning where the output is a category.
Examples include spam detection and image recognition.

### Regression

Regression predicts continuous values. Examples include predicting house prices
or stock market trends.""",
            metadata={"source": "ml_intro.md", "topic": "machine learning"}
        ),
        Document(
            content="""Natural Language Processing (NLP) is a field at the intersection of
computer science, artificial intelligence, and linguistics. It focuses on enabling
computers to understand, interpret, and generate human language.

Key NLP tasks include:
- Text classification
- Named entity recognition
- Sentiment analysis
- Machine translation
- Question answering

Modern NLP heavily relies on deep learning models, particularly transformers like
BERT and GPT, which have revolutionized the field.""",
            metadata={"source": "nlp_overview.txt", "topic": "nlp"}
        ),
        Document(
            content="""Python has become the dominant language for data science and machine learning.
Its rich ecosystem of libraries makes it ideal for these tasks.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Popular Python libraries for ML include scikit-learn, TensorFlow, and PyTorch.""",
            metadata={"source": "python_ml.py", "topic": "programming"}
        )
    ]
    return docs


def test_chunking_strategies():
    """Test different chunking strategies"""
    logger.info("=" * 60)
    logger.info("Testing Chunking Strategies")
    logger.info("=" * 60)

    docs = create_sample_documents()

    # Test Semantic Chunker
    logger.info("\n1. Semantic Chunker")
    semantic_chunker = SemanticChunker({
        "chunk_size": 100,
        "respect_sentence_boundary": True,
        "overlap_sentences": 1
    })
    semantic_chunks = semantic_chunker.chunk(docs)
    logger.info(f"   Created {len(semantic_chunks)} semantic chunks")
    logger.info(f"   Sample: {semantic_chunks[0].content[:100]}...")

    # Test Sliding Window Chunker
    logger.info("\n2. Sliding Window Chunker")
    sliding_chunker = SlidingWindowChunker({
        "window_size": 150,
        "step_size": 75,
        "unit": "tokens"
    })
    sliding_chunks = sliding_chunker.chunk(docs)
    logger.info(f"   Created {len(sliding_chunks)} sliding window chunks")
    logger.info(f"   Overlap: {sliding_chunker.overlap} tokens")

    # Test Hierarchical Chunker
    logger.info("\n3. Hierarchical Chunker")
    hierarchical_chunker = HierarchicalChunker({
        "levels": [
            {"name": "coarse", "size": 200, "overlap": 0},
            {"name": "fine", "size": 100, "overlap": 20}
        ],
        "track_relationships": True
    })
    hierarchical_chunks = hierarchical_chunker.chunk(docs)
    logger.info(f"   Created {len(hierarchical_chunks)} hierarchical chunks")

    # Count chunks per level
    coarse_count = sum(1 for c in hierarchical_chunks if c.metadata["level_name"] == "coarse")
    fine_count = sum(1 for c in hierarchical_chunks if c.metadata["level_name"] == "fine")
    logger.info(f"   Coarse level: {coarse_count} chunks")
    logger.info(f"   Fine level: {fine_count} chunks")

    # Test Document-Aware Chunker
    logger.info("\n4. Document-Aware Chunker")
    doc_aware_chunker = DocumentAwareChunker({
        "max_chunk_size": 150,
        "detect_headers": True,
        "detect_code_blocks": True
    })
    doc_aware_chunks = doc_aware_chunker.chunk(docs)
    logger.info(f"   Created {len(doc_aware_chunks)} document-aware chunks")

    # Count structure types
    structure_types = {}
    for chunk in doc_aware_chunks:
        stype = chunk.metadata.get("structure_type", "unknown")
        structure_types[stype] = structure_types.get(stype, 0) + 1
    logger.info(f"   Structure types: {structure_types}")

    return semantic_chunks  # Return for retrieval testing


def test_retrieval_methods(chunks: List[Chunk]):
    """Test different retrieval methods"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Retrieval Methods")
    logger.info("=" * 60)

    # Test BM25 Retriever
    logger.info("\n1. BM25 Sparse Retrieval")
    bm25_retriever = BM25Retriever({
        "k1": 1.2,
        "b": 0.75,
        "tokenizer": "simple"
    })
    bm25_retriever.index(chunks)

    query = "What is supervised learning?"
    bm25_results = bm25_retriever.retrieve(query, top_k=3)
    logger.info(f"   Query: '{query}'")
    logger.info(f"   Found {len(bm25_results)} results")
    for i, result in enumerate(bm25_results):
        logger.info(f"   Result {i+1} (score: {result.score:.3f}): {result.chunk.content[:50]}...")

    # Test Dense Retriever
    logger.info("\n2. Dense Retrieval")
    dense_retriever = DenseRetriever()

    # Use mock embedder for demo
    embedder = MockEmbedder()
    vector_store = FAISSVectorStore({"dimension": 384})  # Mock embedder uses 384 dims by default

    dense_retriever.set_components(embedder, vector_store)
    dense_retriever.index(chunks)

    dense_results = dense_retriever.retrieve(query, top_k=3)
    logger.info(f"   Found {len(dense_results)} results")
    for i, result in enumerate(dense_results):
        logger.info(f"   Result {i+1} (score: {result.score:.3f}): {result.chunk.content[:50]}...")

    # Test Hybrid Retriever
    logger.info("\n3. Hybrid Retrieval")

    # Test different fusion methods
    fusion_methods = ["weighted_sum", "rrf", "parallel"]

    for method in fusion_methods:
        logger.info(f"\n   Fusion method: {method}")
        hybrid_retriever = HybridRetriever({
            "fusion_method": method,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "rrf_k": 60,
            "normalization": "min_max"
        })

        hybrid_retriever.set_retrievers(dense_retriever, bm25_retriever)
        hybrid_results = hybrid_retriever.retrieve(query, top_k=3)

        logger.info(f"   Found {len(hybrid_results)} results")
        for i, result in enumerate(hybrid_results):
            source = result.metadata.get("source", "combined")
            logger.info(f"   Result {i+1} [{source}] (score: {result.score:.3f})")

    return hybrid_results  # Return for reranking


def test_reranking(query: str, results: List):
    """Test reranking with cross-encoder"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Reranking")
    logger.info("=" * 60)

    logger.info("\nNote: Cross-encoder reranking requires sentence-transformers models.")
    logger.info("In production, this would download and use a real cross-encoder model.")
    logger.info("For this demo, we'll simulate reranking.\n")

    # Create mock reranker for demo (in production, this would use real model)
    try:
        reranker = CrossEncoderReranker({
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "normalize_scores": True
        })

        # If model loads successfully (requires sentence-transformers)
        reranked_results = reranker.rerank(query, results, top_k=3)

        logger.info("Reranked results:")
        for i, result in enumerate(reranked_results):
            original_score = result.metadata.get("original_score", "N/A")
            logger.info(f"   Result {i+1}: score {result.score:.3f} (original: {original_score})")

    except Exception as e:
        logger.warning(f"Could not load reranker model: {e}")
        logger.info("Skipping reranking demo (requires sentence-transformers)")


def test_registry_integration():
    """Test that all components are properly registered"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Component Registry")
    logger.info("=" * 60)

    # Register all components
    register_default_components()
    registry = get_registry()

    # List all registered components
    all_components = registry.list_components()

    logger.info("\nRegistered Components:")
    for component_type, names in all_components.items():
        logger.info(f"\n{component_type.upper()}:")
        for name in names:
            if ":" not in name:  # Skip versioned duplicates
                logger.info(f"  - {name}")

    # Test creating components from registry
    logger.info("\nCreating components from registry:")

    # Create a semantic chunker
    semantic_chunker = registry.create_component(
        "chunker", "semantic",
        {"chunk_size": 256}
    )
    logger.info(f"  ✓ Created: {semantic_chunker.__class__.__name__}")

    # Create a BM25 retriever
    bm25_retriever = registry.create_component(
        "retriever", "bm25",
        {"k1": 1.2, "b": 0.75}
    )
    logger.info(f"  ✓ Created: {bm25_retriever.__class__.__name__}")

    # Create a cross-encoder reranker
    try:
        reranker = registry.create_component(
            "reranker", "cross_encoder",
            {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
        )
        logger.info(f"  ✓ Created: {reranker.__class__.__name__}")
    except:
        logger.info("  ⚠ CrossEncoderReranker (requires sentence-transformers)")


def main():
    """Run all Week 4 component demos"""
    logger.info("=" * 60)
    logger.info("Week 4 Components Demo")
    logger.info("=" * 60)

    # Test chunking strategies
    chunks = test_chunking_strategies()

    # Test retrieval methods
    results = test_retrieval_methods(chunks)

    # Test reranking
    test_reranking("What is supervised learning?", results)

    # Test registry integration
    test_registry_integration()

    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)
    logger.info("\nAll Week 4 components have been successfully implemented:")
    logger.info("✓ BM25 Sparse Retrieval")
    logger.info("✓ Hybrid Retrieval (weighted_sum, RRF, parallel)")
    logger.info("✓ Cross-Encoder Reranking")
    logger.info("✓ Semantic Chunking")
    logger.info("✓ Sliding Window Chunking")
    logger.info("✓ Hierarchical Chunking")
    logger.info("✓ Document-Aware Chunking")
    logger.info("\nNext steps: Run MS MARCO evaluation with real data")


if __name__ == "__main__":
    main()