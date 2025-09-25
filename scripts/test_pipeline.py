"""Quick test script to verify RAG pipeline works"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from autorag.core.rag_pipeline import RAGPipeline
from autorag.core.document_processor import Document
from loguru import logger


def test_pipeline():
    """Test the RAG pipeline with minimal data"""

    logger.info("Testing RAG pipeline with minimal data")

    # Create test documents
    documents = [
        Document("Python is a programming language known for simplicity."),
        Document("Machine learning uses algorithms to learn from data."),
        Document("The Eiffel Tower is located in Paris, France."),
    ]

    # Initialize pipeline
    logger.info("Initializing pipeline")
    pipeline = RAGPipeline()

    # Index documents
    logger.info("Indexing documents")
    pipeline.index(documents)

    # Test query
    test_query = "What is Python?"
    logger.info(f"Testing query: {test_query}")

    result = pipeline.query(test_query, top_k=2)

    # Display results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"\nQuery: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nRetrieved Contexts:")
    for i, ctx in enumerate(result['contexts']):
        print(f"  {i+1}. Score: {ctx['score']:.3f}")
        print(f"     Content: {ctx['content'][:100]}...")
    print("="*60)

    logger.info("Pipeline test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        test_pipeline()
        print("\n[SUCCESS] Pipeline test passed! Ready to run full evaluation.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline test failed: {e}")
        raise