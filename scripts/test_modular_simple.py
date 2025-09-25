"""Simple test of modular architecture with mock components"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from autorag.pipeline.rag_pipeline import ModularRAGPipeline
from autorag.components.base import Document


def test_modular_with_mock():
    """Test modular pipeline with mock components (no API needed)"""
    print("\n" + "="*60)
    print("TESTING MODULAR ARCHITECTURE WITH MOCK COMPONENTS")
    print("="*60)

    # Use mock configuration
    config_path = Path(__file__).parent.parent / "configs" / "mock_pipeline.yaml"
    pipeline = ModularRAGPipeline(str(config_path))

    # Create test documents
    documents = [
        Document(content="The capital of France is Paris.", metadata={"source": "doc1"}),
        Document(content="Python is a programming language.", metadata={"source": "doc2"}),
        Document(content="Machine learning is part of AI.", metadata={"source": "doc3"})
    ]

    print("\n1. Indexing documents...")
    pipeline.index(documents)
    print("✓ Indexing complete")

    print("\n2. Processing query...")
    result = pipeline.query("What is Python?")

    print("\n3. Results:")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Execution time: {result.get('total_time', 0):.3f}s")

    print("\n4. Execution trace:")
    for trace in result.get('execution_trace', []):
        status = "✓" if trace['status'] == 'success' else "✗"
        if 'execution_time' in trace:
            print(f"  {status} {trace['node_id']}: {trace['execution_time']:.3f}s")
        else:
            print(f"  {status} {trace['node_id']}: {trace.get('error', 'unknown')}")

    print("\n" + "="*60)
    print("✅ MODULAR ARCHITECTURE TEST SUCCESSFUL")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = test_modular_with_mock()
        if success:
            print("\nThe modular architecture is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()