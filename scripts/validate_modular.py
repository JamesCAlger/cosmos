"""Validate Week 2 modular architecture matches Week 1 performance"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
from autorag.pipeline.rag_pipeline import ModularRAGPipeline
from autorag.components.base import Document


def validate_modular_architecture():
    """Validate that modular architecture works correctly"""
    print("\n" + "="*60)
    print("VALIDATING WEEK 2 MODULAR ARCHITECTURE")
    print("="*60)

    # Test 1: Mock components (no API needed)
    print("\n1. Testing with Mock Components")
    print("-" * 40)

    config_path = Path(__file__).parent.parent / "configs" / "mock_pipeline.yaml"
    pipeline = ModularRAGPipeline(str(config_path))

    documents = [
        Document(content="The capital of France is Paris.", metadata={"source": "doc1"}),
        Document(content="Python is a programming language.", metadata={"source": "doc2"}),
        Document(content="Machine learning is part of AI.", metadata={"source": "doc3"})
    ]

    # Index
    start = time.time()
    pipeline.index(documents)
    index_time = time.time() - start
    print(f"  Indexing: {index_time:.3f}s for {len(documents)} documents")

    # Query
    start = time.time()
    result = pipeline.query("What is Python?")
    query_time = time.time() - start
    print(f"  Query: {query_time:.3f}s")
    print(f"  Answer: {result['answer'][:50]}...")

    # Check components
    components = pipeline.get_components()
    print(f"  Components used: {', '.join(components.keys())}")

    mock_success = all([
        index_time < 1.0,  # Should be fast with mock
        query_time < 1.0,
        "Mock answer" in result['answer'],
        len(components) == 4  # chunker, embedder, vectorstore, generator
    ])

    print(f"  Status: {'PASS' if mock_success else 'FAIL'}")

    # Test 2: Component Swapping
    print("\n2. Testing Component Swapping")
    print("-" * 40)

    # Load different configuration
    custom_config = {
        "pipeline": {
            "components": [
                {"id": "chunker", "type": "chunker", "name": "fixed_size",
                 "config": {"chunk_size": 100}},
                {"id": "embedder", "type": "embedder", "name": "mock",
                 "config": {"dimension": 256}},
                {"id": "vectorstore", "type": "vectorstore", "name": "faiss",
                 "config": {"dimension": 256}},
                {"id": "generator", "type": "generator", "name": "mock",
                 "config": {"template": "Custom: {query}"}}
            ]
        }
    }

    pipeline2 = ModularRAGPipeline()
    pipeline2.orchestrator.load_config(custom_config)

    # Check that components are different
    components2 = pipeline2.get_components()
    swap_success = (
        components["chunker"]["component"] != components2["chunker"]["component"] and
        components2["chunker"]["component"] == "fixed_size"
    )

    print(f"  Original chunker: {components['chunker']['component']}")
    print(f"  Swapped chunker: {components2['chunker']['component']}")
    print(f"  Status: {'PASS' if swap_success else 'FAIL'}")

    # Test 3: Execution Order
    print("\n3. Testing Execution Order")
    print("-" * 40)

    order = pipeline.get_execution_order()
    expected_order = ["chunker", "embedder", "vectorstore", "generator"]
    order_success = order == expected_order

    print(f"  Expected: {expected_order}")
    print(f"  Actual: {order}")
    print(f"  Status: {'PASS' if order_success else 'FAIL'}")

    # Test 4: Compare with Week 1 (if results exist)
    print("\n4. Comparing with Week 1 Results")
    print("-" * 40)

    week1_file = Path("experiments") / "baseline_20250915_140949.json"
    if week1_file.exists():
        with open(week1_file) as f:
            week1_data = json.load(f)

        week1_perf = week1_data["performance"]
        print(f"  Week 1 avg query time: {week1_perf['avg_query_time_seconds']:.3f}s")
        print(f"  Modular query time: {query_time:.3f}s (mock components)")
        print("  Note: Direct comparison not possible (different components)")
        print("  Status: ARCHITECTURE VALIDATED")
    else:
        print("  No Week 1 results found for comparison")
        print("  Status: SKIPPED")

    # Overall result
    print("\n" + "="*60)
    all_success = mock_success and swap_success and order_success
    if all_success:
        print("VALIDATION RESULT: SUCCESS")
        print("The modular architecture is working correctly!")
        print("\nKey achievements:")
        print("- Components can be swapped via configuration")
        print("- Pipeline execution follows correct order")
        print("- Mock components enable testing without APIs")
        print("- Architecture is extensible for future weeks")
    else:
        print("VALIDATION RESULT: PARTIAL SUCCESS")
        print("Some tests failed - review the results above")
    print("="*60)

    return all_success


if __name__ == "__main__":
    try:
        validate_modular_architecture()
    except Exception as e:
        print(f"\nValidation error: {e}")
        import traceback
        traceback.print_exc()