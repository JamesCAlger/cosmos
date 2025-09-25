"""Demonstrate Week 2 modular pipeline architecture"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from autorag.pipeline.rag_pipeline import ModularRAGPipeline
from autorag.components.base import Document


def demo_component_swapping():
    """Demonstrate swapping components via configuration"""
    print("\n" + "="*60)
    print("WEEK 2: MODULAR PIPELINE DEMONSTRATION")
    print("="*60)

    # Test documents
    documents = [
        Document(content="The capital of France is Paris. It is known for the Eiffel Tower.",
                metadata={"source": "doc1"}),
        Document(content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "doc2"}),
        Document(content="Python is a popular programming language for data science.",
                metadata={"source": "doc3"})
    ]

    print("\n1. Testing with PRODUCTION Components (OpenAI)")
    print("-" * 40)

    try:
        # Load production configuration
        config_path = Path(__file__).parent.parent / "configs" / "baseline_linear.yaml"
        pipeline = ModularRAGPipeline(str(config_path))

        # Show components
        components = pipeline.get_components()
        print("Components loaded:")
        for comp_id, info in components.items():
            print(f"  - {comp_id}: {info['component']} ({info['type']})")

        # Index and query
        pipeline.index(documents)
        result = pipeline.query("What is the capital of France?")
        print(f"\nQuery: What is the capital of France?")
        print(f"Answer: {result['answer'][:100]}...")

    except Exception as e:
        logger.warning(f"OpenAI components not available: {e}")
        print("Skipping OpenAI components (API key required)")

    print("\n2. Testing with MOCK Components")
    print("-" * 40)

    # Load mock configuration
    config_path = Path(__file__).parent.parent / "configs" / "mock_pipeline.yaml"
    pipeline = ModularRAGPipeline(str(config_path))

    # Show components
    components = pipeline.get_components()
    print("Components loaded:")
    for comp_id, info in components.items():
        print(f"  - {comp_id}: {info['component']} ({info['type']})")

    # Index and query
    pipeline.index(documents)
    result = pipeline.query("What is Python used for?")
    print(f"\nQuery: What is Python used for?")
    print(f"Answer: {result['answer']}")
    print(f"Execution time: {result.get('total_time', 0):.3f}s")

    print("\n3. Demonstrating Configuration Flexibility")
    print("-" * 40)

    # Create custom configuration programmatically
    custom_config = {
        "pipeline": {
            "components": [
                {
                    "id": "chunker",
                    "type": "chunker",
                    "name": "fixed_size",
                    "config": {"chunk_size": 50, "overlap": 10}
                },
                {
                    "id": "embedder",
                    "type": "embedder",
                    "name": "mock",
                    "config": {"dimension": 256}
                },
                {
                    "id": "vectorstore",
                    "type": "vectorstore",
                    "name": "faiss",
                    "config": {"dimension": 256}
                },
                {
                    "id": "generator",
                    "type": "generator",
                    "name": "mock",
                    "config": {"template": "Custom response for: {query}"}
                }
            ]
        }
    }

    pipeline = ModularRAGPipeline()
    pipeline.orchestrator.load_config(custom_config)

    print("Custom configuration loaded:")
    components = pipeline.get_components()
    for comp_id, info in components.items():
        print(f"  - {comp_id}: {info['component']}")
        if info['config']:
            for key, value in info['config'].items():
                print(f"      {key}: {value}")

    # Test the custom pipeline
    pipeline.index(documents)
    result = pipeline.query("Test query")
    print(f"\nCustom pipeline result: {result['answer']}")

    print("\n4. Execution Order Visualization")
    print("-" * 40)

    execution_order = pipeline.get_execution_order()
    print("Pipeline execution order:")
    for i, component in enumerate(execution_order, 1):
        print(f"  {i}. {component}")

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("✓ Component swapping via configuration")
    print("✓ YAML configuration loading")
    print("✓ Mock components for testing")
    print("✓ Programmatic configuration")
    print("✓ Pipeline execution tracing")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    demo_component_swapping()