"""Test OpenAI API connection and components"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
import openai
from loguru import logger

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if API key is valid"""
    print("\n" + "="*60)
    print("TESTING OPENAI API CONNECTION")
    print("="*60)

    # Check if API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found in environment")
        return False

    print(f"[OK] API key found: {api_key[:10]}...{api_key[-4:]}")

    # Test API connection
    try:
        client = openai.OpenAI(api_key=api_key)

        # Test with a simple completion
        print("\nTesting chat completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API working'"}],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"[OK] Chat completion works: {result}")

        # Test embeddings
        print("\nTesting embeddings...")
        embedding_response = client.embeddings.create(
            input="Test text",
            model="text-embedding-ada-002"
        )

        embedding_dim = len(embedding_response.data[0].embedding)
        print(f"[OK] Embeddings work: dimension={embedding_dim}")

        return True

    except openai.AuthenticationError as e:
        print(f"[FAIL] Authentication failed: {e}")
        print("   API key is invalid or expired")
        return False
    except openai.RateLimitError as e:
        print(f"[WARN] Rate limit reached: {e}")
        print("   API key is valid but rate limited")
        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_pipeline_components():
    """Test if pipeline components can initialize"""
    print("\n" + "="*60)
    print("TESTING PIPELINE COMPONENTS")
    print("="*60)

    # Test OpenAI Embedder
    print("\n1. Testing OpenAI Embedder...")
    try:
        from autorag.components.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder({
            "model": "text-embedding-ada-002",
            "batch_size": 1
        })

        # Test embedding
        embeddings = embedder.embed(["Test text"])
        print(f"[OK] OpenAI Embedder works: {len(embeddings)} embeddings generated")

    except Exception as e:
        print(f"[FAIL] OpenAI Embedder failed: {e}")

    # Test OpenAI Generator
    print("\n2. Testing OpenAI Generator...")
    try:
        from autorag.components.generators.openai import OpenAIGenerator

        generator = OpenAIGenerator({
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 50
        })

        # Test generation
        answer = generator.generate(
            query="What is Python?",
            contexts=["Python is a programming language"]
        )
        print(f"[OK] OpenAI Generator works: {answer[:50]}...")

    except Exception as e:
        print(f"[FAIL] OpenAI Generator failed: {e}")

    # Test Full Pipeline
    print("\n3. Testing Full Pipeline...")
    try:
        from autorag.pipeline.simple_rag import SimpleRAGPipeline
        from autorag.components.base import Document

        config_path = Path(__file__).parent.parent / "configs" / "baseline_rag.yaml"
        pipeline = SimpleRAGPipeline(str(config_path))

        # Test indexing
        docs = [Document(content="Python is a programming language", metadata={})]
        result = pipeline.index(docs)
        print(f"[OK] Pipeline indexing works: {result}")

        # Test query
        query_result = pipeline.query("What is Python?")
        print(f"[OK] Pipeline query works: {query_result.get('answer', 'No answer')[:50]}...")

        return True

    except Exception as e:
        print(f"[FAIL] Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_fallback_conditions():
    """Check why pipeline might be falling back to mocks"""
    print("\n" + "="*60)
    print("CHECKING FALLBACK CONDITIONS")
    print("="*60)

    # Check if mock components are registered
    from autorag.pipeline.registry import ComponentRegistry

    registry = ComponentRegistry()

    print("\nRegistered components:")
    for comp_type in ["embedder", "generator", "chunker", "vectorstore"]:
        components = registry.list_components(comp_type)
        print(f"\n{comp_type}:")
        for name, info in components.items():
            print(f"  - {name}: {info.get('class', 'Unknown')}")

    # Check config file
    config_path = Path(__file__).parent.parent / "configs" / "baseline_rag.yaml"
    print(f"\nConfig file exists: {config_path.exists()}")

    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print("\nPipeline components in config:")
        for component in config.get("pipeline", {}).get("components", []):
            print(f"  - {component['id']}: {component['type']}/{component['name']}")


def main():
    """Run all tests"""
    # Test API connection
    api_works = test_api_key()

    if api_works:
        # Test components
        components_work = test_pipeline_components()

        # Check fallback conditions
        check_fallback_conditions()

        print("\n" + "="*60)
        print("DIAGNOSIS SUMMARY")
        print("="*60)

        if components_work:
            print("[OK] All components working - pipeline should use real models")
            print("\nTo run real evaluation:")
            print("  python scripts/run_week3_msmarco_evaluation.py")
        else:
            print("[WARN] Components failed to initialize")
            print("Check the error messages above for details")
    else:
        print("\n" + "="*60)
        print("API KEY ISSUE DETECTED")
        print("="*60)
        print("\nThe API key in .env appears to be invalid.")
        print("Please get a new key from: https://platform.openai.com/api-keys")
        print("\nUpdate .env with:")
        print("  OPENAI_API_KEY=your_new_api_key_here")


if __name__ == "__main__":
    main()