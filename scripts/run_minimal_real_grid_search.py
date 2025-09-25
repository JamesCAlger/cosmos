"""Minimal real 3x3 grid search with actual OpenAI API calls - cost-optimized version"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import json
import time
from typing import Dict, Any, List
from loguru import logger
import numpy as np
from itertools import product
import openai
from datasets import load_dataset
import traceback
import datetime

# Setup logging
logger.add("minimal_real_grid_search.log", rotation="10 MB")

# Import our architecture
from autorag.components.descriptors import ComponentDescriptor, ParamSpec, ParamType, SelfDescribingComponent
from autorag.components.auto_register import auto_register
from autorag.components.base import Component, Chunker, Embedder, Retriever, Generator
from autorag.optimization.configuration_bridge import ConfigurationBridge
from autorag.pipeline.registry import get_registry

# Clear registry
registry = get_registry()
registry.clear()

# ==============================================================================
# Real Components with Minimal API Usage
# ==============================================================================

@auto_register("chunker", "fixed_size")
class RealFixedSizeChunker(Chunker, SelfDescribingComponent):
    """Real fixed-size chunker"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="FixedSizeChunker",
            type="chunker",
            parameters={
                "chunk_size": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[256, 512, 1024],
                    default=512,
                    tunable=True,
                    description="Size of chunks in characters"
                )
            },
            inputs=["documents"],
            outputs=["chunks"],
            tunable_params=["chunk_size"],
            estimated_cost=0.0,
            estimated_latency=0.01
        )

    def chunk(self, documents: List[str]) -> List[str]:
        chunk_size = self.config.get("chunk_size", 512)
        chunks = []
        for doc in documents:
            # Simple character-based chunking
            for i in range(0, len(doc), chunk_size):
                chunk = doc[i:i+chunk_size]
                if len(chunk) > 50:  # Skip very small chunks
                    chunks.append(chunk)
        return chunks


@auto_register("embedder", "mock")
class MockEmbedder(Embedder, SelfDescribingComponent):
    """Mock embedder to avoid costs during testing"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="MockEmbedder",
            type="embedder",
            parameters={},
            inputs=["chunks"],
            outputs=["embeddings"],
            tunable_params=[],
            estimated_cost=0.0,
            estimated_latency=0.01
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Return mock embeddings (768-dim vectors)
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, query: str) -> List[float]:
        # Return mock embedding for a single query (768-dim vector)
        return [0.1] * 768


@auto_register("retriever", "simple")
class SimpleRetriever(Retriever, SelfDescribingComponent):
    """Simple retriever using cosine similarity"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="SimpleRetriever",
            type="retriever",
            parameters={
                "top_k": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[3, 5, 10],
                    default=5,
                    tunable=True,
                    description="Number of chunks to retrieve"
                )
            },
            inputs=["query", "chunks"],
            outputs=["retrieved_documents"],
            tunable_params=["top_k"],
            estimated_cost=0.0,
            estimated_latency=0.01
        )

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.openai_client = None  # Store client for reuse
        self.chunks = []

    def index(self, chunks: List[str]):
        """Store chunks for retrieval"""
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        if top_k is None:
            top_k = self.config.get("top_k", 5)

        # Simple keyword matching for demo
        query_words = set(query.lower().split())
        scores = []

        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scores.append(overlap)

        # Get top-k chunks
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices if i < len(self.chunks)]


@auto_register("generator", "openai_mini")
class OpenAIMiniGenerator(Generator, SelfDescribingComponent):
    """OpenAI generator using GPT-3.5-turbo (cheapest option)"""

    # Class-level client to share across all instances
    _shared_client = None
    _api_call_count = 0  # Track total API calls
    _success_count = 0   # Track successful calls
    _error_count = 0     # Track failed calls
    _first_error_config = None  # Track when errors start

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="OpenAIMiniGenerator",
            type="generator",
            parameters={
                "temperature": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[0.0, 0.3, 0.7],
                    default=0.3,
                    tunable=True,
                    description="Generation temperature"
                )
            },
            inputs=["query", "retrieved_documents"],
            outputs=["answer"],
            tunable_params=["temperature"],
            estimated_cost=0.0005,  # ~$0.0005 per call
            estimated_latency=1.0
        )

    def generate(self, query: str, contexts: List[str]) -> str:
        temperature = self.config.get("temperature", 0.3)

        # Increment call counter
        OpenAIMiniGenerator._api_call_count += 1
        call_num = OpenAIMiniGenerator._api_call_count

        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        logger.info(f"[Generator] Call #{call_num} - API key present: {bool(api_key)}, Query: {query[:50]}...")
        if not api_key:
            # Return mock response if no API key with realistic answers
            logger.warning("[Generator] No API key found, using mock responses")
            mock_answers = {
                "python": "Python is a programming language known for its simplicity.",
                "eiffel": "The Eiffel Tower is located in Paris, France.",
                "light": "The speed of light is approximately 300,000 km/s."
            }
            for key, answer in mock_answers.items():
                if key in query.lower():
                    logger.info(f"[Generator] Mock response for '{key}': {answer}")
                    return answer
            return f"Mock answer for: {query} (temp={temperature})"

        try:
            # Prepare context
            context_text = "\n\n".join(contexts[:3])  # Limit context to save tokens

            # Create prompt
            prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_text[:1500]}  # Limit to 1500 chars

Question: {query}

Answer:"""

            # Use shared client across all instances
            if not OpenAIMiniGenerator._shared_client:
                OpenAIMiniGenerator._shared_client = openai.OpenAI(
                    api_key=api_key,
                    max_retries=2,  # Reduce retries to fail faster
                    timeout=20.0    # Overall timeout
                )
                logger.info(f"[Generator] Created SHARED OpenAI client at call #{call_num}")
            else:
                logger.debug(f"[Generator] Call #{call_num} - Reusing existing shared OpenAI client")

            # Call OpenAI API with timeout
            response = OpenAIMiniGenerator._shared_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=100,  # Limit tokens to save costs
                timeout=30.0  # 30 second timeout
            )

            answer = response.choices[0].message.content
            OpenAIMiniGenerator._success_count += 1
            logger.info(f"[Generator] Call #{call_num} SUCCESS ({OpenAIMiniGenerator._success_count} total) - Response: {answer[:50]}...")
            return answer

        except Exception as e:
            OpenAIMiniGenerator._error_count += 1
            if OpenAIMiniGenerator._first_error_config is None:
                OpenAIMiniGenerator._first_error_config = call_num

            error_str = str(e).lower()
            logger.error(f"[Generator] Call #{call_num} FAILED (Error #{OpenAIMiniGenerator._error_count})")
            logger.error(f"[Generator] Error type: {type(e).__name__}")
            logger.error(f"[Generator] Error message: {e}")
            logger.error(f"[Generator] First error occurred at call #{OpenAIMiniGenerator._first_error_config}")
            logger.error(f"[Generator] Success/Error ratio: {OpenAIMiniGenerator._success_count}/{OpenAIMiniGenerator._error_count}")

            # Log full traceback for first few errors
            if OpenAIMiniGenerator._error_count <= 3:
                logger.error(f"[Generator] Full traceback:\n{traceback.format_exc()}")

            # Handle different error types
            if "rate" in error_str:
                # Rate limit error - wait longer
                wait_time = 5
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                # Try once more after waiting
                try:
                    if not OpenAIMiniGenerator._shared_client:
                        OpenAIMiniGenerator._shared_client = openai.OpenAI(
                            api_key=api_key,
                            max_retries=1,
                            timeout=20.0
                        )
                    response = OpenAIMiniGenerator._shared_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=100,
                        timeout=30.0
                    )
                    answer = response.choices[0].message.content
                    logger.info(f"[Generator] Retry successful: {answer[:100]}...")
                    return answer
                except Exception as retry_error:
                    logger.error(f"[Generator] Retry failed: {retry_error}")
                    return f"Error generating answer after retry: {str(retry_error)}"
            elif "connection" in error_str or "timeout" in error_str:
                # Connection error - shorter wait
                logger.info("Connection issue detected, waiting 2 seconds...")
                time.sleep(2)
            return f"Error generating answer: {str(e)}"


# ==============================================================================
# Minimal Grid Search Runner
# ==============================================================================

class MinimalRealGridSearch:
    """Minimal real grid search with cost optimization"""

    def __init__(self, num_docs: int = 10, num_queries: int = 3, api_delay: float = 0.15,
                 evaluation_method: str = "semantic_fixed", semantic_threshold: float = 0.75):
        """
        Initialize minimal grid search
        api_delay: Delay between API calls to avoid rate limiting (seconds)
            - Free tier: 500 RPM = ~8.3 RPS = 0.12s minimum between calls
            - Using 0.15s for safety margin

        Args:
            num_docs: Number of documents (keep small to minimize costs)
            num_queries: Number of test queries (keep small to minimize costs)
            evaluation_method: Evaluation method to use ("keyword" or "semantic_fixed")
            semantic_threshold: Threshold for semantic similarity matching (default 0.75)
        """
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.api_delay = api_delay
        self.evaluation_method = evaluation_method
        self.semantic_threshold = semantic_threshold
        self.bridge = ConfigurationBridge()
        self.openai_client = None  # Reuse client across calls

        # Initialize semantic evaluator if needed
        self.semantic_evaluator = None
        if evaluation_method == "semantic_fixed":
            from autorag.evaluation.semantic_metrics import SemanticMetrics
            self.semantic_evaluator = SemanticMetrics(
                model_name='all-MiniLM-L6-v2',
                similarity_threshold=semantic_threshold
            )
            logger.info(f"Using semantic evaluation with threshold {semantic_threshold}")

    def load_minimal_data(self):
        """Load minimal dataset for testing"""
        logger.info(f"Loading minimal dataset: {self.num_docs} docs, {self.num_queries} queries")

        # Sample documents
        documents = [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning is a subset of artificial intelligence.",
            "The Eiffel Tower is located in Paris, France.",
            "Climate change affects global weather patterns.",
            "DNA contains genetic information.",
            "Shakespeare wrote many famous plays.",
            "The speed of light is approximately 300,000 km/s.",
            "The Pacific Ocean is the largest ocean.",
            "Photosynthesis converts light into energy.",
            "The human brain has billions of neurons."
        ][:self.num_docs]

        # Sample queries with ground truth answers for semantic evaluation
        queries = [
            {
                "question": "What is Python?",
                "expected": "programming language",  # For keyword matching
                "ground_truth_answer": "Python is a high-level programming language known for its simplicity and readability."
            },
            {
                "question": "Where is the Eiffel Tower?",
                "expected": "Paris",  # For keyword matching
                "ground_truth_answer": "The Eiffel Tower is located in Paris, France."
            },
            {
                "question": "What is the speed of light?",
                "expected": "300,000",  # For keyword matching
                "ground_truth_answer": "The speed of light is approximately 300,000 kilometers per second."
            }
        ][:self.num_queries]

        return documents, queries

    def evaluate_configuration(self, params: Dict[str, Any], documents: List[str], queries: List[Dict], config_num: int = 0) -> Dict[str, float]:
        """Evaluate a single configuration"""
        logger.info(f"\n" + "="*60)
        logger.info(f"Evaluating Config #{config_num}: {params}")
        logger.info(f"[Evaluation] Method: {self.evaluation_method}")
        logger.info(f"[Evaluation] API key status: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
        logger.info(f"[Evaluation] Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"="*60)

        # Create components
        chunker = RealFixedSizeChunker({"chunk_size": params.get("chunker.fixed_size.chunk_size", 512)})
        embedder = MockEmbedder({})
        retriever = SimpleRetriever({"top_k": params.get("retriever.simple.top_k", 5)})
        generator = OpenAIMiniGenerator({"temperature": params.get("generator.openai_mini.temperature", 0.3)})

        # Process documents
        chunks = chunker.chunk(documents)
        retriever.index(chunks)

        # Evaluate on queries
        scores = []
        answers = []
        ground_truths = []
        total_time = 0

        for query_data in queries:
            query = query_data["question"]
            expected = query_data.get("expected", "")
            # For semantic evaluation, use full ground truth if available
            ground_truth = query_data.get("ground_truth_answer", expected)

            start = time.time()

            # RAG pipeline
            retrieved = retriever.retrieve(query)

            # Always add delay before API call to respect rate limits
            # Free tier: 500 RPM = need at least 0.12s between calls
            time.sleep(self.api_delay)

            answer = generator.generate(query, retrieved)
            answers.append(answer)
            ground_truths.append(ground_truth)

            total_time += time.time() - start

            # Evaluation based on method
            if self.evaluation_method == "semantic_fixed" and self.semantic_evaluator:
                # Compute semantic similarity
                similarity = self.semantic_evaluator.semantic_similarity_batch([answer], [ground_truth])[0]
                # Binary scoring based on threshold
                score = 1.0 if similarity >= self.semantic_threshold else 0.0
                logger.info(f"Q: {query[:50]}...")
                logger.info(f"  Ground truth: '{ground_truth[:100]}...'")
                logger.info(f"  Answer: {answer[:100]}...")
                logger.info(f"  Similarity: {similarity:.3f} (threshold: {self.semantic_threshold})")
                logger.info(f"  Score: {score:.1f}")
            else:
                # Fallback to keyword matching
                score = 1.0 if expected.lower() in answer.lower() else 0.5
                logger.info(f"Q: {query[:50]}...")
                logger.info(f"  Expected keyword: '{expected}'")
                logger.info(f"  Answer: {answer[:100]}...")
                logger.info(f"  Score: {score:.2f}")

            scores.append(score)

        metrics = {
            "accuracy": np.mean(scores),
            "avg_latency": total_time / len(queries),
            "total_cost": 0.0005 * len(queries) if os.getenv('OPENAI_API_KEY') else 0.0,  # Cost only if using API
            "evaluation_method": self.evaluation_method
        }

        # Add semantic-specific metrics if using semantic evaluation
        if self.evaluation_method == "semantic_fixed" and self.semantic_evaluator:
            # Compute all similarities for statistics
            similarities = self.semantic_evaluator.semantic_similarity_batch(answers, ground_truths)
            metrics["similarity_mean"] = np.mean(similarities)
            metrics["similarity_std"] = np.std(similarities)
            metrics["similarity_min"] = np.min(similarities)
            metrics["similarity_max"] = np.max(similarities)
            metrics["semantic_threshold"] = self.semantic_threshold

        logger.info(f"[Evaluation] Final metrics: {metrics}")
        return metrics

    def run(self):
        """Run the minimal grid search"""
        print("\n" + "=" * 80)
        print("MINIMAL REAL 3x3 GRID SEARCH")
        print("=" * 80)
        print(f"Documents: {self.num_docs}")
        print(f"Queries: {self.num_queries}")
        print(f"Estimated total cost: ~${0.0005 * self.num_queries * 27:.2f}")
        print("=" * 80)

        # Load data
        documents, queries = self.load_minimal_data()

        # Define search space (3x3x3 = 27 combinations)
        search_space = {
            "chunker.fixed_size.chunk_size": [256, 512, 1024],
            "retriever.simple.top_k": [3, 5, 10],
            "generator.openai_mini.temperature": [0.0, 0.3, 0.7]
        }

        # Generate combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        all_combinations = list(product(*param_values))

        results = []
        best_score = -1
        best_config = None

        print("\nConfig | Chunk | TopK | Temp | Accuracy | Latency | Cost")
        print("-" * 60)

        for i, combination in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combination))

            # Evaluate with config number
            metrics = self.evaluate_configuration(params, documents, queries, config_num=i)

            # Track results
            result = {
                "config_id": i,
                "params": params,
                "metrics": metrics,
                "score": metrics["accuracy"]
            }
            results.append(result)

            if metrics["accuracy"] > best_score:
                best_score = metrics["accuracy"]
                best_config = result

            # Display
            print(f"{i:3d}/27 | {params['chunker.fixed_size.chunk_size']:5d} | "
                  f"{params['retriever.simple.top_k']:4d} | "
                  f"{params['generator.openai_mini.temperature']:.1f} | "
                  f"{metrics['accuracy']:.3f}    | "
                  f"{metrics['avg_latency']:.2f}s    | "
                  f"${metrics['total_cost']:.4f}")

        print("=" * 60)

        # Display top 5 configurations with all metrics
        print("\n" + "=" * 80)
        print("TOP 5 CONFIGURATIONS (Ranked by Score)")
        print("=" * 80)

        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]

        for i, config in enumerate(sorted_results, 1):
            print(f"\n{i}. Configuration #{config['config_id']}")
            print("-" * 60)
            print(f"   Overall Score:        {config['score']:.3f}")

            # Display only computed metrics (not missing ones)
            metrics = config.get('metrics', {})
            print(f"   Accuracy:            {metrics.get('accuracy', 0):.3f}")
            # Only show these metrics if they were actually computed
            if 'semantic_similarity' in metrics:
                print(f"   Semantic Similarity: {metrics.get('semantic_similarity'):.3f}")
            if 'retrieval_precision' in metrics:
                print(f"   Retrieval Precision: {metrics.get('retrieval_precision'):.3f}")
            if 'answer_completeness' in metrics:
                print(f"   Answer Completeness: {metrics.get('answer_completeness'):.3f}")
            print(f"   Average Latency:     {metrics.get('avg_latency', 0):.3f}s")
            if 'total_latency' in metrics:
                print(f"   Total Latency:       {metrics.get('total_latency'):.3f}s")
            print(f"   Total Cost:          ${metrics.get('total_cost', 0):.4f}")
            print(f"   Parameters:")
            for param, value in config['params'].items():
                param_name = param.split('.')[-1]
                print(f"      - {param_name}: {value}")

        print("\n" + "=" * 80)
        print("BEST CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"Best Configuration: #{best_config['config_id']}")
        print(f"Best Score: {best_config['score']:.3f}")
        print(f"Parameters: {best_config['params']}")

        # Print debug summary
        print("\n" + "=" * 80)
        print("DEBUG SUMMARY")
        print("=" * 80)
        print(f"Total API calls attempted: {OpenAIMiniGenerator._api_call_count}")
        print(f"Successful calls: {OpenAIMiniGenerator._success_count}")
        print(f"Failed calls: {OpenAIMiniGenerator._error_count}")
        if OpenAIMiniGenerator._first_error_config:
            print(f"First error at call: #{OpenAIMiniGenerator._first_error_config}")
        print(f"Success rate: {OpenAIMiniGenerator._success_count / max(1, OpenAIMiniGenerator._api_call_count) * 100:.1f}%")
        print("=" * 80)

        # Save results
        with open("minimal_real_results.json", "w") as f:
            json.dump({
                "results": results,
                "best": best_config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        print("\nResults saved to minimal_real_results.json")
        return results, best_config


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MINIMAL REAL GRID SEARCH - COST-OPTIMIZED VERSION")
    print("=" * 80)

    # Load .env file if it exists
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"\n[STATUS] Loaded .env file from {env_path}")

    # Check API key status
    api_key_status = "SET" if os.getenv("OPENAI_API_KEY") else "NOT SET"
    print(f"[STATUS] OPENAI_API_KEY: {api_key_status}")
    if os.getenv("OPENAI_API_KEY"):
        key_preview = os.getenv("OPENAI_API_KEY")[:10] + "..."
        print(f"[STATUS] Key preview: {key_preview}")

    print("\nThis version:")
    print("- Uses only 10 documents and 3 queries")
    print("- Uses mock embeddings (no API calls)")
    print("- Uses simple keyword retrieval")
    print("- Only calls OpenAI for answer generation")
    print("- Estimated total cost: ~$0.04 for all 27 configs")
    print("\nTo run with OpenAI:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Run this script")
    print("\nTo run without OpenAI (mock mode):")
    print("  1. Don't set OPENAI_API_KEY")
    print("  2. Run this script (will use mock responses)")
    print("=" * 80)

    response = input("\nProceed with grid search? (y/n): ")
    if response.lower() == 'y':
        # Use appropriate delay for OpenAI free tier (500 RPM = ~8.3 RPS)
        # 0.15 seconds between calls provides safety margin
        # Default to semantic evaluation with 0.75 threshold
        grid_search = MinimalRealGridSearch(
            num_docs=10,
            num_queries=3,
            api_delay=0.15,
            evaluation_method="semantic_fixed",  # Use semantic evaluation by default
            semantic_threshold=0.75
        )
        results, best = grid_search.run()
        print("\nGrid search complete!")
    else:
        print("Grid search cancelled.")