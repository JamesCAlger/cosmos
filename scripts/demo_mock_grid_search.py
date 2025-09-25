"""Mock 3x3 grid search demonstration using self-describing architecture"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, Any, List
from loguru import logger
import time
import random

# Setup logging
logger.add("mock_grid_search.log", rotation="10 MB")

# Import our new architecture components
from autorag.components.descriptors import ComponentDescriptor, ParamSpec, ParamType, SelfDescribingComponent
from autorag.components.auto_register import auto_register
from autorag.components.base import Component, Chunker, Retriever, Generator
from autorag.pipeline.auto_pipeline_builder import AutoPipelineBuilder
from autorag.optimization.configuration_bridge import ConfigurationBridge
from autorag.pipeline.registry import get_registry


# Clear registry for clean demo
registry = get_registry()
registry.clear()


# ==============================================================================
# STEP 1: Define Mock Components with Self-Description
# ==============================================================================

@auto_register("chunker", "mock")
class MockChunker(Chunker, SelfDescribingComponent):
    """Mock chunker for demonstration"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="MockChunker",
            type="chunker",
            parameters={
                "strategy": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=["fixed", "semantic", "sliding"],
                    default="fixed",
                    tunable=True,
                    description="Chunking strategy"
                )
            },
            inputs=["documents"],
            outputs=["chunks"],
            tunable_params=["strategy"],
            estimated_cost=0.0,
            estimated_latency=0.01
        )

    def chunk(self, documents: List[str]) -> List[str]:
        strategy = self.config.get("strategy", "fixed")
        # Mock chunking
        chunks = []
        for doc in documents:
            if strategy == "fixed":
                chunks.extend([f"chunk_{i}" for i in range(3)])
            elif strategy == "semantic":
                chunks.extend([f"semantic_chunk_{i}" for i in range(4)])
            else:  # sliding
                chunks.extend([f"sliding_chunk_{i}" for i in range(5)])
        return chunks


@auto_register("retriever", "mock")
class MockRetriever(Retriever, SelfDescribingComponent):
    """Mock retriever for demonstration"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="MockRetriever",
            type="retriever",
            parameters={
                "method": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=["dense", "sparse", "hybrid"],
                    default="dense",
                    tunable=True,
                    description="Retrieval method"
                )
            },
            inputs=["query", "chunks"],
            outputs=["retrieved_documents"],
            tunable_params=["method"],
            estimated_cost=0.001,
            estimated_latency=0.05
        )

    def retrieve(self, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        method = self.config.get("method", "dense")
        # Mock retrieval with method-dependent "quality"
        if method == "dense":
            return [f"dense_doc_{i}" for i in range(top_k)]
        elif method == "sparse":
            return [f"sparse_doc_{i}" for i in range(top_k)]
        else:  # hybrid
            return [f"hybrid_doc_{i}" for i in range(top_k)]

    def process(self, *args, **kwargs):
        return self.retrieve(*args, **kwargs)


@auto_register("generator", "mock")
class MockGenerator(Generator, SelfDescribingComponent):
    """Mock generator for demonstration"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="MockGenerator",
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
            estimated_cost=0.01,
            estimated_latency=0.1
        )

    def generate(self, query: str, context: List[str]) -> str:
        temperature = self.config.get("temperature", 0.3)
        # Mock generation with temperature-dependent "quality"
        if temperature == 0.0:
            return f"Precise answer to: {query}"
        elif temperature == 0.3:
            return f"Balanced answer to: {query}"
        else:  # 0.7
            return f"Creative answer to: {query}"

    def process(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


# ==============================================================================
# STEP 2: Mock Evaluator
# ==============================================================================

class MockEvaluator:
    """Mock evaluator that simulates evaluation metrics"""

    def evaluate(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Simulate evaluation with configuration-dependent scores"""
        # Extract configuration values
        chunking = config.get("chunker.mock.strategy", "fixed")
        retrieval = config.get("retriever.mock.method", "dense")
        temperature = config.get("generator.mock.temperature", 0.3)

        # Simulate "best" configuration: semantic + hybrid + 0.3
        base_accuracy = 0.5

        # Chunking contribution
        if chunking == "semantic":
            base_accuracy += 0.15
        elif chunking == "sliding":
            base_accuracy += 0.10
        else:  # fixed
            base_accuracy += 0.05

        # Retrieval contribution
        if retrieval == "hybrid":
            base_accuracy += 0.20
        elif retrieval == "dense":
            base_accuracy += 0.15
        else:  # sparse
            base_accuracy += 0.10

        # Temperature contribution
        if temperature == 0.3:
            base_accuracy += 0.10
        elif temperature == 0.0:
            base_accuracy += 0.05
        else:  # 0.7
            base_accuracy += 0.02

        # Add some noise
        noise = random.uniform(-0.05, 0.05)
        accuracy = min(max(base_accuracy + noise, 0.0), 1.0)

        # Simulate other metrics
        latency = 0.01 + 0.05 + 0.1 + random.uniform(0, 0.05)
        cost = 0.0 + 0.001 + 0.01

        return {
            "accuracy": accuracy,
            "latency": latency,
            "cost": cost
        }


# ==============================================================================
# STEP 3: Grid Search Implementation
# ==============================================================================

def run_mock_grid_search():
    """Run mock 3x3 grid search"""

    logger.info("=" * 80)
    logger.info("MOCK 3x3 GRID SEARCH DEMONSTRATION")
    logger.info("=" * 80)

    # Initialize components
    bridge = ConfigurationBridge()
    evaluator = MockEvaluator()

    # Define search space (3x3x3 = 27 combinations)
    search_space = {
        "chunker.mock.strategy": ["fixed", "semantic", "sliding"],
        "retriever.mock.method": ["dense", "sparse", "hybrid"],
        "generator.mock.temperature": [0.0, 0.3, 0.7]
    }

    logger.info(f"Search space: {json.dumps(search_space, indent=2)}")
    logger.info(f"Total combinations: 3 × 3 × 3 = 27")

    # Generate all combinations
    from itertools import product

    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    all_combinations = list(product(*param_values))

    # Track results
    results = []
    best_config = None
    best_score = -float('inf')

    # Run grid search
    logger.info("\nStarting grid search...")
    print("\n" + "=" * 80)
    print("Configuration # | Chunking  | Retrieval | Temp | Accuracy | Latency | Cost")
    print("-" * 80)

    for i, combination in enumerate(all_combinations, 1):
        # Create parameter dictionary
        params = dict(zip(param_names, combination))

        # Convert to pipeline configuration
        pipeline_config = bridge.params_to_pipeline(params)

        # Evaluate (mock)
        metrics = evaluator.evaluate(params)

        # Track results
        result = {
            "config_id": i,
            "params": params,
            "metrics": metrics
        }
        results.append(result)

        # Check if best
        if metrics["accuracy"] > best_score:
            best_score = metrics["accuracy"]
            best_config = result

        # Display progress
        print(f"Config {i:2d}/27   | {params['chunker.mock.strategy']:9s} | {params['retriever.mock.method']:9s} | "
              f"{params['generator.mock.temperature']:.1f}  | "
              f"{metrics['accuracy']:.3f}    | {metrics['latency']:.3f}s  | ${metrics['cost']:.3f}")

        # Simulate processing time
        time.sleep(0.01)

    print("=" * 80)

    # Display results summary
    logger.info("\n" + "=" * 80)
    logger.info("GRID SEARCH RESULTS")
    logger.info("=" * 80)

    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x["metrics"]["accuracy"], reverse=True)

    print("\nTop 5 Configurations:")
    print("-" * 80)
    print("Rank | Chunking  | Retrieval | Temp | Accuracy | Config ID")
    print("-" * 80)

    for rank, result in enumerate(sorted_results[:5], 1):
        params = result["params"]
        print(f" {rank}   | {params['chunker.mock.strategy']:9s} | {params['retriever.mock.method']:9s} | "
              f"{params['generator.mock.temperature']:.1f}  | "
              f"{result['metrics']['accuracy']:.3f}    | #{result['config_id']}")

    # Best configuration
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"Configuration ID: #{best_config['config_id']}")
    print(f"Parameters:")
    for param, value in best_config["params"].items():
        print(f"  {param}: {value}")
    print(f"\nMetrics:")
    for metric, value in best_config["metrics"].items():
        if metric == "cost":
            print(f"  {metric}: ${value:.4f}")
        elif metric == "latency":
            print(f"  {metric}: {value:.3f}s")
        else:
            print(f"  {metric}: {value:.3f}")

    # Demonstrate pipeline building
    print("\n" + "=" * 80)
    print("PIPELINE GENERATION DEMO")
    print("=" * 80)
    print("Converting best configuration to executable pipeline...")

    best_pipeline = bridge.params_to_pipeline(best_config["params"])
    print(f"\nGenerated pipeline with {len(best_pipeline['pipeline']['nodes'])} nodes:")
    for node in best_pipeline["pipeline"]["nodes"]:
        print(f"  - {node['type']}: {node['component']} (config: {node['config']})")

    print(f"\nAuto-wired {len(best_pipeline['pipeline']['edges'])} edges:")
    for edge in best_pipeline["pipeline"]["edges"]:
        print(f"  - {edge['from']} -> {edge['to']}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)

    return results, best_config


if __name__ == "__main__":
    results, best_config = run_mock_grid_search()

    # Save results
    with open("mock_grid_search_results.json", "w") as f:
        json.dump({
            "all_results": results,
            "best_config": best_config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    logger.info("\nResults saved to mock_grid_search_results.json")