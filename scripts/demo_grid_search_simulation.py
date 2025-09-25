"""Simulated grid search demonstration with MS MARCO data structure"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
from typing import Dict, Any, List
from loguru import logger
import numpy as np
from itertools import product

# Setup logging
logger.add("simulated_grid_search.log", rotation="10 MB")

# Import our architecture
from autorag.components.descriptors import ComponentDescriptor, ParamSpec, ParamType, SelfDescribingComponent
from autorag.components.auto_register import auto_register
from autorag.components.base import Component, Chunker, Retriever, Generator
from autorag.optimization.configuration_bridge import ConfigurationBridge
from autorag.pipeline.registry import get_registry


# Clear registry
registry = get_registry()
registry.clear()


# ==============================================================================
# Simulated Components (Real structure, simulated execution)
# ==============================================================================

@auto_register("chunker", "fixed_size")
class SimulatedChunker(Chunker, SelfDescribingComponent):
    """Simulated chunker for demonstration"""

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
                    description="Size of chunks in tokens"
                )
            },
            inputs=["documents"],
            outputs=["chunks"],
            tunable_params=["chunk_size"],
            estimated_cost=0.0,
            estimated_latency=0.01
        )

    def chunk(self, documents):
        chunk_size = self.config.get("chunk_size", 512)
        # Simulate chunking - more chunks for smaller size
        num_chunks = len(documents) * (2048 // chunk_size)
        return [f"chunk_{i}" for i in range(num_chunks)]


@auto_register("retriever", "dense")
class SimulatedRetriever(Retriever, SelfDescribingComponent):
    """Simulated retriever for demonstration"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="DenseRetriever",
            type="retriever",
            parameters={
                "top_k": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[3, 5, 10],
                    default=5,
                    tunable=True,
                    description="Number of documents to retrieve"
                )
            },
            inputs=["query", "chunks"],
            outputs=["retrieved_documents"],
            tunable_params=["top_k"],
            estimated_cost=0.0001,
            estimated_latency=0.05
        )

    def retrieve(self, query, chunks, top_k=5):
        top_k = self.config.get("top_k", 5)
        return [f"doc_{i}" for i in range(top_k)]

    def process(self, *args, **kwargs):
        return self.retrieve(*args, **kwargs)


@auto_register("generator", "gpt35")
class SimulatedGenerator(Generator, SelfDescribingComponent):
    """Simulated generator for demonstration"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="GPT35Generator",
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
            estimated_cost=0.002,
            estimated_latency=1.0
        )

    def generate(self, query, contexts):
        temperature = self.config.get("temperature", 0.3)
        return f"Answer with temp={temperature}"

    def process(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


# ==============================================================================
# Simulated Evaluator with realistic scoring
# ==============================================================================

class SimulatedEvaluator:
    """Simulates RAGAS evaluation with realistic patterns"""

    def evaluate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Simulate evaluation metrics based on configuration"""

        # Extract parameters
        chunk_size = params.get("chunker.fixed_size.chunk_size", 512)
        top_k = params.get("retriever.dense.top_k", 5)
        temperature = params.get("generator.gpt35.temperature", 0.3)

        # Simulate realistic scoring patterns
        # Chunk size: 512 is optimal, too small or too large hurts
        if chunk_size == 256:
            chunk_score = 0.75  # Too granular
        elif chunk_size == 512:
            chunk_score = 0.90  # Optimal
        else:  # 1024
            chunk_score = 0.80  # Too coarse

        # Top-k: 5 is optimal for precision/recall balance
        if top_k == 3:
            retrieval_score = 0.70  # Too few
        elif top_k == 5:
            retrieval_score = 0.85  # Optimal
        else:  # 10
            retrieval_score = 0.75  # Too many, adds noise

        # Temperature: 0.3 is optimal for factual Q&A
        if temperature == 0.0:
            generation_score = 0.80  # Too rigid
        elif temperature == 0.3:
            generation_score = 0.90  # Optimal
        else:  # 0.7
            generation_score = 0.65  # Too creative for factual

        # Calculate metrics with some noise
        noise = lambda: random.uniform(-0.05, 0.05)

        answer_relevancy = (chunk_score * 0.3 + retrieval_score * 0.4 + generation_score * 0.3) + noise()
        faithfulness = (chunk_score * 0.2 + retrieval_score * 0.3 + generation_score * 0.5) + noise()
        context_relevancy = (chunk_score * 0.4 + retrieval_score * 0.5 + generation_score * 0.1) + noise()

        # Clamp to [0, 1]
        answer_relevancy = max(0, min(1, answer_relevancy))
        faithfulness = max(0, min(1, faithfulness))
        context_relevancy = max(0, min(1, context_relevancy))

        # Calculate costs (realistic)
        cost = 0.0
        cost += 0.0001 * (1000 * 2048 // chunk_size)  # Embedding cost
        cost += 0.002 * 20  # Generation cost for 20 queries

        # Calculate latency
        latency = 0.01 + 0.05 + 1.0 + random.uniform(0, 0.2)

        return {
            "answer_relevancy": answer_relevancy,
            "faithfulness": faithfulness,
            "context_relevancy": context_relevancy,
            "total_cost": cost,
            "avg_latency": latency
        }


# ==============================================================================
# Grid Search Runner
# ==============================================================================

def run_simulated_grid_search():
    """Run simulated grid search with realistic patterns"""

    logger.info("=" * 80)
    logger.info("SIMULATED GRID SEARCH (MS MARCO Structure)")
    logger.info("=" * 80)

    # Initialize
    bridge = ConfigurationBridge()
    evaluator = SimulatedEvaluator()

    # Define search space (3x3x3 = 27 combinations)
    search_space = {
        "chunker.fixed_size.chunk_size": [256, 512, 1024],
        "retriever.dense.top_k": [3, 5, 10],
        "generator.gpt35.temperature": [0.0, 0.3, 0.7]
    }

    print(f"\nSearch Space Configuration:")
    print(f"  Chunk Sizes: {search_space['chunker.fixed_size.chunk_size']}")
    print(f"  Top-K Values: {search_space['retriever.dense.top_k']}")
    print(f"  Temperatures: {search_space['generator.gpt35.temperature']}")
    print(f"  Total Combinations: 3 × 3 × 3 = 27")

    print(f"\nSimulated Environment:")
    print(f"  Documents: 1000 (simulated MS MARCO passages)")
    print(f"  Queries: 20 (simulated MS MARCO questions)")
    print(f"  Metrics: RAGAS (answer relevancy, faithfulness, context relevancy)")

    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    all_combinations = list(product(*param_values))

    # Track results
    results = []
    best_config = None
    best_score = -float('inf')

    # Run grid search
    print("\n" + "=" * 110)
    print("Config | Chunk Size | Top-K | Temp | Answer Rel | Faithfulness | Context Rel | Score | Cost   | Status")
    print("-" * 110)

    for i, combination in enumerate(all_combinations, 1):
        # Create parameter dictionary
        params = dict(zip(param_names, combination))

        # Convert to pipeline (demonstrates auto-wiring)
        pipeline_config = bridge.params_to_pipeline(params)

        # Evaluate
        metrics = evaluator.evaluate(params)

        # Calculate composite score
        score = (
            metrics["answer_relevancy"] * 0.4 +
            metrics["faithfulness"] * 0.3 +
            metrics["context_relevancy"] * 0.3
        )

        # Track results
        result = {
            "config_id": i,
            "params": params,
            "metrics": metrics,
            "score": score,
            "pipeline": pipeline_config  # Include auto-generated pipeline
        }
        results.append(result)

        # Check if best
        if score > best_score:
            best_score = score
            best_config = result
            status = "* NEW BEST"
        else:
            status = ""

        # Display progress
        print(f"{i:3d}/27 | {params['chunker.fixed_size.chunk_size']:10d} | {params['retriever.dense.top_k']:5d} | "
              f"{params['generator.gpt35.temperature']:.1f}  | "
              f"{metrics['answer_relevancy']:.3f}      | "
              f"{metrics['faithfulness']:.3f}        | "
              f"{metrics['context_relevancy']:.3f}       | "
              f"{score:.3f} | "
              f"${metrics['total_cost']:.3f} | {status}")

        # Simulate processing time
        time.sleep(0.05)

    print("=" * 110)

    # Display results
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\nRank | Chunk Size | Top-K | Temp | Score | Improvement | Config ID")
    print("-" * 80)

    baseline_score = sorted_results[-1]["score"]  # Worst score as baseline

    for rank, result in enumerate(sorted_results[:5], 1):
        params = result["params"]
        improvement = ((result["score"] - baseline_score) / baseline_score) * 100
        print(f" {rank}   | {params['chunker.fixed_size.chunk_size']:10d} | "
              f"{params['retriever.dense.top_k']:5d} | "
              f"{params['generator.gpt35.temperature']:.1f}  | "
              f"{result['score']:.3f} | "
              f"+{improvement:5.1f}%     | #{result['config_id']}")

    # Best configuration details
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION DETAILS")
    print("=" * 80)
    print(f"Configuration ID: #{best_config['config_id']}")
    print(f"Overall Score: {best_config['score']:.3f}")

    print(f"\nOptimal Parameters:")
    for param, value in best_config["params"].items():
        param_name = param.split(".")[-1]
        print(f"  {param_name}: {value}")

    print(f"\nPerformance Metrics:")
    for metric, value in best_config["metrics"].items():
        if metric == "total_cost":
            print(f"  {metric}: ${value:.4f}")
        elif metric == "avg_latency":
            print(f"  {metric}: {value:.3f}s")
        else:
            print(f"  {metric}: {value:.3f}")

    # Show auto-generated pipeline
    print("\n" + "=" * 80)
    print("AUTO-GENERATED PIPELINE")
    print("=" * 80)
    pipeline = best_config["pipeline"]["pipeline"]
    print(f"Nodes ({len(pipeline['nodes'])}):")
    for node in pipeline["nodes"]:
        print(f"  - {node['type']}: {node['component']} (config: {node['config']})")

    print(f"\nAuto-wired Edges ({len(pipeline['edges'])}):")
    for edge in pipeline["edges"]:
        print(f"  - {edge['from']} -> {edge['to']}")

    # Statistical summary
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    scores = [r["score"] for r in results]
    print(f"Score Distribution:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std Dev: {np.std(scores):.3f}")
    print(f"  Min: {np.min(scores):.3f}")
    print(f"  Max: {np.max(scores):.3f}")
    print(f"  Range: {np.max(scores) - np.min(scores):.3f}")

    # Parameter importance (simplified)
    print(f"\nParameter Impact Analysis:")

    # Chunk size impact
    chunk_scores = {}
    for size in [256, 512, 1024]:
        size_results = [r for r in results if r["params"]["chunker.fixed_size.chunk_size"] == size]
        chunk_scores[size] = np.mean([r["score"] for r in size_results])

    best_chunk = max(chunk_scores, key=chunk_scores.get)
    print(f"  Best Chunk Size: {best_chunk} (avg score: {chunk_scores[best_chunk]:.3f})")

    # Top-k impact
    topk_scores = {}
    for k in [3, 5, 10]:
        k_results = [r for r in results if r["params"]["retriever.dense.top_k"] == k]
        topk_scores[k] = np.mean([r["score"] for r in k_results])

    best_topk = max(topk_scores, key=topk_scores.get)
    print(f"  Best Top-K: {best_topk} (avg score: {topk_scores[best_topk]:.3f})")

    # Temperature impact
    temp_scores = {}
    for t in [0.0, 0.3, 0.7]:
        t_results = [r for r in results if r["params"]["generator.gpt35.temperature"] == t]
        temp_scores[t] = np.mean([r["score"] for r in t_results])

    best_temp = max(temp_scores, key=temp_scores.get)
    print(f"  Best Temperature: {best_temp} (avg score: {temp_scores[best_temp]:.3f})")

    # Save results
    with open("simulated_grid_search_results.json", "w") as f:
        json.dump({
            "all_results": results,
            "best_config": best_config,
            "search_space": search_space,
            "statistics": {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores))
            },
            "parameter_impact": {
                "chunk_size": {str(k): v for k, v in chunk_scores.items()},
                "top_k": {str(k): v for k, v in topk_scores.items()},
                "temperature": {str(k): v for k, v in temp_scores.items()}
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Results saved to simulated_grid_search_results.json")
    print("\nKey Insights:")
    print("  1. Chunk size of 512 provides optimal granularity")
    print("  2. Top-K of 5 balances precision and recall")
    print("  3. Temperature of 0.3 is best for factual Q&A")
    print("  4. Self-describing architecture automatically handled all configurations")
    print("  5. No manual pipeline wiring was required!")

    return results, best_config


if __name__ == "__main__":
    results, best_config = run_simulated_grid_search()