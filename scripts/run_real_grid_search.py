"""Real 3x3 grid search with MS MARCO data using self-describing architecture"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from typing import Dict, Any, List
from loguru import logger
import numpy as np
from itertools import product

# Setup logging
logger.add("real_grid_search.log", rotation="10 MB")

# Import our architecture components
from autorag.components.descriptors import ComponentDescriptor, ParamSpec, ParamType, SelfDescribingComponent
from autorag.components.auto_register import auto_register
from autorag.components.base import Component, Chunker, Embedder, VectorStore, Retriever, Generator
from autorag.pipeline.auto_pipeline_builder import AutoPipelineBuilder
from autorag.optimization.configuration_bridge import ConfigurationBridge
from autorag.pipeline.registry import get_registry

# Import existing components
from autorag.components.chunkers.fixed_size import FixedSizeChunker as OldFixedChunker
from autorag.components.embedders.openai import OpenAIEmbedder as OldOpenAIEmbedder
from autorag.components.retrievers.dense import DenseRetriever as OldDenseRetriever
from autorag.components.generators.openai import OpenAIGenerator as OldOpenAIGenerator

# Import evaluation components
from autorag.datasets.msmarco_loader import MSMARCOLoader
from autorag.evaluation.ragas_evaluator import RAGASEvaluator
from autorag.evaluation.cost_tracker import CostTracker

# Clear registry for clean start
registry = get_registry()
registry.clear()


# ==============================================================================
# STEP 1: Migrate Existing Components to Self-Describing
# ==============================================================================

@auto_register("chunker", "fixed_size")
class FixedSizeChunkerSD(OldFixedChunker, SelfDescribingComponent):
    """Fixed-size chunker with self-description"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="FixedSizeChunker",
            type="chunker",
            parameters={
                "chunk_size": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[256, 512, 1024],  # 3 options for grid search
                    default=512,
                    tunable=True,
                    description="Size of chunks in tokens"
                ),
                "overlap": ParamSpec(
                    type=ParamType.INT,
                    default=50,
                    min_value=0,
                    max_value=100,
                    tunable=False  # Fixed for this experiment
                )
            },
            inputs=["documents"],
            outputs=["chunks"],
            tunable_params=["chunk_size"],
            estimated_cost=0.0,
            estimated_latency=0.01
        )

    def process(self, documents):
        # Use parent class implementation
        return self.chunk(documents)


@auto_register("embedder", "openai")
class OpenAIEmbedderSD(OldOpenAIEmbedder, SelfDescribingComponent):
    """OpenAI embedder with self-description"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="OpenAIEmbedder",
            type="embedder",
            parameters={
                "model": ParamSpec(
                    type=ParamType.STRING,
                    default="text-embedding-ada-002",
                    tunable=False  # Fixed for cost reasons
                ),
                "batch_size": ParamSpec(
                    type=ParamType.INT,
                    default=100,
                    tunable=False
                )
            },
            inputs=["chunks"],
            outputs=["embeddings"],
            tunable_params=[],  # No tunable params for embedder
            estimated_cost=0.0001,  # Per embedding
            estimated_latency=0.1
        )

    def process(self, texts):
        return self.embed(texts)


@auto_register("retriever", "dense")
class DenseRetrieverSD(OldDenseRetriever, SelfDescribingComponent):
    """Dense retriever with self-description"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="DenseRetriever",
            type="retriever",
            parameters={
                "top_k": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[3, 5, 10],  # 3 options for grid search
                    default=5,
                    tunable=True,
                    description="Number of documents to retrieve"
                ),
                "similarity_metric": ParamSpec(
                    type=ParamType.STRING,
                    default="cosine",
                    tunable=False
                )
            },
            inputs=["query", "embeddings"],
            outputs=["retrieved_documents"],
            tunable_params=["top_k"],
            estimated_cost=0.0001,
            estimated_latency=0.05
        )

    def process(self, query, index):
        return self.retrieve(query, top_k=self.config.get("top_k", 5))


@auto_register("generator", "openai")
class OpenAIGeneratorSD(OldOpenAIGenerator, SelfDescribingComponent):
    """OpenAI generator with self-description"""

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return ComponentDescriptor(
            name="OpenAIGenerator",
            type="generator",
            parameters={
                "model": ParamSpec(
                    type=ParamType.STRING,
                    default="gpt-3.5-turbo",
                    tunable=False  # Fixed for cost
                ),
                "temperature": ParamSpec(
                    type=ParamType.CHOICE,
                    choices=[0.0, 0.3, 0.7],  # 3 options for grid search
                    default=0.3,
                    tunable=True,
                    description="Generation temperature"
                ),
                "max_tokens": ParamSpec(
                    type=ParamType.INT,
                    default=300,
                    tunable=False  # Fixed for consistency
                )
            },
            inputs=["query", "retrieved_documents"],
            outputs=["answer"],
            tunable_params=["temperature"],
            estimated_cost=0.002,  # Per generation
            estimated_latency=1.0
        )

    def process(self, query, contexts):
        return self.generate(query, contexts)


# ==============================================================================
# STEP 2: Grid Search Implementation
# ==============================================================================

class RealGridSearch:
    """Real grid search with MS MARCO data"""

    def __init__(self, num_docs: int = 1000, num_queries: int = 20):
        """
        Initialize grid search

        Args:
            num_docs: Number of documents to index
            num_queries: Number of queries to evaluate
        """
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.bridge = ConfigurationBridge()
        self.cost_tracker = CostTracker()

        logger.info(f"Initializing real grid search with {num_docs} docs, {num_queries} queries")

    def load_data(self):
        """Load MS MARCO data"""
        logger.info("Loading MS MARCO data...")
        loader = MSMARCOLoader()

        # Load subset
        documents, queries = loader.load_subset(
            num_docs=self.num_docs,
            num_queries=self.num_queries,
            include_answers=True
        )

        logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
        return documents, queries

    def create_pipeline(self, params: Dict[str, Any]):
        """Create pipeline from parameters"""
        # Convert params to pipeline config
        pipeline_config = self.bridge.params_to_pipeline(params)

        # Create actual component instances
        from autorag.pipeline.rag_pipeline import SimpleRAGPipeline

        # Initialize components with config
        chunker = FixedSizeChunkerSD({
            "chunk_size": params.get("chunker.fixed_size.chunk_size", 512),
            "overlap": 50
        })

        embedder = OpenAIEmbedderSD({
            "model": "text-embedding-ada-002",
            "batch_size": 100
        })

        retriever = DenseRetrieverSD({
            "top_k": params.get("retriever.dense.top_k", 5)
        })

        generator = OpenAIGeneratorSD({
            "model": "gpt-3.5-turbo",
            "temperature": params.get("generator.openai.temperature", 0.3),
            "max_tokens": 300
        })

        # Create pipeline
        pipeline = SimpleRAGPipeline(
            chunker=chunker,
            embedder=embedder,
            retriever=retriever,
            generator=generator
        )

        return pipeline

    def evaluate_configuration(self, params: Dict[str, Any], documents: List, queries: List) -> Dict[str, float]:
        """Evaluate a single configuration"""
        try:
            logger.info(f"Evaluating configuration: {params}")

            # Create pipeline
            pipeline = self.create_pipeline(params)

            # Index documents
            logger.info("Indexing documents...")
            pipeline.index_documents(documents)

            # Initialize evaluator
            evaluator = RAGASEvaluator()

            # Evaluate on queries
            results = []
            total_cost = 0.0
            total_time = 0.0

            for i, query_data in enumerate(queries[:self.num_queries]):
                query = query_data["question"]

                # Time the query
                start_time = time.time()

                # Get answer
                answer, contexts = pipeline.answer_question(query)

                # Track time
                query_time = time.time() - start_time
                total_time += query_time

                # Evaluate answer
                metrics = evaluator.evaluate_single(
                    question=query,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=query_data.get("wellFormedAnswers", [""])[0] if query_data.get("wellFormedAnswers") else None
                )

                results.append(metrics)

                # Track costs
                total_cost += 0.0001 * len(documents)  # Embedding cost
                total_cost += 0.002  # Generation cost

                logger.info(f"Query {i+1}/{self.num_queries}: {metrics.get('answer_relevancy', 0):.3f}")

            # Aggregate metrics
            avg_metrics = {}
            for key in results[0].keys():
                values = [r.get(key, 0) for r in results if r.get(key) is not None]
                if values:
                    avg_metrics[key] = np.mean(values)

            # Add cost and time metrics
            avg_metrics["total_cost"] = total_cost
            avg_metrics["avg_latency"] = total_time / len(queries)

            logger.info(f"Configuration metrics: {avg_metrics}")
            return avg_metrics

        except Exception as e:
            logger.error(f"Error evaluating configuration: {e}")
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "context_relevancy": 0.0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "error": str(e)
            }

    def run_grid_search(self):
        """Run the grid search"""
        logger.info("=" * 80)
        logger.info("REAL 3x3 GRID SEARCH WITH MS MARCO DATA")
        logger.info("=" * 80)

        # Load data
        documents, queries = self.load_data()

        # Define search space
        search_space = {
            "chunker.fixed_size.chunk_size": [256, 512, 1024],
            "retriever.dense.top_k": [3, 5, 10],
            "generator.openai.temperature": [0.0, 0.3, 0.7]
        }

        logger.info(f"Search space: {json.dumps(search_space, indent=2)}")
        logger.info(f"Total combinations: 3 × 3 × 3 = 27")

        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        all_combinations = list(product(*param_values))

        # Track results
        results = []
        best_config = None
        best_score = -float('inf')

        # Run grid search
        print("\n" + "=" * 100)
        print("Config | Chunk Size | Top-K | Temp | Relevancy | Faithfulness | Context | Cost   | Latency")
        print("-" * 100)

        for i, combination in enumerate(all_combinations, 1):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))

            # Evaluate configuration
            metrics = self.evaluate_configuration(params, documents, queries)

            # Calculate composite score (weighted average)
            score = (
                metrics.get("answer_relevancy", 0) * 0.4 +
                metrics.get("faithfulness", 0) * 0.3 +
                metrics.get("context_relevancy", 0) * 0.3
            )

            # Track results
            result = {
                "config_id": i,
                "params": params,
                "metrics": metrics,
                "score": score
            }
            results.append(result)

            # Check if best
            if score > best_score:
                best_score = score
                best_config = result

            # Display progress
            print(f"{i:3d}/27 | {params['chunker.fixed_size.chunk_size']:10d} | {params['retriever.dense.top_k']:5d} | "
                  f"{params['generator.openai.temperature']:.1f} | "
                  f"{metrics.get('answer_relevancy', 0):.3f}     | "
                  f"{metrics.get('faithfulness', 0):.3f}        | "
                  f"{metrics.get('context_relevancy', 0):.3f}   | "
                  f"${metrics.get('total_cost', 0):.3f} | "
                  f"{metrics.get('avg_latency', 0):.2f}s")

            # Save intermediate results
            with open(f"grid_search_checkpoint_{i}.json", "w") as f:
                json.dump(result, f, indent=2)

        print("=" * 100)

        # Display final results
        print("\n" + "=" * 80)
        print("GRID SEARCH RESULTS")
        print("=" * 80)

        # Sort by score
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        print("\nTop 5 Configurations:")
        print("-" * 80)
        print("Rank | Chunk Size | Top-K | Temp | Score | Config ID")
        print("-" * 80)

        for rank, result in enumerate(sorted_results[:5], 1):
            params = result["params"]
            print(f" {rank}   | {params['chunker.fixed_size.chunk_size']:10d} | "
                  f"{params['retriever.dense.top_k']:5d} | "
                  f"{params['generator.openai.temperature']:.1f}  | "
                  f"{result['score']:.3f} | #{result['config_id']}")

        # Best configuration
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        print(f"Configuration ID: #{best_config['config_id']}")
        print(f"Score: {best_config['score']:.3f}")
        print(f"\nParameters:")
        for param, value in best_config["params"].items():
            print(f"  {param}: {value}")
        print(f"\nMetrics:")
        for metric, value in best_config["metrics"].items():
            if metric in ["total_cost"]:
                print(f"  {metric}: ${value:.4f}")
            elif metric in ["avg_latency"]:
                print(f"  {metric}: {value:.3f}s")
            elif metric != "error":
                print(f"  {metric}: {value:.3f}")

        # Save final results
        with open("real_grid_search_results.json", "w") as f:
            json.dump({
                "all_results": results,
                "best_config": best_config,
                "search_space": search_space,
                "num_docs": self.num_docs,
                "num_queries": self.num_queries,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        logger.info("\nResults saved to real_grid_search_results.json")

        return results, best_config


if __name__ == "__main__":
    # Configure parameters
    NUM_DOCS = 100  # Start small to manage costs
    NUM_QUERIES = 20  # 20 queries for evaluation

    print(f"\nStarting real grid search with {NUM_DOCS} documents and {NUM_QUERIES} queries")
    print("Estimated cost: ~$2-5 depending on API usage")
    print("Estimated time: ~10-20 minutes\n")

    # Run grid search
    grid_search = RealGridSearch(num_docs=NUM_DOCS, num_queries=NUM_QUERIES)
    results, best_config = grid_search.run_grid_search()

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)