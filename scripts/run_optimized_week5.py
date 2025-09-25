"""
Optimized Week 5 Evaluation with Embedding Cache

This script minimizes embedding costs by caching embeddings for identical
chunking strategies and embedding models.
"""

import os
import sys
from pathlib import Path
import json
import hashlib
from typing import Dict, Any, List, Optional
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.demo_week5_config_search import main as original_main
import argparse


class EmbeddingCache:
    """Cache embeddings to avoid redundant API calls"""

    def __init__(self, cache_dir: str = "cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key based on chunking and embedding params"""
        # Only chunking and embedding params affect embeddings
        key_params = {
            "chunking": params.get("chunking", {}),
            "embedding": params.get("embedding", {})
        }
        key_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def has_embeddings(self, cache_key: str) -> bool:
        """Check if embeddings exist for this configuration"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        return cache_file.exists()

    def load_embeddings(self, cache_key: str) -> Optional[Dict]:
        """Load cached embeddings"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def save_embeddings(self, cache_key: str, embeddings: Dict):
        """Save embeddings to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(embeddings, f)


def estimate_costs_with_cache(num_configs: int, num_queries: int, num_docs: int = 10000):
    """Estimate costs considering embedding cache"""

    # Unique chunking strategies in search space
    unique_chunking_configs = 4  # 2 strategies × 2 sizes

    # Embedding costs (only for unique chunking configs)
    tokens_per_doc = 100
    embedding_cost_per_config = (num_docs * tokens_per_doc / 1000) * 0.0001
    total_embedding_cost = embedding_cost_per_config * unique_chunking_configs

    # Generation costs (per query per configuration)
    tokens_per_query = 500
    generation_cost_per_query = (tokens_per_query / 1000) * 0.002
    total_generation_cost = generation_cost_per_query * num_queries * num_configs

    print("\n" + "="*60)
    print("COST OPTIMIZATION ANALYSIS")
    print("="*60)

    print(f"\nWithout Embedding Cache:")
    print(f"- Embedding cost: ${embedding_cost_per_config * num_configs:.2f}")
    print(f"  ({num_configs} configs × ${embedding_cost_per_config:.2f} each)")

    print(f"\nWith Embedding Cache:")
    print(f"- Embedding cost: ${total_embedding_cost:.2f}")
    print(f"  ({unique_chunking_configs} unique chunking configs × ${embedding_cost_per_config:.2f} each)")

    print(f"\nSavings: ${(embedding_cost_per_config * num_configs - total_embedding_cost):.2f}")

    print(f"\nGeneration costs (same for both):")
    print(f"- Total: ${total_generation_cost:.2f}")
    print(f"  ({num_configs} configs × {num_queries} queries × ${generation_cost_per_query:.4f})")

    print(f"\nTotal Estimated Costs:")
    print(f"- Without cache: ${embedding_cost_per_config * num_configs + total_generation_cost:.2f}")
    print(f"- With cache: ${total_embedding_cost + total_generation_cost:.2f}")


def main():
    """Main function with embedding cache optimization"""

    parser = argparse.ArgumentParser(description="Optimized Week 5 Evaluation")
    parser.add_argument("--estimate-only", action="store_true",
                       help="Only show cost estimates")
    parser.add_argument("--num-queries", type=int, default=30,
                       help="Number of queries per configuration")
    parser.add_argument("--max-configs", type=int, default=10,
                       help="Maximum configurations to evaluate")
    parser.add_argument("--num-docs", type=int, default=10000,
                       help="Number of documents to index")

    args = parser.parse_args()

    if args.estimate_only:
        estimate_costs_with_cache(
            num_configs=args.max_configs,
            num_queries=args.num_queries,
            num_docs=args.num_docs
        )

        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"\nFor {args.num_docs:,} documents:")
        print(f"1. Use the existing demo script with --real-eval")
        print(f"2. The script will automatically load from MS MARCO")
        print(f"3. Each unique chunking strategy needs separate embeddings")
        print(f"4. Consider reducing to 5-6 configs to stay under budget")
        print(f"\nRun command:")
        print(f"python scripts/demo_week5_config_search.py \\")
        print(f"    --real-eval \\")
        print(f"    --num-queries {args.num_queries} \\")
        print(f"    --max-configs {args.max_configs} \\")
        print(f"    --budget 25.0 \\")
        print(f"    --use-ragas")

        return 0

    # For actual run, use the original demo script
    print("\nRedirecting to original demo script with optimizations...")
    sys.argv = [
        "demo_week5_config_search.py",
        "--real-eval",
        f"--num-queries={args.num_queries}",
        f"--max-configs={args.max_configs}",
        "--budget=25.0",
        "--use-ragas"
    ]

    return original_main()


if __name__ == "__main__":
    exit(main())