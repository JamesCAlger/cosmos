"""Cost tracking system for RAG evaluation"""

import tiktoken
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from loguru import logger


@dataclass
class ModelPricing:
    """Pricing information for a model"""
    model_name: str
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    embedding_cost_per_1k: Optional[float] = None  # For embedding models


@dataclass
class CostRecord:
    """Record of costs for a single operation"""
    timestamp: datetime
    operation: str  # e.g., "embedding", "generation", "retrieval"
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Track and manage costs for RAG operations"""

    # Default pricing (as of 2024)
    DEFAULT_PRICING = {
        # OpenAI models
        "gpt-3.5-turbo": ModelPricing(
            model_name="gpt-3.5-turbo",
            input_cost_per_1k=0.0005,
            output_cost_per_1k=0.0015
        ),
        "gpt-4": ModelPricing(
            model_name="gpt-4",
            input_cost_per_1k=0.03,
            output_cost_per_1k=0.06
        ),
        "gpt-4o-mini": ModelPricing(
            model_name="gpt-4o-mini",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.0006
        ),
        "text-embedding-ada-002": ModelPricing(
            model_name="text-embedding-ada-002",
            input_cost_per_1k=0.0001,
            output_cost_per_1k=0.0,
            embedding_cost_per_1k=0.0001
        ),
        "text-embedding-3-small": ModelPricing(
            model_name="text-embedding-3-small",
            input_cost_per_1k=0.00002,
            output_cost_per_1k=0.0,
            embedding_cost_per_1k=0.00002
        ),
        "text-embedding-3-large": ModelPricing(
            model_name="text-embedding-3-large",
            input_cost_per_1k=0.00013,
            output_cost_per_1k=0.0,
            embedding_cost_per_1k=0.00013
        ),

        # Anthropic models
        "claude-3-haiku": ModelPricing(
            model_name="claude-3-haiku",
            input_cost_per_1k=0.00025,
            output_cost_per_1k=0.00125
        ),
        "claude-3-sonnet": ModelPricing(
            model_name="claude-3-sonnet",
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015
        ),
        "claude-3-opus": ModelPricing(
            model_name="claude-3-opus",
            input_cost_per_1k=0.015,
            output_cost_per_1k=0.075
        ),

        # Local/open models (compute cost estimates)
        "e5-small-v2": ModelPricing(
            model_name="e5-small-v2",
            input_cost_per_1k=0.00001,  # Estimated compute cost
            output_cost_per_1k=0.0,
            embedding_cost_per_1k=0.00001
        ),
        "bge-small-en": ModelPricing(
            model_name="bge-small-en",
            input_cost_per_1k=0.00001,
            output_cost_per_1k=0.0,
            embedding_cost_per_1k=0.00001
        ),
    }

    def __init__(self,
                 pricing: Optional[Dict[str, ModelPricing]] = None,
                 budget_limit: Optional[float] = None,
                 track_history: bool = True):
        """
        Initialize cost tracker

        Args:
            pricing: Custom pricing dictionary (uses defaults if None)
            budget_limit: Maximum budget allowed
            track_history: Whether to keep detailed history
        """
        self.pricing = pricing or self.DEFAULT_PRICING
        self.budget_limit = budget_limit
        self.track_history = track_history

        self.total_cost = 0.0
        self.costs_by_operation: Dict[str, float] = {}
        self.costs_by_model: Dict[str, float] = {}
        self.history: List[CostRecord] = []

        # Token counting
        self.tokenizers = {}

    def get_tokenizer(self, model: str):
        """Get or create tokenizer for a model"""
        if model not in self.tokenizers:
            try:
                # Try to get the specific encoding for the model
                if "gpt" in model.lower():
                    self.tokenizers[model] = tiktoken.encoding_for_model(model)
                else:
                    # Default to cl100k_base for other models
                    self.tokenizers[model] = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {model}: {e}")
                # Fallback to cl100k_base
                self.tokenizers[model] = tiktoken.get_encoding("cl100k_base")

        return self.tokenizers[model]

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text for a specific model"""
        try:
            tokenizer = self.get_tokenizer(model)
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback: rough estimation
            return len(text.split()) * 1.3  # Rough approximation

    def estimate_cost(self,
                      text: str,
                      model: str,
                      operation: str = "generation",
                      output_text: Optional[str] = None) -> float:
        """
        Estimate cost for processing text

        Args:
            text: Input text
            model: Model name
            operation: Type of operation (generation, embedding, etc.)
            output_text: Output text for generation operations

        Returns:
            Estimated cost
        """
        # Get pricing info
        pricing = self._get_pricing(model)
        if not pricing:
            logger.warning(f"No pricing info for {model}, using default")
            pricing = ModelPricing(model, 0.0005, 0.0015)  # Default to GPT-3.5 pricing

        # Count tokens
        input_tokens = self.count_tokens(text, model)

        if operation == "embedding":
            cost = (input_tokens / 1000) * (pricing.embedding_cost_per_1k or pricing.input_cost_per_1k)
            output_tokens = 0
        else:
            # Generation operation
            if output_text:
                output_tokens = self.count_tokens(output_text, model)
            else:
                # Estimate output tokens (rough heuristic)
                output_tokens = min(input_tokens * 2, 500)  # Assume 2x input or max 500

            input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
            output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
            cost = input_cost + output_cost

        # Track the cost
        self.track_cost(
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens if operation != "embedding" else 0,
            cost=cost
        )

        return cost

    def track_cost(self,
                   operation: str,
                   model: str,
                   input_tokens: int,
                   output_tokens: int,
                   cost: float,
                   metadata: Optional[Dict] = None):
        """Track a cost record"""
        # Update totals
        self.total_cost += cost

        # Update by operation
        if operation not in self.costs_by_operation:
            self.costs_by_operation[operation] = 0
        self.costs_by_operation[operation] += cost

        # Update by model
        if model not in self.costs_by_model:
            self.costs_by_model[model] = 0
        self.costs_by_model[model] += cost

        # Add to history if tracking
        if self.track_history:
            record = CostRecord(
                timestamp=datetime.now(),
                operation=operation,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                metadata=metadata or {}
            )
            self.history.append(record)

        # Check budget
        if self.budget_limit and self.total_cost > self.budget_limit:
            logger.warning(f"Budget limit exceeded! Current: ${self.total_cost:.4f}, Limit: ${self.budget_limit:.4f}")

    def estimate_pipeline_cost(self,
                                config: Dict,
                                num_queries: int,
                                avg_doc_length: int = 500,
                                avg_query_length: int = 20,
                                avg_chunks_per_query: int = 5) -> float:
        """
        Estimate cost for running a pipeline configuration

        Args:
            config: Pipeline configuration
            num_queries: Number of queries to process
            avg_doc_length: Average document length in tokens
            avg_query_length: Average query length in tokens
            avg_chunks_per_query: Average chunks retrieved per query

        Returns:
            Estimated total cost
        """
        total_cost = 0.0

        # Embedding costs (if using dense retrieval)
        if config.get("retrieval", {}).get("method") in ["dense", "hybrid"]:
            embedding_model = config.get("embedding", {}).get("model", "text-embedding-ada-002")
            embedding_pricing = self._get_pricing(embedding_model)

            if embedding_pricing:
                # Cost for embedding chunks
                embedding_cost = (avg_doc_length / 1000) * embedding_pricing.embedding_cost_per_1k
                # Cost for embedding queries
                query_embedding_cost = (avg_query_length / 1000) * embedding_pricing.embedding_cost_per_1k * num_queries
                total_cost += embedding_cost + query_embedding_cost

        # Generation costs
        generation_model = config.get("generation", {}).get("model", "gpt-3.5-turbo")
        generation_pricing = self._get_pricing(generation_model)

        if generation_pricing:
            # Input: query + retrieved chunks
            input_tokens_per_query = avg_query_length + (avg_chunks_per_query * avg_doc_length)
            # Output: generated answer
            output_tokens_per_query = config.get("generation", {}).get("max_tokens", 300)

            input_cost = (input_tokens_per_query / 1000) * generation_pricing.input_cost_per_1k * num_queries
            output_cost = (output_tokens_per_query / 1000) * generation_pricing.output_cost_per_1k * num_queries
            total_cost += input_cost + output_cost

        return total_cost

    def _get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a model"""
        # Try exact match first
        if model in self.pricing:
            return self.pricing[model]

        # Try partial match
        for key, pricing in self.pricing.items():
            if key in model or model in key:
                return pricing

        return None

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if an operation would exceed budget"""
        if not self.budget_limit:
            return True

        return (self.total_cost + estimated_cost) <= self.budget_limit

    def get_remaining_budget(self) -> Optional[float]:
        """Get remaining budget"""
        if not self.budget_limit:
            return None

        return max(0, self.budget_limit - self.total_cost)

    def reset(self):
        """Reset all cost tracking"""
        self.total_cost = 0.0
        self.costs_by_operation = {}
        self.costs_by_model = {}
        self.history = []

    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        summary = {
            "total_cost": self.total_cost,
            "budget_limit": self.budget_limit,
            "remaining_budget": self.get_remaining_budget(),
            "costs_by_operation": self.costs_by_operation,
            "costs_by_model": self.costs_by_model,
            "num_operations": len(self.history),
        }

        if self.history:
            total_input_tokens = sum(r.input_tokens for r in self.history)
            total_output_tokens = sum(r.output_tokens for r in self.history)

            summary.update({
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "avg_cost_per_operation": self.total_cost / len(self.history)
            })

        return summary

    def save_history(self, filepath: str):
        """Save cost history to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "history": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "operation": r.operation,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost": r.cost,
                    "metadata": r.metadata
                }
                for r in self.history
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_history(self, filepath: str):
        """Load cost history from file"""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"History file not found: {filepath}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Restore summary
        if "summary" in data:
            self.total_cost = data["summary"].get("total_cost", 0)
            self.costs_by_operation = data["summary"].get("costs_by_operation", {})
            self.costs_by_model = data["summary"].get("costs_by_model", {})

        # Restore history
        if "history" in data:
            self.history = []
            for record_data in data["history"]:
                record = CostRecord(
                    timestamp=datetime.fromisoformat(record_data["timestamp"]),
                    operation=record_data["operation"],
                    model=record_data["model"],
                    input_tokens=record_data["input_tokens"],
                    output_tokens=record_data["output_tokens"],
                    cost=record_data["cost"],
                    metadata=record_data.get("metadata", {})
                )
                self.history.append(record)

    def format_summary(self) -> str:
        """Format summary as human-readable string"""
        summary = self.get_summary()

        lines = ["Cost Tracking Summary"]
        lines.append("=" * 40)
        lines.append(f"Total Cost: ${summary['total_cost']:.4f}")

        if summary['budget_limit']:
            lines.append(f"Budget Limit: ${summary['budget_limit']:.2f}")
            lines.append(f"Remaining: ${summary['remaining_budget']:.4f}")
            usage_pct = (summary['total_cost'] / summary['budget_limit']) * 100
            lines.append(f"Budget Used: {usage_pct:.1f}%")

        lines.append("\nCosts by Operation:")
        for op, cost in summary['costs_by_operation'].items():
            lines.append(f"  {op}: ${cost:.4f}")

        lines.append("\nCosts by Model:")
        for model, cost in summary['costs_by_model'].items():
            lines.append(f"  {model}: ${cost:.4f}")

        if "total_input_tokens" in summary:
            lines.append(f"\nTotal Tokens:")
            lines.append(f"  Input: {summary['total_input_tokens']:,}")
            lines.append(f"  Output: {summary['total_output_tokens']:,}")
            lines.append(f"\nAvg Cost per Operation: ${summary['avg_cost_per_operation']:.4f}")

        return "\n".join(lines)