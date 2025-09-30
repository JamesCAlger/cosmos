"""
Optimization Task Definition

Defines what to optimize for a single component.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional


@dataclass
class OptimizationTask:
    """
    Defines a single component optimization task

    This encapsulates everything needed to optimize one component:
    - Which component to optimize
    - What parameter space to search
    - How to evaluate configurations
    - How many evaluations to perform
    - Context from upstream components
    """

    component_id: str
    """Component identifier (e.g., 'chunker', 'retriever', 'generator')"""

    search_space: Dict[str, Any]
    """
    Parameter search space
    Format: {
        'param_name': [value1, value2, ...],  # Categorical
        'param_name': (min, max),              # Continuous range
    }
    """

    evaluator: Callable[[Dict[str, Any]], float]
    """
    Evaluation function
    Input: Configuration dict
    Output: Quality score [0, 1] where higher is better
    """

    budget: int
    """Number of configurations to evaluate"""

    context: Optional[Dict[str, Any]] = field(default_factory=dict)
    """
    Context from upstream optimizations
    Contains results, best configs, or component instances
    """

    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Additional metadata about the task"""

    def __repr__(self) -> str:
        return f"OptimizationTask(component={self.component_id}, budget={self.budget}, search_space_size={len(self.search_space)})"