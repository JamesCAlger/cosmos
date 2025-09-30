"""
Optimization Strategy Abstraction

Defines interface for different optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from autorag.optimization.bayesian_search import OptimizationResult
from autorag.cosmos.optimization.task import OptimizationTask


class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization algorithms

    This allows swapping different optimization strategies
    (Bayesian, Genetic, Random, etc.) without changing other code.
    """

    @abstractmethod
    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        """
        Run optimization on the given task

        Args:
            task: OptimizationTask defining what to optimize

        Returns:
            OptimizationResult with best configuration and metrics
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get strategy name for logging

        Returns:
            Human-readable strategy name
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"