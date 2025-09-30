"""
COSMOS Optimization Module

Provides optimization framework for compositional RAG systems.
"""

from .evaluators import ComponentEvaluator, build_component
from .task import OptimizationTask
from .strategy import OptimizationStrategy
from .bayesian_strategy import BayesianStrategy
from .random_strategy import RandomSearchStrategy
from .compositional_optimizer import CompositionalOptimizer, CompositionalOptimizerBuilder

__all__ = [
    'ComponentEvaluator',
    'build_component',
    'OptimizationTask',
    'OptimizationStrategy',
    'BayesianStrategy',
    'RandomSearchStrategy',
    'CompositionalOptimizer',
    'CompositionalOptimizerBuilder'
]