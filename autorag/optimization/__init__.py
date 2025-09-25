"""Configuration search and optimization module for Week 5"""

from .search_space import SearchSpace, ParameterRange, ComponentSearchSpace
from .config_generator import ConfigurationGenerator
from .grid_search import GridSearchOptimizer
from .result_manager import ResultManager
from .statistical_comparison import StatisticalComparison, StatisticalTest

__all__ = [
    "SearchSpace",
    "ParameterRange",
    "ComponentSearchSpace",
    "ConfigurationGenerator",
    "GridSearchOptimizer",
    "ResultManager",
    "StatisticalComparison",
    "StatisticalTest"
]