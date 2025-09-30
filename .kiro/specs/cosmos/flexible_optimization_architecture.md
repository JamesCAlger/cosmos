# COSMOS-Enabled Flexible Optimization Architecture

## Executive Summary

This document outlines a **COSMOS-aligned, research-oriented architecture** for optimizing compositional ML systems. The design combines:
1. **COSMOS Foundation**: Component-intrinsic metrics, interface contracts, hierarchical optimization
2. **Strategy Pattern**: Pluggable algorithms (Bayesian, Genetic, RL-based) for research flexibility
3. **Compositional Decomposition**: Micro/meso/macro optimization levels reduce complexity from O(∏|Cᵢ|) to O(Σ|Cᵢ|)

**Core Decoupling**:
- **What to optimize**: COSMOS components with self-contained metrics
- **How to optimize**: Pluggable optimization strategies
- **In what order**: Orchestration strategies (forward, backward, inside-out)
- **At what level**: Hierarchical scope (component, workflow, system)

**Key Philosophy:** Build COSMOS abstractions from day 1, use RAG components to validate, maintain research flexibility for algorithm comparison.

---

## Motivation: Research Flexibility

### The Problem
Current implementation:
- Bayesian optimization as the only algorithm (limits research)
- Forward (input→output) as the only optimization order
- Component-level as the only optimization scope
- **External evaluators** instead of component-intrinsic metrics
- **No interface contracts** for composition validation
- **Single-objective** only (ignoring cost/latency tradeoffs)
- **Monolithic optimization** (O(∏|Cᵢ|) complexity)

### The Research Questions
1. **Which algorithm works best?** Bayesian, Genetic, CMA-ES, or hybrid approaches?
2. **Does optimization order matter?** Forward vs backward vs inside-out?
3. **What's the optimal scope?** Independent components vs joint workflows?
4. **Can hierarchical optimization reduce complexity?** Micro/meso/macro levels?
5. **How to handle multi-objective tradeoffs?** Accuracy vs latency vs cost?
6. **Can bounded agentic components improve performance?** Internal strategy selection?
7. **Does transfer learning across domains work?** Pattern reuse?

### The Solution
**COSMOS-aligned architecture** with:
- ✅ **Component-intrinsic metrics** (self-reporting performance)
- ✅ **Interface contracts** (type-safe composition)
- ✅ **Hierarchical optimization** (component → workflow → system)
- ✅ **Multi-objective** from day 1 (Pareto frontiers)
- ✅ **Pluggable algorithms** (research flexibility)
- ✅ **Bounded agency support** (agentic components)
- ✅ **Transfer learning hooks** (domain adaptation)

**Benefits**:
- Same evaluation infrastructure for fair comparisons
- MLflow tracking for all experiments
- True bridge to COSMOS generalization
- Research flexibility maintained

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    COSMOS-Enabled Compositional Optimizer                 │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   COSMOS Component Layer                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │    │
│  │  │  Component   │  │  Component   │  │  Agentic     │          │    │
│  │  │  + Metrics   │  │  + Metrics   │  │  Component   │          │    │
│  │  │  + Interface │  │  + Interface │  │  + Strategies│          │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                        │                        │            │
│           └────────────────────────┴────────────────────────┘            │
│                                    │                                     │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │              Hierarchical Optimization Controller                │   │
│  │                                                                   │   │
│  │  Level 1: MICRO   → Component parameters                         │   │
│  │  Level 2: MESO    → Workflow configurations                      │   │
│  │  Level 3: MACRO   → Pipeline assembly                            │   │
│  └───────────────────────────────────────────────────────────────────   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────┐         ┌───────────────────────┐            │
│  │  Orchestration       │         │   Algorithm           │            │
│  │  Strategy            │────────►│   Strategy            │            │
│  │                      │         │                       │            │
│  │  • Forward           │         │  • Bayesian           │            │
│  │  • Backward          │         │  • Genetic            │            │
│  │  • Inside-Out        │         │  • CMA-ES             │            │
│  │  • Workflow-First    │         │  • Hybrid             │            │
│  └──────────────────────┘         │  • Your Novel Algo    │            │
│           │                        └───────────────────────┘            │
│           │                                   │                         │
│           ▼                                   ▼                         │
│  ┌────────────────────────────────────────────────────────┐            │
│  │         Multi-Objective Optimization Tasks             │            │
│  │  Task = Component × SearchSpace × Objectives × Budget  │            │
│  │  Objectives = [Accuracy, Latency, Cost]                │            │
│  └────────────────────────────────────────────────────────┘            │
│                           │                                             │
│                           ▼                                             │
│  ┌────────────────────────────────────────────────────────┐            │
│  │       MLflow Experiment Tracker + Transfer DB          │            │
│  │  • Log all configurations + domain features            │            │
│  │  • Log multi-objective metrics                         │            │
│  │  • Store optimization patterns for transfer            │            │
│  │  • Generate Pareto frontiers                           │            │
│  └────────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core Abstractions

### 0. COSMOS Component (Foundation)
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type
from dataclasses import dataclass

@dataclass
class InterfaceContract:
    """Defines component input/output contract"""
    input_type: Type
    output_type: Type
    required_metrics: List[str]
    invariants: List[Callable[[Any], bool]]  # e.g., ordering preserved

class COSMOSComponent(ABC):
    """
    Base class for all COSMOS-compatible components.
    Components are self-contained with intrinsic metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = []

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Core processing logic"""
        pass

    @abstractmethod
    def process_with_metrics(self, input_data: Any) -> Tuple[Any, Dict[str, float]]:
        """
        Process and return intrinsic metrics.

        Returns:
            (output, metrics) where metrics includes:
            - Quality metrics (accuracy, precision, etc.)
            - Efficiency metrics (latency, throughput)
            - Cost metrics (API calls, tokens)
        """
        pass

    @abstractmethod
    def get_config_space(self) -> Dict[str, Any]:
        """
        Return optimization parameter space.

        Format:
        {
            'param_name': [val1, val2, val3],  # Categorical
            'param_name2': (low, high),         # Continuous/Integer
        }
        """
        pass

    @abstractmethod
    def get_interface_spec(self) -> InterfaceContract:
        """Return interface contract for composition validation"""
        pass

    def validate_compatibility(self, upstream: 'COSMOSComponent') -> float:
        """
        Check compatibility with upstream component.

        Returns:
            Compatibility score [0, 1] where:
            - 1.0 = fully compatible
            - 0.0 = incompatible
        """
        my_spec = self.get_interface_spec()
        upstream_spec = upstream.get_interface_spec()

        # Check type compatibility
        if my_spec.input_type != upstream_spec.output_type:
            return 0.0

        # Check metric availability
        missing_metrics = set(my_spec.required_metrics) - set(upstream_spec.required_metrics)
        metric_score = 1.0 - (len(missing_metrics) / max(len(my_spec.required_metrics), 1))

        return metric_score

    def update_config(self, new_config: Dict[str, Any]):
        """Update component configuration"""
        self.config.update(new_config)
```

**Why This is Critical:**
- **Intrinsic metrics**: Components self-report performance (not external evaluators)
- **Interface contracts**: Type-safe composition and validation
- **Self-contained**: Each component knows its own optimization space
- **COSMOS-ready**: Matches COSMOS architecture exactly

### 1. Optimization Task
```python
from enum import Enum

class OptimizationScope(Enum):
    """Hierarchical optimization levels"""
    MICRO = "component"     # Single component parameters
    MESO = "workflow"       # Workflow configurations
    MACRO = "system"        # Pipeline assembly

@dataclass
class MultiObjective:
    """Multi-objective optimization specification"""
    metrics: List[str]              # e.g., ['accuracy', 'latency', 'cost']
    weights: List[float]            # Relative importance
    minimize: List[bool]            # True if metric should be minimized

    def score(self, metrics_dict: Dict[str, float]) -> float:
        """Compute weighted score"""
        total = 0.0
        for metric, weight, should_minimize in zip(self.metrics, self.weights, self.minimize):
            value = metrics_dict.get(metric, 0.0)
            total += weight * (-value if should_minimize else value)
        return total

    def compute_pareto_frontier(self, results: List[Tuple[Dict, Dict]]) -> List[Tuple[Dict, Dict]]:
        """
        Compute Pareto-optimal configurations.

        Args:
            results: List of (config, metrics) tuples

        Returns:
            List of non-dominated (config, metrics) tuples
        """
        pareto_front = []
        for config, metrics in results:
            dominated = False
            for _, other_metrics in pareto_front:
                if self._dominates(other_metrics, metrics):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append((config, metrics))
        return pareto_front

    def _dominates(self, metrics1: Dict, metrics2: Dict) -> bool:
        """Check if metrics1 dominates metrics2"""
        better_in_one = False
        for metric, should_minimize in zip(self.metrics, self.minimize):
            val1, val2 = metrics1[metric], metrics2[metric]
            if should_minimize:
                if val1 > val2:
                    return False
                if val1 < val2:
                    better_in_one = True
            else:
                if val1 < val2:
                    return False
                if val1 > val2:
                    better_in_one = True
        return better_in_one

@dataclass
class OptimizationTask:
    """A unit of optimization work"""
    scope: OptimizationScope        # Micro/meso/macro
    component: COSMOSComponent      # Component to optimize (not just ID!)
    objectives: MultiObjective      # Multi-objective specification
    budget: int                     # Number of evaluations
    context: Dict = None            # Shared state (e.g., upstream components)
    workflow_id: str = None         # For meso-level optimization
```

**Example:**
```python
chunker_task = OptimizationTask(
    scope=OptimizationScope.MICRO,
    component=cosmos_chunker,  # COSMOSComponent instance
    objectives=MultiObjective(
        metrics=['chunk_quality', 'processing_time'],
        weights=[0.7, 0.3],
        minimize=[False, True]
    ),
    budget=20
)
```

### 2. Optimization Strategy
```python
class OptimizationStrategy(ABC):
    """Abstract optimization algorithm"""

    @abstractmethod
    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        """Execute optimization and return best configuration"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Strategy name for logging"""
        pass
```

**Concrete Strategies:**
- `BayesianStrategy` - Gaussian Process optimization
- `GeneticStrategy` - Genetic algorithm
- `RandomSearchStrategy` - Baseline random sampling
- `HybridStrategy` - Combine multiple algorithms
- **Your research:** Custom algorithms

### 3. Orchestration Strategy
```python
class OrchestrationStrategy(ABC):
    """How to order and group optimization tasks"""

    @abstractmethod
    def plan(self,
             components: List[str],
             dependencies: Dict[str, List[str]]) -> List[OptimizationTask]:
        """Generate task sequence based on pipeline structure"""
        pass
```

**Concrete Orchestrators:**
- `ForwardOrchestrator` - Optimize input→output order
- `BackwardOrchestrator` - Optimize output→input order (backprop-inspired)
- `InsideOutOrchestrator` - Optimize critical components first
- `WorkflowGroupOrchestrator` - Optimize tightly-coupled groups jointly
- **Your research:** Novel ordering strategies

### 4. Compositional Optimizer
```python
class CompositionalOptimizer:
    """Combines orchestration + algorithm + tracking"""

    def __init__(self,
                 orchestration_strategy: OrchestrationStrategy,
                 algorithm_strategy: OptimizationStrategy,
                 tracker: ExperimentTracker):
        self.orchestrator = orchestration_strategy
        self.algorithm = algorithm_strategy
        self.tracker = tracker

    def optimize(self,
                 components: List[str],
                 dependencies: Dict[str, List[str]],
                 total_budget: int) -> Dict[str, OptimizationResult]:
        """Execute compositional optimization with tracking"""
        # 1. Generate optimization plan
        tasks = self.orchestrator.plan(components, dependencies)

        # 2. Allocate budget
        self._allocate_budget(tasks, total_budget)

        # 3. Execute each task with algorithm strategy
        for task in tasks:
            result = self.algorithm.optimize(task)
            self.tracker.log_trial(task.target_id, result)

        # 4. Return all component configurations
        return self.results
```

---

## Implementation Roadmap

### **Phase 0: COSMOS Component Foundation (Week 1 - Critical!)**

**Goal:** Build COSMOS-compatible component abstraction before optimization strategies.

**Why First:** All optimization depends on component-intrinsic metrics and interface contracts.

#### 0.1 Create Base COSMOS Component (3 days)

**File:** `autorag/components/cosmos_base.py`
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type, List, Callable
from dataclasses import dataclass
from enum import Enum
import time

@dataclass
class InterfaceContract:
    """Component interface specification"""
    input_type: Type
    output_type: Type
    required_metrics: List[str]
    invariants: List[Callable[[Any], bool]] = None

    def validate_input(self, data: Any) -> bool:
        """Validate input matches contract"""
        return isinstance(data, self.input_type)

    def validate_output(self, data: Any) -> bool:
        """Validate output matches contract"""
        return isinstance(data, self.output_type)

class COSMOSComponent(ABC):
    """Base class for all optimizable components"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_history = []
        self._interface_spec = self.get_interface_spec()

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Core processing logic"""
        pass

    def process_with_metrics(self, input_data: Any) -> Tuple[Any, Dict[str, float]]:
        """
        Process and collect metrics.
        Default implementation adds timing, subclasses add quality metrics.
        """
        # Validate input
        if not self._interface_spec.validate_input(input_data):
            raise ValueError(f"Input type mismatch: expected {self._interface_spec.input_type}")

        # Process with timing
        start_time = time.time()
        result = self.process(input_data)
        processing_time = time.time() - start_time

        # Collect metrics
        metrics = {
            'latency': processing_time,
            **self._compute_quality_metrics(input_data, result)
        }

        # Validate output
        if not self._interface_spec.validate_output(result):
            raise ValueError(f"Output type mismatch: expected {self._interface_spec.output_type}")

        # Store metrics
        self.metrics_history.append(metrics)

        return result, metrics

    @abstractmethod
    def _compute_quality_metrics(self, input_data: Any, result: Any) -> Dict[str, float]:
        """Compute component-specific quality metrics"""
        pass

    @abstractmethod
    def get_config_space(self) -> Dict[str, Any]:
        """Return optimization parameter space"""
        pass

    @abstractmethod
    def get_interface_spec(self) -> InterfaceContract:
        """Return interface contract"""
        pass

    def validate_compatibility(self, upstream: 'COSMOSComponent') -> float:
        """Check compatibility with upstream component"""
        my_spec = self.get_interface_spec()
        upstream_spec = upstream.get_interface_spec()

        # Type compatibility
        if my_spec.input_type != upstream_spec.output_type:
            return 0.0

        # Metric availability
        if my_spec.required_metrics:
            available = set(upstream_spec.required_metrics or [])
            required = set(my_spec.required_metrics)
            missing = required - available
            metric_score = 1.0 - (len(missing) / len(required)) if required else 1.0
        else:
            metric_score = 1.0

        return metric_score

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(new_config)

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics from history"""
        if not self.metrics_history:
            return {}

        import numpy as np
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history]
            avg_metrics[key] = float(np.mean(values))

        return avg_metrics
```

#### 0.2 Wrap Existing RAG Components (3 days)

**File:** `autorag/components/cosmos_wrappers.py`
```python
from typing import List
from .cosmos_base import COSMOSComponent, InterfaceContract
from ..components.chunkers.base import Chunker
from ..components.retrievers.base import Retriever

class COSMOSChunker(COSMOSComponent):
    """COSMOS wrapper for existing chunker components"""

    def __init__(self, base_chunker: Chunker, config: Dict = None):
        super().__init__(config)
        self.chunker = base_chunker
        if config:
            # Update base chunker config
            for key, value in config.items():
                setattr(self.chunker, key, value)

    def process(self, documents: List) -> List:
        """Chunk documents"""
        return self.chunker.chunk(documents)

    def _compute_quality_metrics(self, documents: List, chunks: List) -> Dict[str, float]:
        """Compute chunking quality metrics"""
        import numpy as np

        if not chunks:
            return {'chunk_quality': 0.0, 'num_chunks': 0, 'avg_chunk_size': 0}

        chunk_sizes = [len(c.content) for c in chunks]

        return {
            'num_chunks': len(chunks),
            'avg_chunk_size': float(np.mean(chunk_sizes)),
            'std_chunk_size': float(np.std(chunk_sizes)),
            'chunk_quality': self._heuristic_quality_score(chunk_sizes)
        }

    def _heuristic_quality_score(self, chunk_sizes: List[int]) -> float:
        """Heuristic quality based on size distribution"""
        import numpy as np

        avg_size = np.mean(chunk_sizes)
        std_size = np.std(chunk_sizes)

        # Prefer moderate sizes (200-400 chars) with low variance
        size_score = 1.0 - min(abs(avg_size - 300) / 300, 1.0)
        consistency_score = 1.0 - min(std_size / 100, 1.0)

        return 0.6 * size_score + 0.4 * consistency_score

    def get_config_space(self) -> Dict[str, Any]:
        """Return chunker config space"""
        return {
            'chunk_size': (128, 512),
            'overlap': (0, 100),
            'strategy': ['fixed', 'semantic', 'sliding_window']
        }

    def get_interface_spec(self) -> InterfaceContract:
        """Return interface contract"""
        return InterfaceContract(
            input_type=list,  # List of documents
            output_type=list,  # List of chunks
            required_metrics=['num_chunks', 'avg_chunk_size', 'chunk_quality']
        )

class COSMOSRetriever(COSMOSComponent):
    """COSMOS wrapper for retriever components"""

    def __init__(self, base_retriever: Retriever, config: Dict = None):
        super().__init__(config)
        self.retriever = base_retriever
        if config:
            for key, value in config.items():
                setattr(self.retriever, key, value)

    def process(self, query: str) -> List:
        """Retrieve relevant chunks"""
        top_k = self.config.get('top_k', 5)
        return self.retriever.retrieve(query, top_k=top_k)

    def _compute_quality_metrics(self, query: str, results: List) -> Dict[str, float]:
        """Compute retrieval quality metrics"""
        import numpy as np

        if not results:
            return {'num_results': 0, 'avg_score': 0.0, 'retrieval_quality': 0.0}

        scores = [r.score for r in results]

        return {
            'num_results': len(results),
            'avg_score': float(np.mean(scores)),
            'min_score': float(min(scores)),
            'max_score': float(max(scores)),
            'score_variance': float(np.var(scores)),
            'retrieval_quality': float(np.mean(scores))  # Simplified
        }

    def get_config_space(self) -> Dict[str, Any]:
        """Return retriever config space"""
        return {
            'top_k': [3, 5, 10, 20],
            'method': ['dense', 'sparse', 'hybrid'],
            'rerank': [True, False]
        }

    def get_interface_spec(self) -> InterfaceContract:
        """Return interface contract"""
        return InterfaceContract(
            input_type=str,   # Query
            output_type=list, # Retrieved chunks
            required_metrics=['num_results', 'avg_score', 'retrieval_quality']
        )
```

#### 0.3 Test COSMOS Components (1 day)

**File:** `tests/test_cosmos_components.py`
```python
import pytest
from autorag.components.cosmos_wrappers import COSMOSChunker, COSMOSRetriever
from autorag.components.chunkers.fixed import FixedChunker
from autorag.components.base import Document

def test_cosmos_chunker_metrics():
    """Test that chunker reports intrinsic metrics"""
    base_chunker = FixedChunker({'chunk_size': 200, 'overlap': 50})
    cosmos_chunker = COSMOSChunker(base_chunker)

    documents = [Document(content="Test document " * 50, doc_id="1")]

    chunks, metrics = cosmos_chunker.process_with_metrics(documents)

    # Check metrics exist
    assert 'latency' in metrics
    assert 'num_chunks' in metrics
    assert 'chunk_quality' in metrics
    assert metrics['num_chunks'] > 0

def test_interface_compatibility():
    """Test interface validation between components"""
    chunker = COSMOSChunker(FixedChunker({}))

    # Get interface specs
    chunker_spec = chunker.get_interface_spec()

    assert chunker_spec.input_type == list
    assert chunker_spec.output_type == list
    assert 'num_chunks' in chunker_spec.required_metrics

def test_config_space():
    """Test config space definition"""
    chunker = COSMOSChunker(FixedChunker({}))

    config_space = chunker.get_config_space()

    assert 'chunk_size' in config_space
    assert 'overlap' in config_space
    assert isinstance(config_space['chunk_size'], tuple)  # Range
    assert isinstance(config_space['strategy'], list)     # Categorical
```

**Success Criteria:**
- ✅ COSMOS components self-report metrics
- ✅ Interface contracts validate composition
- ✅ Existing RAG components wrapped and functional
- ✅ Tests pass

---

### **Phase 1: Multi-Objective Strategy Abstractions (Week 2)**

**Goal:** Create optimization strategy abstractions with multi-objective support.

#### 1.1 Create Base Classes (2 days)

**File:** `autorag/optimization/strategy.py`
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from dataclasses import dataclass

@dataclass
class OptimizationTask:
    """Minimal task definition"""
    target_id: str
    search_space: Dict[str, Any]
    evaluator: Callable[[Dict], float]
    budget: int
    context: Dict[str, Any] = None

@dataclass
class OptimizationResult:
    """Minimal result definition"""
    best_config: Dict[str, Any]
    best_score: float
    all_configs: List[Dict[str, Any]]
    all_scores: List[float]
    metadata: Dict[str, Any]

class OptimizationStrategy(ABC):
    """Base strategy interface"""

    @abstractmethod
    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
```

#### 1.2 Implement 2 Simple Strategies (3 days)

**File:** `autorag/optimization/strategies/bayesian.py`
```python
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from ..strategy import OptimizationStrategy, OptimizationTask, OptimizationResult

class BayesianStrategy(OptimizationStrategy):
    """Bayesian optimization using scikit-optimize"""

    def __init__(self, n_initial_points: int = 5):
        self.n_initial_points = n_initial_points

    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        # Convert search space to skopt format
        dimensions = self._build_dimensions(task.search_space)

        # Track evaluations
        configs, scores = [], []

        @use_named_args(dimensions)
        def objective(**params):
            config = self._params_to_config(params, task.search_space)
            score = task.evaluator(config)
            configs.append(config)
            scores.append(score)
            return -score  # Minimize

        # Run optimization
        gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=task.budget,
            n_initial_points=self.n_initial_points,
            random_state=42
        )

        best_idx = scores.index(max(scores))
        return OptimizationResult(
            best_config=configs[best_idx],
            best_score=scores[best_idx],
            all_configs=configs,
            all_scores=scores,
            metadata={'strategy': 'bayesian'}
        )

    def get_name(self) -> str:
        return "Bayesian"
```

**File:** `autorag/optimization/strategies/random.py`
```python
import random
from ..strategy import OptimizationStrategy, OptimizationTask, OptimizationResult

class RandomSearchStrategy(OptimizationStrategy):
    """Baseline random search"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def optimize(self, task: OptimizationTask) -> OptimizationResult:
        random.seed(self.random_state)
        configs, scores = [], []

        for _ in range(task.budget):
            config = self._sample_random_config(task.search_space)
            score = task.evaluator(config)
            configs.append(config)
            scores.append(score)

        best_idx = scores.index(max(scores))
        return OptimizationResult(
            best_config=configs[best_idx],
            best_score=scores[best_idx],
            all_configs=configs,
            all_scores=scores,
            metadata={'strategy': 'random'}
        )

    def get_name(self) -> str:
        return "RandomSearch"

    def _sample_random_config(self, search_space: Dict) -> Dict:
        """Sample random configuration from search space"""
        config = {}
        for param_name, param_spec in search_space.items():
            if isinstance(param_spec, list):
                # Categorical
                config[param_name] = random.choice(param_spec)
            elif isinstance(param_spec, tuple) and len(param_spec) == 2:
                # Range
                low, high = param_spec
                if isinstance(low, int) and isinstance(high, int):
                    config[param_name] = random.randint(low, high)
                else:
                    config[param_name] = random.uniform(low, high)
        return config
```

**Why start with just 2?**
- Bayesian is your main algorithm
- Random is essential baseline
- Proves the abstraction works
- Can add more later (Genetic, CMA-ES, custom)

---

### **Phase 2: Simple Orchestration (Week 1-2 - Start Simple!)**

**Goal:** Support forward-order optimization only (easiest to reason about).

#### 2.1 Create Forward Orchestrator (2 days)

**File:** `autorag/optimization/orchestration.py`
```python
from abc import ABC, abstractmethod
from typing import List, Dict
from .strategy import OptimizationTask

class OrchestrationStrategy(ABC):
    """How to order optimization tasks"""

    @abstractmethod
    def plan(self,
             components: List[str],
             dependencies: Dict[str, List[str]],
             search_spaces: Dict[str, Dict],
             evaluators: Dict[str, Callable]) -> List[OptimizationTask]:
        pass

class ForwardOrchestrator(OrchestrationStrategy):
    """Optimize components in pipeline order (input → output)"""

    def plan(self, components, dependencies, search_spaces, evaluators):
        # Topological sort to get execution order
        ordered = self._topological_sort(components, dependencies)

        # Create optimization task for each component
        tasks = []
        for component_id in ordered:
            task = OptimizationTask(
                target_id=component_id,
                search_space=search_spaces[component_id],
                evaluator=evaluators[component_id],
                budget=0  # Will be allocated later
            )
            tasks.append(task)

        return tasks

    def _topological_sort(self, components, dependencies):
        """Standard topological sort"""
        visited = set()
        order = []

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in dependencies.get(node, []):
                visit(dep)
            order.append(node)

        for comp in components:
            visit(comp)

        return order
```

**Why forward only initially?**
- Matches intuition (how you'd manually tune)
- Easiest to debug
- Can add backward/inside-out later to test research hypotheses

---

### **Phase 3: Compositional Optimizer (Week 2)**

**Goal:** Wire everything together with tracking.

#### 3.1 Create Compositional Optimizer (3 days)

**File:** `autorag/optimization/compositional_optimizer.py`
```python
from typing import Dict, List
from loguru import logger
from .strategy import OptimizationTask, OptimizationResult
from .orchestration import OrchestrationStrategy
from .strategies.base import OptimizationStrategy
from .tracking import ExperimentTracker

class CompositionalOptimizer:
    """Combines orchestration + algorithm + tracking"""

    def __init__(self,
                 orchestration_strategy: OrchestrationStrategy,
                 algorithm_strategy: OptimizationStrategy,
                 tracker: ExperimentTracker):
        self.orchestrator = orchestration_strategy
        self.algorithm = algorithm_strategy
        self.tracker = tracker
        self.results = {}

    def optimize(self,
                 components: List[str],
                 dependencies: Dict[str, List[str]],
                 search_spaces: Dict[str, Dict],
                 evaluators: Dict[str, Callable],
                 total_budget: int) -> Dict[str, OptimizationResult]:
        """
        Execute compositional optimization

        Args:
            components: List of component IDs
            dependencies: Dict mapping component_id -> [upstream_ids]
            search_spaces: Dict mapping component_id -> search_space
            evaluators: Dict mapping component_id -> evaluation_function
            total_budget: Total number of evaluations across all components

        Returns:
            Dict mapping component_id -> OptimizationResult
        """
        logger.info("Starting compositional optimization")

        # Log experiment configuration
        self.tracker.log_params({
            'orchestration_strategy': self.orchestrator.__class__.__name__,
            'algorithm_strategy': self.algorithm.get_name(),
            'total_budget': total_budget,
            'num_components': len(components)
        })

        # Phase 1: Generate optimization plan
        tasks = self.orchestrator.plan(
            components,
            dependencies,
            search_spaces,
            evaluators
        )
        logger.info(f"Generated {len(tasks)} optimization tasks")

        # Phase 2: Allocate budget across tasks
        self._allocate_budget(tasks, total_budget)

        # Phase 3: Execute tasks sequentially
        for i, task in enumerate(tasks):
            logger.info(f"Optimizing {task.target_id} ({i+1}/{len(tasks)})")

            # Inject context from previous results
            task.context = {'previous_results': self.results}

            # Run optimization with tracking
            with self.tracker.start_run(run_name=f"{task.target_id}_optimization"):
                result = self.algorithm.optimize(task)

                # Log results
                self.tracker.log_params({
                    'component': task.target_id,
                    'budget': task.budget
                })
                self.tracker.log_params(result.best_config)
                self.tracker.log_metrics({
                    'best_score': result.best_score,
                    'mean_score': np.mean(result.all_scores),
                    'std_score': np.std(result.all_scores)
                })

            # Store results
            self.results[task.target_id] = result
            logger.info(f"Best score for {task.target_id}: {result.best_score:.4f}")

        # Phase 4: Return all results
        logger.info("Compositional optimization complete")
        return self.results

    def _allocate_budget(self, tasks: List[OptimizationTask], total_budget: int):
        """Simple uniform budget allocation"""
        budget_per_task = total_budget // len(tasks)
        for task in tasks:
            task.budget = budget_per_task

        logger.info(f"Allocated {budget_per_task} evaluations per component")
```

#### 3.2 Create Domain-Agnostic Tracker (2 days)

**File:** `autorag/optimization/tracking.py`
```python
import mlflow
from mlflow.tracking import MlflowClient
from contextlib import contextmanager
from typing import Dict, Any

class ExperimentTracker:
    """Domain-agnostic MLflow wrapper"""

    def __init__(self, experiment_name: str, tracking_uri: str = "file:./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        self.client = MlflowClient(tracking_uri)
        self.experiment_name = experiment_name

    @contextmanager
    def start_run(self, run_name: str = None):
        """Context manager for MLflow runs"""
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=True
        )
        try:
            yield run
        finally:
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters (flattens nested dicts)"""
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics"""
        mlflow.log_metrics(metrics)

    def log_dict(self, data: Dict, filename: str):
        """Log dictionary as JSON artifact"""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            temp_path = f.name

        mlflow.log_artifact(temp_path, "data")
        os.unlink(temp_path)

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
```

---

### **Phase 4: RAG-Specific Integration (Week 2-3)**

**Goal:** Test the architecture with actual RAG components.

#### 4.1 Create Component Evaluators (3 days)

**File:** `autorag/optimization/rag/component_evaluators.py`
```python
from typing import Dict, List, Callable
import numpy as np
from loguru import logger

class ComponentEvaluatorFactory:
    """Creates evaluators for RAG components"""

    def __init__(self, pipeline_config: Dict, test_data: Dict):
        self.pipeline_config = pipeline_config
        self.test_data = test_data
        self.semantic_metrics = None  # Initialize when needed

    def create_chunker_evaluator(self) -> Callable:
        """
        Evaluator for chunker component
        Measures: chunk quality (coherence, size distribution)
        """
        def evaluate_chunker(config: Dict) -> float:
            from ...components.chunkers.fixed import FixedChunker
            from ...components.chunkers.semantic import SemanticChunker

            # Create chunker with config
            if config['strategy'] == 'fixed':
                chunker = FixedChunker(config)
            elif config['strategy'] == 'semantic':
                chunker = SemanticChunker(config)
            else:
                raise ValueError(f"Unknown strategy: {config['strategy']}")

            # Chunk test documents
            from ...components.base import Document
            documents = [Document(content=doc, doc_id=str(i))
                        for i, doc in enumerate(self.test_data['documents'][:10])]
            chunks = chunker.chunk(documents)

            # Compute metrics
            chunk_sizes = [len(c.content) for c in chunks]

            # Quality score (heuristic - can be improved)
            avg_size = np.mean(chunk_sizes)
            std_size = np.std(chunk_sizes)

            # Prefer moderate-sized chunks with low variance
            size_score = 1.0 - abs(avg_size - 300) / 300  # Target ~300 chars
            consistency_score = 1.0 - min(std_size / 100, 1.0)  # Penalize high variance

            score = 0.6 * size_score + 0.4 * consistency_score

            logger.debug(f"Chunker config {config}: score={score:.3f}")
            return score

        return evaluate_chunker

    def create_retriever_evaluator(self, chunker_config: Dict) -> Callable:
        """
        Evaluator for retriever component
        Measures: retrieval accuracy (uses ground truth)
        """
        def evaluate_retriever(config: Dict) -> float:
            # Build pipeline with chunker + embedder + retriever
            from ...pipeline.orchestrator import PipelineOrchestrator

            pipeline = PipelineOrchestrator()
            # ... setup pipeline with configs ...

            # Index documents
            documents = [...]
            pipeline.index_documents(documents)

            # Evaluate on queries
            accuracies = []
            for query_item in self.test_data['queries'][:10]:
                results = pipeline.components['retriever'].retrieve(
                    query_item['query'],
                    top_k=config.get('top_k', 5)
                )

                # Check if relevant document is in results
                relevant_doc_id = query_item.get('relevant_doc_id')
                retrieved_ids = [r.chunk.doc_id for r in results]

                accuracy = 1.0 if relevant_doc_id in retrieved_ids else 0.0
                accuracies.append(accuracy)

            score = np.mean(accuracies)
            logger.debug(f"Retriever config {config}: score={score:.3f}")
            return score

        return evaluate_retriever

    def create_generator_evaluator(self, upstream_configs: Dict) -> Callable:
        """
        Evaluator for generator component
        Measures: answer quality (semantic similarity to ground truth)
        """
        def evaluate_generator(config: Dict) -> float:
            if self.semantic_metrics is None:
                from ...evaluation.semantic_metrics import SemanticMetrics
                self.semantic_metrics = SemanticMetrics()

            # Build full pipeline
            from ...pipeline.orchestrator import PipelineOrchestrator
            pipeline = PipelineOrchestrator()
            # ... setup with all upstream configs ...

            # Evaluate end-to-end
            similarities = []
            for query_item in self.test_data['queries'][:10]:
                result = pipeline.query(query_item['query'])

                similarity = self.semantic_metrics.similarity_score(
                    result['output'],
                    query_item['expected_answer']
                )
                similarities.append(similarity)

            score = np.mean(similarities)
            logger.debug(f"Generator config {config}: score={score:.3f}")
            return score

        return evaluate_generator
```

#### 4.2 Create Experiment Script (2 days)

**File:** `experiments/test_compositional_optimization.py`
```python
#!/usr/bin/env python
"""
Test compositional optimization with RAG components
Start small: optimize 3 components with 2 strategies
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autorag.optimization.compositional_optimizer import CompositionalOptimizer
from autorag.optimization.orchestration import ForwardOrchestrator
from autorag.optimization.strategies.bayesian import BayesianStrategy
from autorag.optimization.strategies.random import RandomSearchStrategy
from autorag.optimization.tracking import ExperimentTracker
from autorag.optimization.rag.component_evaluators import ComponentEvaluatorFactory

def load_test_data():
    """Load small test dataset"""
    # For now, use minimal data
    return {
        'documents': [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text.",
            # ... add 10-20 documents
        ],
        'queries': [
            {
                'query': "What is machine learning?",
                'expected_answer': "Machine learning is a subset of artificial intelligence.",
                'relevant_doc_id': "0"
            },
            # ... add 5-10 queries
        ]
    }

def main():
    print("=" * 60)
    print("Compositional Optimization Test - RAG Components")
    print("=" * 60)

    # Load test data
    test_data = load_test_data()
    print(f"Loaded {len(test_data['documents'])} documents, {len(test_data['queries'])} queries")

    # Define components and dependencies
    components = ['chunker', 'retriever', 'generator']
    dependencies = {
        'chunker': [],
        'retriever': ['chunker'],
        'generator': ['retriever']
    }

    # Define search spaces (start simple!)
    search_spaces = {
        'chunker': {
            'strategy': ['fixed', 'semantic'],
            'chunk_size': (200, 400),
            'overlap': (0, 50)
        },
        'retriever': {
            'top_k': [3, 5, 10],
            'method': ['dense', 'hybrid']
        },
        'generator': {
            'temperature': (0.3, 0.9),
            'max_tokens': [256, 512]
        }
    }

    # Create component evaluators
    evaluator_factory = ComponentEvaluatorFactory(
        pipeline_config={},
        test_data=test_data
    )

    evaluators = {
        'chunker': evaluator_factory.create_chunker_evaluator(),
        'retriever': evaluator_factory.create_retriever_evaluator(
            chunker_config={}  # Will be updated with optimized config
        ),
        'generator': evaluator_factory.create_generator_evaluator(
            upstream_configs={}
        )
    }

    # Test 2 strategies
    strategies = [
        ('random', RandomSearchStrategy()),
        ('bayesian', BayesianStrategy(n_initial_points=5))
    ]

    for strategy_name, strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Testing Strategy: {strategy_name.upper()}")
        print(f"{'=' * 60}\n")

        # Create tracker
        tracker = ExperimentTracker(
            f"rag_compositional_{strategy_name}"
        )

        # Create optimizer
        optimizer = CompositionalOptimizer(
            orchestration_strategy=ForwardOrchestrator(),
            algorithm_strategy=strategy,
            tracker=tracker
        )

        # Run optimization (small budget for testing)
        results = optimizer.optimize(
            components=components,
            dependencies=dependencies,
            search_spaces=search_spaces,
            evaluators=evaluators,
            total_budget=30  # 10 evaluations per component
        )

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Results for {strategy_name.upper()}")
        print(f"{'=' * 60}\n")

        for component_id, result in results.items():
            print(f"{component_id}:")
            print(f"  Best Score: {result.best_score:.4f}")
            print(f"  Best Config: {result.best_config}")
            print(f"  Evaluations: {len(result.all_configs)}")
            print()

if __name__ == "__main__":
    main()
```

---

### **Phase 5: Compare Strategies (Week 3)**

**Goal:** Validate that the architecture enables fair comparisons.

#### 5.1 Create Comparison Script

**File:** `experiments/compare_optimization_strategies.py`
```python
#!/usr/bin/env python
"""
Compare different optimization strategies on RAG components
Research question: Which algorithm works best for component optimization?
"""

import numpy as np
import json
from pathlib import Path
from test_compositional_optimization import (
    load_test_data, components, dependencies, search_spaces, evaluators
)

from autorag.optimization.compositional_optimizer import CompositionalOptimizer
from autorag.optimization.orchestration import ForwardOrchestrator
from autorag.optimization.strategies.bayesian import BayesianStrategy
from autorag.optimization.strategies.random import RandomSearchStrategy
from autorag.optimization.tracking import ExperimentTracker

def compare_strategies(budget: int = 60):
    """
    Compare optimization strategies with same budget

    Args:
        budget: Total evaluations (distributed across components)
    """

    strategies = {
        'random': RandomSearchStrategy(),
        'bayesian_5init': BayesianStrategy(n_initial_points=5),
        'bayesian_10init': BayesianStrategy(n_initial_points=10),
    }

    results = {}

    for strategy_name, strategy in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {strategy_name}")
        print(f"{'=' * 60}\n")

        tracker = ExperimentTracker(f"strategy_comparison_{strategy_name}")

        optimizer = CompositionalOptimizer(
            orchestration_strategy=ForwardOrchestrator(),
            algorithm_strategy=strategy,
            tracker=tracker
        )

        strategy_results = optimizer.optimize(
            components=components,
            dependencies=dependencies,
            search_spaces=search_spaces,
            evaluators=evaluators,
            total_budget=budget
        )

        results[strategy_name] = strategy_results

    # Analyze results
    analysis = analyze_comparison(results)

    # Save comparison
    output_file = Path("results/strategy_comparison.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}\n")
    print(json.dumps(analysis, indent=2))

    return analysis

def analyze_comparison(results: dict) -> dict:
    """Analyze and rank strategies"""

    analysis = {}

    for strategy_name, strategy_results in results.items():
        # Aggregate component scores
        component_scores = {
            comp_id: result.best_score
            for comp_id, result in strategy_results.items()
        }

        # Overall metrics
        analysis[strategy_name] = {
            'component_scores': component_scores,
            'mean_score': np.mean(list(component_scores.values())),
            'min_score': min(component_scores.values()),
            'max_score': max(component_scores.values()),
            'total_evaluations': sum(
                len(result.all_configs)
                for result in strategy_results.values()
            )
        }

    # Rank strategies
    ranked = sorted(
        analysis.items(),
        key=lambda x: x[1]['mean_score'],
        reverse=True
    )

    analysis['ranking'] = [name for name, _ in ranked]

    return analysis

if __name__ == "__main__":
    compare_strategies(budget=60)
```

---

## Testing Strategy

### Week 1-2: Prove the Abstraction Works
```bash
# Step 1: Test with minimal components (chunker only)
python experiments/test_single_component.py --component chunker --budget 20

# Step 2: Test with 2 components (chunker + retriever)
python experiments/test_two_components.py --budget 40

# Step 3: Test full pipeline (3 components)
python experiments/test_compositional_optimization.py --budget 60
```

### Week 3: Compare Strategies
```bash
# Compare Random vs Bayesian
python experiments/compare_optimization_strategies.py --budget 60

# Analyze results in MLflow UI
mlflow ui --port 5000
# Open: http://localhost:5000
```

### Week 4+: Add New Strategies
```bash
# Add Genetic algorithm
# Implement in: autorag/optimization/strategies/genetic.py

# Test it
python experiments/compare_optimization_strategies.py --strategies random,bayesian,genetic
```

---

## Expected Results

### Hypothesis 1: Bayesian > Random
**Test:** Compare convergence speed
```
Random Search (20 evals):     Score = 0.65 ± 0.15
Bayesian (20 evals):          Score = 0.72 ± 0.08
```
**Expected:** Bayesian finds better configurations with less variance

### Hypothesis 2: Forward is Sufficient (Initially)
**Test:** Start with forward orchestration only
```
Forward Optimization:
  - Chunker optimized independently
  - Retriever optimized with fixed chunker
  - Generator optimized with fixed upstream
```
**Later Research:** Test backward orchestration to see if it improves

### Hypothesis 3: Component Optimization Reduces Search Space
**Test:** Compare joint vs compositional
```
Joint Optimization:    4,320 possible configs → sample 60
Compositional:         24 + 15 + 12 = 51 configs → evaluate all

Result: Compositional likely to find better configs
```

---

## Path to COSMOS

### What Transfers Directly ✅
1. **All strategy abstractions** - Work with any domain
2. **MLflow tracking** - Domain-agnostic
3. **Compositional optimizer** - Just needs new evaluators
4. **Comparison infrastructure** - Reusable

### What Needs Domain Adaptation ⚠️
1. **Component evaluators** - RAG-specific → Create ETL/video evaluators
2. **Search space definitions** - RAG params → Domain-specific params
3. **Test data loading** - MS MARCO → Domain datasets

### COSMOS Transition (Week 5+)
```python
# Replace RAG evaluator
from autorag.optimization.rag.component_evaluators import ComponentEvaluatorFactory

# With generic COSMOS evaluator
from cosmos.optimization.evaluators import COSMOSEvaluatorFactory

# Everything else stays the same!
optimizer = CompositionalOptimizer(
    orchestration_strategy=ForwardOrchestrator(),  # Same
    algorithm_strategy=BayesianStrategy(),         # Same
    tracker=ExperimentTracker("cosmos_experiment") # Same
)
```

---

## Success Metrics

### Phase 1 Success (Week 1-2)
- ✅ Can optimize single component with 2 strategies
- ✅ Strategies return consistent results
- ✅ MLflow logs all experiments
- ✅ Can switch strategies without changing other code

### Phase 2 Success (Week 3)
- ✅ Can optimize full RAG pipeline compositionally
- ✅ Results better than random baseline
- ✅ Can compare strategies fairly
- ✅ Clear winner emerges (likely Bayesian)

### Phase 3 Success (Week 4+)
- ✅ Can add new strategy in < 1 day
- ✅ Can test research ideas (backward optimization, hybrid algorithms)
- ✅ Easy to explain and reproduce experiments
- ✅ Ready to generalize to COSMOS

---

## File Structure

```
autorag/
├── optimization/
│   ├── strategy.py                    # Base abstractions (Phase 1)
│   ├── orchestration.py               # Orchestration strategies (Phase 2)
│   ├── compositional_optimizer.py     # Main optimizer (Phase 3)
│   ├── tracking.py                    # MLflow wrapper (Phase 3)
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── bayesian.py               # Bayesian strategy (Phase 1)
│   │   ├── random.py                 # Random baseline (Phase 1)
│   │   ├── genetic.py                # Future: Week 4+
│   │   └── cmaes.py                  # Future: Week 4+
│   └── rag/
│       ├── __init__.py
│       └── component_evaluators.py   # RAG evaluators (Phase 4)
│
experiments/
├── test_single_component.py          # Phase 1 test
├── test_two_components.py            # Phase 2 test
├── test_compositional_optimization.py # Phase 4 test
└── compare_optimization_strategies.py # Phase 5 comparison

results/
├── strategy_comparison.json
└── experiment_logs/

mlruns/                                # MLflow tracking data
```

---

## Research Questions Timeline

### Immediate (Week 1-3)
1. ✅ Does the abstraction work?
2. ✅ Is Bayesian better than Random for component optimization?
3. ✅ What's the right budget allocation per component?

### Near-term (Week 4-6)
4. Does forward vs backward orchestration matter?
5. Are genetic algorithms better for discrete parameters?
6. Can hybrid strategies combine strengths?

### Long-term (Week 7+)
7. Can we use gradient signals from downstream components?
8. Does component importance-based ordering improve results?
9. Can we transfer optimization knowledge across domains?

---

## Next Steps

### Immediate Action (Today)
1. Create directory structure
2. Implement base abstractions (`strategy.py`)
3. Implement `RandomSearchStrategy` (baseline)
4. Test on single component

### Week 1 Deliverables
- [ ] Base strategy abstractions working
- [ ] 2 strategies implemented (Random, Bayesian)
- [ ] Can optimize single component
- [ ] MLflow tracking integrated

### Week 2 Deliverables
- [ ] Forward orchestration working
- [ ] Can optimize 3-component pipeline
- [ ] Results logged to MLflow
- [ ] Basic analysis script

### Week 3 Deliverables
- [ ] Strategy comparison complete
- [ ] Clear results showing Bayesian > Random
- [ ] Ready to add new strategies
- [ ] Documentation of findings

---

## Key Principles

1. **Start Small** - 1 component, 2 strategies, small budget
2. **Test Incrementally** - Validate each phase before moving on
3. **Track Everything** - MLflow for all experiments
4. **Fair Comparisons** - Same budget, same data, same evaluation
5. **Research-First** - Architecture supports experimentation
6. **COSMOS-Ready** - Design for generalization from day 1

---

## Conclusion

This architecture enables **research-driven optimization** by decoupling algorithm choice from pipeline structure. Starting with simple RAG components proves the approach works before scaling to COSMOS-level architecture search.

**Core Innovation:** Strategy pattern lets you test different algorithms, orchestrations, and hybrid approaches without rewriting infrastructure.

**Path Forward:** Start with 2 strategies, validate on RAG, then expand to test research hypotheses about component optimization.