# Bayesian Optimization Implementation Guide for auto-RAG System

## Executive Summary

This guide provides a detailed roadmap for replacing the current grid search optimization in the auto-RAG system with Bayesian optimization. The implementation leverages the existing modular architecture while adding sophisticated multi-stage metrics collection and multi-objective optimization capabilities.

## Table of Contents
1. [Current System Analysis](#current-system-analysis)
2. [Implementation Architecture](#implementation-architecture)
3. [Phase 1: Minimal Implementation](#phase-1-minimal-implementation-3-4-days)
4. [Phase 2: Enhanced Metrics](#phase-2-enhanced-metrics-week-2)
5. [Phase 3: Production System](#phase-3-production-system-week-3-4)
6. [Integration Strategy](#integration-strategy)
7. [Migration Path](#migration-path)
8. [Technical Details](#technical-details)
9. [Testing Strategy](#testing-strategy)
10. [Performance Expectations](#performance-expectations)

## Current System Analysis

### Existing Components

#### 1. Grid Search Infrastructure (`autorag/optimization/grid_search.py`)
- **GridSearchOptimizer**: Systematic enumeration of configurations
- **Budget management**: Cost tracking and limits
- **Checkpointing**: Resume from interruptions
- **Parallel evaluation**: ThreadPoolExecutor for concurrent runs
- **Result tracking**: ResultManager for storing and analyzing results

#### 2. Search Space Definition (`autorag/optimization/search_space.py`)
- **ComponentSearchSpace**: Parameter ranges per component
- **ParameterRange**: Categorical, numerical, boolean parameters
- **Conditional parameters**: Dependencies between parameters
- **Total combinations**: ~960 configurations in default space

#### 3. Pipeline Architecture
- **ModularRAGPipeline** (`autorag/pipeline/rag_pipeline.py`): High-level interface
- **Component Registry**: Auto-registration of components
- **Base Components**: Chunker, Embedder, Retriever, Generator
- **Configuration Bridge**: Maps search space to component configs

#### 4. Current Metrics
- Simple binary metrics: retrieval_success, answer_generated
- RAGAS metrics when available: faithfulness, answer_relevancy, context_relevance
- Cost tracking: API costs per configuration
- Execution time: Total pipeline runtime

### Limitations of Current Approach

1. **Exhaustive Search**: Tests all configurations regardless of performance patterns
2. **Single-Stage Metrics**: Only measures final output quality
3. **No Learning**: Doesn't use results to guide search
4. **Limited Insights**: Minimal understanding of component interactions
5. **High Cost**: Evaluates poor configurations fully

## Implementation Architecture

### Core Components for Bayesian Optimization

```
autorag/
├── optimization/
│   ├── bayesian_search.py         # Main Bayesian optimizer
│   ├── surrogate_models.py        # Gaussian Process models
│   ├── acquisition.py             # Acquisition functions (EI, EHVI, UCB)
│   ├── multi_objective.py         # Pareto frontier tracking
│   └── search_space_converter.py  # Convert to skopt format
├── metrics/
│   ├── __init__.py
│   ├── stage_metrics.py           # Per-stage metric definitions
│   ├── collectors.py              # External metrics collection
│   ├── mixins.py                  # Component metric mixins
│   └── decorators.py              # Metric collection decorators
├── evaluation/
│   ├── multi_stage_evaluator.py   # Stage-wise evaluation orchestration
│   └── metric_aggregator.py       # Combine metrics across stages
└── visualization/
    ├── pareto_plot.py              # Pareto frontier visualization
    └── optimization_dashboard.py   # Real-time monitoring

```

### Detailed Component Specifications

#### 1. BayesianSearchOptimizer (`bayesian_search.py`)

```python
class BayesianSearchOptimizer:
    """
    Drop-in replacement for GridSearchOptimizer with Bayesian optimization.

    Key differences from grid search:
    - Iterative selection of configurations
    - Uses surrogate models to predict performance
    - Balances exploration and exploitation
    - Supports multi-objective optimization
    """

    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: Callable,
                 objectives: List[str] = ["accuracy"],
                 n_initial_points: int = 20,
                 n_calls: int = 100,
                 acquisition_func: str = "EI",
                 metrics_mode: str = "hybrid"):
        """
        Args:
            search_space: Same as grid search
            evaluator: Same interface as grid search
            objectives: List of objectives to optimize
            n_initial_points: Random exploration phase
            n_calls: Total optimization budget
            acquisition_func: EI, EHVI, UCB, or custom
            metrics_mode: none, external, internal, hybrid
        """
```

#### 2. Stage Metrics System (`metrics/stage_metrics.py`)

```python
# Metric definitions for each pipeline stage

CHUNKING_METRICS = {
    "quality": {
        "semantic_coherence": "Average similarity within chunks",
        "boundary_quality": "Clean sentence/paragraph boundaries",
        "information_density": "Content words ratio",
        "completeness_score": "Complete semantic units"
    },
    "efficiency": {
        "chunking_time": "Processing time",
        "chunks_created": "Output count",
        "avg_chunk_size": "Tokens per chunk",
        "size_variance": "Consistency"
    },
    "predictive": {
        "retrieval_difficulty": "Estimated retrieval challenge",
        "coverage_ratio": "Content preservation",
        "redundancy_score": "Information overlap"
    }
}

RETRIEVAL_METRICS = {
    "quality": {
        "recall@k": [5, 10, 20],
        "precision@k": [5, 10, 20],
        "mrr": "Mean Reciprocal Rank",
        "coverage": "Queries with results"
    },
    "efficiency": {
        "query_encoding_time": "Encoding latency",
        "search_time": "Search latency",
        "num_candidates": "Documents considered",
        "index_memory": "Memory usage"
    },
    "predictive": {
        "result_diversity": "Document variety",
        "confidence_scores": "Retrieval confidence",
        "relevance_gap": "Score separation"
    }
}

GENERATION_METRICS = {
    "quality": {
        "answer_relevance": "Query relevance",
        "answer_faithfulness": "Grounding score",
        "answer_completeness": "Coverage",
        "fluency_score": "Language quality"
    },
    "efficiency": {
        "generation_time": "Response time",
        "tokens_used": "Token consumption",
        "retry_count": "Attempts needed"
    },
    "cost": {
        "api_cost": "Direct costs",
        "token_cost": "Usage-based costs"
    }
}
```

## Phase 1: Minimal Implementation (3-4 days)

### Day 1-2: External Metrics Collection

#### Step 1: Create External Metrics Collector

**File**: `autorag/evaluation/external_metrics.py`

```python
class ExternalMetricsCollector:
    """
    Collects metrics without modifying existing components.
    This is the quickest way to add metrics to the pipeline.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.metrics = {}

    def evaluate_with_metrics(self, query, documents, ground_truth=None):
        """
        Wraps pipeline execution with metric collection.

        Returns:
            answer: Generated answer
            metrics: Dict with stage-wise metrics
        """
        metrics = {}

        # Chunking metrics
        start = time.time()
        chunks = self.pipeline.chunker.chunk(documents)
        metrics['chunking'] = {
            'time': time.time() - start,
            'chunks_count': len(chunks),
            'avg_size': np.mean([len(c.split()) for c in chunks]),
            'total_chars': sum(len(c) for c in chunks)
        }

        # Retrieval metrics
        start = time.time()
        retrieved = self.pipeline.retriever.retrieve(query, top_k=10)
        metrics['retrieval'] = {
            'time': time.time() - start,
            'docs_retrieved': len(retrieved),
            'top_score': retrieved[0].score if retrieved else 0,
            'score_spread': max(r.score for r in retrieved) - min(r.score for r in retrieved) if retrieved else 0
        }

        # Generation metrics
        start = time.time()
        answer = self.pipeline.generator.generate(query, retrieved[:5])
        metrics['generation'] = {
            'time': time.time() - start,
            'answer_length': len(answer.split()),
            'estimated_tokens': len(answer.split()) * 1.3  # Rough token estimate
        }

        # Aggregate metrics
        metrics['total'] = {
            'pipeline_time': sum(m.get('time', 0) for m in metrics.values()),
            'estimated_cost': self._estimate_cost(metrics)
        }

        return answer, metrics
```

#### Step 2: Create Simple Bayesian Optimizer

**File**: `autorag/optimization/bayesian_search.py`

```python
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

class SimpleBayesianOptimizer:
    """
    Initial Bayesian optimizer using scikit-optimize.
    Single-objective optimization for quick implementation.
    """

    def __init__(self, search_space, evaluator, n_calls=50):
        self.search_space = search_space
        self.evaluator = evaluator
        self.n_calls = n_calls
        self.results = []

    def optimize(self):
        """Run Bayesian optimization"""

        # Convert search space to skopt format
        skopt_space = self._convert_search_space()

        # Define objective function
        @use_named_args(skopt_space)
        def objective(**params):
            # Convert params to configuration
            config = self._params_to_config(params)

            # Evaluate configuration
            result = self.evaluator(config)

            # Store result
            self.results.append(result)

            # Return negative score for minimization
            return -result['metrics'].get('accuracy', 0)

        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=skopt_space,
            n_calls=self.n_calls,
            n_initial_points=10,
            acq_func='EI',
            random_state=42
        )

        return self._format_results(result)
```

### Day 3-4: Integration and Testing

#### Step 3: Integrate with Existing Pipeline

**Modification to**: `scripts/run_minimal_real_grid_search.py`

Create new file: `scripts/run_bayesian_search.py`

```python
# Key changes from grid search script:

# 1. Replace GridSearchOptimizer with BayesianSearchOptimizer
from autorag.optimization.bayesian_search import SimpleBayesianOptimizer

# 2. Wrap evaluator with metrics collector
from autorag.evaluation.external_metrics import ExternalMetricsCollector

def enhanced_evaluator(config):
    """Evaluator with metric collection"""
    pipeline = create_pipeline(config)
    collector = ExternalMetricsCollector(pipeline)

    # Run evaluation on test queries
    results = []
    for query, ground_truth in test_data:
        answer, metrics = collector.evaluate_with_metrics(
            query, documents, ground_truth
        )
        results.append({
            'answer': answer,
            'metrics': metrics,
            'ground_truth': ground_truth
        })

    # Aggregate results
    return aggregate_results(results)

# 3. Run Bayesian optimization
optimizer = SimpleBayesianOptimizer(
    search_space=search_space,
    evaluator=enhanced_evaluator,
    n_calls=50  # Much fewer than grid search's 960
)

best_config = optimizer.optimize()
```

## Phase 2: Enhanced Metrics (Week 2)

### Component Metrics Mixins

#### Step 1: Create Metric Mixins

**File**: `autorag/metrics/mixins.py`

```python
class ChunkerMetricsMixin:
    """
    Mixin to add metrics to any Chunker component.
    Use multiple inheritance to add to existing chunkers.
    """

    def chunk_with_metrics(self, documents):
        """Enhanced chunking with metrics collection"""
        start_time = time.time()
        start_memory = get_memory_usage()

        # Call original chunk method
        chunks = self.chunk(documents)

        # Collect metrics
        metrics = self._collect_chunk_metrics(
            documents, chunks, start_time, start_memory
        )

        return chunks, metrics

    def _collect_chunk_metrics(self, documents, chunks, start_time, start_memory):
        """Detailed chunk metrics"""
        metrics = {
            'efficiency': {
                'chunking_time': time.time() - start_time,
                'memory_used': get_memory_usage() - start_memory,
                'chunks_created': len(chunks),
                'compression_ratio': len(''.join(chunks)) / len(''.join(documents))
            },
            'quality': {
                'avg_chunk_size': np.mean([len(c.split()) for c in chunks]),
                'size_std': np.std([len(c.split()) for c in chunks]),
                'size_cv': np.std([len(c.split()) for c in chunks]) / np.mean([len(c.split()) for c in chunks])
            }
        }

        # Semantic coherence (requires embeddings)
        if hasattr(self, 'embedder'):
            embeddings = self.embedder.embed(chunks[:10])  # Sample for efficiency
            coherence = self._calculate_coherence(embeddings)
            metrics['quality']['semantic_coherence'] = coherence

        # Boundary quality
        metrics['quality']['clean_boundaries'] = sum(
            1 for c in chunks if c.rstrip().endswith(('.', '!', '?', '\n'))
        ) / len(chunks)

        return metrics

class RetrieverMetricsMixin:
    """Mixin for retriever metrics"""

    def retrieve_with_metrics(self, query, top_k=10, ground_truth=None):
        """Enhanced retrieval with metrics"""
        start_time = time.time()

        # Original retrieval
        results = self.retrieve(query, top_k)

        # Collect metrics
        metrics = {
            'efficiency': {
                'retrieval_time': time.time() - start_time,
                'candidates_scored': len(self.chunks) if hasattr(self, 'chunks') else 0
            },
            'quality': {
                'results_returned': len(results),
                'top_score': results[0].score if results else 0,
                'score_distribution': {
                    'mean': np.mean([r.score for r in results]),
                    'std': np.std([r.score for r in results]),
                    'min': min(r.score for r in results) if results else 0,
                    'max': max(r.score for r in results) if results else 0
                }
            }
        }

        # Calculate recall/precision if ground truth available
        if ground_truth:
            relevant_ids = set(ground_truth.get('relevant_chunks', []))
            retrieved_ids = set(r.id for r in results)

            metrics['quality']['recall@k'] = len(relevant_ids & retrieved_ids) / len(relevant_ids) if relevant_ids else 0
            metrics['quality']['precision@k'] = len(relevant_ids & retrieved_ids) / len(retrieved_ids) if retrieved_ids else 0

            # MRR (Mean Reciprocal Rank)
            for i, result in enumerate(results):
                if result.id in relevant_ids:
                    metrics['quality']['mrr'] = 1 / (i + 1)
                    break
            else:
                metrics['quality']['mrr'] = 0

        return results, metrics

class GeneratorMetricsMixin:
    """Mixin for generator metrics"""

    def generate_with_metrics(self, query, context):
        """Enhanced generation with metrics"""
        start_time = time.time()

        # Token counting (before)
        input_tokens = self._count_tokens(query + ' '.join(context))

        # Original generation
        answer = self.generate(query, context)

        # Token counting (after)
        output_tokens = self._count_tokens(answer)

        # Collect metrics
        metrics = {
            'efficiency': {
                'generation_time': time.time() - start_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            },
            'cost': {
                'estimated_cost': self._calculate_cost(input_tokens, output_tokens)
            },
            'quality': {
                'answer_length': len(answer.split()),
                'uses_context': self._check_context_usage(answer, context)
            }
        }

        # Advanced quality metrics (optional, requires additional API calls)
        if self.config.get('calculate_quality_metrics', False):
            metrics['quality'].update(self._calculate_quality_metrics(
                query, answer, context
            ))

        return answer, metrics
```

#### Step 2: Create Enhanced Components

**File**: `autorag/components/enhanced/chunkers.py`

```python
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.metrics.mixins import ChunkerMetricsMixin

class EnhancedFixedSizeChunker(FixedSizeChunker, ChunkerMetricsMixin):
    """
    Fixed-size chunker with integrated metrics.
    Backward compatible - still supports original chunk() method.
    """

    def process(self, documents, collect_metrics=False):
        """
        Process documents with optional metrics collection.

        Args:
            documents: Input documents
            collect_metrics: Whether to collect metrics

        Returns:
            If collect_metrics=False: chunks
            If collect_metrics=True: (chunks, metrics)
        """
        if collect_metrics:
            return self.chunk_with_metrics(documents)
        else:
            return self.chunk(documents)
```

### Multi-Objective Optimization

#### Step 3: Implement Pareto Optimization

**File**: `autorag/optimization/multi_objective.py`

```python
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ParetoPoint:
    """Single point on Pareto frontier"""
    config: Dict[str, Any]
    objectives: Dict[str, float]
    dominated_by: List[int] = None
    dominates: List[int] = None

    def dominates_point(self, other: 'ParetoPoint') -> bool:
        """Check if this point dominates another"""
        better_in_at_least_one = False
        for obj_name, obj_value in self.objectives.items():
            other_value = other.objectives[obj_name]

            # Assuming maximization for all objectives
            if obj_value < other_value:
                return False  # Worse in this objective
            elif obj_value > other_value:
                better_in_at_least_one = True

        return better_in_at_least_one

class ParetoFrontier:
    """Tracks and updates Pareto frontier"""

    def __init__(self, objectives: List[str], directions: Dict[str, str] = None):
        """
        Args:
            objectives: List of objective names
            directions: Dict mapping objective to 'maximize' or 'minimize'
        """
        self.objectives = objectives
        self.directions = directions or {obj: 'maximize' for obj in objectives}
        self.points = []
        self.frontier = []

    def add_point(self, config: Dict[str, Any], metrics: Dict[str, float]):
        """Add new point and update frontier"""
        # Normalize objectives based on direction
        normalized_objectives = {}
        for obj in self.objectives:
            value = metrics.get(obj, 0)
            if self.directions[obj] == 'minimize':
                value = -value  # Convert to maximization
            normalized_objectives[obj] = value

        point = ParetoPoint(config=config, objectives=normalized_objectives)
        self.points.append(point)

        # Update frontier
        self._update_frontier()

        return point in self.frontier

    def _update_frontier(self):
        """Recalculate Pareto frontier"""
        self.frontier = []

        for i, point in enumerate(self.points):
            is_dominated = False

            for j, other in enumerate(self.points):
                if i != j and other.dominates_point(point):
                    is_dominated = True
                    break

            if not is_dominated:
                self.frontier.append(point)

    def get_frontier(self) -> List[ParetoPoint]:
        """Get current Pareto frontier"""
        return self.frontier

    def get_hypervolume(self, reference_point: Dict[str, float] = None) -> float:
        """Calculate hypervolume indicator"""
        if not self.frontier:
            return 0.0

        # Default reference point (worst case)
        if reference_point is None:
            reference_point = {
                obj: min(p.objectives[obj] for p in self.points)
                for obj in self.objectives
            }

        # Simplified 2D hypervolume calculation
        if len(self.objectives) == 2:
            return self._calculate_2d_hypervolume(reference_point)
        else:
            # For >2D, use pymoo or other library
            return self._calculate_nd_hypervolume(reference_point)
```

#### Step 4: Advanced Bayesian Optimizer

**File**: `autorag/optimization/bayesian_search_advanced.py`

```python
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf

class AdvancedBayesianOptimizer:
    """
    Multi-objective Bayesian optimization with stage-wise metrics.
    Uses BoTorch for advanced acquisition functions.
    """

    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: Callable,
                 objectives: List[str] = ["accuracy", "latency", "cost"],
                 n_initial: int = 20,
                 n_calls: int = 100,
                 metrics_mode: str = "hybrid"):

        self.search_space = search_space
        self.evaluator = evaluator
        self.objectives = objectives
        self.n_initial = n_initial
        self.n_calls = n_calls
        self.metrics_mode = metrics_mode

        # Initialize components
        self.pareto_frontier = ParetoFrontier(objectives)
        self.surrogate_models = {}
        self.search_data = {'X': [], 'Y': {obj: [] for obj in objectives}}

    def optimize(self):
        """Run multi-objective Bayesian optimization"""

        # Phase 1: Random exploration
        logger.info(f"Phase 1: Random exploration ({self.n_initial} points)")
        for i in range(self.n_initial):
            config = self.search_space.sample(n=1, method="random")[0]
            self._evaluate_and_update(config)

        # Phase 2: Bayesian optimization
        logger.info(f"Phase 2: Bayesian optimization ({self.n_calls - self.n_initial} points)")
        for i in range(self.n_initial, self.n_calls):
            # Train surrogate models
            self._train_surrogates()

            # Select next configuration
            next_config = self._select_next_config()

            # Evaluate and update
            self._evaluate_and_update(next_config)

            # Log progress
            if (i + 1) % 10 == 0:
                self._log_progress(i + 1)

        return self._generate_report()

    def _evaluate_and_update(self, config: Dict[str, Any]):
        """Evaluate configuration and update data"""
        # Run evaluation with metrics
        result = self.evaluator(config)

        # Extract objectives
        objectives_values = {}
        for obj in self.objectives:
            if obj == "accuracy":
                objectives_values[obj] = result['metrics'].get('accuracy', 0)
            elif obj == "latency":
                objectives_values[obj] = result['metrics'].get('total_time', float('inf'))
            elif obj == "cost":
                objectives_values[obj] = result['metrics'].get('total_cost', float('inf'))

        # Update Pareto frontier
        is_pareto = self.pareto_frontier.add_point(config, objectives_values)

        # Store data for surrogate training
        config_vector = self._config_to_vector(config)
        self.search_data['X'].append(config_vector)
        for obj in self.objectives:
            self.search_data['Y'][obj].append(objectives_values[obj])

        logger.info(f"Config evaluated - Accuracy: {objectives_values.get('accuracy', 0):.3f}, "
                   f"Latency: {objectives_values.get('latency', 0):.2f}s, "
                   f"Cost: ${objectives_values.get('cost', 0):.4f}, "
                   f"Pareto: {is_pareto}")

    def _train_surrogates(self):
        """Train Gaussian Process surrogate models"""
        X = torch.tensor(self.search_data['X'])

        for obj in self.objectives:
            Y = torch.tensor(self.search_data['Y'][obj]).unsqueeze(-1)

            # Standardize Y
            Y_mean = Y.mean()
            Y_std = Y.std()
            Y_standardized = (Y - Y_mean) / (Y_std + 1e-6)

            # Train GP model
            model = SingleTaskGP(X, Y_standardized)
            model.likelihood.train()
            model.train()

            # Optimize hyperparameters
            from botorch.fit import fit_gpytorch_model
            from gpytorch.mlls import ExactMarginalLogLikelihood

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Store model with standardization params
            self.surrogate_models[obj] = {
                'model': model,
                'mean': Y_mean,
                'std': Y_std
            }

    def _select_next_config(self):
        """Select next configuration using EHVI acquisition"""
        # Get reference point for hypervolume
        ref_point = self._get_reference_point()

        # Create acquisition function
        acq_function = ExpectedHypervolumeImprovement(
            model=self._create_multi_objective_model(),
            ref_point=ref_point,
            partitioning=self.pareto_frontier
        )

        # Optimize acquisition function
        bounds = self._get_search_bounds()
        candidate, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100
        )

        # Convert back to configuration
        return self._vector_to_config(candidate.squeeze().numpy())
```

## Phase 3: Production System (Week 3-4)

### Advanced Features

#### 1. Constraint Handling

**File**: `autorag/optimization/constraints.py`

```python
class ConstraintManager:
    """Manages optimization constraints"""

    def __init__(self):
        self.constraints = []

    def add_budget_constraint(self, max_cost: float):
        """Add cost budget constraint"""
        self.constraints.append({
            'type': 'budget',
            'max_cost': max_cost,
            'cumulative': True
        })

    def add_latency_constraint(self, max_latency: float):
        """Add latency constraint"""
        self.constraints.append({
            'type': 'latency',
            'max_latency': max_latency,
            'per_query': True
        })

    def add_accuracy_constraint(self, min_accuracy: float):
        """Add minimum accuracy constraint"""
        self.constraints.append({
            'type': 'accuracy',
            'min_accuracy': min_accuracy
        })

    def is_feasible(self, metrics: Dict[str, Any]) -> bool:
        """Check if configuration satisfies constraints"""
        for constraint in self.constraints:
            if constraint['type'] == 'budget':
                if metrics.get('total_cost', 0) > constraint['max_cost']:
                    return False
            elif constraint['type'] == 'latency':
                if metrics.get('latency', float('inf')) > constraint['max_latency']:
                    return False
            elif constraint['type'] == 'accuracy':
                if metrics.get('accuracy', 0) < constraint['min_accuracy']:
                    return False
        return True
```

#### 2. Early Stopping

**File**: `autorag/optimization/early_stopping.py`

```python
class EarlyStopping:
    """Early stopping based on stage metrics"""

    def __init__(self, thresholds: Dict[str, float]):
        """
        Args:
            thresholds: Min thresholds for stage metrics
        """
        self.thresholds = thresholds

    def should_stop(self, stage: str, metrics: Dict[str, Any]) -> bool:
        """Check if should stop after stage"""
        if stage == "chunking":
            if metrics.get('chunks_created', 0) == 0:
                return True
            if metrics.get('avg_chunk_size', 0) < self.thresholds.get('min_chunk_size', 50):
                return True

        elif stage == "retrieval":
            if metrics.get('recall@10', 0) < self.thresholds.get('min_recall', 0.1):
                return True
            if metrics.get('top_score', 0) < self.thresholds.get('min_score', 0.01):
                return True

        elif stage == "reranking":
            if metrics.get('ndcg@5', 0) < self.thresholds.get('min_ndcg', 0.1):
                return True

        return False

    def evaluate_with_early_stopping(self, pipeline, query, documents):
        """Run pipeline with early stopping"""
        results = {}

        # Chunking
        chunks, chunk_metrics = pipeline.chunker.chunk_with_metrics(documents)
        results['chunking'] = chunk_metrics

        if self.should_stop('chunking', chunk_metrics):
            return results, "STOPPED_AT_CHUNKING"

        # Retrieval
        retrieved, retrieval_metrics = pipeline.retriever.retrieve_with_metrics(query)
        results['retrieval'] = retrieval_metrics

        if self.should_stop('retrieval', retrieval_metrics):
            return results, "STOPPED_AT_RETRIEVAL"

        # Continue with remaining stages...
        return results, "COMPLETED"
```

#### 3. Visualization Dashboard

**File**: `autorag/visualization/optimization_dashboard.py`

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

class OptimizationDashboard:
    """Real-time monitoring dashboard for optimization"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Bayesian Optimization Dashboard"),

            # Progress indicators
            html.Div([
                html.H3("Progress"),
                dcc.Graph(id='progress-chart'),
                dcc.Interval(id='interval-component', interval=5000)  # Update every 5s
            ]),

            # Pareto frontier
            html.Div([
                html.H3("Pareto Frontier"),
                dcc.Graph(id='pareto-chart')
            ]),

            # Stage metrics
            html.Div([
                html.H3("Stage Metrics"),
                dcc.Graph(id='stage-metrics-chart')
            ]),

            # Best configurations
            html.Div([
                html.H3("Top Configurations"),
                html.Table(id='config-table')
            ])
        ])

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            Output('pareto-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_pareto_chart(n):
            """Update Pareto frontier visualization"""
            frontier = self.optimizer.pareto_frontier.get_frontier()

            if len(self.optimizer.objectives) == 2:
                # 2D Pareto plot
                fig = go.Figure()

                # All points
                all_points = self.optimizer.pareto_frontier.points
                fig.add_trace(go.Scatter(
                    x=[p.objectives[self.optimizer.objectives[0]] for p in all_points],
                    y=[p.objectives[self.optimizer.objectives[1]] for p in all_points],
                    mode='markers',
                    name='Evaluated',
                    marker=dict(size=8, color='lightgray')
                ))

                # Pareto frontier
                fig.add_trace(go.Scatter(
                    x=[p.objectives[self.optimizer.objectives[0]] for p in frontier],
                    y=[p.objectives[self.optimizer.objectives[1]] for p in frontier],
                    mode='markers+lines',
                    name='Pareto Frontier',
                    marker=dict(size=12, color='red'),
                    line=dict(dash='dash')
                ))

                fig.update_layout(
                    xaxis_title=self.optimizer.objectives[0],
                    yaxis_title=self.optimizer.objectives[1],
                    hovermode='closest'
                )

            elif len(self.optimizer.objectives) == 3:
                # 3D Pareto plot
                fig = go.Figure()

                # All points
                all_points = self.optimizer.pareto_frontier.points
                fig.add_trace(go.Scatter3d(
                    x=[p.objectives[self.optimizer.objectives[0]] for p in all_points],
                    y=[p.objectives[self.optimizer.objectives[1]] for p in all_points],
                    z=[p.objectives[self.optimizer.objectives[2]] for p in all_points],
                    mode='markers',
                    name='Evaluated',
                    marker=dict(size=4, color='lightgray')
                ))

                # Pareto frontier
                fig.add_trace(go.Scatter3d(
                    x=[p.objectives[self.optimizer.objectives[0]] for p in frontier],
                    y=[p.objectives[self.optimizer.objectives[1]] for p in frontier],
                    z=[p.objectives[self.optimizer.objectives[2]] for p in frontier],
                    mode='markers',
                    name='Pareto Frontier',
                    marker=dict(size=8, color='red')
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title=self.optimizer.objectives[0],
                        yaxis_title=self.optimizer.objectives[1],
                        zaxis_title=self.optimizer.objectives[2]
                    )
                )

            return fig

    def run(self, port=8050):
        """Run dashboard server"""
        self.app.run_server(debug=False, port=port)
```

## Integration Strategy

### 1. Backward Compatibility

Maintain compatibility with existing code:

```python
class BayesianSearchOptimizer:
    """Drop-in replacement for GridSearchOptimizer"""

    def search(self, **kwargs):
        """Same interface as GridSearchOptimizer.search()"""
        # Internal implementation uses Bayesian optimization
        return self.optimize()

    def _generate_report(self, best_config):
        """Same report format as grid search"""
        return {
            "summary": {...},
            "best_configuration": best_config,
            "top_5_configurations": [...],
            "parameter_importance": {...},
            "cost_breakdown": {...}
        }
```

### 2. Configuration Flag

Allow switching between optimizers via configuration:

```python
# config.yaml
optimization:
  method: "bayesian"  # or "grid"

  # Bayesian-specific settings
  bayesian:
    n_initial: 20
    n_calls: 100
    objectives: ["accuracy", "latency", "cost"]
    acquisition: "EHVI"
    metrics_mode: "hybrid"

  # Grid search settings (existing)
  grid:
    max_configurations: 960
    early_stopping: true

# Usage in code
def create_optimizer(config, search_space, evaluator):
    method = config['optimization']['method']

    if method == 'bayesian':
        return BayesianSearchOptimizer(
            search_space=search_space,
            evaluator=evaluator,
            **config['optimization']['bayesian']
        )
    else:
        return GridSearchOptimizer(
            search_space=search_space,
            evaluator=evaluator,
            **config['optimization']['grid']
        )
```

### 3. Gradual Migration

Start with hybrid approach:

```python
class HybridOptimizer:
    """Combines grid search and Bayesian optimization"""

    def optimize(self, grid_configs=20, bayesian_calls=80):
        """
        Run grid search first, then Bayesian optimization.
        This provides baseline data for better surrogate models.
        """
        # Phase 1: Grid search for initial data
        grid_optimizer = GridSearchOptimizer(...)
        grid_results = grid_optimizer.search(max_configurations=grid_configs)

        # Phase 2: Bayesian optimization
        bayesian_optimizer = BayesianSearchOptimizer(...)
        bayesian_optimizer.initialize_with_data(grid_results)
        final_results = bayesian_optimizer.optimize(n_calls=bayesian_calls)

        return final_results
```

## Migration Path

### Week 1: Foundation
1. **Day 1-2**: Implement external metrics collector
2. **Day 3-4**: Create simple Bayesian optimizer with scikit-optimize
3. **Day 5**: Test on small subset, compare with grid search

### Week 2: Enhancement
1. **Day 1-2**: Add component mixins for richer metrics
2. **Day 3-4**: Implement multi-objective optimization
3. **Day 5**: Create Pareto visualization

### Week 3: Production
1. **Day 1-2**: Add constraints and early stopping
2. **Day 3-4**: Build monitoring dashboard
3. **Day 5**: Full system testing

### Week 4: Optimization
1. **Day 1-2**: Performance tuning
2. **Day 3-4**: Documentation and examples
3. **Day 5**: Deployment preparation

## Technical Details

### Search Space Conversion

```python
def convert_to_skopt_space(search_space: SearchSpace):
    """Convert custom SearchSpace to scikit-optimize format"""
    skopt_dimensions = []

    for component_name, component in search_space.components.items():
        for param in component.parameters:
            # Create dimension name
            dim_name = f"{component_name}.{param.name}"

            # Convert based on parameter type
            if param.parameter_type == "categorical":
                dimension = Categorical(param.values, name=dim_name)
            elif param.parameter_type == "numerical":
                if all(isinstance(v, int) for v in param.values):
                    dimension = Integer(min(param.values), max(param.values), name=dim_name)
                else:
                    dimension = Real(min(param.values), max(param.values), name=dim_name)
            elif param.parameter_type == "boolean":
                dimension = Categorical([True, False], name=dim_name)

            skopt_dimensions.append(dimension)

    return skopt_dimensions
```

### Surrogate Model Training

```python
def train_surrogate_with_stage_metrics(X, Y, stage_metrics):
    """
    Train surrogate model using both configuration and stage metrics.
    This improves prediction accuracy.
    """
    # Combine configuration features with stage metrics
    enhanced_features = []

    for i, config_vector in enumerate(X):
        # Configuration features
        features = list(config_vector)

        # Add stage metric features
        if stage_metrics and i < len(stage_metrics):
            metrics = stage_metrics[i]
            features.extend([
                metrics.get('chunking', {}).get('chunks_created', 0),
                metrics.get('retrieval', {}).get('recall@10', 0),
                metrics.get('retrieval', {}).get('top_score', 0),
                metrics.get('generation', {}).get('tokens_used', 0)
            ])

        enhanced_features.append(features)

    # Train GP with enhanced features
    X_enhanced = np.array(enhanced_features)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=10
    )
    gp.fit(X_enhanced, Y)

    return gp
```

### Acquisition Function Selection

```python
def select_acquisition_function(iteration, n_initial, exploration_weight=0.1):
    """
    Dynamically select acquisition function based on optimization phase.
    Early: more exploration, Late: more exploitation
    """
    exploration_ratio = max(0, (n_initial - iteration) / n_initial)

    if iteration < n_initial * 0.3:
        # Early phase: Pure exploration
        return "UCB", {"kappa": 2.0}
    elif iteration < n_initial * 0.7:
        # Middle phase: Balanced
        return "EI", {"xi": 0.01}
    else:
        # Late phase: Exploitation
        return "PI", {"xi": 0.001}
```

## Testing Strategy

### Unit Tests

```python
# tests/test_bayesian_optimizer.py

def test_search_space_conversion():
    """Test conversion to skopt format"""
    search_space = SearchSpace()
    search_space.define_chunking_space(strategies=["fixed"], sizes=[256, 512])

    skopt_space = convert_to_skopt_space(search_space)
    assert len(skopt_space) == 2
    assert skopt_space[0].name == "chunking.strategy"

def test_pareto_dominance():
    """Test Pareto dominance calculation"""
    p1 = ParetoPoint(config={}, objectives={"acc": 0.8, "cost": 0.1})
    p2 = ParetoPoint(config={}, objectives={"acc": 0.7, "cost": 0.2})

    assert p1.dominates_point(p2) == True
    assert p2.dominates_point(p1) == False

def test_early_stopping():
    """Test early stopping logic"""
    early_stopper = EarlyStopping({"min_recall": 0.2})

    metrics = {"recall@10": 0.1}
    assert early_stopper.should_stop("retrieval", metrics) == True

    metrics = {"recall@10": 0.3}
    assert early_stopper.should_stop("retrieval", metrics) == False
```

### Integration Tests

```python
# tests/test_integration.py

def test_bayesian_vs_grid():
    """Compare Bayesian optimization with grid search"""
    search_space = create_test_search_space()
    evaluator = create_mock_evaluator()

    # Grid search
    grid_opt = GridSearchOptimizer(search_space, evaluator)
    grid_results = grid_opt.search(max_configurations=50)

    # Bayesian optimization
    bayes_opt = BayesianSearchOptimizer(search_space, evaluator)
    bayes_results = bayes_opt.optimize(n_calls=50)

    # Bayesian should achieve similar or better performance
    assert bayes_results['best_score'] >= grid_results['best_score'] * 0.95

    # Bayesian should evaluate fewer configurations
    assert bayes_results['configurations_evaluated'] <= grid_results['configurations_evaluated']
```

### Performance Tests

```python
# tests/test_performance.py

def test_optimization_efficiency():
    """Test optimization efficiency"""
    import time

    search_space = create_large_search_space()  # 1000+ configurations
    evaluator = create_fast_mock_evaluator()

    start = time.time()
    optimizer = BayesianSearchOptimizer(search_space, evaluator)
    results = optimizer.optimize(n_calls=100)
    elapsed = time.time() - start

    # Should complete 100 evaluations in reasonable time
    assert elapsed < 300  # 5 minutes

    # Should find good configuration
    assert results['best_score'] > 0.7
```

## Performance Expectations

### Comparison with Grid Search

| Metric | Grid Search | Bayesian Optimization | Improvement |
|--------|------------|----------------------|-------------|
| Configurations Evaluated | 960 | 100 | 89.6% reduction |
| Time to Best Config | ~480 evaluations | ~40 evaluations | 91.7% faster |
| Final Best Score | 0.25 | 0.40+ | 60% better |
| Cost (API calls) | $4.80 | $0.50 | 89.6% reduction |
| Insights Generated | Minimal | Rich (Pareto, importance) | Significant |

### Expected Milestones

1. **After 20 evaluations**: Match grid search baseline performance
2. **After 50 evaluations**: Exceed grid search best score
3. **After 100 evaluations**: Find multiple Pareto optimal configurations
4. **Final outcome**: 5-10 production-ready configurations with different trade-offs

### Resource Requirements

- **Memory**: ~2GB for GP models with 100 data points
- **CPU**: Moderate (GP training takes seconds)
- **Storage**: ~100MB for checkpoints and results
- **API Costs**: 80-90% reduction vs grid search

## Troubleshooting Guide

### Common Issues

1. **Surrogate model convergence issues**
   - Solution: Increase n_initial points for more training data
   - Alternative: Use simpler kernel (RBF instead of Matern)

2. **Acquisition function optimization stuck**
   - Solution: Increase num_restarts in optimize_acqf
   - Alternative: Use simpler acquisition (UCB instead of EHVI)

3. **Memory issues with large search spaces**
   - Solution: Use sparse GPs or local models
   - Alternative: Limit search space dimensionality

4. **Poor initial exploration**
   - Solution: Increase n_initial or use Latin Hypercube Sampling
   - Alternative: Warm-start with grid search results

## Conclusion

This implementation guide provides a complete roadmap for transitioning from grid search to Bayesian optimization in the auto-RAG system. The phased approach ensures minimal disruption while delivering significant improvements in optimization efficiency and solution quality. The modular design maintains compatibility with existing code while enabling advanced features like multi-objective optimization and stage-wise metrics collection.

Key success factors:
1. Start simple with Phase 1 implementation
2. Maintain backward compatibility
3. Focus on metrics that matter
4. Use visualization for insights
5. Iterate based on results

Expected outcome: 90% reduction in optimization cost while finding 60% better configurations, with rich insights into component interactions and trade-offs.