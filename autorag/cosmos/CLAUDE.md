# COSMOS Framework - Compositional Optimization Architecture

## When to Read This

**Read this doc if you're**:
- ✅ Understanding how COSMOS works (sequential optimization, breaking circular dependencies)
- ✅ Adding a new component type to COSMOS (e.g., reranker)
- ✅ Debugging optimization issues or understanding component evaluation
- ✅ Modifying optimization strategies (Bayesian, Random)

**Skip this doc if you're**:
- ❌ Just running COSMOS optimizations → see root `CLAUDE.md` for usage instructions
- ❌ Understanding component architecture → see `autorag/components/CLAUDE.md` first
- ❌ Working only with existing components → root `CLAUDE.md` has what you need

**Prerequisites**:
- Basic understanding of RAG pipeline (chunker → retriever → generator)
- Familiarity with hyperparameter optimization concepts

**Recommended reading order for adding new components**:
1. Read this doc first → understand COSMOS framework
2. Read component-specific doc (e.g., `autorag/components/rerankers/CLAUDE.md`)
3. Follow step-by-step guide in "How to Add a New Component Type" section

---

## What is COSMOS?

**COSMOS** (Compositional Optimization with Sequential Metrics and Orchestration System) is a framework for optimizing multi-component architectures by breaking circular dependencies.

**Core idea**: Instead of optimizing the entire pipeline jointly (expensive, circular dependencies), optimize components **sequentially** using component-intrinsic metrics and context passing.

## Why COSMOS? The Circular Dependency Problem

### Traditional Approach (Bayesian on Full Pipeline)
```
Config → Build Pipeline → Evaluate (run full pipeline) → Score
```
**Problem**: To evaluate chunker config, you need retriever and generator. But those aren't optimized yet!

### COSMOS Approach
```
1. Optimize chunker → get best_chunker
2. Optimize retriever (using best_chunker) → get best_retriever
3. Optimize generator (using best_chunker + best_retriever) → get best_generator
```
**Solution**: Each component optimized independently with **component-intrinsic metrics** + context from upstream

## Architecture

### Core Components

#### 1. **COSMOSComponent** (`component_wrapper.py`)
Wraps existing components to add `process_with_metrics()` capability.

**Supported types** (currently):
- `chunker`: Wraps `BaseChunker` → measures chunking metrics
- `retriever`: Wraps `BaseRetriever` → measures retrieval metrics
- `generator`: Wraps `BaseGenerator` → measures generation metrics

**Missing**: `reranker` support (needs to be added)

**Key method**:
```python
output, metrics = cosmos_component.process_with_metrics(*args, **kwargs)
```

#### 2. **ComponentMetrics** (`metrics/component_metrics.py`)
Computes component-intrinsic metrics without full pipeline context.

**Chunking metrics**: count, avg_length, size_variance, coherence
**Retrieval metrics**: latency, retrieval_rate, coverage_score
**Generation metrics**: semantic_similarity, latency, answer_length

#### 3. **ComponentEvaluator** (`optimization/evaluators.py`)
Evaluates component configurations using intrinsic metrics + context.

**Key function**: `build_component(component_type, config, context)`
- Takes component type, config dict, and context from upstream
- Builds the component instance
- **Currently supports**: chunker, retriever, generator
- **Needs**: reranker case added

#### 4. **CompositionalOptimizer** (`optimization/compositional_optimizer.py`)
Orchestrates sequential optimization across components.

**Flow**:
```python
for component in ['chunker', 'retriever', 'generator']:
    task = create_task(component, search_space, context)
    best_config = strategy.optimize(task)
    context.update({'best_' + component: best_config})
```

#### 5. **Optimization Strategies** (`optimization/strategy.py`)
- `RandomStrategy`: Random search over hyperparameters
- `BayesianStrategy`: Bayesian optimization for sample efficiency

## Component Flow & Context Passing

### Current RAG Flow
```
documents → [chunker] → chunks (context for retriever)
                           ↓
query → [retriever] → results (context for generator)
                         ↓
         [generator] → answer
```

### Adding Reranker to Flow
```
documents → [chunker] → chunks (context for retriever)
                           ↓
query → [retriever] → results (context for reranker)
                         ↓
         [reranker] → reranked_results (context for generator)
                         ↓
         [generator] → answer
```

**Key insight**: Reranker sits **between retriever and generator**, needs retriever output as input.

## How to Add a New Component Type (Example: Reranker)

### Step 1: Add to COSMOSComponent

**File**: `component_wrapper.py` → `COSMOSComponent.process_with_metrics()` method → add 'reranker' case to if-elif chain

```python
def process_with_metrics(self, *args, **kwargs):
    if self.type == 'chunker':
        return self._process_chunker(*args, **kwargs)
    elif self.type == 'retriever':
        return self._process_retriever(*args, **kwargs)
    elif self.type == 'reranker':  # ADD THIS
        return self._process_reranker(*args, **kwargs)
    elif self.type == 'generator':
        return self._process_generator(*args, **kwargs)
```

### Step 2: Implement processor method

```python
def _process_reranker(self, query: str, results: List, top_k: int = 5, **kwargs):
    start_time = time.time()
    reranked = self.base.rerank(query, results, top_k)
    latency = time.time() - start_time

    metrics = self.metric_collector.compute_reranking_metrics(
        query, results, reranked, latency
    )
    self.metrics_history.append(metrics)
    return reranked, metrics
```

### Step 3: Add metrics computation (`metrics/component_metrics.py`)

```python
def compute_reranking_metrics(self, query, original_results, reranked_results, latency):
    return {
        'latency': latency,
        'score_change': mean_abs_score_change(original_results, reranked_results),
        'rank_correlation': kendall_tau(original_results, reranked_results)
    }
```

### Step 4: Add to build_component (`optimization/evaluators.py`)

```python
def build_component(component_type: str, config: dict, context: dict = None):
    # ... existing cases ...

    elif component_type == 'reranker':
        from autorag.components.rerankers.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(config)
```

### Step 5: Create evaluator class

```python
class RerankerEvaluator(ComponentEvaluator):
    def evaluate(self, config: dict) -> float:
        reranker = build_component('reranker', config)
        cosmos_reranker = COSMOSComponent(reranker, 'reranker', self.metrics)

        # Get context (retriever results)
        retriever = self.context.get('retriever')

        scores = []
        for query in self.queries:
            results = retriever.retrieve(query, top_k=10)
            reranked, metrics = cosmos_reranker.process_with_metrics(query, results, top_k=5)
            score = self.metrics.compute_quality_score('reranker', metrics)
            scores.append(score)

        return np.mean(scores)
```

### Step 6: Add to search space (`scripts/run_cosmos_optimization.py`)

```python
def define_search_spaces():
    return {
        'chunker': {...},
        'retriever': {...},
        'reranker': {  # ADD THIS
            'normalize_scores': [True, False],
            'model_name': ['cross-encoder/ms-marco-MiniLM-L-6-v2']
        },
        'generator': {...}
    }
```

### Step 7: Update optimization sequence

```python
components_to_optimize = ['chunker', 'retriever', 'reranker', 'generator']
```

## Component-Intrinsic Metrics Philosophy

**Goal**: Measure component quality without full pipeline context.

**Principle**: Metrics should be:
1. **Computable** with just component input/output
2. **Predictive** of downstream performance
3. **Fast** to compute (no expensive model calls)

**Examples**:
- ❌ Bad: "Answer quality for chunker" (requires full pipeline)
- ✅ Good: "Chunk coherence" (computable from chunks alone)
- ❌ Bad: "End-to-end latency for retriever" (requires generator)
- ✅ Good: "Retrieval latency + coverage" (intrinsic to retriever)

## When to Use COSMOS vs Bayesian

### Use COSMOS when:
- Many components to optimize (3+)
- Search space is large (10+ dimensions)
- Component metrics are available
- You want to understand component contributions

### Use Bayesian when:
- Joint optimization is critical
- Search space is small (<5 dimensions)
- End-to-end metric is all that matters
- Components are tightly coupled

## Current Limitations

1. **Sequential assumption**: Assumes component order is fixed (chunker → retriever → generator)
2. **No joint optimization**: Components optimized independently (misses interactions)
3. **Metric design**: Requires careful design of component-intrinsic metrics
4. **Greedy approach**: Best chunker + best retriever ≠ best (chunker, retriever) pair

## File Reference

- **Wrapper**: `component_wrapper.py` - Add new component types here
- **Metrics**: `metrics/component_metrics.py` - Add metric computation here
- **Evaluators**: `optimization/evaluators.py` - Add build_component case + evaluator class
- **Optimizer**: `optimization/compositional_optimizer.py` - Main orchestration logic
- **Strategies**: `optimization/strategy.py`, `bayesian_strategy.py`, `random_strategy.py`
- **Demo script**: `scripts/run_cosmos_optimization.py` - Search space + execution

## Quick Start: Adding Reranker

**Files to modify**:
1. `component_wrapper.py` → `COSMOSComponent.process_with_metrics()` → add 'reranker' elif branch
2. `metrics/component_metrics.py` → `ComponentMetrics` class → add `compute_reranking_metrics()` method
3. `optimization/evaluators.py` → `build_component()` function → add 'reranker' case + create `RerankerEvaluator` class
4. `scripts/run_cosmos_optimization.py` → `define_search_spaces()` → add 'reranker' key with search space

**Reference**: See `scripts/bayesian_with_cache/run_optimization.py` for how reranker integrates into pipeline.

---

**Last Updated**: 2025-10-03
**Status**: Fully operational for chunker/retriever/generator, reranker support pending
**Related Docs**: `autorag/components/CLAUDE.md`, `autorag/components/rerankers/CLAUDE.md`
