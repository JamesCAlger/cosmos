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

**Supported types**:
- `chunker`: Wraps `BaseChunker` → measures chunking metrics
- `retriever`: Wraps `BaseRetriever` → measures retrieval metrics
- `reranker`: Wraps `BaseReranker` → measures reranking metrics
- `generator`: Wraps `BaseGenerator` → measures generation metrics

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

### COSMOS Sequential Optimization Flow

The key insight: **break circular dependencies** by optimizing components sequentially, passing context forward:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Optimize Chunker (no dependencies)                  │
│                                                              │
│   Input:  Documents                                         │
│   Metrics: Chunk coherence, size variance, avg length       │
│   Output: best_chunker_config                               │
│                                                              │
│   Example: {'chunk_size': 256, 'overlap': 50}               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Context passed: chunks from best_chunker
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Optimize Retriever (uses best_chunker)              │
│                                                              │
│   Input:  Chunks from best_chunker                          │
│   Metrics: Retrieval latency, coverage score                │
│   Output: best_retriever_config                             │
│                                                              │
│   Example: {'retrieval_method': 'dense', 'top_k': 5}        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Context passed: results from best_retriever
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Optimize Generator (uses best_retriever)            │
│                                                              │
│   Input:  Results from best_retriever                       │
│   Metrics: Answer quality, semantic similarity              │
│   Output: best_generator_config                             │
│                                                              │
│   Example: {'model': 'gpt-3.5-turbo', 'temperature': 0.7}   │
└─────────────────────────────────────────────────────────────┘

Result: Optimized pipeline without circular dependencies
```

**Why sequential works**: Each component optimized with best upstream context, breaking the "need full pipeline to evaluate any component" problem.

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

**⚠️ CRITICAL - Don't Forget CLI Parser**: When adding a new component type, you MUST also update the argparse choices in `scripts/run_cosmos_optimization.py`:

```python
parser.add_argument('--components', nargs='+',
                   choices=['chunker', 'retriever', 'reranker', 'generator'],  # ← Add 'reranker'
                   default=['chunker', 'retriever'],
                   help='Components to optimize')
```

**Why this is easy to miss**: The CLI parser is at the top of the script, far from the search space definition. If you forget this, the feature will be completely unusable from the command line (users can't pass the new component name as an argument).

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

## Design Patterns

### Pattern 1: Environment-Aware Configuration

**Problem**: Code should work seamlessly in development (mock API) and production (real API) without manual config changes.

**Solution**: Auto-detect environment based on API key presence instead of hardcoding config values.

**Example** (`optimization/evaluators.py` → `build_component()` for generator):
```python
elif component_type == 'generator':
    import os
    use_real_api = config.get('use_real_api', True)
    api_key = os.getenv('OPENAI_API_KEY')

    # Smart fallback: use real API if key exists, otherwise use mock
    if use_real_api and api_key:
        from autorag.components.generators.openai import OpenAIGenerator
        return OpenAIGenerator({
            'model': config.get('model', 'gpt-3.5-turbo'),
            'temperature': config.get('temperature', 0.3),
            'max_tokens': config.get('max_tokens', 150),
            'api_key': api_key
        })
    else:
        from autorag.components.generators.mock import MockGenerator
        if use_real_api and not api_key:
            logger.warning("Generator: OpenAI API key not found, falling back to mock generator")
        return MockGenerator({'temperature': config.get('temperature', 0.3)})
```

**Why this matters**:
- ✅ No crashes when API key missing (graceful degradation)
- ✅ Works locally without `.env` file
- ✅ Automatically uses real API in production when key is available
- ✅ No need to modify search spaces based on environment

**Key pattern**: **Extract → Check → Use (or fallback)** instead of assuming the value exists.

### Pattern 2: Upstream Component Re-initialization

**Problem**: When evaluating a component that depends on upstream components, you can't just reuse the upstream component instance - you need to re-initialize its dependencies (embedder, vector store, etc.).

**Why**: Upstream component configs are passed as references (retriever config doesn't serialize embedder/vector_store instances).

**Solution**: Re-initialize required dependencies at evaluation time.

**Example** (`optimization/evaluators.py` → `_evaluate_reranker()`):
```python
def _evaluate_reranker(self, config: Dict[str, Any]) -> float:
    # 1-2. Get upstream components
    chunker = self.upstream_components.get('chunker') or default_chunker()
    retriever = self.upstream_components.get('retriever') or default_retriever()

    # 3. CRITICAL: Re-initialize retriever dependencies
    # Can't use stored retriever directly - it doesn't have embedder/vector_store set
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and config.get('use_real_api', True):
        embedder = CachedEmbedder(OpenAIEmbedder({'api_key': api_key}))
    else:
        embedder = MockEmbedder({})

    vector_store = SimpleVectorStore({})
    retriever.set_components(embedder, vector_store)  # Re-initialize at runtime

    # 4. Now build and evaluate reranker
    reranker = build_component('reranker', config)
    # ... evaluation logic ...
```

**Why this matters**:
- Retriever config is just `{'retrieval_method': 'dense', 'top_k': 5}` - no embedder/vector_store
- Those are runtime dependencies that must be re-created for each evaluation
- Without this, retriever can't actually retrieve (no embeddings, no vector store)

**Key pattern**: **Config → Runtime Dependencies → Set Components** for components with external dependencies.

### Pattern 3: Component Wrapper for Metrics Injection

**Pattern**: Use wrapper pattern to add metrics collection without modifying base components.

**Example** (`component_wrapper.py` → `COSMOSComponent`):
```python
class COSMOSComponent:
    def __init__(self, base_component, component_type, metric_collector):
        self.base = base_component  # Wrap base component
        self.type = component_type
        self.metric_collector = metric_collector

    def process_with_metrics(self, *args, **kwargs):
        # Intercept calls, add metrics collection
        if self.type == 'reranker':
            return self._process_reranker(*args, **kwargs)
        # ... dispatch to correct handler

    def _process_reranker(self, query, results, top_k=5):
        start_time = time.time()
        reranked = self.base.rerank(query, results, top_k)  # Use base component
        latency = time.time() - start_time

        metrics = self.metric_collector.compute_reranking_metrics(...)
        return reranked, metrics  # Return results + metrics
```

**Why this matters**:
- Base components (`CrossEncoderReranker`, `DenseRetriever`, etc.) stay clean - no COSMOS-specific code
- Metrics collection is centralized in wrapper
- Easy to add new component types without modifying existing code

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

## Implementation Verification Checklist

After adding a new component type to COSMOS, verify implementation is complete:

### Code Checklist
- [ ] **Component wrapper** updated (`component_wrapper.py`)
  - [ ] Added elif branch in `process_with_metrics()` dispatcher
  - [ ] Implemented `_process_[component]()` method with metrics collection
- [ ] **Metrics computation** implemented (`metrics/component_metrics.py`)
  - [ ] Added `compute_[component]_metrics()` method
  - [ ] Added quality score case in `compute_quality_score()`
  - [ ] Documented metric interpretation (what values are "good")
- [ ] **Component builder** updated (`optimization/evaluators.py`)
  - [ ] Added elif case in `build_component()` function
  - [ ] Handles environment-aware config (API key detection, mock fallback)
- [ ] **Component evaluator** created (`optimization/evaluators.py`)
  - [ ] Implemented `_evaluate_[component]()` method
  - [ ] Properly re-initializes upstream component dependencies
  - [ ] Uses cache manager if available
- [ ] **Search space** defined (`scripts/run_cosmos_optimization.py`)
  - [ ] Added component key in `define_search_spaces()`
  - [ ] Included relevant hyperparameters
  - [ ] Removed hardcoded environment-specific values (e.g., use_real_api)
- [ ] **CLI parser** updated (`scripts/run_cosmos_optimization.py`)
  - [ ] ⚠️ **CRITICAL**: Added component name to argparse choices
  - [ ] Test: Can you run `--components [new_component]` without error?
- [ ] **Dependencies** added if needed (`requirements.txt`)

### Testing Checklist
- [ ] **Unit tests** (`tests/unit/test_[component]_cosmos.py`)
  - [ ] Test component wrapper basic functionality
  - [ ] Test metrics computation with known inputs
  - [ ] Test quality score edge cases (empty results, single result)
  - [ ] Test component building from config
  - [ ] Minimum: 10-15 unit tests covering core logic
- [ ] **Integration tests** (`tests/integration/test_[component]_optimization.py`)
  - [ ] Test component evaluation with upstream context
  - [ ] Test upstream component re-initialization
  - [ ] Test cache manager integration
  - [ ] Test full optimization sequence
  - [ ] Minimum: 5-10 integration tests covering end-to-end
- [ ] **Manual CLI test**:
  ```bash
  # Can you run this without errors?
  python scripts/run_cosmos_optimization.py --components [new_component] --budget 5
  ```
- [ ] **Run test suite**:
  ```bash
  # All tests passing?
  pytest tests/unit/test_[component]_cosmos.py -v
  pytest tests/integration/test_[component]_optimization.py -v
  ```

### Common Mistakes to Catch
- [ ] CLI parser updated? (Feature unusable from command line if missing)
- [ ] Environment-aware config? (Crashes when API key missing if not handled)
- [ ] Upstream dependencies re-initialized? (Retriever/generator evaluators need this)
- [ ] Metric interpretation documented? (How to read quality score values)
- [ ] **Metric calculation bugs** (Avoid inconsistent denominators in overlap/precision metrics)
  - Fixed bug (2025-10-09): `top_k_overlap` used dynamic k instead of fixed k=5, making metrics incomparable
  - Always use consistent denominators across different config outputs (e.g., k_original=5, not min(5, len(reranked)))

### Ready to Commit When:
- [ ] All code checklist items complete
- [ ] All unit tests passing (14+ tests)
- [ ] All integration tests passing (10+ tests)
- [ ] Manual CLI test works
- [ ] No hardcoded environment-specific values in search spaces

---

**Last Updated**: 2025-10-06
**Status**: Fully operational for chunker/retriever/reranker/generator
**Related Docs**:
- `autorag/components/CLAUDE.md` - Component architecture
- `autorag/components/rerankers/CLAUDE.md` - Reranker specifics
