# COSMOS Framework Implementation Tasks

## Overview

Minimal COSMOS framework for compositional RAG optimization using existing Bayesian optimizer.

**Goal:** Enable component-by-component optimization without modifying existing components.

**Timeline:** 2-3 days

---

## Phase 1: Refactor Metrics Collection (Day 1) ✅

**Objective:** Extract metric computation logic from `EnhancedMetricsCollector` into reusable, isolated functions.

### Tasks

#### 1.1 Create Component Metrics Module
- [ ] Create file: `autorag/cosmos/metrics/__init__.py`
- [ ] Create file: `autorag/cosmos/metrics/component_metrics.py`

#### 1.2 Extract Chunking Metrics
- [ ] Copy chunking metric logic from `EnhancedMetricsCollector` (lines 119-145)
- [ ] Create `compute_chunking_metrics(chunks, latency)` method
- [ ] Returns: `{'time', 'chunks_created', 'avg_chunk_size', 'size_variance', 'semantic_coherence'}`

#### 1.3 Extract Retrieval Metrics
- [ ] Copy retrieval metric logic from `EnhancedMetricsCollector` (lines 166-190)
- [ ] Create `compute_retrieval_metrics(query, results, latency)` method
- [ ] Returns: `{'time', 'docs_retrieved', 'avg_relevance', 'max_relevance', 'precision', 'score_spread'}`

#### 1.4 Extract Generation Metrics
- [ ] Copy generation metric logic from `EnhancedMetricsCollector` (lines 228-257)
- [ ] Create `compute_generation_metrics(query, answer, context, latency)` method
- [ ] Returns: `{'time', 'answer_length', 'answer_relevance', 'context_utilization'}`

#### 1.5 Create Unit Tests
- [ ] Create file: `tests/cosmos/test_component_metrics.py`
- [ ] Test chunking metrics computation
- [ ] Test retrieval metrics computation
- [ ] Test generation metrics computation

**Success Criteria:**
- ✅ All metric methods are pure functions (no side effects)
- ✅ Can compute metrics without full pipeline execution
- ✅ Tests pass for all component types
- ✅ Metrics values match original `EnhancedMetricsCollector` output

**Estimated Time:** 3-4 hours

---

## Phase 2: Component Wrapper (Day 1-2)

**Objective:** Create COSMOS wrapper that adds `process_with_metrics()` to existing components.

### Tasks

#### 2.1 Create Base Wrapper Class
- [ ] Create file: `autorag/cosmos/component_wrapper.py`
- [ ] Implement `COSMOSComponent` class
- [ ] Constructor: `__init__(base_component, component_type, metric_collector)`
- [ ] Properties: `base`, `type`, `config`, `metrics_history`

#### 2.2 Implement Chunker Wrapper
- [ ] Implement `process_with_metrics()` for chunker type
- [ ] Call `base.chunk()` and measure timing
- [ ] Compute metrics using `ComponentMetrics.compute_chunking_metrics()`
- [ ] Return `(chunks, metrics)`

#### 2.3 Implement Retriever Wrapper
- [ ] Implement `process_with_metrics()` for retriever type
- [ ] Call `base.retrieve()` and measure timing
- [ ] Compute metrics using `ComponentMetrics.compute_retrieval_metrics()`
- [ ] Return `(results, metrics)`

#### 2.4 Implement Generator Wrapper
- [ ] Implement `process_with_metrics()` for generator type
- [ ] Call `base.generate()` and measure timing
- [ ] Compute metrics using `ComponentMetrics.compute_generation_metrics()`
- [ ] Return `(answer, metrics)`

#### 2.5 Add Metrics History Tracking
- [ ] Implement `get_average_metrics()` method
- [ ] Implement `clear_metrics()` method
- [ ] Store all metrics in `metrics_history` list

#### 2.6 Create Unit Tests
- [ ] Create file: `tests/cosmos/test_component_wrapper.py`
- [ ] Test wrapping FixedSizeChunker
- [ ] Test wrapping DenseRetriever
- [ ] Test wrapping MockGenerator
- [ ] Test metrics history accumulation

**Success Criteria:**
- ✅ Can wrap any existing component without modification
- ✅ `process_with_metrics()` returns correct output format
- ✅ Metrics match expected values
- ✅ Original component functionality unchanged
- ✅ Can unwrap to get original component

**Estimated Time:** 4-5 hours

---

## Phase 3: Component Evaluators (Day 2-3)

**Objective:** Create evaluators that assess component quality in isolation or with fixed upstream components.

### Tasks

#### 3.1 Create Evaluator Base Class
- [ ] Create file: `autorag/cosmos/optimization/evaluators.py`
- [ ] Implement `ComponentEvaluator` base class
- [ ] Constructor: `__init__(component_type, test_data, upstream_components=None)`

#### 3.2 Implement Chunker Evaluator
- [ ] Create `evaluate_chunker(config)` method
- [ ] Build chunker from config
- [ ] Wrap with COSMOSComponent
- [ ] Execute on test documents
- [ ] Compute quality score (heuristic: size distribution + coherence)
- [ ] Return scalar score [0, 1]

#### 3.3 Implement Retriever Evaluator
- [ ] Create `evaluate_retriever(config)` method
- [ ] Use fixed upstream chunker (from `upstream_components`)
- [ ] Build retriever from config
- [ ] Wrap with COSMOSComponent
- [ ] Execute on test queries
- [ ] Compute quality score (avg relevance)
- [ ] Return scalar score [0, 1]

#### 3.4 Implement Generator Evaluator
- [ ] Create `evaluate_generator(config)` method
- [ ] Use fixed upstream chunker + retriever (from `upstream_components`)
- [ ] Build generator from config
- [ ] Wrap with COSMOSComponent
- [ ] Execute on test queries
- [ ] Compute quality score (semantic similarity to ground truth)
- [ ] Return scalar score [0, 1]

#### 3.5 Create Helper: Component Builder
- [ ] Create `build_component(component_type, config)` function
- [ ] Maps component_type to component class
- [ ] Instantiates component with config
- [ ] Returns component instance

#### 3.6 Create Unit Tests
- [ ] Create file: `tests/cosmos/test_evaluators.py`
- [ ] Test chunker evaluation in isolation
- [ ] Test retriever evaluation with fixed chunker
- [ ] Test generator evaluation with fixed upstream
- [ ] Verify scores are in [0, 1] range

**Success Criteria:**
- ✅ Can evaluate chunker without retriever/generator
- ✅ Can evaluate retriever with fixed chunker
- ✅ Can evaluate generator with fixed upstream components
- ✅ Scores are meaningful and reproducible
- ✅ No circular dependencies

**Estimated Time:** 5-6 hours

---

## Phase 4: Optimization Framework (Day 3)

**Objective:** Create optimization task abstraction and wrap existing Bayesian optimizer.

### Tasks

#### 4.1 Create Optimization Task
- [ ] Create file: `autorag/cosmos/optimization/task.py`
- [ ] Implement `OptimizationTask` dataclass
- [ ] Fields: `component_id`, `search_space`, `evaluator`, `budget`, `context`

#### 4.2 Create Strategy Abstraction
- [ ] Create file: `autorag/cosmos/optimization/strategy.py`
- [ ] Implement `OptimizationStrategy` abstract base class
- [ ] Abstract method: `optimize(task: OptimizationTask) -> OptimizationResult`
- [ ] Abstract method: `get_name() -> str`

#### 4.3 Wrap Existing Bayesian Optimizer
- [ ] Create file: `autorag/cosmos/optimization/bayesian_strategy.py`
- [ ] Implement `BayesianStrategy(OptimizationStrategy)`
- [ ] Constructor: `__init__(n_initial_points=5)`
- [ ] Wrap existing `SimpleBayesianOptimizer` in `optimize()` method
- [ ] Map `OptimizationTask` to `SimpleBayesianOptimizer` parameters
- [ ] Return `OptimizationResult`

#### 4.4 Create Random Search Strategy (Baseline)
- [ ] Create file: `autorag/cosmos/optimization/random_strategy.py`
- [ ] Implement `RandomSearchStrategy(OptimizationStrategy)`
- [ ] Sample random configs from search space
- [ ] Evaluate each config
- [ ] Return best config

#### 4.5 Create Unit Tests
- [ ] Create file: `tests/cosmos/test_optimization_strategies.py`
- [ ] Test BayesianStrategy on simple function
- [ ] Test RandomSearchStrategy on simple function
- [ ] Verify both return valid OptimizationResult
- [ ] Compare convergence (Bayesian should be better)

**Success Criteria:**
- ✅ Strategy abstraction is clean and minimal
- ✅ BayesianStrategy successfully wraps SimpleBayesianOptimizer
- ✅ Can swap strategies without changing other code
- ✅ Both strategies work on same task

**Estimated Time:** 3-4 hours

---

## Phase 5: Compositional Optimizer (Day 3)

**Objective:** Orchestrate component-by-component optimization with context passing.

### Tasks

#### 5.1 Create Compositional Optimizer
- [ ] Create file: `autorag/cosmos/optimization/compositional_optimizer.py`
- [ ] Implement `CompositionalOptimizer` class
- [ ] Constructor: `__init__(optimization_strategy: OptimizationStrategy)`

#### 5.2 Implement Sequential Optimization
- [ ] Implement `optimize(components, search_spaces, test_data, total_budget)` method
- [ ] Allocate budget across components (equal split initially)
- [ ] Loop through components in forward order
- [ ] For each component:
  - Create ComponentEvaluator with upstream context
  - Create OptimizationTask
  - Run optimization with strategy
  - Store result
  - Build best component for downstream use
  - Update upstream_components dict

#### 5.3 Implement Result Tracking
- [ ] Store all component results in `component_results` dict
- [ ] Track total evaluations
- [ ] Track total time
- [ ] Return summary with all component configs

#### 5.4 Add Logging and Progress Tracking
- [ ] Log start of each component optimization
- [ ] Log best score after each component
- [ ] Log final results summary
- [ ] Save results to JSON file

#### 5.5 Create Integration Tests
- [ ] Create file: `tests/cosmos/test_compositional_optimizer.py`
- [ ] Test optimization with 2 components (chunker + retriever)
- [ ] Test optimization with 3 components (chunker + retriever + generator)
- [ ] Verify upstream context is passed correctly
- [ ] Compare with monolithic optimization on same data

**Success Criteria:**
- ✅ Can optimize multiple components sequentially
- ✅ Downstream components use best upstream components
- ✅ Budget is distributed across components
- ✅ Results are stored and retrievable
- ✅ Total evaluations = budget_per_component × num_components

**Estimated Time:** 4-5 hours

---

## Phase 6: End-to-End Script (Day 3-4)

**Objective:** Create runnable script demonstrating compositional optimization on RAG system.

### Tasks

#### 6.1 Create Example Script
- [ ] Create file: `scripts/run_cosmos_optimization.py`
- [ ] Load test data (MS MARCO or fallback)
- [ ] Define search spaces for each component
- [ ] Create test data structure

#### 6.2 Implement Script Logic
- [ ] Initialize ComponentMetrics with SemanticMetrics
- [ ] Create BayesianStrategy
- [ ] Create CompositionalOptimizer
- [ ] Define component list: ['chunker', 'retriever', 'generator']
- [ ] Define search spaces for each
- [ ] Run optimization with budget=30 (10 per component)

#### 6.3 Add Result Display
- [ ] Print optimization progress
- [ ] Display best config for each component
- [ ] Display best score for each component
- [ ] Show total evaluations and time
- [ ] Save results to JSON

#### 6.4 Add Comparison Mode
- [ ] Optional: Run monolithic optimization with same budget
- [ ] Compare final accuracy scores
- [ ] Compare optimization time
- [ ] Display side-by-side comparison

#### 6.5 Add Command-Line Arguments
- [ ] `--budget`: Total evaluation budget
- [ ] `--num-docs`: Number of test documents
- [ ] `--num-queries`: Number of test queries
- [ ] `--strategy`: 'bayesian' or 'random'
- [ ] `--compare-monolithic`: Run comparison with existing optimizer

**Success Criteria:**
- ✅ Script runs without errors
- ✅ Produces meaningful optimization results
- ✅ Results are comparable to existing optimizer
- ✅ Clear output showing component-level insights

**Estimated Time:** 3-4 hours

---

## Phase 7: Validation and Documentation (Day 4)

**Objective:** Validate framework works and document usage.

### Tasks

#### 7.1 Run Validation Tests
- [ ] Run all unit tests: `pytest tests/cosmos/`
- [ ] Run integration script with small dataset (5 docs, 3 queries)
- [ ] Run integration script with larger dataset (20 docs, 10 queries)
- [ ] Compare with existing `run_bayesian_full_space_enhanced.py`

#### 7.2 Performance Validation
- [ ] Measure time for compositional optimization
- [ ] Measure time for monolithic optimization
- [ ] Compare final accuracy scores
- [ ] Verify compositional is faster or comparable

#### 7.3 Create README
- [ ] Create file: `autorag/cosmos/README.md`
- [ ] Document architecture overview
- [ ] Document each module's purpose
- [ ] Provide usage examples
- [ ] Document how to add new strategies

#### 7.4 Create Usage Examples
- [ ] Example 1: Optimize single component
- [ ] Example 2: Optimize two components
- [ ] Example 3: Compare strategies
- [ ] Example 4: Add custom metric

#### 7.5 Update Main Project README
- [ ] Add section on COSMOS framework
- [ ] Link to cosmos/README.md
- [ ] Add example command

**Success Criteria:**
- ✅ All tests pass
- ✅ Compositional optimization produces valid results
- ✅ Performance is acceptable (≤ 2x monolithic time)
- ✅ Documentation is clear and complete

**Estimated Time:** 3-4 hours

---

## File Structure

```
autorag/
├── cosmos/                                    [NEW]
│   ├── __init__.py
│   ├── README.md                             [Phase 7]
│   ├── component_wrapper.py                  [Phase 2]
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── component_metrics.py              [Phase 1]
│   └── optimization/
│       ├── __init__.py
│       ├── task.py                           [Phase 4]
│       ├── strategy.py                       [Phase 4]
│       ├── bayesian_strategy.py              [Phase 4]
│       ├── random_strategy.py                [Phase 4]
│       ├── evaluators.py                     [Phase 3]
│       └── compositional_optimizer.py        [Phase 5]
│
├── components/                               [EXISTING - No changes]
│   └── ...
│
└── optimization/                             [EXISTING - No changes]
    └── bayesian_search.py

scripts/
├── run_bayesian_full_space_enhanced.py      [EXISTING]
└── run_cosmos_optimization.py               [Phase 6]

tests/
└── cosmos/                                   [NEW]
    ├── test_component_metrics.py            [Phase 1]
    ├── test_component_wrapper.py            [Phase 2]
    ├── test_evaluators.py                   [Phase 3]
    ├── test_optimization_strategies.py      [Phase 4]
    └── test_compositional_optimizer.py      [Phase 5]
```

---

## Deferred Features (Future Work)

### Not Included in Minimal MVP:
1. Interface contracts and validation
2. Hierarchical optimization (micro/meso/macro)
3. Multi-objective optimization (Pareto frontiers)
4. Transfer learning across domains
5. Multiple orchestration strategies (backward, inside-out)
6. Agentic component support
7. MLflow integration
8. Adaptive budget allocation
9. Component importance weighting
10. Cross-validation for configuration selection

---

## Key Design Decisions

### 1. Wrapper Pattern (Not Inheritance)
- **Rationale:** Don't modify existing components
- **Benefit:** Zero breaking changes, easy to adopt

### 2. Forward-Only Orchestration
- **Rationale:** Simplest to implement and understand
- **Benefit:** Matches intuition (optimize input→output)

### 3. Equal Budget Allocation
- **Rationale:** No prior knowledge of component importance
- **Benefit:** Simple and fair

### 4. Single-Objective (Accuracy Only)
- **Rationale:** Focus on core optimization first
- **Benefit:** Simpler API, faster implementation

### 5. Reuse Existing Metrics
- **Rationale:** EnhancedMetricsCollector already has good metrics
- **Benefit:** Consistent with existing evaluation

### 6. Wrap SimpleBayesianOptimizer
- **Rationale:** It already works well
- **Benefit:** Don't reinvent the wheel

---

## Success Metrics

### Technical Metrics:
- ✅ All unit tests pass (>90% coverage for new code)
- ✅ Can optimize 3-component pipeline with budget=30
- ✅ Total optimization time < 2x monolithic approach
- ✅ Final accuracy within 5% of monolithic optimization

### Code Quality Metrics:
- ✅ Zero breaking changes to existing components
- ✅ Clean separation of concerns
- ✅ Minimal code duplication
- ✅ Clear, documented interfaces

### Research Enablement Metrics:
- ✅ Can add new optimization strategy in <1 hour
- ✅ Can swap strategies without changing other code
- ✅ Component-level insights are actionable
- ✅ Easy to extend with new component types

---

## Risk Mitigation

### Risk 1: Metrics Don't Correlate with Final Performance
- **Mitigation:** Use semantic similarity (proven in existing system)
- **Validation:** Compare component scores with end-to-end accuracy

### Risk 2: Sequential Optimization Misses Global Optimum
- **Mitigation:** Accept trade-off for speed and interpretability
- **Future:** Add workflow-level joint optimization (meso level)

### Risk 3: Budget Allocation is Suboptimal
- **Mitigation:** Start with equal split (simple baseline)
- **Future:** Implement adaptive budget allocation

### Risk 4: Component Evaluators are Too Slow
- **Mitigation:** Use cached embeddings (existing CachedEmbedder)
- **Mitigation:** Use small test set (10 docs, 5 queries)

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 3-4 hours | Component metrics extraction |
| Phase 2 | 4-5 hours | Component wrapper |
| Phase 3 | 5-6 hours | Component evaluators |
| Phase 4 | 3-4 hours | Strategy abstraction |
| Phase 5 | 4-5 hours | Compositional optimizer |
| Phase 6 | 3-4 hours | End-to-end script |
| Phase 7 | 3-4 hours | Validation & docs |
| **Total** | **25-32 hours** | **Complete COSMOS framework** |

**Estimated Calendar Time:** 3-4 working days

---

## Next Steps

1. ✅ Create this tasks.md document
2. ✅ Commit current state to git
3. ✅ Begin Phase 1 implementation
4. Track progress by checking off tasks
5. Run tests after each phase
6. Update this document with any deviations or insights