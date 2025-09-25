# Multi-Stage Bayesian Optimization for Modular RAG Systems

## Overview

This document outlines how to transition from single-metric optimization (answer semantic accuracy) to a rich multi-stage Bayesian optimization approach while maintaining the modularity of the current RAG system.

## Current State Analysis

### What We Have
- Modular component system with clean interfaces (Chunker, Retriever, Reranker, Generator)
- Component registry for automatic discovery
- Single evaluation metric: semantic similarity of final answers
- Grid search capability testing 960 configurations
- MS MARCO test results showing 16-25% accuracy (using keyword search only)

### What We Need
- Stage-wise evaluation metrics
- Multi-objective optimization capability
- Efficient exploration of configuration space
- Preservation of component modularity

## Stage-wise Metrics Architecture

### Principle: Each Component Reports Its Own Metrics

Instead of only measuring final output, each component should produce evaluation metrics alongside its primary output. This maintains modularity while enabling rich optimization signals.

### Metric Collection Strategy

#### Component Interface Extension
Each component returns both its output AND metrics:
```
Original: chunks = chunker.process(documents)
New: chunks, chunking_metrics = chunker.process_with_metrics(documents)
```

#### Metrics Dictionary Structure
Each stage produces a standardized metrics dictionary:
```
{
    "quality": {...},      # Stage-specific quality measures
    "efficiency": {...},   # Latency and resource usage
    "statistics": {...},   # Descriptive statistics
    "signals": {...}       # Predictive signals for downstream
}
```

### Stage-Specific Metrics

#### Chunking Metrics
**Quality Metrics:**
- `semantic_coherence`: Average cosine similarity within chunks
- `boundary_quality`: Percentage of clean sentence/paragraph boundaries
- `information_density`: Ratio of content words to total words
- `completeness_score`: Percentage of complete semantic units

**Efficiency Metrics:**
- `chunking_time`: Time to process all documents
- `chunks_created`: Total number of chunks
- `avg_chunk_size`: Average tokens per chunk
- `size_variance`: Standard deviation of chunk sizes

**Predictive Signals:**
- `retrieval_difficulty`: Estimated challenge for retrieval (based on chunk similarity)
- `coverage_ratio`: Proportion of original content preserved
- `redundancy_score`: Amount of information overlap

#### Retrieval Metrics
**Quality Metrics:**
- `recall@k`: For k in [5, 10, 20]
- `precision@k`: For k in [5, 10, 20]
- `mrr`: Mean Reciprocal Rank
- `coverage`: Percentage of queries with at least one relevant result

**Efficiency Metrics:**
- `query_encoding_time`: Time to encode query
- `search_time`: Time to search index
- `num_candidates`: Documents considered
- `index_memory`: Memory usage of index

**Predictive Signals:**
- `result_diversity`: Diversity of retrieved documents
- `confidence_scores`: Average retrieval confidence
- `relevance_gap`: Score difference between top results

#### Reranking Metrics
**Quality Metrics:**
- `ndcg@k`: For k in [1, 3, 5]
- `rerank_improvement`: Improvement over initial retrieval
- `top1_accuracy`: Correct document at position 1
- `ranking_stability`: Consistency across similar queries

**Efficiency Metrics:**
- `scoring_time`: Time to score all pairs
- `model_load_time`: One-time model loading cost
- `pairs_evaluated`: Number of query-document pairs

**Predictive Signals:**
- `confidence_spread`: Distribution of reranking scores
- `reorder_distance`: How much reranking changed order
- `discrimination_power`: Ability to separate relevant/irrelevant

#### Generation Metrics
**Quality Metrics:**
- `answer_relevance`: Relevance to query
- `answer_faithfulness`: Grounding in retrieved documents
- `answer_completeness`: Addressing all aspects of query
- `fluency_score`: Language quality

**Efficiency Metrics:**
- `generation_time`: Time to generate
- `tokens_used`: Input + output tokens
- `retry_count`: Number of generation attempts

**Cost Metrics:**
- `api_cost`: Direct API costs
- `token_cost`: Based on token usage
- `total_cost`: Sum of all costs

## Multi-Objective Bayesian Optimization Framework

### Objective Definition

Instead of optimizing single metric, we optimize three objectives simultaneously:

1. **Accuracy Composite Score**
   ```
   accuracy = 0.3 * retrieval_recall@10 +
              0.2 * rerank_ndcg@5 +
              0.5 * answer_relevance
   ```

2. **Latency Composite Score**
   ```
   latency = chunking_time +
             query_encoding_time +
             search_time +
             scoring_time +
             generation_time
   ```

3. **Cost Composite Score**
   ```
   cost = embedding_cost +
          storage_cost +
          api_cost +
          compute_cost
   ```

### Pareto Optimization Approach

#### Using Surrogate Models
Build three Gaussian Process models:
- One predicting accuracy from configuration
- One predicting latency from configuration
- One predicting cost from configuration

Each model uses stage metrics as additional features for better predictions.

#### Acquisition Function: Expected Hypervolume Improvement (EHVI)
EHVI balances:
- Exploring uncertain regions
- Improving individual objectives
- Finding better trade-offs

### Implementation Strategy

#### Phase 1: Baseline Collection (Week 1)
1. Run 30 random configurations
2. Collect all stage metrics
3. Establish baseline Pareto frontier
4. Identify correlation patterns

#### Phase 2: Guided Exploration (Week 2-3)
1. Train initial surrogate models
2. Use EHVI to select next configurations
3. Focus on promising regions
4. Update models after each evaluation

#### Phase 3: Refinement (Week 4)
1. Fine-tune around Pareto optimal points
2. Validate with additional test queries
3. Document trade-offs
4. Select operating points

## Implementation Approaches: From Simple to Complete

### Approach 1: Minimal External Metrics (2-3 days)

The simplest approach that requires NO component modifications:

```python
# File: autorag/evaluation/external_metrics.py

class ExternalMetricsCollector:
    """Collects metrics without modifying components"""

    def evaluate_pipeline_run(self, pipeline, query, documents, ground_truth=None):
        metrics = {}

        # Measure chunking externally
        start = time.time()
        chunks = pipeline.chunker.process(documents)
        metrics['chunking'] = {
            'time': time.time() - start,
            'count': len(chunks),
            'avg_size': np.mean([len(c.content.split()) for c in chunks])
        }

        # Measure retrieval externally
        start = time.time()
        retrieved = pipeline.retriever.retrieve(query, top_k=10)
        metrics['retrieval'] = {
            'time': time.time() - start,
            'top_scores': [r.score for r in retrieved[:5]],
            'score_spread': max(r.score for r in retrieved) - min(r.score for r in retrieved)
        }
        if ground_truth:
            metrics['retrieval']['recall@10'] = self.calculate_recall(retrieved, ground_truth)

        # Measure generation externally
        start = time.time()
        answer = pipeline.generator.generate(query, retrieved[:5])
        metrics['generation'] = {
            'time': time.time() - start,
            'tokens': len(answer.split()),
            'cost_estimate': len(answer.split()) * 0.002 / 1000  # Rough estimate
        }

        # Overall metrics
        metrics['total'] = {
            'latency': sum(m['time'] for m in metrics.values() if 'time' in m),
            'accuracy': self.calculate_accuracy(answer, ground_truth) if ground_truth else None
        }

        return answer, metrics
```

**Advantages:** Zero code changes to existing components, immediate implementation
**Disadvantages:** Limited metrics, no internal component insights

### Approach 2: Hybrid Metrics with Minimal Intrusion (1 week)

A balanced approach using decorators, mixins, and pipeline orchestration:

#### Step 1: Universal Metrics Decorator

```python
# File: autorag/metrics/decorators.py

def collect_basic_metrics(component_type):
    """Decorator that adds basic metrics to any component method"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Pre-execution metrics
            start_time = time.time()
            start_memory = get_memory_usage()

            # Execute original function
            result = func(self, *args, **kwargs)

            # Post-execution metrics
            elapsed = time.time() - start_time
            memory_used = get_memory_usage() - start_memory

            # Basic metrics all components have
            metrics = {
                'execution_time': elapsed,
                'memory_delta': memory_used,
                'timestamp': time.time(),
                'component_type': component_type,
                'success': result is not None
            }

            # Add component-specific external observations
            if component_type == 'chunker':
                metrics['output_count'] = len(result) if hasattr(result, '__len__') else 1
            elif component_type == 'retriever':
                metrics['results_returned'] = len(result) if hasattr(result, '__len__') else 0

            return result, metrics
        return wrapper
    return decorator
```

#### Step 2: Component-Specific Metrics Mixins

```python
# File: autorag/metrics/mixins.py

class MetricsCollectorMixin:
    """Base mixin for metrics collection"""

    def get_base_metrics(self, start_time):
        return {
            'execution_time': time.time() - start_time,
            'component_name': self.__class__.__name__,
            'config': self.config
        }

class ChunkerMetricsMixin(MetricsCollectorMixin):
    """Chunker-specific metrics"""

    def collect_chunk_metrics(self, chunks, start_time):
        metrics = self.get_base_metrics(start_time)

        # Quality metrics
        if chunks:
            metrics['quality'] = {
                'chunk_count': len(chunks),
                'avg_chunk_size': np.mean([len(c.content.split()) for c in chunks]),
                'size_std': np.std([len(c.content.split()) for c in chunks]),
                'min_chunk_size': min(len(c.content.split()) for c in chunks),
                'max_chunk_size': max(len(c.content.split()) for c in chunks)
            }

            # Check for sentence completeness (simple heuristic)
            metrics['quality']['complete_sentences_ratio'] = sum(
                1 for c in chunks if c.content.rstrip().endswith(('.', '!', '?'))
            ) / len(chunks)

        return metrics

class RetrieverMetricsMixin(MetricsCollectorMixin):
    """Retriever-specific metrics"""

    def collect_retrieval_metrics(self, query, results, start_time, ground_truth=None):
        metrics = self.get_base_metrics(start_time)

        if results:
            metrics['quality'] = {
                'num_results': len(results),
                'top_score': results[0].score if results else 0,
                'score_range': max(r.score for r in results) - min(r.score for r in results),
                'avg_score': np.mean([r.score for r in results]),
                'score_std': np.std([r.score for r in results])
            }

            # If ground truth available
            if ground_truth:
                relevant = set(ground_truth.relevant_ids)
                retrieved = set(r.chunk_id for r in results[:10])
                metrics['quality']['recall@10'] = len(relevant & retrieved) / len(relevant) if relevant else 0
                metrics['quality']['precision@10'] = len(relevant & retrieved) / 10

        return metrics
```

#### Step 3: Modified Component Classes Using Mixins

```python
# File: autorag/components/chunkers/enhanced_chunkers.py

from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.metrics.mixins import ChunkerMetricsMixin

class FixedSizeChunkerWithMetrics(FixedSizeChunker, ChunkerMetricsMixin):
    """Fixed-size chunker with metrics collection"""

    def process_with_metrics(self, documents):
        start_time = time.time()

        # Use original chunking logic
        chunks = self.chunk(documents)

        # Collect metrics
        metrics = self.collect_chunk_metrics(chunks, start_time)

        # Add implementation-specific metrics
        metrics['implementation_details'] = {
            'chunking_method': 'fixed_size',
            'chunk_size': self.chunk_size,
            'overlap': self.overlap
        }

        return chunks, metrics

    # Original process method still works normally
    def process(self, documents):
        return self.chunk(documents)
```

#### Step 4: Smart Pipeline Orchestration

```python
# File: autorag/pipeline/metrics_aware_pipeline.py

class MetricsAwareRAGPipeline:
    """Pipeline that intelligently collects metrics from components"""

    def __init__(self, config):
        self.components = self._initialize_components(config)
        self.metrics_mode = config.get('metrics_mode', 'hybrid')  # 'none', 'external', 'internal', 'hybrid'

    def run_with_metrics(self, query, documents, ground_truth=None):
        """Run pipeline collecting metrics based on mode"""
        all_metrics = {'mode': self.metrics_mode}

        # CHUNKING
        if hasattr(self.chunker, 'process_with_metrics') and self.metrics_mode in ['internal', 'hybrid']:
            chunks, chunk_metrics = self.chunker.process_with_metrics(documents)
            all_metrics['chunking_internal'] = chunk_metrics
        else:
            start = time.time()
            chunks = self.chunker.process(documents)
            if self.metrics_mode in ['external', 'hybrid']:
                all_metrics['chunking_external'] = {
                    'time': time.time() - start,
                    'count': len(chunks)
                }

        # RETRIEVAL
        if hasattr(self.retriever, 'process_with_metrics') and self.metrics_mode in ['internal', 'hybrid']:
            results, retrieval_metrics = self.retriever.process_with_metrics(query, chunks)
            all_metrics['retrieval_internal'] = retrieval_metrics
        else:
            start = time.time()
            results = self.retriever.retrieve(query, top_k=10)
            if self.metrics_mode in ['external', 'hybrid']:
                all_metrics['retrieval_external'] = {
                    'time': time.time() - start,
                    'count': len(results)
                }

        # Merge internal and external metrics if in hybrid mode
        if self.metrics_mode == 'hybrid':
            all_metrics = self._merge_metrics(all_metrics)

        # GENERATION
        start = time.time()
        answer = self.generator.generate(query, results[:5])
        all_metrics['generation'] = {
            'time': time.time() - start,
            'answer_length': len(answer.split())
        }

        # OVERALL METRICS
        all_metrics['overall'] = self._calculate_overall_metrics(all_metrics, answer, ground_truth)

        return answer, all_metrics

    def _merge_metrics(self, metrics):
        """Intelligently merge internal and external metrics"""
        merged = {}
        for stage in ['chunking', 'retrieval']:
            internal = metrics.get(f'{stage}_internal', {})
            external = metrics.get(f'{stage}_external', {})

            # Combine both sources, preferring internal when available
            merged[stage] = {**external, **internal}

            # Mark source for transparency
            merged[stage]['metrics_source'] = 'hybrid'

        return merged
```

### Approach 3: Full Implementation with All Patterns (2-3 weeks)

Complete implementation using all three patterns for maximum flexibility:

```python
# File: autorag/metrics/full_implementation.py

class ComponentMetricsFactory:
    """Factory for creating metrics-aware components"""

    @staticmethod
    def create_with_metrics(component_class, metrics_strategy='hybrid'):
        """Dynamically create metrics-aware version of any component"""

        if metrics_strategy == 'decorator':
            # Use decorator pattern
            class DecoratedComponent(component_class):
                @collect_basic_metrics(component_class.__name__)
                def process(self, *args, **kwargs):
                    return super().process(*args, **kwargs)
            return DecoratedComponent

        elif metrics_strategy == 'mixin':
            # Use mixin pattern
            mixin = get_mixin_for_component(component_class)
            class MixedComponent(component_class, mixin):
                def process_with_metrics(self, *args, **kwargs):
                    start = time.time()
                    result = self.process(*args, **kwargs)
                    metrics = self.collect_metrics(result, start, *args, **kwargs)
                    return result, metrics
            return MixedComponent

        elif metrics_strategy == 'hybrid':
            # Combine all patterns
            mixin = get_mixin_for_component(component_class)

            @collect_basic_metrics(component_class.__name__)
            class HybridComponent(component_class, mixin):
                def process_with_metrics(self, *args, **kwargs):
                    # Decorator handles basic metrics
                    result, basic_metrics = self.process(*args, **kwargs)

                    # Mixin provides rich internal metrics
                    internal_metrics = self.collect_internal_metrics(result, *args, **kwargs)

                    # Combine all metrics
                    return result, {**basic_metrics, **internal_metrics}

                @collect_basic_metrics(component_class.__name__)
                def process(self, *args, **kwargs):
                    return super().process(*args, **kwargs)

            return HybridComponent
```

## Integration with Current System

### Choosing the Right Implementation Approach

#### Decision Framework

**Use Approach 1 (External Metrics) when:**
- You need results immediately (2-3 days)
- Components cannot be modified
- Basic optimization is sufficient
- You're validating if Bayesian optimization helps at all

**Use Approach 2 (Hybrid) when:**
- You have a week to implement
- You need richer metrics but want minimal disruption
- You want to gradually enhance metrics over time
- You need both backward compatibility and new capabilities

**Use Approach 3 (Full Implementation) when:**
- Building a production system
- Need comprehensive monitoring
- Have 2-3 weeks for implementation
- Want maximum flexibility and insight

#### Migration Path

Start with Approach 1 and evolve:

```python
# Week 1: External metrics only
evaluator = ExternalMetricsCollector()
answer, metrics = evaluator.evaluate_pipeline_run(pipeline, query, docs)

# Week 2: Add mixins to critical components
class EnhancedRetriever(BM25Retriever, RetrieverMetricsMixin):
    def process_with_metrics(self, query, chunks):
        # Rich internal metrics
        pass

# Week 3: Full hybrid system
pipeline = MetricsAwareRAGPipeline(config={'metrics_mode': 'hybrid'})
answer, comprehensive_metrics = pipeline.run_with_metrics(query, docs)
```

### Implementation Priorities

#### Phase 1: Minimum Viable Metrics (Days 1-3)
1. Implement ExternalMetricsCollector
2. Set up basic Bayesian optimization
3. Run initial experiments
4. Validate approach works

#### Phase 2: Enhanced Metrics (Week 2)
1. Add mixins for components showing high variance
2. Implement hybrid pipeline
3. Collect richer metrics where needed
4. Refine optimization

#### Phase 3: Full System (Optional, Weeks 3-4)
1. Complete metrics factory
2. Add all component metrics
3. Implement advanced acquisition functions
4. Production monitoring

### Optimization Loop

#### Simple Bayesian Optimization Runner
```python
class MultiStageOptimizer:
    def __init__(self, search_space, n_calls=100):
        self.search_space = search_space
        self.n_calls = n_calls
        self.results = []

    def optimize(self):
        # Initial random exploration
        for i in range(20):
            config = self.search_space.sample_random()
            metrics = self.evaluate_configuration(config)
            self.results.append((config, metrics))

        # Bayesian optimization
        for i in range(20, self.n_calls):
            # Train surrogate models on results
            models = self.train_surrogates(self.results)

            # Select next configuration using EHVI
            next_config = self.select_next_ehvi(models)

            # Evaluate
            metrics = self.evaluate_configuration(next_config)
            self.results.append((next_config, metrics))

            # Update Pareto frontier
            self.update_frontier()

        return self.get_pareto_frontier()
```

## Practical Workflow

### Day 1-2: Instrumentation
1. Add metrics collection to each component
2. Verify metrics are meaningful
3. Test metrics aggregation

### Day 3-4: Baseline
1. Run current best configuration
2. Run 20-30 random configurations
3. Analyze metric correlations
4. Identify bottlenecks

### Day 5-7: Optimization
1. Implement Bayesian optimizer
2. Define acquisition function
3. Run optimization loop
4. Monitor convergence

### Day 8-9: Analysis
1. Visualize Pareto frontier
2. Analyze trade-offs
3. Select operating points
4. Document findings

### Day 10: Validation
1. Test selected configurations
2. Verify on held-out data
3. Compare to baseline
4. Report improvements

## Key Design Decisions

### Why Stage Metrics Matter
- **Early failure detection**: Stop evaluating bad configurations early
- **Better surrogate models**: Intermediate signals improve predictions
- **Interpretability**: Understand why configurations succeed/fail
- **Debugging**: Identify which stage causes problems

### Why Multi-Objective
- **No single best configuration**: Different use cases need different trade-offs
- **Realistic constraints**: Cost and latency matter in practice
- **Better exploration**: Finding trade-offs reveals unexpected good configurations

### Why Maintain Modularity
- **Component reusability**: Swap components without changing optimization
- **Independent development**: Improve components separately
- **Clean interfaces**: Metrics don't pollute primary functionality
- **Testing simplicity**: Test components in isolation

## Success Criteria

### Quantitative Goals
- Find configurations with 40%+ accuracy (up from 25%)
- Identify 5-10 Pareto optimal configurations
- Reduce optimization time by 70% vs grid search
- Achieve <1s latency for at least one good configuration

### Qualitative Goals
- Understand component interactions
- Identify bottleneck stages
- Build reusable optimization framework
- Document trade-off decisions

## Common Pitfalls to Avoid

1. **Over-engineering metrics**: Start simple, add complexity as needed
2. **Ignoring correlations**: Stage metrics often correlate strongly
3. **Premature optimization**: Get baseline working before optimizing
4. **Metric gaming**: Ensure metrics align with actual goals
5. **Insufficient exploration**: Don't converge too quickly

## Next Steps

1. Implement metrics collection for one component
2. Verify metrics are informative
3. Extend to all components
4. Build simple optimization loop
5. Run experiments
6. Analyze results
7. Iterate and improve

## Tools and Libraries

### Required
- `scikit-optimize`: Bayesian optimization
- `numpy`: Numerical computations
- `pandas`: Data management
- `matplotlib`: Visualization

### Optional but Helpful
- `optuna`: Alternative optimization framework
- `botorch`: Advanced Bayesian optimization
- `pymoo`: Multi-objective optimization
- `plotly`: Interactive visualizations

## Conclusion

Multi-stage Bayesian optimization provides a principled way to explore the configuration space efficiently while maintaining system modularity. By collecting metrics at each stage, we gain insights into component behavior and can make informed optimization decisions. The result is not just better performance, but understanding of why certain configurations work and what trade-offs are available.