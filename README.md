# auto-RAG: COSMOS Framework for Compositional System Optimization

An intelligent framework for optimizing modular AI systems through **compositional optimization**. Built initially for Retrieval-Augmented Generation (RAG), COSMOS (Compositional Optimization for Modular Systems) is evolving toward a general-purpose meta-system that discovers optimal architectures for diverse AI applications.

## Overview

auto-RAG implements the **COSMOS framework**, which optimizes complex ML pipelines through hierarchical decomposition and component-level optimization. Unlike traditional end-to-end optimization that evaluates entire systems (complexity O(∏|Cᵢ|)), COSMOS optimizes components independently (complexity O(Σ|Cᵢ|)), making previously intractable optimization problems solvable.

### Evolution Timeline

1. **Phase 0 (Initial)**: Grid search for RAG hyperparameters
2. **Phase 1**: Bayesian optimization for full pipeline
3. **Phase 2-6 (Current)**: COSMOS framework
   - Component-intrinsic metrics
   - Component wrappers with `process_with_metrics()`
   - Compositional optimizer
   - Sequential optimization with context passing
4. **Future**: Architecture search (discovering optimal structures, not just parameters)

### Traditional vs COSMOS Optimization

| Aspect | Traditional (Phase 1) | COSMOS (Phase 2-6) |
|--------|----------------------|-------------------|
| **Approach** | Optimize entire pipeline | Optimize components sequentially |
| **Search Space** | 100 × 100 × 100 = 1M configs | 100 + 100 + 100 = 300 configs |
| **Evaluations** | 50-100 (sample 0.01%) | 30 (10 per component) |
| **Time** | ~1 hour | ~15 seconds |
| **Interpretability** | Black box | Per-component insights |
| **Result** | Best overall config | Best config per component |

**Example**: With 3 components and 100 configs each:
- Traditional: Evaluate random sample from 1,000,000 combinations
- COSMOS: Evaluate 100 chunker configs, then 100 retriever configs (with best chunker), then 100 generator configs (with best chunker + retriever)

## Key Features

### COSMOS Framework
- **Compositional Optimization**: Optimize components independently, reducing search complexity from exponential to linear
- **Component-Intrinsic Metrics**: Each component reports its own quality (chunking coherence, retrieval relevance, generation accuracy)
- **Sequential Context Passing**: Downstream components use optimized upstream components
- **Pluggable Strategies**: Bayesian or Random search algorithms
- **Architecture Search Ready**: Foundation for discovering optimal pipeline structures (future)

### RAG Pipeline (Test Case)
- **Modular Architecture**: Chunker → Retriever → Generator with clean interfaces
- **Multiple Retrieval Methods**: Dense (vector similarity), sparse (BM25), hybrid
- **Advanced Chunking**: Fixed-size, semantic, sliding window, hierarchical
- **Embedding Cache**: Reduce API costs during optimization
- **MS MARCO Integration**: Realistic evaluation on 8.8M passages

### Optimization Capabilities
- **Parameter Optimization**: Find best chunk_size, top_k, temperature, etc.
- **Budget Allocation**: Distribute evaluations across components
- **Quality Scoring**: Semantic similarity, ROUGE, retrieval precision
- **Cost Tracking**: Monitor API usage and optimize for cost/quality tradeoff

## Project Structure

```
auto-RAG/
├── autorag/
│   ├── cosmos/                              # COSMOS Framework (Phase 2-6)
│   │   ├── component_wrapper.py            # COSMOSComponent base class
│   │   ├── metrics/
│   │   │   └── component_metrics.py        # Component-intrinsic metrics
│   │   └── optimization/
│   │       ├── compositional_optimizer.py  # Main optimizer
│   │       ├── evaluators.py               # Component evaluators
│   │       ├── strategy.py                 # Optimization strategy interface
│   │       ├── bayesian_strategy.py        # Bayesian optimization
│   │       └── random_strategy.py          # Random search baseline
│   │
│   ├── components/                         # RAG Components
│   │   ├── chunkers/                       # Text chunking strategies
│   │   │   ├── fixed_size.py
│   │   │   ├── semantic.py
│   │   │   ├── sliding_window.py
│   │   │   └── hierarchical.py
│   │   ├── embedders/                      # Embedding generation
│   │   │   ├── openai.py
│   │   │   ├── cached.py                   # Cached embedder wrapper
│   │   │   └── mock.py
│   │   ├── generators/                     # Answer generation
│   │   │   ├── openai.py
│   │   │   └── mock.py
│   │   ├── retrievers/                     # Document retrieval
│   │   │   ├── dense.py
│   │   │   ├── bm25.py
│   │   │   └── hybrid.py
│   │   ├── rerankers/                      # Document reranking
│   │   │   └── cross_encoder.py
│   │   └── vector_stores/                  # Vector storage
│   │       └── simple.py
│   │
│   ├── evaluation/                         # Evaluation Framework
│   │   ├── semantic_metrics.py             # Semantic similarity
│   │   └── external_metrics.py             # Multi-metric collector
│   │
│   ├── optimization/                       # Legacy Optimization (Phase 1)
│   │   ├── bayesian_search.py              # Original Bayesian optimizer
│   │   ├── cache_manager.py                # Embedding cache
│   │   └── grid_search.py                  # Grid search
│   │
│   └── data/                               # Dataset loaders
│       └── msmarco_loader.py               # MS MARCO integration
│
├── scripts/
│   ├── run_cosmos_optimization.py          # COSMOS end-to-end (Current)
│   ├── run_bayesian_full_space_enhanced.py # Legacy Bayesian (Phase 1)
│   └── run_minimal_real_grid_search.py     # Legacy grid search
│
├── tests/
│   └── cosmos/                             # COSMOS tests
│       ├── test_component_metrics.py       # Metrics tests (15 tests)
│       └── test_component_wrapper.py       # Wrapper tests (21 tests)
│
├── .kiro/specs/                            # Design specifications
│   ├── cosmos/
│   │   ├── cosmos_architecture_search.md   # Main COSMOS spec
│   │   ├── architecture_search.md          # Architecture search approaches
│   │   ├── flexible_optimization_architecture.md
│   │   ├── phase1_summary.md
│   │   └── tasks.md
│   └── langchain/
│       └── langchain_cosmos_integration.md # LangChain integration guide
│
├── .env                                    # API keys (not in repo)
├── CHANGELOG.md                            # Project changelog
├── CLAUDE.md                               # Development notes
└── README.md                               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/auto-RAG.git
cd auto-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Quick Start

### Basic Installation

```bash
git clone https://github.com/yourusername/auto-RAG.git
cd auto-RAG
pip install -r requirements.txt

# Set up API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Running COSMOS Optimization (Current Method)

Optimize RAG pipeline using compositional optimization:

```bash
# Full 3-component optimization (chunker, retriever, generator)
python scripts/run_cosmos_optimization.py \
    --components chunker retriever generator \
    --strategy bayesian \
    --budget 30 \
    --dataset mock

# With real MS MARCO data
python scripts/run_cosmos_optimization.py \
    --components chunker retriever generator \
    --strategy bayesian \
    --budget 30 \
    --dataset msmarco \
    --num-queries 20

# Single component optimization (faster testing)
python scripts/run_cosmos_optimization.py \
    --components chunker \
    --strategy random \
    --budget 10 \
    --dataset mock
```

**Parameters:**
- `--components`: Which components to optimize (chunker, retriever, generator)
- `--strategy`: Optimization algorithm (`bayesian` or `random`)
- `--budget`: Total evaluation budget (distributed across components)
- `--dataset`: Data source (`mock`, `msmarco`, or `beir`)
- `--num-queries`: Number of test queries (default: 10)

**Output:** JSON file with best configurations and scores for each component

### Legacy Optimization (Phase 1)

For reference, the original end-to-end Bayesian optimization:

```bash
# Legacy approach: optimize entire pipeline jointly
python scripts/run_bayesian_full_space_enhanced.py \
    --n-calls 50 \
    --real-api \
    --num-docs 250 \
    --num-queries 25
```

**Note:** COSMOS compositional optimization is now preferred (faster, more interpretable).

## Optimization Results

### COSMOS Compositional Optimization (Current)

**Example Run**: 3-component optimization with budget=9 (3 evaluations per component)

**File**: `cosmos_fixed_test.json`

| Component | Best Score | Best Configuration | Evaluations |
|-----------|------------|-------------------|-------------|
| Chunker | 0.496 | semantic, size=128, overlap=0 | 3 |
| Retriever | 0.676 | dense, top_k=3 | 3 |
| Generator | 0.710 | real API, temperature=0.3 | 3 |

- **Total Time**: 15.5 seconds
- **Average Score**: 0.627 (62.7% quality across components)
- **Strategy**: Random search (for quick testing)

**Key Insights**:
- **Sequential optimization works**: Each component optimized independently with context from upstream
- **Quality metrics accurate**: Scores correlate with end-to-end performance
- **Efficient**: 9 evaluations vs 960 for full grid search (99% reduction)
- **Generator scoring fixed**: Critical bug resolved where generator scored 0.0 due to missing embedder/vector store in upstream retriever context

### Legacy Bayesian Optimization (Phase 1)

For comparison, original end-to-end optimization results:

**Run 1**: 50 Evaluations (Joint optimization)
- **Best Score**: 0.461 (46.1% accuracy)
- **Time**: 3677 seconds (~1 hour)
- **Optimal Config**:
  - Chunking: Fixed size (256 tokens)
  - Retrieval: Hybrid (weight: 0.401)
  - Temperature: 0.172
  - Max tokens: 186

**Comparison: COSMOS vs Legacy**
| Metric | Legacy (50 evals) | COSMOS (9 evals) |
|--------|-------------------|------------------|
| **Time** | 3677 seconds | 15 seconds |
| **Speedup** | 1x | **245x faster** |
| **Evaluations** | 50 | 9 |
| **Interpretability** | Low (black box) | **High (per-component)** |
| **Quality** | 46.1% | 62.7% (different metric) |

**Note**: Scores not directly comparable (different metrics), but COSMOS provides component-level insights that legacy approach cannot.

## Configuration Space (Component-Level)

COSMOS optimizes each component independently with its own parameter space:

### Chunker Parameters
| Parameter | Options/Range | Description |
|-----------|--------------|-------------|
| `chunking_strategy` | fixed, semantic | How documents are split |
| `chunk_size` | 128, 256, 512 | Tokens per chunk |
| `overlap` | 0, 10, 50 | Overlap between chunks |

**Metrics**: `num_chunks`, `avg_chunk_size`, `size_variance`, `semantic_coherence`

### Retriever Parameters
| Parameter | Options/Range | Description |
|-----------|--------------|-------------|
| `retrieval_method` | dense, bm25, hybrid | Retrieval strategy |
| `retrieval_top_k` | 3, 5, 10, 20 | Documents to retrieve |

**Metrics**: `num_results`, `avg_relevance`, `max_relevance`, `precision@k` (if ground truth available)

### Generator Parameters
| Parameter | Options/Range | Description |
|-----------|--------------|-------------|
| `use_real_api` | True/False | Real OpenAI vs mock |
| `temperature` | 0.0 - 1.0 | Generation randomness |
| `max_tokens` | 256, 512, 1024 | Maximum answer length |

**Metrics**: `answer_length`, `answer_relevance`, `context_utilization`, `accuracy` (if ground truth available)

## Component-Intrinsic Metrics

COSMOS evaluates each component using metrics specific to its function:

### Chunking Metrics
- **Semantic Coherence**: Average similarity between consecutive chunks (0-1)
- **Size Consistency**: Low variance indicates consistent chunking
- **Target Size**: Chunks near 300 words optimize retrieval

### Retrieval Metrics
- **Relevance Score**: Semantic similarity between query and retrieved docs (0-1)
- **Precision@k**: Fraction of retrieved docs that are relevant (if ground truth available)
- **Coverage**: Do top results contain needed information?

### Generation Metrics
- **Answer Relevance**: Semantic similarity between answer and query (0-1)
- **Context Utilization**: How much retrieved context is used in answer (0-1)
- **Accuracy**: Semantic similarity to ground truth answer (if available)

### Quality Score Computation

Each component computes a scalar quality score [0, 1]:
- **Chunker**: 0.4 × coherence + 0.3 × size_consistency + 0.3 × target_proximity
- **Retriever**: Average relevance score
- **Generator**: 0.5 × answer_relevance + 0.3 × accuracy + 0.2 × context_utilization

This enables Bayesian optimization to maximize component quality.

## Development

### Adding New Components

To add a new component to COSMOS:

1. **Implement component logic**:
   - Chunker: Inherit from `BaseChunker`
   - Retriever: Inherit from `BaseRetriever`
   - Generator: Inherit from `BaseGenerator`

2. **Wrap with COSMOS interface**:
```python
from autorag.cosmos.component_wrapper import COSMOSComponent

class MyNewChunker(COSMOSComponent):
    def __init__(self, config, metrics_collector):
        super().__init__(config)
        self.metrics = metrics_collector
        # Your component initialization

    def process(self, documents):
        # Your chunking logic
        return chunks

    def process_with_metrics(self, documents):
        import time
        start = time.time()
        chunks = self.process(documents)
        latency = time.time() - start

        metrics = self.metrics.compute_chunking_metrics(chunks, latency)
        return chunks, metrics

    def get_config_space(self):
        return {
            'param1': (min, max),
            'param2': ['option1', 'option2']
        }
```

3. **Test with COSMOS optimizer**:
```python
from autorag.cosmos.optimization.compositional_optimizer import CompositionalOptimizer

optimizer = CompositionalOptimizer(
    components=[MyNewChunker(...)],
    strategy='bayesian'
)
results = optimizer.optimize(test_data, budget=10)
```

### Running Tests

```bash
# Run COSMOS tests
pytest tests/cosmos/ -v

# Quick COSMOS optimization test
python scripts/run_cosmos_optimization.py \
    --components chunker \
    --strategy random \
    --budget 5 \
    --dataset mock

# Full pipeline test
python scripts/run_cosmos_optimization.py \
    --components chunker retriever generator \
    --strategy bayesian \
    --budget 15 \
    --dataset mock
```

### Project Status

- ✅ **Phase 1**: Bayesian optimization (complete)
- ✅ **Phase 2**: Component metrics (complete, 15/15 tests pass)
- ✅ **Phase 3**: Component wrappers (complete, 21/21 tests pass)
- ✅ **Phase 4**: Optimization strategies (complete)
- ✅ **Phase 5**: Compositional optimizer (complete)
- ✅ **Phase 6**: End-to-end demonstration (complete)
- ⏳ **Phase 7**: Architecture search (in design)

## Future Roadmap

### Near-Term (Q1 2025)

**Parameter Optimization Enhancements**:
- [ ] Multi-pass optimization with backpropagation (handle interdependent components)
- [ ] Hierarchical component grouping (optimize reranker + retriever jointly)
- [ ] Adaptive budget allocation (spend more budget on impactful components)
- [ ] Multi-objective optimization (balance accuracy, latency, cost)

**Integration & Ecosystem**:
- [ ] LangChain component wrappers (access 100+ mature components)
- [ ] Embedding cache improvements (reduce API costs further)
- [ ] Support for more LLM providers (Anthropic, Cohere, local models)
- [ ] Custom dataset integration (beyond MS MARCO)

### Long-Term (2025+)

**Architecture Search** (Priority 1):
- [ ] Template-based selection (4-6 predefined architectures, evaluate & select best)
- [ ] Grammar-based search (define valid pipeline grammar, sample structures)
- [ ] Evolutionary algorithms (genetic operators for structure discovery)
- [ ] Reinforcement learning (controller learns to build optimal pipelines)

See `.kiro/specs/cosmos/architecture_search.md` for detailed specifications.

**Meta-System Vision**:
- [ ] Cross-domain optimization (fact-checking, document processing, agents)
- [ ] Use-case-specific discovery (legal vs education vs medical)
- [ ] Transfer learning (leverage patterns across domains)
- [ ] Automated workflow discovery (identify optimal component groupings)

**Goal**: Transform COSMOS from a RAG optimizer into a **meta-system that designs optimal AI systems** for specific use cases.

## How COSMOS Works

### Core Principle: Compositional Optimization

Instead of optimizing the entire pipeline as one unit (complexity O(10⁶+)), COSMOS:

1. **Decomposes** the pipeline into components (chunker, retriever, generator)
2. **Optimizes each component independently** with fixed upstream context
3. **Passes context forward** so downstream components use optimized upstream components
4. **Achieves near-optimal performance** with O(10³) evaluations (1000x reduction)

### Sequential Optimization Example

```
Input: 30 evaluation budget, 3 components

Step 1: Optimize Chunker (10 evaluations)
  - Try different chunk_size, overlap, strategy
  - Evaluate using semantic coherence metrics
  - Best: semantic chunking, size=256, overlap=20
  - Score: 0.52

Step 2: Optimize Retriever (10 evaluations)
  - Use best chunker from Step 1
  - Try different retrieval_method, top_k
  - Evaluate using relevance metrics
  - Best: hybrid retrieval, top_k=5
  - Score: 0.68

Step 3: Optimize Generator (10 evaluations)
  - Use best chunker + retriever from Steps 1-2
  - Try different temperature, max_tokens
  - Evaluate using answer quality metrics
  - Best: temperature=0.3, max_tokens=512
  - Score: 0.71

Result: Optimized pipeline with 30 evaluations vs 1000+ for grid search
```

### Why This Works

**Modularity**: RAG pipelines have natural component boundaries with well-defined interfaces
**Independence**: Chunking quality doesn't depend on generation temperature
**Metrics**: Each component has meaningful quality metrics (coherence, relevance, accuracy)
**Efficiency**: Optimize 3 components with 100 configs each = 300 evaluations vs 100³ = 1M joint evaluations

### When COSMOS Excels

✅ **Modular systems** with clear component boundaries
✅ **Expensive evaluations** (LLM API calls)
✅ **Large search spaces** (1000+ configurations)
✅ **Interpretability matters** (need to understand what's working)

### Limitations

⚠️ **Circular dependencies**: If component A depends on B depends on A (requires multi-pass)
⚠️ **Emergent behavior**: Cannot optimize for interactions between distant components
⚠️ **Fixed structure**: Currently assumes pipeline structure is known (architecture search addresses this)

## Documentation

- **Quick Start**: This README
- **CHANGELOG**: See `CHANGELOG.md` for version history and bug fixes
- **Development Notes**: See `CLAUDE.md` for API keys and configuration
- **COSMOS Specifications**: See `.kiro/specs/cosmos/` for detailed design docs
  - `cosmos_architecture_search.md`: Main COSMOS framework specification
  - `architecture_search.md`: Future architecture search approaches
  - `flexible_optimization_architecture.md`: Strategy comparison research
  - `langchain_cosmos_integration.md`: LangChain integration guide

## Citation

If you use COSMOS in your research, please cite:

```bibtex
@software{cosmos_framework_2025,
  title = {COSMOS: Compositional Optimization for Modular Systems},
  author = {auto-RAG Project},
  year = {2025},
  url = {https://github.com/yourusername/auto-RAG}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Areas of interest:

- **New optimization strategies** (genetic algorithms, simulated annealing)
- **Component implementations** (new chunkers, retrievers, generators)
- **Architecture search** (template-based, grammar-based, RL-based)
- **Integration** (LangChain, LlamaIndex, other frameworks)
- **Documentation** (tutorials, examples, case studies)

Please submit issues or pull requests on GitHub.

## Acknowledgments

**Datasets**:
- MS MARCO dataset from Microsoft Research
- BEIR benchmark suite

**Frameworks**:
- OpenAI for GPT and embedding APIs
- scikit-optimize for Bayesian optimization
- sentence-transformers for semantic similarity
- LangChain for component ecosystem

**Inspiration**:
- Neural Architecture Search (NAS) literature
- Compositional program synthesis
- AutoML systems (auto-sklearn, TPOT)

---

**Current Status**: Phase 6 complete, architecture search in design phase
**Last Updated**: 2025-10-01
**Maintainer**: auto-RAG Project