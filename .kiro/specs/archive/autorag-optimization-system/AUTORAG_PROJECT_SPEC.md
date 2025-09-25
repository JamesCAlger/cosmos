# AutoRAG: Automatic RAG Architecture Optimization System
## Project Specification for New Development

### Executive Summary

This document specifies a **focused, achievable** system for automatically optimizing Retrieval-Augmented Generation (RAG) architectures based on input datasets. The system will discover optimal configurations through intelligent search, starting with a simple implementation and gradually adding sophistication.

**Core Innovation**: A system that automatically finds the best RAG configuration for any given dataset, eliminating manual tuning and discovering non-obvious optimal architectures.

### Project Philosophy

1. **Start Simple, Evolve Intelligently**: Begin with a working end-to-end system, then add complexity
2. **Research First, Infrastructure Later**: Focus on the optimization algorithms, not deployment
3. **Measure Everything**: Every decision should be data-driven
4. **Fail Fast**: Quick experiments to validate ideas before deep implementation

### Phase 1: Minimal Viable Product (4 weeks)

#### Goal
Build a system that can evaluate 10 different RAG configurations on MS MARCO subset and identify the best one.

#### Core Components

```python
# project structure
autorag/
├── core/
│   ├── rag_pipeline.py      # Simple, configurable RAG implementation
│   ├── evaluator.py          # Evaluation metrics (accuracy, latency, cost)
│   └── config.py             # Configuration management
├── optimization/
│   ├── grid_search.py        # Start with simple grid search
│   └── optimizer.py          # Base optimization interface
├── datasets/
│   ├── loader.py             # MS MARCO data loading
│   └── sampler.py            # Intelligent sampling for fast evaluation
├── experiments/
│   └── tracker.py            # Simple experiment tracking (JSON/CSV)
└── run_optimization.py       # Main entry point
```

#### Implementation Details

##### 1. RAG Pipeline (`core/rag_pipeline.py`)
```python
class RAGPipeline:
    """Configurable RAG pipeline with swappable components."""

    def __init__(self, config: Dict[str, Any]):
        self.chunker = self._create_chunker(config['chunking'])
        self.embedder = self._create_embedder(config['embedding'])
        self.retriever = self._create_retriever(config['retrieval'])
        self.generator = self._create_generator(config['generation'])

    def index(self, documents: List[Document]) -> None:
        """Index documents for retrieval."""
        chunks = self.chunker.chunk(documents)
        embeddings = self.embedder.embed(chunks)
        self.retriever.index(embeddings, chunks)

    def query(self, question: str, top_k: int = 5) -> Answer:
        """Answer a question using RAG."""
        query_embedding = self.embedder.embed_query(question)
        contexts = self.retriever.retrieve(query_embedding, top_k)
        answer = self.generator.generate(question, contexts)
        return answer
```

##### 2. Configuration Space
```python
# Initial search space - kept deliberately simple
SEARCH_SPACE = {
    'chunking': {
        'strategy': ['fixed', 'semantic', 'sliding'],
        'size': [128, 256, 512],
        'overlap': [0, 50, 100]
    },
    'embedding': {
        'model': ['ada-002', 'e5-small-v2', 'bge-small-en'],
        'batch_size': [32, 64, 128]
    },
    'retrieval': {
        'method': ['dense', 'sparse', 'hybrid'],
        'top_k': [3, 5, 10]
    },
    'generation': {
        'model': ['gpt-3.5-turbo', 'gpt-4o-mini'],
        'temperature': [0, 0.3, 0.7],
        'max_tokens': [150, 300, 500]
    }
}
```

##### 3. Evaluation Metrics
```python
class Evaluator:
    """Evaluate RAG pipeline performance."""

    def evaluate(self, pipeline: RAGPipeline, test_set: TestSet) -> Metrics:
        """Run evaluation on test set."""
        results = []

        for question, expected_answer, relevant_docs in test_set:
            start_time = time.time()
            answer = pipeline.query(question)
            latency = time.time() - start_time

            # Calculate metrics
            accuracy = self.calculate_accuracy(answer, expected_answer)
            relevance = self.calculate_relevance(answer, relevant_docs)
            cost = self.calculate_cost(pipeline.config)

            results.append({
                'accuracy': accuracy,
                'relevance': relevance,
                'latency': latency,
                'cost': cost
            })

        return self.aggregate_metrics(results)
```

##### 4. Simple Grid Search Optimizer
```python
class GridSearchOptimizer:
    """Simple grid search to establish baseline."""

    def optimize(self, dataset: Dataset, search_space: Dict) -> BestConfig:
        """Find best configuration through grid search."""
        configurations = self.generate_configs(search_space)
        results = []

        for config in configurations:
            pipeline = RAGPipeline(config)
            pipeline.index(dataset.documents)

            metrics = self.evaluator.evaluate(pipeline, dataset.test_set)
            results.append((config, metrics))

            # Save intermediate results
            self.save_result(config, metrics)

        return self.find_best(results)
```

#### Deliverables for Phase 1
1. Working RAG pipeline with 3-5 component variants
2. Evaluation on MS MARCO subset (1000 docs, 100 queries)
3. Grid search over ~100 configurations
4. Simple CSV/JSON experiment tracking
5. Clear winner configuration with metrics

### Phase 2: Intelligent Optimization (4 weeks)

#### Goal
Replace grid search with Bayesian optimization to find good configurations 10x faster.

#### New Components
```python
optimization/
├── bayesian.py              # Bayesian optimization with Gaussian Processes
├── acquisition.py           # Acquisition functions (EI, UCB, PI)
└── surrogate_model.py       # Gaussian Process for predicting performance
```

#### Key Features
1. **Bayesian Optimization**: Use past evaluations to guide search
2. **Multi-objective**: Balance accuracy, cost, and latency
3. **Early Stopping**: Skip bad configurations quickly
4. **Progressive Scaling**: Test on small data first

#### Implementation Sketch
```python
class BayesianOptimizer:
    """Smart optimization using Gaussian Processes."""

    def optimize(self, dataset: Dataset, n_iterations: int = 50) -> BestConfig:
        """Find optimal configuration with fewer evaluations."""

        # Start with random configurations
        initial_configs = self.random_sample(n=10)
        observations = self.evaluate_configs(initial_configs)

        # Gaussian Process model
        gp = GaussianProcess()
        gp.fit(observations)

        for i in range(n_iterations - 10):
            # Select next configuration using acquisition function
            next_config = self.acquisition_function(gp, method='EI')

            # Evaluate with progressive scaling
            score = self.progressive_evaluate(next_config, dataset)

            # Update model
            gp.add_observation(next_config, score)

            # Early stopping if converged
            if self.has_converged(gp):
                break

        return gp.best_config()
```

### Phase 3: Meta-Learning and Transfer (4 weeks)

#### Goal
Learn from past experiments to warm-start new optimizations.

#### New Components
```python
meta_learning/
├── dataset_analyzer.py      # Extract dataset characteristics
├── config_predictor.py      # Predict good configs from dataset features
└── knowledge_base.py        # Store and retrieve past experiments
```

#### Key Innovation
```python
class MetaLearner:
    """Learn to predict good configurations from dataset characteristics."""

    def predict_initial_configs(self, dataset: Dataset) -> List[Config]:
        """Suggest good starting points based on dataset analysis."""

        # Analyze dataset
        features = self.extract_features(dataset)
        # - Document length distribution
        # - Vocabulary size
        # - Query complexity
        # - Domain (medical, legal, general)

        # Find similar past experiments
        similar_experiments = self.knowledge_base.find_similar(features)

        # Adapt configurations
        adapted_configs = []
        for exp in similar_experiments:
            config = self.adapt_config(exp.best_config, features)
            adapted_configs.append(config)

        return adapted_configs
```

### Phase 4: Architecture Search (4 weeks)

#### Goal
Discover novel RAG architectures beyond standard retrieve-then-generate.

#### New Components
```python
architecture/
├── space.py                 # Define architecture search space
├── evolutionary.py          # Evolutionary algorithm for architecture search
└── novel_architectures.py   # New architecture patterns
```

#### Novel Architectures to Explore
1. **Iterative Refinement**: Query → Retrieve → Generate → Refine Query → Retrieve Again
2. **Multi-Stage Retrieval**: Coarse retrieval → Rerank → Fine retrieval
3. **Hybrid Fusion**: Combine multiple retrieval methods intelligently
4. **Adaptive Routing**: Route queries to different pipelines based on type

### Technical Stack (Minimal)

#### Required Libraries
```python
# Core
numpy
scipy
pandas
scikit-learn

# RAG Components
openai  # For embeddings and generation
faiss-cpu  # Vector search
rank-bm25  # Sparse retrieval

# Optimization
scikit-optimize  # Bayesian optimization
optuna  # Alternative optimization framework

# Evaluation
nltk  # Text processing
rouge-score  # Evaluation metrics

# Simple ML
lightgbm  # For meta-learning

# Utilities
pyyaml  # Configuration
tqdm  # Progress bars
loguru  # Logging
```

#### Data Storage
- **Experiments**: Simple JSON/CSV files initially
- **Embeddings**: In-memory FAISS index
- **Documents**: Local file storage
- **No databases initially** - add PostgreSQL later if needed

### Development Workflow

#### Week 1-2: Core RAG Pipeline
```bash
# First milestone: Can run one evaluation
python run_optimization.py --config configs/single_eval.yaml

# Output:
# Accuracy: 0.75, Latency: 1.2s, Cost: $0.02
```

#### Week 3-4: Grid Search
```bash
# Second milestone: Compare multiple configs
python run_optimization.py --strategy grid --configs 20

# Output:
# Best config: {chunking: semantic, embedding: ada-002, ...}
# Improvement: 15% over baseline
```

#### Week 5-6: Bayesian Optimization
```bash
# Third milestone: Smart search
python run_optimization.py --strategy bayesian --budget 50

# Output:
# Found optimal in 35 evaluations (vs 100 for grid)
# Best score: 0.82
```

#### Week 7-8: Progressive Features
- Add multi-objective optimization
- Implement early stopping
- Add cost-aware optimization

#### Week 9-12: Advanced Features
- Meta-learning from past experiments
- Architecture search
- Cross-dataset evaluation

### Success Metrics

#### Phase 1 Success (MVP)
- ✅ System runs end-to-end
- ✅ Evaluates 10+ configurations
- ✅ Finds configuration better than default
- ✅ Results reproducible

#### Phase 2 Success (Intelligent)
- ✅ 5x fewer evaluations than grid search
- ✅ Handles accuracy/cost/latency trade-offs
- ✅ Saves 50% evaluation time with early stopping

#### Phase 3 Success (Meta-Learning)
- ✅ 50% reduction in optimization time for new datasets
- ✅ Successful transfer between similar datasets
- ✅ Automated dataset characterization

#### Phase 4 Success (Novel)
- ✅ Discovers architecture that beats standard RAG
- ✅ Novel architectures are interpretable
- ✅ Consistent improvements across datasets

### Key Design Principles

1. **Modularity**: Every component should be swappable
```python
# Good
retriever = RetrieverFactory.create(config['retrieval_type'])

# Bad
if config['retrieval_type'] == 'dense':
    # 100 lines of dense retrieval code
```

2. **Configuration-Driven**: Everything controlled by config files
```yaml
# experiment.yaml
dataset:
  name: msmarco
  subset_size: 1000

search_space:
  chunking:
    size: [128, 256, 512]

optimization:
  strategy: bayesian
  n_iterations: 50
```

3. **Fail Fast**: Quick validation before expensive operations
```python
def optimize(self, dataset):
    # Validate configuration first
    self.validate_config()

    # Test on tiny subset
    if not self.smoke_test(dataset.sample(10)):
        raise ValueError("Pipeline fails on sample data")

    # Now run full optimization
    return self._optimize_impl(dataset)
```

4. **Incremental Complexity**: Start simple, add features gradually
```python
# Version 1: Simple
class Optimizer:
    def optimize(self, configs):
        return max(configs, key=self.evaluate)

# Version 2: Add caching
class Optimizer:
    def __init__(self):
        self.cache = {}

    def optimize(self, configs):
        # Check cache first...

# Version 3: Add parallelization
# Version 4: Add early stopping
# etc.
```

### Common Pitfalls to Avoid

1. **Don't Over-Engineer Early**
   - No Kubernetes
   - No microservices
   - No complex databases
   - No web UI (initially)

2. **Don't Optimize Prematurely**
   - Get it working first
   - Profile before optimizing
   - Simple code > clever code

3. **Don't Skip Evaluation**
   - Every change needs metrics
   - A/B test new features
   - Keep baseline for comparison

4. **Don't Ignore Costs**
   - Track API costs from day 1
   - Set budget limits
   - Optimize for cost/quality trade-off

### Project Structure

```
autorag/
├── README.md                 # Project overview and setup
├── requirements.txt          # Minimal dependencies
├── setup.py                  # Package setup
├── configs/
│   ├── default.yaml         # Default configuration
│   ├── search_spaces/       # Different search space definitions
│   └── datasets/            # Dataset configurations
├── autorag/
│   ├── __init__.py
│   ├── core/                # RAG pipeline components
│   ├── optimization/        # Optimization algorithms
│   ├── evaluation/          # Metrics and evaluation
│   ├── datasets/            # Data loading and sampling
│   └── utils/               # Helpers and utilities
├── scripts/
│   ├── run_optimization.py  # Main entry point
│   ├── analyze_results.py   # Result analysis
│   └── compare_configs.py   # Configuration comparison
├── experiments/              # Experiment results (gitignored)
├── data/                     # Datasets (gitignored)
└── tests/                    # Unit tests
    ├── test_pipeline.py
    ├── test_optimizer.py
    └── test_evaluation.py
```

### Getting Started Commands

```bash
# Clone and setup
git clone https://github.com/yourusername/autorag.git
cd autorag
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Download MS MARCO subset
python scripts/download_data.py --dataset msmarco --size small

# Run first evaluation
python scripts/run_optimization.py \
    --config configs/default.yaml \
    --strategy single \
    --output experiments/first_run.json

# Run grid search
python scripts/run_optimization.py \
    --config configs/search_spaces/small.yaml \
    --strategy grid \
    --max_configs 20 \
    --output experiments/grid_search.json

# Analyze results
python scripts/analyze_results.py experiments/grid_search.json
```

### Testing Strategy

#### Unit Tests (Required)
```python
def test_rag_pipeline():
    """Test basic pipeline functionality."""
    config = {'chunking': {'size': 256}, ...}
    pipeline = RAGPipeline(config)

    # Test indexing
    docs = [Document("Test document")]
    pipeline.index(docs)

    # Test query
    answer = pipeline.query("What is in the document?")
    assert answer is not None

def test_optimizer_convergence():
    """Test that optimizer improves over time."""
    # Use simple test function
    optimizer = BayesianOptimizer()
    results = optimizer.optimize(test_function, n_iter=20)

    # Should improve
    assert results[-1]['score'] > results[0]['score']
```

#### Integration Tests
```python
def test_end_to_end_optimization():
    """Test complete optimization workflow."""
    dataset = load_test_dataset()  # Small dataset
    optimizer = create_optimizer('bayesian')

    best_config = optimizer.optimize(dataset, n_iterations=10)

    # Verify result structure
    assert 'accuracy' in best_config.metrics
    assert best_config.accuracy > 0.5  # Reasonable baseline
```

### Documentation Requirements

#### Code Documentation
```python
def optimize(self, dataset: Dataset, n_iterations: int = 50) -> OptimizationResult:
    """
    Find optimal RAG configuration for given dataset.

    Args:
        dataset: Dataset to optimize for
        n_iterations: Number of configurations to try

    Returns:
        OptimizationResult with best configuration and metrics

    Example:
        >>> dataset = load_dataset('msmarco')
        >>> result = optimizer.optimize(dataset, n_iterations=30)
        >>> print(f"Best accuracy: {result.best_config.accuracy}")
    """
```

#### README Structure
1. **Quick Start** - Get running in 5 minutes
2. **Core Concepts** - What is AutoRAG?
3. **Installation** - Requirements and setup
4. **Usage Examples** - Common workflows
5. **Configuration** - How to configure
6. **Results** - What to expect
7. **Contributing** - How to contribute

### Future Extensions (After Core Works)

1. **Web Dashboard** (Month 4)
   - Simple Flask app
   - Experiment tracking UI
   - Real-time optimization progress

2. **Advanced Optimizers** (Month 5)
   - Population-based training
   - Reinforcement learning
   - Neural architecture search

3. **Production Features** (Month 6)
   - API for optimization as a service
   - Model registry
   - A/B testing framework

### Key Differentiators

1. **Automatic**: No manual tuning required
2. **Intelligent**: Learns from experience
3. **Comprehensive**: Optimizes entire pipeline, not just parts
4. **Novel**: Discovers new architectures
5. **Practical**: Considers cost and latency, not just accuracy

### Final Notes

This is a **research project** that could become a product. Start with research goals:

1. **Can we automatically find good RAG configurations?** (Phase 1)
2. **Can we do it efficiently?** (Phase 2)
3. **Can we transfer knowledge between datasets?** (Phase 3)
4. **Can we discover novel architectures?** (Phase 4)

Each phase answers a research question. If successful, the system becomes valuable for:
- **Researchers**: Baseline for RAG experiments
- **Developers**: Quick RAG optimization for new domains
- **Companies**: Cost-optimized RAG deployments

Remember: **Simple and working beats complex and broken.** Start with the simplest possible implementation that could work, then iterate based on results.