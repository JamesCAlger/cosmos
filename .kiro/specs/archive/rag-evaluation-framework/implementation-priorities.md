# Implementation Priority Guide for AutoRAG

## Overview

This document outlines the prioritized implementation order for building a modular, extensible RAG optimization system. The approach emphasizes early validation of core assumptions, progressive complexity, and maintaining flexibility for future enhancements like rerankers, graph databases, and agentic elements.

## Core Implementation Principles

1. **Validate Before Optimize**: Ensure evaluation works before building optimization
2. **Foundation Before Features**: Build modular architecture before component variety
3. **Simple Before Complex**: Master basic RAG before advanced architectures
4. **Measure Everything**: Every addition must be measurable and provide value
5. **Maintain Extensibility**: Design for future additions from day one

## Priority 1: Minimal RAG Baseline + Basic Evaluation

### Timeline: Week 1

### Objectives
- Establish working end-to-end RAG pipeline
- Validate evaluation methodology
- Create performance baseline

### What to Build

#### Minimal RAG Pipeline
- **Document Processing**: Simple fixed-size text splitting (e.g., 256 tokens)
- **Embedding**: Single model (OpenAI text-embedding-ada-002)
- **Storage**: In-memory FAISS or simple numpy arrays
- **Retrieval**: Basic cosine similarity search
- **Generation**: Simple prompt template with GPT-3.5-turbo
- **No Configuration System**: Hardcode everything initially

#### Basic Evaluation
- **RAGAS Integration**: Minimal setup with 2-3 metrics
  - Faithfulness (no ground truth needed)
  - Answer relevancy (no ground truth needed)
  - Context relevancy (optional)
- **Test Dataset**: 10-20 MS MARCO queries
- **Success Metrics**: Establish baseline scores

### Success Criteria
- Can process documents and index them
- Can retrieve relevant contexts for queries
- Can generate answers using retrieved contexts
- Can obtain RAGAS evaluation scores
- Baseline metrics documented

### What NOT to Build Yet
- Configuration management system
- Multiple component implementations
- Optimization framework
- Complex evaluation infrastructure
- Caching systems
- Error handling beyond basic try-catch

## Priority 2: Modular Architecture Foundation

### Timeline: Week 2

### Objectives
- Create extensible component architecture
- Enable component swapping via configuration
- Design for future complexity (graphs, agents)

### What to Build

#### Abstract Component Interfaces

```
Core Interfaces Hierarchy:
├── DocumentProcessor
│   ├── Chunker (multiple strategies)
│   └── Preprocessor (future: cleaning, normalization)
├── Embedder
│   ├── DenseEmbedder (various models)
│   └── SparseEmbedder (future: BM25 vectors)
├── Retriever
│   ├── DenseRetriever
│   ├── SparseRetriever
│   ├── HybridRetriever
│   └── GraphRetriever (future)
├── Reranker (interface only)
│   └── (implementations in Priority 4)
├── Generator
│   ├── SimpleGenerator
│   └── AgenticGenerator (future)
└── PostProcessor (interface only)
    └── (future: fact checking, refinement)
```

#### Pipeline Orchestrator
- **Graph-Based Architecture**: Not just linear pipeline
  - Support for branching paths
  - Conditional routing capability
  - Parallel processing paths
  - Feedback loops (for future agentic behavior)
- **Component Chaining**: Define execution order
- **Data Flow Management**: Pass context between components
- **Metadata Propagation**: Track decisions through pipeline

#### Component Registry
- **Dynamic Loading**: Register components at runtime
- **Version Management**: Support multiple versions of same component
- **Dependency Resolution**: Handle component requirements
- **Discovery Mechanism**: Find available components

#### Simple Configuration System
- **YAML/JSON Based**: Human-readable configuration
- **Schema Validation**: Ensure valid configurations
- **Default Values**: Sensible defaults for all parameters
- **Configuration Inheritance**: Base configs with overrides

### Key Design Decisions

#### Make Pipeline a Directed Graph
```yaml
# Example configuration structure
pipeline:
  nodes:
    - id: retriever_1
      type: dense_retriever
    - id: reranker_1
      type: cross_encoder
    - id: retriever_2
      type: graph_retriever
    - id: generator
      type: simple_generator

  edges:
    - from: query
      to: [retriever_1, retriever_2]  # Parallel retrieval
    - from: retriever_1
      to: reranker_1
    - from: [reranker_1, retriever_2]
      to: generator  # Merge paths
```

### Success Criteria
- Can swap components via configuration
- Same configuration produces same results
- New components can be added without modifying core
- Pipeline supports non-linear flows

## Priority 3: Evaluation Infrastructure

### Timeline: Week 3

### Objectives
- Build robust evaluation system
- Enable statistical comparisons
- Implement cost tracking
- Create progressive evaluation strategy

### What to Build

#### RAGAS Integration Layer
- **Caching System**: Avoid redundant evaluations
  - Cache key: (config_hash, query, context, answer)
  - Tiered caching: Memory → Disk → Database
- **Batch Processing**: Evaluate multiple queries efficiently
- **Async Evaluation**: Non-blocking evaluation calls
- **Error Recovery**: Handle API failures gracefully

#### Progressive Evaluation System
```
Evaluation Levels:
├── Level 0: Smoke Test (5 queries)
│   ├── Purpose: Catch configuration errors
│   ├── Duration: < 30 seconds
│   └── Cost: < $0.01
├── Level 1: Quick Evaluation (20 queries)
│   ├── Purpose: Fast pass/fail decision
│   ├── Duration: < 2 minutes
│   └── Cost: < $0.05
├── Level 2: Standard Evaluation (100 queries)
│   ├── Purpose: Statistical comparison
│   ├── Duration: < 10 minutes
│   └── Cost: < $0.25
├── Level 3: Comprehensive (1000 queries)
│   ├── Purpose: Final validation
│   ├── Duration: < 1 hour
│   └── Cost: < $2.50
└── Level 4: Cross-Dataset (5000+ queries)
    ├── Purpose: Generalization testing
    ├── Duration: Multiple hours
    └── Cost: < $10.00
```

#### Statistical Framework
- **Significance Testing**: Compare configurations rigorously
  - Paired t-tests for same test set
  - Effect size calculation (Cohen's d)
  - Confidence intervals via bootstrap
- **Variance Analysis**: Understand score stability
  - Within-configuration variance
  - Between-configuration variance
  - Required sample size calculation
- **Multiple Comparison Correction**: Bonferroni adjustment

#### Cost Tracking System
- **Component-Level Tracking**: Cost per embedding, retrieval, generation
- **Cumulative Monitoring**: Track total spend
- **Budget Enforcement**: Stop when limits reached
- **Cost Prediction**: Estimate before running

#### Data Management
- **MS MARCO Integration**: Efficient loading and sampling
- **Stratified Sampling**: Maintain query difficulty distribution
- **Train/Dev/Test Splits**: Prevent overfitting
- **Query Categorization**: Simple/medium/complex

### Critical Design Decision
**Evaluation as a Service**: Build evaluation as separate module that can evaluate any pipeline configuration, not embedded within pipeline itself. This enables:
- Independent evaluation of different systems
- Comparison with external baselines
- Batch evaluation of multiple configurations
- Clean separation of concerns

### Success Criteria
- Can statistically compare two configurations
- Progressive evaluation saves 80% of cost on bad configs
- Evaluation results are reproducible
- Cost tracking is accurate within 5%

## Priority 4: Component Variety - Retrieval Focus

### Timeline: Week 4

### Objectives
- Add retrieval alternatives
- Implement reranking
- Explore chunking strategies
- Validate modular architecture

### Implementation Order and Rationale

#### 4.1 BM25 Sparse Retrieval
**Why First**: Strong baseline, no embeddings needed, often beats dense retrieval
- Implement using rank-bm25 or Elasticsearch
- Add as alternative to dense retrieval
- Enable A/B testing vs dense retrieval

#### 4.2 Hybrid Retrieval
**Why Second**: Typically improves over both individual methods
- Combine dense and sparse scores
- Implement score normalization
- Add weight parameter for tuning
- Expected improvement: 10-20% over single method

#### 4.3 Simple Reranker
**Why Third**: High impact, modular addition
- Cross-encoder for top-k reranking
- Start with sentence-transformers models
- Rerank top 20-50 to select top 5
- Expected improvement: 15-25% in precision

#### 4.4 Multiple Chunking Strategies
**Why Fourth**: Affects entire downstream pipeline
- **Semantic Chunking**: Use sentence boundaries, paragraph structure
- **Sliding Window**: Overlapping chunks for context preservation
- **Hierarchical**: Multiple chunk sizes for coarse-to-fine retrieval
- **Document-Aware**: Respect document structure (headers, sections)

### Validation Requirements
- Each component must be evaluatable independently
- Must show measurable improvement over baseline
- Must not break existing functionality
- Must integrate cleanly via configuration

### Success Criteria
- Hybrid retrieval beats both dense and sparse alone
- Reranker improves precision@5 by >15%
- At least one chunking strategy beats baseline
- All components configurable via YAML

## Priority 5: Configuration Search Space

### Timeline: Week 5

### Objectives
- Define search space
- Implement configuration generation
- Build simple optimization
- Track and compare results

### What to Build

#### Search Space Definition
```yaml
# Start small - 2-3 options per component
search_space:
  chunking:
    strategy: [fixed, semantic]
    size: [256, 512]
  retrieval:
    method: [dense, sparse, hybrid]
    top_k: [3, 5]
  reranking:
    enabled: [true, false]
    model: [cross-encoder/ms-marco-MiniLM-L-6-v2]
  generation:
    temperature: [0, 0.3]
    max_tokens: [150, 300]

# Total configurations: 2×2×3×2×2×2×2 = 192 (still manageable)
```

#### Configuration Generator
- Generate valid configurations from search space
- Handle dependencies (e.g., reranker model only if enabled)
- Support for conditional parameters
- Configuration validation

#### Simple Grid Search
- Exhaustive search over small space
- Parallel evaluation support
- Progress tracking and checkpointing
- Early stopping for bad configurations

#### Result Management
- Store all evaluation results
- Compare configurations statistically
- Identify best configuration
- Generate optimization report

### Success Criteria
- Find configuration 20% better than baseline
- Complete grid search within budget
- Results reproducible
- Clear winner identified with statistical significance

## Priority 6: Advanced Components

### Timeline: Weeks 6-12

### Implementation Order

#### Week 6: Advanced Rerankers
**Why**: Immediate quality improvement, easy to evaluate

**Options to Implement**:
- ColBERT for efficient deep reranking
- BGE-Reranker for multilingual support
- Custom cross-encoders trained on domain data
- Listwise reranking (beyond pairwise)

**Integration Pattern**: Add as drop-in replacement for simple reranker

#### Week 7: Query Enhancement
**Why**: Improves retrieval without architecture changes

**Options to Implement**:
- **Query Expansion**: Add synonyms and related terms
- **HyDE**: Hypothetical Document Embeddings
- **Query Decomposition**: Break complex queries into sub-queries
- **Query Rewriting**: Reformulate for better retrieval

**Integration Pattern**: Add as preprocessing layer before retrieval

#### Week 8: Multi-stage Retrieval
**Why**: Leverages graph architecture design

**Options to Implement**:
- **Coarse-to-Fine**: Fast initial retrieval, then focused search
- **Multi-Hop**: Follow references and links
- **Iterative Refinement**: Use initial results to improve query
- **Cascade Retrieval**: Multiple retrievers in sequence

**Integration Pattern**: Use pipeline graph structure for complex flows

#### Week 9-10: Graph Integration
**Why**: High potential value for connected information

**Implementation Strategy**:
- Add as **parallel path**, not replacement
- Start with simple knowledge graph (entities and relations)
- Implement entity linking from documents
- Add graph traversal for multi-hop reasoning
- Combine with vector retrieval for hybrid approach

**Key Components**:
- Graph construction from documents
- Entity recognition and linking
- Graph query language integration
- Subgraph retrieval strategies

#### Week 11-12: Agentic Elements
**Why**: Most complex, needs solid foundation

**Implementation Strategy**:
- Implement as **optional post-processing layer**
- Start with simple self-reflection
- Add iterative refinement based on confidence
- Implement tool use for specific queries

**Agentic Capabilities**:
- **Self-Reflection**: Evaluate own answers
- **Iterative Refinement**: Improve based on self-critique
- **Selective Retrieval**: Decide when more context needed
- **Tool Use**: Calculator, code execution, web search
- **Confidence Calibration**: Know when uncertain

**Integration Pattern**: Add as optional pipeline stages with feedback loops

## What NOT to Prioritize Early

### Avoid in First Month
1. **Complex Optimization Algorithms**
   - Bayesian optimization
   - Evolutionary algorithms
   - Meta-learning
   - Why: Grid search is sufficient initially

2. **UI/Visualization**
   - Web dashboards
   - Interactive visualizations
   - Real-time monitoring
   - Why: CLI and JSON outputs are enough

3. **Production Features**
   - Authentication/authorization
   - Deployment infrastructure
   - Monitoring/alerting
   - Why: Research validation first

4. **Premature Generalization**
   - Multi-modal support
   - Multi-language (beyond English)
   - Domain-specific adaptations
   - Why: Prove concept on single domain first

### Avoid Until Phase 2
1. **Meta-Learning**
   - Need many experiments first
   - Requires stable evaluation
   - Complex to validate

2. **Novel Architectures**
   - Master standard RAG first
   - Need solid baselines
   - Research papers can wait

3. **Extensive Hyperparameter Tuning**
   - Focus on architectural choices first
   - Fine-tuning comes after structure

## Critical Architecture Decisions

### Component Interface Design

#### Async-Ready Interfaces
```python
class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> List[Document]:
        """Support both sync and async execution"""
        pass

    @abstractmethod
    def retrieve_batch(self, queries: List[str], top_k: int) -> List[List[Document]]:
        """Batch processing for efficiency"""
        pass
```

#### Metadata Propagation
- Every component input/output includes metadata
- Track decision provenance through pipeline
- Enable debugging and analysis
- Support for confidence scores

### Configuration Schema

#### Hierarchical Configuration
```yaml
base_config:
  chunking:
    strategy: semantic
    size: 256

experiments:
  exp1:
    extends: base_config
    chunking:
      size: 512  # Override specific parameter
```

#### Component Versioning
```yaml
retriever:
  type: dense_retriever
  version: 1.2.0  # Version tracking from start
  config:
    model: text-embedding-ada-002
```

### Data Flow Design

#### Intermediate Data Format
```python
@dataclass
class PipelineData:
    query: Query
    documents: List[Document]
    chunks: List[Chunk]
    retrieved_contexts: List[Context]
    reranked_contexts: Optional[List[Context]]
    answer: Answer
    metadata: Dict[str, Any]
    trace: List[ComponentTrace]  # For debugging
```

#### Observability Design
- Ability to inspect data between any components
- Optional data dumping for debugging
- Performance profiling hooks
- Error context preservation

### Extension Mechanism

#### Plugin System Design
```python
class ComponentRegistry:
    def register(self,
                component_type: str,
                component_class: Type[Component],
                version: str):
        """Register new components dynamically"""

    def discover(self, package: str):
        """Auto-discover components in package"""

    def get_component(self,
                     component_type: str,
                     version: Optional[str] = None):
        """Retrieve registered component"""
```

## Success Metrics by Phase

### Week 1 Success Metrics
- ✅ Baseline RAGAS scores established
- ✅ End-to-end pipeline working
- ✅ Evaluation methodology validated
- ✅ MS MARCO subset processed

### Week 2 Success Metrics
- ✅ Can swap components via configuration
- ✅ Graph-based pipeline architecture working
- ✅ Component registry functional
- ✅ Configuration validation working

### Week 3 Success Metrics
- ✅ Can statistically compare two configurations
- ✅ Progressive evaluation reduces costs by >50%
- ✅ Evaluation results reproducible
- ✅ Cost tracking accurate

### Week 4 Success Metrics
- ✅ Hybrid retrieval beats single-method by >10%
- ✅ Reranker improves precision by >15%
- ✅ Multiple chunking strategies evaluated
- ✅ All components configurable

### Week 5 Success Metrics
- ✅ Found configuration 20% better than baseline
- ✅ Grid search completed within budget
- ✅ Statistical significance achieved
- ✅ Optimization report generated

### Weeks 6+ Success Metrics
- ✅ Each new component provides measurable improvement
- ✅ Graph integration working in parallel
- ✅ Agentic elements show value on complex queries
- ✅ System remains modular and maintainable

## Common Pitfalls to Avoid

### Technical Pitfalls
1. **Over-Engineering Early**
   - Don't build for 1000 users when you have 1
   - Don't optimize prematurely
   - Don't abstract everything immediately

2. **Under-Engineering Foundations**
   - Don't skip interface design
   - Don't hardcode assumptions
   - Don't ignore configuration management

3. **Evaluation Mistakes**
   - Don't optimize on test set
   - Don't ignore variance
   - Don't trust single runs
   - Don't skip statistical tests

### Process Pitfalls
1. **Scope Creep**
   - Resist adding "just one more feature"
   - Maintain focus on core objectives
   - Document but defer good ideas

2. **Premature Optimization**
   - Don't optimize before measuring
   - Don't tune hyperparameters before architecture
   - Don't add caching before profiling

3. **Insufficient Validation**
   - Don't skip evaluation for new components
   - Don't assume improvements translate
   - Don't ignore edge cases

## Risk Mitigation Strategies

### Technical Risks
1. **Evaluation Unreliability**
   - Mitigation: Multiple metrics, statistical validation
   - Fallback: Human evaluation on small sample

2. **Component Integration Issues**
   - Mitigation: Strong interface contracts, extensive testing
   - Fallback: Adapter pattern for incompatible components

3. **Scalability Problems**
   - Mitigation: Design for scale, implement for current needs
   - Fallback: Vertical scaling before horizontal

### Project Risks
1. **Scope Explosion**
   - Mitigation: Strict prioritization, time-boxed experiments
   - Fallback: Feature freeze, focus on core

2. **Cost Overruns**
   - Mitigation: Progressive evaluation, cost tracking
   - Fallback: Smaller test sets, cheaper models

3. **Timeline Slippage**
   - Mitigation: Conservative estimates, buffer time
   - Fallback: Reduce scope, not quality

## Conclusion

This implementation priority guide provides a clear path from minimal viable RAG to sophisticated optimization system. Key principles:

1. **Validate core assumptions first** (evaluation reliability)
2. **Build modular foundation early** (graph-based architecture)
3. **Add complexity progressively** (simple → hybrid → agentic)
4. **Measure everything** (statistical rigor throughout)
5. **Maintain flexibility** (extensible design patterns)

Following this guide will result in a robust, extensible, and scientifically valid RAG optimization system that can grow to incorporate advanced features while maintaining a solid foundation.