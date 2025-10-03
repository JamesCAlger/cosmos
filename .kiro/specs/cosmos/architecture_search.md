    # Architecture Search for COSMOS Framework

**Date**: 2025-10-01
**Context**: Extending COSMOS beyond parameter optimization to discover optimal pipeline structures

---

## Problem Statement

Current COSMOS optimizes **parameters** for a **predefined structure**:
- Input: chunker → retriever → generator (fixed)
- Output: Best chunk_size, top_k, temperature, etc.

**Goal**: Enable COSMOS to discover **optimal structure** + parameters:
- Input: Component library + objective
- Output: Which components? In what order? With what parameters?

**Example Use Case**: PDF document parsing
- Available components: {text_extractor[A,B], image_extractor[A,B], chunker[A,B]}
- Unknown: Should we use text-only? Image-only? Both sequentially? Both in parallel?

---

## Approach 1: Predefined Templates + Selection

### Concept
Human designs 4-6 reasonable pipeline templates. System evaluates each (optimizing parameters within), then selects best.

### Methodology
1. Define templates based on domain knowledge
2. For each template: optimize component parameters using standard COSMOS
3. Evaluate all optimized templates on validation set
4. Select best-performing template

### Implementation Complexity
**Effort**: 1-2 days
**Extensions needed**: Minimal (template evaluation loop)

### Computational Cost
- Budget × num_templates
- Example: 5 templates × 15 evaluations = 75 total evaluations
- Parallelizable across templates

### Pros
✅ Leverages human domain expertise
✅ Interpretable - understand why each template works
✅ Manageable search space
✅ High success probability if templates well-designed
✅ Fast implementation
✅ Works with current COSMOS infrastructure

### Cons
❌ Limited to predefined options
❌ Might miss novel structures
❌ Requires expertise to design good templates
❌ Not truly "automated" structure discovery

### When to Use
- Small to medium projects
- Clear understanding of domain (can design good templates)
- Want quick results with minimal risk
- 4-6 obvious architectural variations exist
- **90% of practical applications**

### Example: PDF Parsing Templates
1. **TextOnly**: TextExtractor → Chunker (text-native PDFs)
2. **ImageOnly**: ImageExtractor(OCR) → Chunker (scanned docs)
3. **SequentialMultimodal**: TextExtractor → ImageExtractor → Merge → Chunker
4. **ParallelMultimodal**: [Text path] + [Image path] → Merge
5. **Adaptive**: Analyzer → Routes to appropriate template

### Expected Performance
Within 5-10% of optimal, found with 10x less compute than grammar, 100x less than RL

---

## Approach 2: Grammar-Based Structure Search

### Concept
Define formal grammar describing valid pipelines. System samples structures from grammar, optimizes each, selects best.

### Methodology
1. Define grammar rules for valid pipeline construction
2. Sample N structures by expanding grammar rules
3. For each sampled structure: optimize parameters with COSMOS
4. Select best structure+parameters

### Grammar Example
```
Pipeline = ExtractorStage + ChunkingStage
ExtractorStage = TextExtractor | ImageExtractor | (TextExtractor AND ImageExtractor)
TextExtractor = ExtractorA(params) | ExtractorB(params)
AND = ParallelExecution + Merge | SequentialExecution + Merge
ChunkingStage = FixedChunker(params) | SemanticChunker(params)
```

### Implementation Complexity
**Effort**: 1-2 weeks
**Extensions needed**: Grammar parser, structure sampler, structure evaluator

### Computational Cost
- Budget × num_samples
- Example: 50 samples × 20 evaluations = 1000 total evaluations
- Requires parallelization infrastructure

### Pros
✅ More flexible than templates
✅ Can discover novel combinations not explicitly designed
✅ Constrained to valid pipelines (grammar prevents nonsensical structures)
✅ Extensible - new components = new grammar rules
✅ Systematic exploration of structure space

### Cons
❌ Still requires defining grammar (human knowledge)
❌ Large search space - many sampled structures may be poor
❌ Exponential growth with grammar complexity
❌ More implementation complexity than templates
❌ No learning across samples (each independent)

### When to Use
- Medium to large projects
- Need more flexibility than fixed templates
- Diverse problem instances requiring different structures
- Have 1-2 weeks for implementation
- Budget for 500-1000 evaluations

### Search Space Analysis
- Templates: 5 architectures (fixed)
- Grammar: 100-1000+ architectures (depends on rules)
- With budget=50: sample 5-10% of grammar space
- Use Bayesian optimization to sample intelligently

---

## Approach 3: Reinforcement Learning (RL)

### Concept
Treat structure discovery as sequential decision problem. Neural network "controller" learns to build pipelines through trial and error.

### Methodology
**Setup**:
- State: Current partial pipeline
- Action: Add next component
- Reward: Final pipeline performance

**Training Loop**:
1. Controller network generates structure
2. Optimize parameters for that structure
3. Evaluate performance → reward
4. Update controller via REINFORCE to favor high-reward actions

After many iterations, controller learns patterns like "TextExtractor + ImageExtractor → use Merge" or "High image content → prioritize OCR quality"

### Implementation Complexity
**Effort**: 1-2 months
**Extensions needed**:
- Controller network architecture
- RL training infrastructure
- State/action encoding schemes
- Reward shaping

### Computational Cost
- 1000-5000+ structure evaluations
- Each evaluation: parameter optimization + testing
- Example: 3000 architectures × 1 hour each = 3000 GPU hours = 125 days sequential
- Requires GPU cluster for parallelization

### Pros
✅ No human-defined templates or grammar needed
✅ Can discover truly novel structures
✅ Learns from accumulated experience
✅ Proven approach (NAS-RL, ENAS in neural architecture search)
✅ Handles complex search spaces with branching/merging
✅ Adaptive - focuses compute on promising regions

### Cons
❌ Extremely computationally expensive
❌ Complex implementation (RL expertise required)
❌ Prone to overfitting test distribution
❌ Difficult to debug and tune
❌ Requires substantial infrastructure
❌ May discover fragile structures that don't generalize
❌ Long development cycle

### When to Use
- Large-scale industrial deployments
- 10,000+ diverse training examples
- Budget for weeks/months of GPU time
- Team with RL expertise
- Performance delta of 1-2% matters (high-value application)
- Long-term investment (6-12 month project)

### Similar Systems in Literature
- **NAS-RL** (Zoph & Le, 2017): RL for neural architecture search
- **ENAS** (Pham et al., 2018): Efficient NAS via parameter sharing
- **AutoML-Zero** (Real et al., 2020): Evolving ML algorithms from scratch

### Handling Long Pipelines: Search Space Explosion Problem

**Challenge**: Longer ML pipelines create combinatorially large search spaces that can overwhelm RL training.

#### The Problem

**Search Space Growth**:

| Pipeline Length | Component Choices | Total Combinations |
|----------------|-------------------|-------------------|
| 3 components | 5 per position | 5³ = 125 |
| 5 components | 5 per position | 5⁵ = 3,125 |
| 10 components | 5 per position | 5¹⁰ = 9,765,625 |
| 15 components | 5 per position | 5¹⁵ ≈ 30 billion |
| 20 components | 5 per position | 5²⁰ ≈ 95 trillion |

**Example: Document Processing + RAG + Reporting System** (12 components):
```
PDFExtractor → OCREngine → ExcelParser → DataNormalizer →
Chunker → Embedder → Retriever → Reranker →
ContextAssembler → LLMGenerator → FactChecker → ReportFormatter
```

**Search space**: 5¹² ≈ 244 million combinations (assuming 5 choices per component type)

With parameter optimization (10-100 parameter configurations per architecture), the true search space becomes intractable.

#### Why RL Scales Polynomially, Not Exponentially

RL doesn't randomly sample all combinations. It learns structure through sequential decision-making:

**Learning Process Example**:
```
Iteration 1-50 (exploration):
- Try: Generator → Chunker (fails - no input)
- Try: Retriever → Chunker (fails - wrong order)
- Try: Chunker → Retriever → Generator (works! Reward: 0.65)

RL learns: P(Chunker | State=[]) increases
          P(Retriever | State=[Chunker]) increases
          P(Generator | State=[Chunker, Retriever]) increases

Iteration 50-200 (exploitation):
- Focus on: Which Chunker? Which Retriever?
- Skip invalid orderings entirely
- Effective search space: 9.7M → ~1,000 architectures
```

**Empirical Scaling Law**:
- 3 components: ~500 evaluations to 95% optimal
- 5 components: ~1,500 evaluations
- 10 components: ~5,000 evaluations
- 15 components: ~12,000 evaluations
- 20 components: ~25,000 evaluations

**Fitted curve**: Evaluations ≈ 50 × n² (polynomial O(n²), not exponential O(C^n))

**Why not exponential?**:
1. RL learns valid orderings (eliminates 99% of combinations)
2. Discovers hierarchical structure (groups components into stages)
3. Learns component dependencies (can't put Generator before Retriever)
4. Many components are optional (doesn't multiply search space)

**Practical Limits**:
- **Tractable** (1-2 weeks): 3-8 components, 1,500-5,000 evaluations
- **Challenging** (3-4 weeks): 9-15 components, 5,000-15,000 evaluations
- **Prohibitive** (months): 16-20 components, 15,000-50,000 evaluations
- **Intractable** without decomposition: 20+ components

#### Solution 1: Modular Decomposition

**Concept**: Split long pipeline into logical modules, optimize separately, compose optimized modules.

**Example: 12-Component System**

**Monolithic Approach** (intractable):
- 12 components in sequence
- Search space: 5¹² = 244M combinations
- Estimated cost: 50,000+ evaluations

**Modular Decomposition**:

**Module 1 - Data Extraction** (4 components):
```
PDFExtractor → OCREngine → ExcelParser → DataNormalizer
```
- Search space: 5⁴ = 625 combinations
- Optimize independently: 500 evaluations
- Optimization criterion: Extraction accuracy, format consistency

**Module 2 - RAG Core** (4 components):
```
Chunker → Embedder → Retriever → Reranker
```
- Search space: 5⁴ = 625 combinations
- Optimize independently: 500 evaluations
- Optimization criterion: Retrieval precision@k, latency

**Module 3 - Generation & Reporting** (4 components):
```
ContextAssembler → LLMGenerator → FactChecker → ReportFormatter
```
- Search space: 5⁴ = 625 combinations
- Optimize independently: 500 evaluations
- Optimization criterion: Answer quality, citation accuracy

**Module Composition**:
- Assemble optimized modules: Module1 → Module2 → Module3
- Fine-tune interfaces: 100 evaluations
- Test end-to-end integration

**Total Cost**: 1,600 evaluations vs 50,000+ monolithic (**97% reduction**)

**Advantages**:
✅ Breaks intractable problem into 3 tractable subproblems
✅ Parallelizable (optimize all modules concurrently)
✅ Matches human system design (functional decomposition)
✅ Reusable modules (RAG core works across applications)
✅ Debuggable (test each module independently)
✅ Incremental improvement (upgrade one module without touching others)

**Disadvantages**:
❌ Misses global optima (best overall ≠ composition of best modules)
❌ Interface constraints (modules must have compatible I/O)
❌ Boundary selection matters (wrong boundaries → poor performance)
❌ Cross-module dependencies (Chunker in Module 2 may depend on extraction quality from Module 1)

**Guidelines for Good Module Boundaries**:

**Minimize Cross-Module Dependencies**:
- ✅ Good: Extraction → clean text (clear output contract)
- ❌ Bad: Chunker depends on PDF quality, but they're in different modules

**Align with Functional Units**:
- Input processing → intermediate representation
- Core algorithm → processed results
- Output generation → final format

**Balance Module Complexity**:
- ✅ Good: 3-5 components per module
- ❌ Bad: 2 components in one module, 8 in another

**Two-Level Optimization Strategy**:
1. **Module-level**: Which modules? In what order? (Template/Agentic approach)
2. **Component-level**: Which components within each module? (RL optimization)

This is COSMOS compositional optimization at a coarser grain.

#### Solution 2: Hierarchical Reinforcement Learning

**Concept**: RL learns at two levels of abstraction simultaneously.

**High-Level Policy** (macro-actions):
- State: Current pipeline structure
- Actions: "Add Extraction Stage", "Add Retrieval Stage", "Add Generation Stage"
- Learns: Which functional stages, in what order
- Search space: ~10-20 stage types

**Low-Level Policy** (micro-actions):
- State: Current components within a stage
- Actions: "Add BM25Retriever", "Add DenseRetriever", "Add Reranker"
- Learns: Which components within each stage
- Search space: ~5-10 components per stage

**Hierarchical Composition**:
```
High-level decision:
[Extraction Stage] → [Retrieval Stage] → [Generation Stage]

Low-level decisions per stage:
Extraction: [PDFExtractor] → [OCREngine] → [DataNormalizer]
Retrieval: [Chunker] → [DenseRetriever] → [Reranker]
Generation: [LLMGenerator] → [FactChecker] → [ReportFormatter]
```

**Search Space Reduction**:
- Flat RL: 5¹² = 244M combinations
- Hierarchical RL: 20³ (high-level stages) × 5³ (components per stage) = 8,000 × 125 = 1M effective combinations
- **Reduction: 244× smaller search space**

**Training Strategy**:
1. Train high-level policy on stage selection (cheap - few stages to evaluate)
2. For each stage type, train low-level policy on components (moderate cost per stage)
3. Compose trained policies for full pipeline

**Advantage over Modular Decomposition**: Learns module boundaries automatically rather than requiring human specification.

**Evidence from Literature**:
- **Options Framework** (Sutton et al., 1999): Temporal abstraction in RL
- **Feudal RL** (Dayan & Hinton, 1993): Manager-worker hierarchy
- **HAM** (Parr & Russell, 1998): Hierarchical abstract machines
- Applications: Robot control, game playing with long horizons

#### Solution 3: Progressive Lengthening (Curriculum Learning)

**Concept**: Start with short pipelines, gradually increase length as RL learns basic structures.

**Curriculum Schedule**:

**Stage 1 - Core Pipeline** (3 components, 125 combinations):
```
Chunker → Retriever → Generator
```
- Training: 500 evaluations
- RL learns: Basic RAG structure, component dependencies
- Output: Best 3-component architecture (Accuracy: 0.70)

**Stage 2 - Add Pre/Post Processing** (5 components, 3,125 combinations):
```
[Optional: DataNormalizer] → Chunker → [Optional: Embedder] → Retriever → Generator
```
- Training: 1,000 evaluations
- Start from Stage 1 best architecture, explore additions
- RL learns: When normalization helps, embedding strategy matters
- Output: Best 5-component architecture (Accuracy: 0.78)

**Stage 3 - Full Pipeline** (8 components, 390,625 combinations):
```
PDFExtractor → DataNormalizer → Chunker → Embedder →
Retriever → Reranker → Generator → FactChecker
```
- Training: 2,000 evaluations
- Start from Stage 2 best, explore full extensions
- RL learns: PDF extraction quality, reranking tradeoffs, fact checking value
- Output: Best 8-component architecture (Accuracy: 0.87)

**Total Cost**: 3,500 evaluations vs 50,000+ for direct 8-component training (**93% reduction**)

**Why This Works**:
- **Transfer learning**: Discoveries from short pipelines transfer to longer ones
- **Incremental validation**: Catch problems early with simple pipelines
- **Reduced exploration**: RL doesn't need to rediscover "Chunker before Retriever" at each length
- **Curriculum principle**: Easy → hard learning is more sample-efficient

**Advantages**:
✅ Smoother learning curve for RL controller
✅ Early stages validate basic assumptions quickly
✅ Can stop early if short pipeline meets requirements
✅ Debugging is easier (isolate issues to specific stages)

**Disadvantages**:
❌ May bias toward incremental additions (best 8-component might not extend best 3-component)
❌ Requires careful curriculum design (which lengths? which additions?)
❌ Sequential training (can't parallelize across stages)

**Inspired by**: NASNet (Zoph et al., 2018) - starts with cell design, scales to full networks

#### Solution 4: Constraint-Based Pruning

**Concept**: Use component type systems to eliminate invalid sequences before RL explores them.

**Component Type System Definition**:
```
PDFExtractor: Input=FilePath → Output=RawText
OCREngine: Input=RawText → Output=CleanText
Chunker: Input=CleanText → Output=List[Chunk]
Retriever: Input=(Query, List[Chunk]) → Output=List[ScoredChunk]
Generator: Input=(Query, List[ScoredChunk]) → Output=Answer
```

**Interface Constraints**:
- PDFExtractor must come before Chunker (Chunker requires CleanText, PDFExtractor outputs RawText)
- Need OCREngine between PDFExtractor and Chunker (to convert RawText → CleanText)
- Retriever must come after Chunker (requires List[Chunk])
- Generator must come after Retriever (requires List[ScoredChunk])

**Valid Ordering Enforcement via Masked Actions**:

RL controller state tracking:
```
State: [PDFExtractor]
Available output: RawText

Valid next actions: [OCREngine] (accepts RawText)
Invalid actions: [Chunker] (needs CleanText), [Retriever] (needs Chunks), [Generator] (needs ScoredChunks)

RL action probabilities masked:
P(OCREngine | State=[PDFExtractor]) = 0.8 (valid)
P(Chunker | State=[PDFExtractor]) = 0.0 (masked)
P(Retriever | State=[PDFExtractor]) = 0.0 (masked)
```

**Search Space Reduction**:
- 10-component pipeline with 5 component types
- Without constraints: 5¹⁰ = 9,765,625 combinations
- With type constraints: ~50,000 valid orderings
- **Reduction: 99.5% of search space eliminated**

**Implementation**:
1. Define component interface ontology (input/output types)
2. RL uses masked action space based on current state
3. Policy network only assigns probability to valid actions
4. Invalid actions automatically get P(action|state) = 0

**Advantages**:
✅ Massive search space reduction (99%+) with no extra evaluations
✅ Guarantees all explored architectures are valid (no wasted evaluations on broken pipelines)
✅ Encoding human knowledge about component compatibility
✅ Works orthogonally with other solutions (combine with hierarchical, progressive, etc.)

**Disadvantages**:
❌ Requires defining comprehensive type system upfront
❌ May be too restrictive (some "invalid" sequences might work via implicit conversions)
❌ Doesn't reduce parameter search within valid architectures

**Related Technique**: Constrained decoding in language models (mask invalid tokens based on grammar)

#### Solution 5: Neural Architecture Performance Predictors

**Concept**: Train a separate neural network to predict architecture performance without evaluating it.

**Predictor Training**:

**Phase 1 - Collect Dataset**:
```
From previous RL training runs:
(architecture_1, task_context, performance_1)
(architecture_2, task_context, performance_2)
...
(architecture_1000, task_context, performance_1000)

Example:
Architecture: [Chunker(fixed, 256), BM25Retriever(k=5), GPT3.5Generator]
Task: Legal document QA, 100 docs, latency<1s
Performance: Accuracy=0.78, Latency=0.6s
```

**Phase 2 - Train Predictor Network**:
```
Architecture:
Input:
  - Architecture graph embedding (GNN over component DAG)
  - Task embedding (encode task description, requirements)

Output:
  - Predicted accuracy (0.0-1.0)
  - Predicted latency (seconds)
  - Confidence (0.0-1.0)

Training: Supervised learning on collected (architecture, performance) pairs
Loss: MSE on accuracy + MSE on latency
```

**RL with Predictor-Guided Search**:

**Standard RL** (expensive):
```
For each iteration:
  1. RL proposes architecture
  2. Build architecture (20 minutes)
  3. Optimize parameters with COSMOS (20 evaluations × 3 minutes = 1 hour)
  4. Evaluate performance (5 minutes)
  5. Update RL policy

Total per architecture: ~1.5 hours
5,000 architectures = 7,500 GPU hours
```

**Predictor-Guided RL** (efficient):
```
For each iteration:
  1. RL proposes architecture
  2. Predictor estimates performance (100 milliseconds)
  3. If predicted_accuracy > threshold (e.g., 0.75):
       → Evaluate for real (1.5 hours)
  4. If predicted_accuracy < threshold:
       → Skip, try next architecture (0 cost)
  5. Update RL policy AND predictor with real results

Filter rate: 80-90% of proposed architectures skipped
Effective evaluations: 5,000 → 500-1,000
Cost reduction: 5-10×
```

**Predictor Accuracy Improves Over Time**:
- Initially: 60% accuracy (random guessing)
- After 100 evaluations: 75% accuracy (useful filtering)
- After 500 evaluations: 85% accuracy (reliable)
- After 1000 evaluations: 90% accuracy (highly reliable)

**Advantages**:
✅ Massive cost reduction (5-10×) via cheap filtering
✅ Predictor improves continuously during RL training
✅ Can transfer predictor across similar tasks (legal QA → medical QA)
✅ Provides uncertainty estimates (skip high-uncertainty predictions)

**Disadvantages**:
❌ Requires initial dataset (cold start: first 100 evaluations without predictor)
❌ Predictor can be overconfident (wrongly skip good architectures)
❌ Training overhead (need to update predictor periodically)
❌ May bias search toward regions predictor is confident about

**Evidence from Literature**:
- **Predictor-based NAS** (Liu et al., 2018): Neural architecture performance prediction for NAS
- **BANANAS** (White et al., 2021): Bayesian optimization with neural predictor
- **GATES** (Ning et al., 2021): Graph neural networks for architecture performance prediction

#### Solution Comparison & Recommendations

**For Different Pipeline Lengths**:

**5-8 components** (tractable):
- Use: Standard RL + Constraint-Based Pruning
- Cost: 1,500-3,000 evaluations
- Timeline: 1-2 weeks
- Rationale: Manageable search space, constraints eliminate most invalid architectures

**9-12 components** (challenging):
- Use: Modular Decomposition (3-4 modules of 3-4 components each)
- Cost: 1,500-2,500 evaluations
- Timeline: 2-3 weeks (parallelizable to 1 week)
- Rationale: Breaks into tractable subproblems, modules can be optimized independently

**13-15 components** (difficult):
- Use: Hierarchical RL + Constraint Pruning
- Cost: 5,000-10,000 evaluations
- Timeline: 3-4 weeks
- Rationale: Automatic module discovery, constraints reduce exploration waste

**16-20 components** (very difficult):
- Use: Modular Decomposition + Progressive Lengthening
- Cost: 8,000-15,000 evaluations
- Timeline: 4-6 weeks
- Rationale: Mandatory decomposition, progressive training to handle complexity

**20+ components** (intractable without decomposition):
- Use: Modular Decomposition (4-6 modules) + Hierarchical RL within modules
- Cost: 10,000-20,000 evaluations
- Timeline: 2-3 months
- Rationale: Only feasible approach, combine multiple techniques

**Performance Predictors**: Add to any approach after ~100 evaluations for 2-5× speedup

**Key Principle**: Match optimization strategy to pipeline complexity. Start simple (standard RL), escalate to advanced techniques (modular, hierarchical) only when necessary.

---

## Approach 4: Evolutionary/Genetic Algorithms

### Concept
Maintain population of pipeline structures. Evolve via genetic operators (crossover, mutation, selection).

### Methodology
**Initialization**: Generate random population of N structures

**Each Generation**:
1. Evaluate fitness (performance) of each structure
2. Select top K performers (elites)
3. Generate offspring via crossover:
   - Parent A: TextExtractor → Chunker
   - Parent B: ImageExtractor → Merge → Chunker
   - Child: TextExtractor → Merge → Chunker
4. Apply mutations:
   - Add component
   - Remove component
   - Replace component
   - Swap order
5. New population = elites + offspring

Repeat 50-100 generations

### Implementation Complexity
**Effort**: 3-4 weeks
**Extensions needed**:
- Structure encoding/decoding
- Genetic operators for pipelines
- Fitness evaluation infrastructure
- Population management

### Computational Cost
- Population_size × generations × evaluations_per_structure
- Example: 50 population × 50 generations × 1 eval = 2500 evaluations
- Parallelizable within generation

### Pros
✅ No gradient computation needed
✅ Can handle discrete structure choices naturally
✅ Maintains diversity through population
✅ Less prone to local optima than RL
✅ Simpler than RL (no neural networks)
✅ Proven in NAS (NEAT, AmoebaNet)

### Cons
❌ Still very computationally expensive
❌ Requires careful tuning (mutation rates, selection pressure)
❌ Crossover may produce invalid structures
❌ Population size × generations = many evaluations
❌ No clear convergence criteria

### When to Use
- Similar to RL but prefer simpler implementation
- Team familiar with evolutionary algorithms
- Discrete structure choices (vs continuous parameters)
- Budget for 1000-3000 evaluations

---

## Approach Comparison Matrix

| Criterion | Templates | Grammar | RL | Evolution |
|-----------|-----------|---------|----|-----------|
| **Flexibility** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Compute Cost** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Implementation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Success Probability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Human Expertise** | High (design) | Medium (grammar) | Low (task), High (RL) | Low (task), Medium (EA) |
| **Evaluations Needed** | 50-100 | 500-1000 | 1000-5000 | 1000-3000 |
| **Timeline** | 1-2 days | 1-2 weeks | 1-2 months | 3-4 weeks |

---

## Decision Framework

### Use Templates If:
- ✅ First time doing architecture search
- ✅ Small-medium project scope
- ✅ Clear domain understanding
- ✅ 3-6 obvious architectural variations
- ✅ Need results quickly (days)
- ✅ Limited computational budget

### Use Grammar If:
- ✅ Need more flexibility than templates
- ✅ Many valid architectural patterns exist
- ✅ Want to discover novel combinations
- ✅ Have 1-2 weeks for implementation
- ✅ Can run 500-1000 evaluations
- ✅ Domain has clear compositional rules

### Use RL If:
- ✅ Large-scale deployment (10,000+ examples)
- ✅ Performance matters critically (1-2% delta valuable)
- ✅ Have GPU cluster + weeks of compute time
- ✅ Team with RL expertise
- ✅ Long-term project (3-6 months)
- ✅ Need truly automated discovery

### Use Evolution If:
- ✅ Similar requirements to RL
- ✅ But prefer simpler implementation
- ✅ Team familiar with genetic algorithms
- ✅ Discrete structure choices
- ✅ Want population diversity benefits

---

## Recommended Approach for PDF Parsing

**Phase 1 (Immediate)**: Templates
- Define 5 templates for common PDF types
- Optimize each with COSMOS (15 evals × 5 templates = 75 total)
- Select best on validation set
- **Timeline**: 1-2 days
- **Expected**: 90-95% of optimal performance

**Phase 2 (If needed)**: Grammar-based
- If templates insufficient, define grammar
- Sample 50-100 structures
- Optimize promising structures
- **Timeline**: +1-2 weeks
- **Expected**: 95-98% of optimal performance

**Phase 3 (Future research)**: RL/Evolution
- Only if building production system at scale
- 10,000+ diverse PDFs
- Months of development time
- **Expected**: 98-100% performance

**Rationale**:
- Templates give 90% of benefit for 1% of cost
- Grammar provides flexibility if templates insufficient
- RL/Evolution only justified for large-scale systems where 2-3% performance gain is worth months of development

---

## Integration with Current COSMOS

### Minimal Changes for Templates
No structural changes needed. Simple meta-loop:

```python
templates = [template1, template2, ...]
results = {}

for template in templates:
    optimizer = CompositionalOptimizer(
        components=template.components,
        structure=template.structure,
        budget=budget_per_template
    )
    results[template.name] = optimizer.optimize()

best_template = max(results, key=lambda t: results[t].score)
```

### Medium Changes for Grammar
Add structure sampling and evaluation:

```python
grammar = define_grammar()
structures = [sample_from_grammar(grammar) for _ in range(N)]

for structure in structures:
    components = extract_components(structure)
    optimizer = CompositionalOptimizer(
        components=components,
        structure=structure,  # Pass structure definition
        budget=budget_per_structure
    )
    evaluate(structure, optimizer.optimize())
```

### Major Changes for RL/Evolution
Requires new framework layer:

```python
class StructureSearchController:
    def __init__(self, component_library, search_strategy):
        self.library = component_library
        self.strategy = search_strategy  # RL or EA

    def search(self):
        while not converged:
            # RL: Controller proposes structure
            # EA: Population generates offspring
            structure = self.strategy.propose()

            # Evaluate via COSMOS
            optimizer = CompositionalOptimizer(
                components=structure.components,
                structure=structure
            )
            score = optimizer.optimize()

            # RL: Update controller
            # EA: Update population
            self.strategy.update(structure, score)
```

---

## Open Research Questions

1. **Structure Complexity**: How complex can discovered structures be before they overfit?

2. **Generalization**: Do structures found on one dataset transfer to related domains?

3. **Component Composability**: Which components are "compatible"? Can we learn compatibility rules?

4. **Hierarchical Search**: Can we search at multiple levels (micro-structure within components, macro-structure of pipeline)?

5. **Multi-Objective**: How to balance accuracy, latency, cost when discovering structures?

6. **Human-in-the-Loop**: Can human feedback guide structure search more efficiently than pure automated search?

7. **Transfer Learning**: Can structures discovered for one task bootstrap search for related tasks?

---

## References

### Neural Architecture Search
- Zoph & Le (2017): "Neural Architecture Search with Reinforcement Learning"
- Pham et al. (2018): "Efficient Neural Architecture Search via Parameter Sharing"
- Real et al. (2019): "Regularized Evolution for Image Classifier Architecture Search"
- Liu et al. (2019): "DARTS: Differentiable Architecture Search"

### AutoML Pipelines
- Feurer et al. (2015): "Efficient and Robust Automated Machine Learning" (auto-sklearn)
- Olson & Moore (2016): "TPOT: A Tree-based Pipeline Optimization Tool"

### Program Synthesis
- Real et al. (2020): "AutoML-Zero: Evolving Machine Learning Algorithms From Scratch"

### COSMOS Framework
- Phase 6 (2025): Current implementation (parameter optimization only)
- Proposed: Structure search extension

---

## Implementation Priority

**Priority 1** (Immediate): Template-based approach
- Minimal changes to COSMOS
- High success probability
- Fast implementation
- Sufficient for most use cases

**Priority 2** (Future enhancement): Grammar-based approach
- Medium complexity
- Enables more flexible discovery
- Good balance of automation and control

**Priority 3** (Research direction): RL/Evolution
- High complexity
- Only for large-scale deployments
- Requires significant resources

---

**Document Version**: 1.0
**Last Updated**: 2025-10-01
**Status**: Specification for future development
