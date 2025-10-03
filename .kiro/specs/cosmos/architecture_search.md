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
