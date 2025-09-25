# Evaluation System Improvements: From Keyword Matching to Semantic Evaluation

## Overview

This specification outlines a three-phase approach to improve the RAG evaluation system by moving away from the current keyword-matching approach (which gives 0.5 minimum score and penalizes temperature variation) to more accurate semantic evaluation methods.

## Current State

### Problems with Current Keyword Matching
- **Minimum score of 0.5**: Even completely wrong answers get 50% credit
- **Exact substring matching**: Penalizes paraphrasing and creative answers
- **Temperature bias**: High-temperature models unfairly penalized for varied vocabulary
- **Not true binary scoring**: Should be 0 or 1, not 0.5 or 1.0

### Example of Current Limitations
```python
# Current approach
Query: "What is the capital of France?"
Expected keyword: "paris"

Answer: "The capital of France is Paris"          → Score: 1.0 ✓
Answer: "France's seat of government is in Paris"  → Score: 1.0 ✓
Answer: "The capital city is Paris"                → Score: 1.0 ✓
Answer: "The capital of France is Lyon"            → Score: 0.5 ✗ (should be 0!)
Answer: "I don't know"                             → Score: 0.5 ✗ (should be 0!)
```

## Proposed Modular Implementation

The implementation should be **modular**, allowing each phase to work independently. The evaluation method should be configurable via parameters, not hard-coded.

### Architecture
```python
class EvaluationMethod(Enum):
    KEYWORD = "keyword"           # Current method
    SEMANTIC_FIXED = "semantic_fixed"     # Phase 1
    SEMANTIC_DYNAMIC = "semantic_dynamic" # Phase 2
    LLM_JUDGE = "llm_judge"              # Phase 3
    HYBRID = "hybrid"                    # Future: combination

class GridSearchEvaluator:
    def __init__(self,
                 evaluation_method: EvaluationMethod = EvaluationMethod.SEMANTIC_FIXED,
                 semantic_threshold: float = 0.75,
                 llm_judge_top_n: int = 5):
        self.method = evaluation_method
        self.threshold = semantic_threshold
        self.llm_top_n = llm_judge_top_n
```

---

## Phase 1: Fixed Threshold Semantic Similarity ✅ (Implement First)

### Objective
Replace keyword matching with semantic similarity using a fixed threshold for binary scoring.

### Implementation
```python
def evaluate_semantic_fixed(answer: str, ground_truth: str, threshold: float = 0.75) -> float:
    """
    Binary evaluation using semantic similarity with fixed threshold.

    Returns:
        1.0 if similarity >= threshold (correct)
        0.0 if similarity < threshold (incorrect)
    """
    # Use existing SemanticMetrics class
    from autorag.evaluation.semantic_metrics import SemanticMetrics

    evaluator = SemanticMetrics(similarity_threshold=threshold)
    similarity = evaluator.compute_similarity(
        evaluator.model.encode(answer),
        evaluator.model.encode(ground_truth)
    )

    return 1.0 if similarity >= threshold else 0.0
```

### Configuration
- **Default threshold**: 0.75 (configurable)
- **Model**: `all-MiniLM-L6-v2` (lightweight, fast)
- **Caching**: Reuse embeddings across evaluations

### Advantages
- True binary scoring (0 or 1)
- Handles paraphrasing and synonyms
- Temperature-agnostic
- Fast and deterministic
- No API costs

### Integration Points
- Modify `evaluate_configuration()` in `run_minimal_real_grid_search.py`
- Add `evaluation_method` parameter to `MinimalRealGridSearch.__init__()`
- Keep backward compatibility with keyword method

---

## Phase 2: Dynamic Threshold Semantic Similarity

### Objective
Find optimal threshold automatically based on validation data, and fix the F1 score calculation.

### Current F1 Problem
```python
# Current implementation incorrectly assumes:
y_true = [1] * len(ground_truths)  # All answers should be correct

# Should be:
y_true = human_validated_correctness  # Actual 0/1 labels
```

### Implementation
```python
def find_optimal_threshold(validation_answers: List[str],
                          validation_truths: List[str],
                          human_labels: Optional[List[int]] = None) -> float:
    """
    Find threshold that maximizes accuracy on validation set.

    If human_labels not provided, uses high similarity (>0.9) as proxy for correct.
    """
    from autorag.evaluation.semantic_metrics import SemanticMetrics

    evaluator = SemanticMetrics()

    # Get similarities for validation set
    similarities = evaluator.semantic_similarity_batch(
        validation_answers,
        validation_truths
    )

    # Test multiple thresholds
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    best_threshold = 0.75
    best_accuracy = 0

    for threshold in thresholds:
        predictions = [1 if sim >= threshold else 0 for sim in similarities]

        if human_labels:
            # Use actual human judgments
            accuracy = sum(p == h for p, h in zip(predictions, human_labels)) / len(predictions)
        else:
            # Use very high similarity as proxy for correctness
            proxy_labels = [1 if sim > 0.9 else 0 for sim in similarities]
            accuracy = sum(p == l for p, l in zip(predictions, proxy_labels)) / len(predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold
```

### Workflow
1. Run small validation set (50-100 Q&A pairs)
2. Find optimal threshold
3. Use this threshold for full grid search
4. Report threshold used in results

### Configuration
- **Validation size**: 5-10% of test set
- **Threshold range**: 0.5 to 0.9 in 0.05 increments
- **Optimization metric**: Accuracy (or F1 if labels available)

---

## Phase 3: LLM-as-Judge for Top Architectures

### Objective
Re-evaluate top N configurations using LLM-as-judge for final ranking.

### Implementation
```python
def llm_judge_evaluation(question: str, answer: str, ground_truth: str) -> float:
    """
    Use LLM to judge answer correctness.

    Returns:
        Score from 0.0 to 1.0 based on factual accuracy
    """
    prompt = f"""
    Evaluate if the generated answer is factually correct compared to the expected answer.

    Question: {question}
    Expected Answer: {ground_truth}
    Generated Answer: {answer}

    Scoring criteria:
    - 1.0: Completely correct, all key facts present
    - 0.8: Mostly correct, minor details missing
    - 0.6: Partially correct, some key facts present
    - 0.4: Somewhat correct, major gaps
    - 0.2: Mostly incorrect
    - 0.0: Completely incorrect or irrelevant

    Respond with only a number between 0 and 1.
    """

    # Use GPT-3.5 for cost efficiency
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # Deterministic
        max_tokens=10
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    except:
        return 0.0  # Default to incorrect if parsing fails
```

### Workflow
1. Complete grid search with semantic similarity
2. Select top N configurations (default: 5)
3. Re-evaluate these N configs with LLM-judge
4. Generate final ranking report

### Configuration
- **Top N**: 5 configurations (configurable)
- **Model**: GPT-3.5-turbo (balance cost/quality)
- **Temperature**: 0 for consistency
- **Retry logic**: 3 attempts with exponential backoff
- **Cache results**: Store LLM judgments for reuse

### Cost Management
```python
def estimate_llm_judge_cost(n_configs: int, n_queries: int) -> float:
    """Estimate cost before running LLM judge."""
    # GPT-3.5: ~$0.002 per 1K tokens
    # Assume ~200 tokens per evaluation
    tokens_per_eval = 200
    cost_per_1k_tokens = 0.002
    total_evals = n_configs * n_queries
    total_cost = (total_evals * tokens_per_eval / 1000) * cost_per_1k_tokens
    return total_cost

# Example: 5 configs × 100 queries × 200 tokens = 100K tokens = $0.20
```

---

## Implementation Priority

1. **Phase 1** (Immediate): Implement semantic_fixed - simple, immediate improvement
2. **Phase 2** (Next Sprint): Add dynamic threshold finding
3. **Phase 3** (Future): Add LLM-judge for final validation

## Migration Path

### Step 1: Add to existing code without breaking changes
```python
# In test_grid_search_msmarco.ps1, add parameter:
param(
    [string]$EvaluationMethod = "semantic_fixed",  # New parameter
    [float]$SemanticThreshold = 0.75,              # New parameter
    ...
)
```

### Step 2: Update evaluate_configuration
```python
def evaluate_configuration(self, params, documents, queries, config_num,
                          evaluation_method="keyword"):
    if evaluation_method == "semantic_fixed":
        score = evaluate_semantic_fixed(answer, expected, self.semantic_threshold)
    else:
        # Keep existing keyword logic for backward compatibility
        score = 1.0 if expected.lower() in answer.lower() else 0.5
```

### Step 3: Gradual rollout
- Run both methods in parallel initially
- Compare results
- Validate semantic threshold on your domain
- Switch default once confident

## Success Metrics

### Phase 1 Success
- [ ] Semantic evaluation implemented and working
- [ ] Grid search identifies different "best" architecture than keyword method
- [ ] Scores show true 0/1 distribution (not 0.5/1)
- [ ] High-temperature models no longer unfairly penalized

### Phase 2 Success
- [ ] Optimal threshold discovered automatically
- [ ] F1 score correctly calculated
- [ ] Threshold documented and justified

### Phase 3 Success
- [ ] Top 5 architectures re-ranked by LLM
- [ ] Final ranking more aligned with human judgment
- [ ] Cost remains under $1 for typical evaluation

## Testing Strategy

### Unit Tests
```python
def test_semantic_evaluation():
    # Test paraphrases score high
    assert evaluate_semantic_fixed(
        "The capital of France is Paris",
        "Paris is France's capital city",
        threshold=0.7
    ) == 1.0

    # Test wrong answers score zero
    assert evaluate_semantic_fixed(
        "The capital of France is London",
        "The capital of France is Paris",
        threshold=0.7
    ) == 0.0
```

### Integration Tests
- Run same queries through all three methods
- Compare score distributions
- Validate no method gives 0.5 scores

### A/B Testing
- Run grid search with keyword method
- Run grid search with semantic method
- Compare which architectures are selected
- Validate semantic choices with human review

## Notes

- **Modularity is critical**: Each phase should work independently
- **Backward compatibility**: Keep keyword method available via parameter
- **Start simple**: Phase 1 with fixed threshold is good enough for most cases
- **Document choices**: Log which evaluation method was used in results
- **Monitor metrics**: Track how scores change with new methods