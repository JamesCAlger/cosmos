# Query Expansion for RAG Systems

## Overview

Query expansion is a preprocessing technique that transforms or augments user queries before retrieval to improve the relevance and recall of retrieved documents. It addresses the fundamental vocabulary mismatch problem where users and documents use different terms to describe the same concepts.

## Core Problem

Users often express their information needs using different vocabulary than what appears in documents:
- User: "How do I make my computer faster?"
- Documents: "system optimization", "performance tuning", "reduce latency"
- Result: Poor retrieval due to vocabulary mismatch

## Types of Query Expansion

### 1. Synonym Expansion
Adds semantically related terms to broaden the search:
- Input: "car insurance"
- Output: "car insurance" OR "auto insurance" OR "vehicle coverage" OR "automobile policy"

### 2. Query Decomposition
Breaks complex multi-hop questions into simpler sub-queries:
- Input: "What is the population of the capital of France?"
- Decomposed:
  1. "What is the capital of France?" → Paris
  2. "What is the population of Paris?" → 2.1M

### 3. Hypothetical Document Embeddings (HyDE)
Generates a hypothetical answer to create better embedding-based queries:
- Original query: "How does photosynthesis work?"
- Generated hypothetical: "Photosynthesis is the process by which plants convert sunlight..."
- Use hypothetical's embedding for similarity search

### 4. Contextual Expansion
Incorporates conversation history or session context:
- Previous: "Tell me about Einstein"
- Current: "What was his most famous equation?"
- Expanded: "What was Einstein's most famous equation? Einstein relativity E=mc²"

### 5. Question Generation
Creates multiple question variants to capture different phrasings:
- Input: "Python list comprehension"
- Generated variants:
  - "How to use list comprehension in Python?"
  - "Python list comprehension syntax examples"
  - "List comprehension vs for loop Python"

## Benefits for RAG Systems

### Improved Recall
- Captures 30-40% more relevant documents
- Reduces false negatives in retrieval
- Handles vocabulary variations across domains

### Robustness to User Input
- Accommodates different phrasings
- Handles incomplete or ambiguous queries
- Bridges technical vs. colloquial language gaps

### Better Semantic Coverage
- Explores related concepts
- Captures implicit information needs
- Addresses query intent beyond literal terms

## Implementation Considerations

### Performance Trade-offs
- **Synonym expansion**: Fast (10ms), moderate improvement (15-25% recall boost)
- **HyDE**: Slower (200ms for LLM call), significant improvement (30-40% recall boost)
- **Query decomposition**: Slowest (500ms+), best for complex queries (35-50% improvement)

### When to Use
- **High-stakes domains**: Medical, legal, financial (prioritize recall)
- **Technical content**: Documentation, research papers (vocabulary variation)
- **Conversational systems**: Chatbots, assistants (context matters)
- **Sparse data**: Small corpora where every match counts

### Integration Points
Query expansion should occur:
1. After query preprocessing (spelling correction, normalization)
2. Before retrieval execution
3. With caching for repeated expansions
4. With option to bypass for exact match requirements

## Evaluation Metrics

### Expansion Quality
- **Coverage**: Percentage of relevant terms captured
- **Precision**: Ratio of useful to noisy expansions
- **Diversity**: Variety of expansion strategies triggered

### System Impact
- **Recall improvement**: Increase in relevant documents retrieved
- **Latency overhead**: Additional processing time
- **Cost implications**: Extra embeddings or LLM calls

### End-to-End Metrics
- **Answer quality**: Final response accuracy improvement
- **User satisfaction**: Reduced query reformulation
- **System efficiency**: Balance of quality vs. computational cost

## Best Practices

1. **Combine multiple strategies**: Use lightweight expansions first, expensive ones selectively
2. **Cache expansions**: Store common query expansions
3. **Monitor effectiveness**: Track which expansions improve outcomes
4. **Allow user control**: Provide options to disable or tune expansion
5. **Domain adaptation**: Customize expansion strategies per use case

## Future Directions

- **Learned expansion**: Train models on query-document pairs
- **Personalized expansion**: Adapt to user vocabulary preferences
- **Multi-modal expansion**: Extend to image, video queries
- **Feedback loops**: Learn from successful/failed expansions
- **Efficient implementations**: Reduce computational overhead