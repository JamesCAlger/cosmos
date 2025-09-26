# auto-RAG: Automated RAG Pipeline Optimization

An intelligent Retrieval-Augmented Generation (RAG) system with automated hyperparameter optimization using Bayesian search. This project demonstrates how to systematically optimize RAG pipeline configurations to achieve the best performance on question-answering tasks.

## Overview

auto-RAG provides a modular RAG pipeline with automated optimization capabilities to find the best configuration for your specific use case. The system evaluates different combinations of chunking strategies, retrieval methods, reranking options, and generation parameters to maximize answer accuracy.

## Key Features

- **Modular RAG Pipeline**: Flexible architecture supporting multiple chunking, retrieval, and generation strategies
- **Bayesian Optimization**: Intelligent search through hyperparameter space to find optimal configurations
- **Multiple Retrieval Methods**: Dense, sparse (BM25), and hybrid retrieval with configurable weighting
- **Advanced Metrics**: Comprehensive evaluation including semantic similarity, ROUGE scores, and retrieval precision
- **Real-time Rate Limiting**: Handles OpenAI API rate limits with automatic retries and delays
- **MS MARCO Integration**: Benchmarked against the MS MARCO dataset for realistic evaluation

## Project Structure

```
auto-RAG/
├── autorag/
│   ├── components/
│   │   ├── chunkers/           # Text chunking strategies
│   │   │   ├── base.py
│   │   │   ├── fixed_size.py
│   │   │   └── semantic.py
│   │   ├── embedders/          # Embedding generation
│   │   │   ├── base.py
│   │   │   └── openai.py
│   │   ├── generators/         # Answer generation
│   │   │   ├── base.py
│   │   │   └── openai.py
│   │   ├── rerankers/          # Document reranking
│   │   │   ├── base.py
│   │   │   └── cross_encoder.py
│   │   ├── retrievers/         # Document retrieval
│   │   │   ├── base.py
│   │   │   ├── bm25.py
│   │   │   ├── dense.py
│   │   │   └── hybrid.py
│   │   └── vector_stores/      # Vector storage
│   │       ├── base.py
│   │       └── simple.py
│   ├── evaluation/
│   │   └── metrics.py          # Evaluation metrics
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── grid_search.py     # Grid search optimization
│   │   └── bayesian_search.py # Bayesian optimization
│   ├── pipelines/
│   │   └── rag_pipeline.py    # Main RAG pipeline
│   └── utils/
│       └── logging.py          # Logging utilities
├── scripts/
│   ├── run_bayesian_full_space.py         # Basic Bayesian optimization
│   ├── run_bayesian_full_space_enhanced.py # Enhanced with median scoring
│   ├── run_bayesian_msmarco.py            # MS MARCO specific optimization
│   └── run_minimal_real_grid_search.py    # Grid search implementation
├── bayesian_enhanced_results/              # Optimization results directory
│   ├── bayesian_optimization_results.json # Latest 50-evaluation run
│   └── intermediate_100.json              # 100-evaluation run
├── .env                                    # API keys (not in repo)
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

## Usage

### Running Bayesian Optimization

The primary optimization script uses Bayesian search to find optimal RAG configurations:

```bash
# Run enhanced Bayesian optimization with median scoring
python scripts/run_bayesian_full_space_enhanced.py \
    --n-calls 50 \
    --real-api \
    --num-docs 250 \
    --num-queries 25
```

Parameters:
- `--n-calls`: Number of configurations to evaluate
- `--real-api`: Use actual OpenAI API (vs mock mode)
- `--num-docs`: Number of documents from MS MARCO
- `--num-queries`: Number of test queries

### Running Grid Search

For exhaustive search through a predefined parameter grid:

```bash
python scripts/run_minimal_real_grid_search.py
```

## Optimization Results

The system has been extensively tested with Bayesian optimization. Here are two example runs demonstrating the optimization process:

### Run 1: 100 Evaluations (Mean-based scoring)
- **File**: `bayesian_enhanced_results/intermediate_100.json`
- **Best Score**: 0.454 (45.4% accuracy)
- **Time**: ~7903 seconds
- **Optimal Config**:
  - Chunking: Fixed size (256 tokens)
  - Retrieval: Hybrid (weight: 0.661)
  - Temperature: 0.285
  - Max tokens: 172

### Run 2: 50 Evaluations (Recent optimization)
- **File**: `bayesian_enhanced_results/bayesian_optimization_results.json`
- **Best Score**: 0.461 (46.1% accuracy)
- **Time**: 3677 seconds
- **Optimal Config**:
  - Chunking: Fixed size (256 tokens)
  - Retrieval: Hybrid (weight: 0.401)
  - Temperature: 0.172
  - Max tokens: 186

Key findings:
- Hybrid retrieval consistently outperforms pure dense or sparse methods
- Fixed-size chunking with 256 tokens provides good balance
- Lower temperatures (0.17-0.28) improve accuracy
- Reranking did not improve performance in these tests

## Configuration Space

The optimizer explores the following hyperparameters:

| Parameter | Options/Range | Description |
|-----------|--------------|-------------|
| `chunking_strategy` | 'fixed', 'semantic' | How documents are split |
| `chunk_size` | 128, 256, 512 | Tokens per chunk |
| `retrieval_method` | 'dense', 'bm25', 'hybrid' | Retrieval strategy |
| `retrieval_top_k` | 3, 5, 10 | Documents to retrieve |
| `hybrid_weight` | 0.0 - 1.0 | Dense vs sparse balance |
| `reranking_enabled` | True/False | Use cross-encoder reranking |
| `temperature` | 0.0 - 0.3 | Generation randomness |
| `max_tokens` | 150 - 300 | Maximum answer length |

## Evaluation Metrics

The system evaluates configurations using multiple metrics:

1. **Semantic Similarity**: Cosine similarity between generated and ground-truth answers
2. **ROUGE-L Score**: Longest common subsequence-based similarity
3. **Retrieval Precision**: Relevance of retrieved documents
4. **Context Utilization**: How well the answer uses retrieved context
5. **Latency**: End-to-end response time

The optimizer can use either mean or median aggregation across queries, with median being more robust to outliers.

## API Rate Limiting

The system includes built-in rate limiting for OpenAI's API:
- 500 requests per minute limit (free tier)
- 0.15-second delay between calls
- Exponential backoff on rate limit errors
- Shared client instance to avoid connection issues

## Development

### Adding New Components

1. **New Chunker**: Inherit from `BaseChunker` in `autorag/components/chunkers/base.py`
2. **New Retriever**: Inherit from `BaseRetriever` in `autorag/components/retrievers/base.py`
3. **New Generator**: Inherit from `BaseGenerator` in `autorag/components/generators/base.py`

### Running Tests

```bash
# Test with mock API (no charges)
python scripts/run_bayesian_full_space.py --n-calls 10

# Test with real API (small scale)
python scripts/run_bayesian_full_space.py --n-calls 5 --real-api --num-queries 5
```

## Performance Notes

- The system achieves 45-46% accuracy on MS MARCO dataset
- Individual queries can achieve up to 95% accuracy
- Median-based optimization (vs mean) better handles query difficulty variations
- Hybrid retrieval consistently outperforms single-method approaches

## Future Improvements

- [ ] Add support for more embedding models (Cohere, Anthropic)
- [ ] Implement query expansion techniques
- [ ] Add multi-stage retrieval pipelines
- [ ] Support for custom datasets beyond MS MARCO
- [ ] Implement parallel evaluation for faster optimization
- [ ] Add visualization tools for optimization convergence

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- MS MARCO dataset from Microsoft Research
- OpenAI for GPT and embedding APIs
- scikit-optimize for Bayesian optimization framework