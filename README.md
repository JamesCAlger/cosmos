# AutoRAG: Automatic RAG Architecture Optimization System

## Week 1: Minimal RAG Baseline Implementation

This is the Week 1 implementation of the AutoRAG system, providing a minimal working RAG pipeline with RAGAS evaluation.

### Features Implemented (Week 1)

- ✅ **Document Processing**: Fixed-size chunking (256 tokens)
- ✅ **Embedding**: OpenAI text-embedding-ada-002
- ✅ **Vector Storage**: In-memory FAISS index
- ✅ **Retrieval**: Cosine similarity search
- ✅ **Generation**: GPT-3.5-turbo with simple prompt template
- ✅ **Evaluation**: RAGAS metrics (faithfulness, answer_relevancy, context_relevancy)
- ✅ **Dataset**: MS MARCO subset loader

### Quick Start

#### 1. Setup Environment

```bash
# Clone the repository (if applicable)
cd auto-RAG

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

#### 3. Run Baseline Evaluation

```bash
python scripts/run_baseline.py
```

This will:
- Load a subset of MS MARCO (100 documents, 20 queries)
- Index the documents using OpenAI embeddings
- Process queries through the RAG pipeline
- Evaluate results using RAGAS metrics
- Save results to `experiments/baseline_[timestamp].json`

### Project Structure

```
auto-RAG/
├── autorag/
│   ├── core/              # Core RAG components
│   │   ├── document_processor.py  # Document chunking
│   │   ├── embedder.py           # OpenAI embeddings
│   │   ├── retriever.py          # FAISS retrieval
│   │   ├── generator.py          # GPT-3.5 generation
│   │   └── rag_pipeline.py       # Main pipeline
│   ├── evaluation/        # Evaluation components
│   │   └── ragas_evaluator.py    # RAGAS integration
│   └── datasets/          # Dataset loaders
│       └── msmarco_loader.py     # MS MARCO loader
├── scripts/
│   └── run_baseline.py    # Main evaluation script
├── experiments/           # Evaluation results (gitignored)
├── requirements.txt       # Python dependencies
├── .env                  # API keys (gitignored)
└── README.md            # This file
```

### Baseline Configuration

The Week 1 implementation uses a fixed configuration:

```python
{
    "chunking": {
        "strategy": "fixed",
        "size": 256,
        "overlap": 0
    },
    "embedding": {
        "model": "text-embedding-ada-002"
    },
    "retrieval": {
        "method": "dense",
        "top_k": 5
    },
    "generation": {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 300
    }
}
```

### Expected Output

After running the baseline evaluation, you'll see:

1. **Console Output**: Summary of results including:
   - Number of documents indexed
   - Number of queries processed
   - Performance metrics (indexing time, query time)
   - RAGAS evaluation scores

2. **JSON File**: Detailed results saved to `experiments/baseline_[timestamp].json` containing:
   - Full configuration
   - Performance metrics
   - RAGAS evaluation scores
   - Sample query results

### Sample Results Format

```json
{
    "timestamp": "2024-01-XX",
    "configuration": {...},
    "dataset": {
        "num_documents": 100,
        "num_queries": 20
    },
    "performance": {
        "indexing_time_seconds": X.XX,
        "avg_query_time_seconds": X.XX
    },
    "evaluation": {
        "ragas_metrics": {
            "faithfulness": 0.XX,
            "answer_relevancy": 0.XX,
            "context_relevancy": 0.XX
        }
    }
}
```

### Cost Estimation

For the baseline evaluation with 100 documents and 20 queries:
- Embedding costs: ~$0.01
- Generation costs: ~$0.02
- **Total estimated cost: < $0.05**

### Next Steps (Week 2)

- Implement modular architecture with configurable components
- Add multiple chunking strategies
- Support for different embedding models
- Implement configuration management system

### Troubleshooting

1. **OpenAI API Key Error**: Make sure your `.env` file contains a valid API key
2. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
3. **FAISS Installation Issues**: On Windows, you might need: `pip install faiss-cpu --no-cache-dir`
4. **MS MARCO Loading Issues**: The script includes fallback synthetic data if dataset loading fails

### Development Notes

This is a minimal Week 1 implementation focusing on:
- Getting a working end-to-end pipeline
- Establishing baseline metrics
- Validating the evaluation methodology

The code is intentionally simple with hardcoded configurations. Week 2 will introduce the modular architecture and configuration system.