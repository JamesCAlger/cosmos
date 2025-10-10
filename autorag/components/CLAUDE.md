# RAG Components - Architecture & Design Philosophy

## When to Read This

**Read this doc if you're**:
- ✅ Adding a new component implementation (e.g., new chunker, retriever)
- ✅ Understanding base classes and data structures (Document, Chunk, QueryResult)
- ✅ Learning component design patterns (wrapper, composite)
- ✅ Figuring out how components fit together in the pipeline

**Skip this doc if you're**:
- ❌ Just running optimizations → see root `CLAUDE.md`
- ❌ Understanding COSMOS framework → see `autorag/cosmos/CLAUDE.md`
- ❌ Working specifically with rerankers → see `autorag/components/rerankers/CLAUDE.md` first

**Prerequisites**: Familiarity with RAG concepts (retrieval, generation, embeddings)

---

## Component Abstraction

The auto-RAG system uses a **modular component architecture** where each stage of the RAG pipeline is an independent, swappable component.

**Design principle**: Components should be:
1. **Self-contained**: No hard dependencies on other components
2. **Configurable**: Accept config dict for all hyperparameters
3. **Testable**: Can be tested in isolation
4. **Swappable**: Multiple implementations of same interface

## Base Classes (`base.py`)

All components inherit from abstract base classes defining standard interfaces:

### Component Hierarchy
```
Component (abstract base)
  ├── Chunker
  ├── Embedder
  ├── VectorStore
  ├── Retriever
  ├── Reranker
  ├── Generator
  └── PostProcessor
```

### Core Data Structures

#### `Document` (base.py:9-18)
```python
@dataclass
class Document:
    content: str                    # Full document text
    metadata: Dict[str, Any]        # Source, title, etc.
    doc_id: Optional[str]           # Unique identifier
```

#### `Chunk` (base.py:21-29)
```python
@dataclass
class Chunk:
    content: str                    # Chunk text
    doc_id: str                     # Parent document
    chunk_id: str                   # Unique chunk identifier
    start_char: int                 # Position in original doc
    end_char: int
    metadata: Dict[str, Any]        # Inherited + chunk-specific
```

#### `QueryResult` (base.py:32-37)
```python
@dataclass
class QueryResult:
    chunk: Chunk                    # Retrieved chunk
    score: float                    # Relevance score
    metadata: Optional[Dict]        # Retrieval-specific metadata
```

## Component Types & Pipeline Flow

### Standard RAG Pipeline
```
Documents → Chunker → Chunks
                        ↓
                    Embedder → Embeddings
                        ↓
                   VectorStore (index)

Query → Embedder → Query Embedding
                        ↓
                   VectorStore (search)
                        ↓
                    Retriever → QueryResults (List[QueryResult])
                        ↓
                   [Reranker] → Reranked QueryResults (optional)
                        ↓
                    Generator → Answer (str)
```

### Component Responsibilities

#### **Chunker** (`base.py:67-77`)
**Purpose**: Split documents into retrievable chunks
**Input**: `List[Document]`
**Output**: `List[Chunk]`
**Key method**: `chunk(documents) -> List[Chunk]`

**Implementations**:
- `FixedSizeChunker`: Fixed token/character windows
- `SemanticChunker`: Sentence-boundary aware splitting
- `SlidingWindowChunker`: Overlapping windows

#### **Embedder** (`base.py:80-95`)
**Purpose**: Convert text to dense vector representations
**Input**: `List[str]` or `str`
**Output**: `List[List[float]]` or `List[float]`
**Key methods**:
- `embed(texts) -> List[List[float]]`
- `embed_query(query) -> List[float]`

**Implementations**:
- `OpenAIEmbedder`: OpenAI embedding models
- `CachedEmbedder`: Wrapper with disk caching
- `MockEmbedder`: Fast random embeddings for testing

#### **VectorStore** (`base.py:98-118`)
**Purpose**: Store and search embeddings
**Input**: Embeddings + Chunks
**Output**: `List[QueryResult]`
**Key methods**:
- `add(embeddings, chunks)`
- `search(query_embedding, top_k) -> List[QueryResult]`

**Implementations**:
- `SimpleVectorStore`: In-memory cosine similarity
- `FAISSVectorStore`: FAISS-based for scale

#### **Retriever** (`base.py:121-131`)
**Purpose**: Retrieve relevant chunks for queries
**Input**: `query: str, top_k: int`
**Output**: `List[QueryResult]`
**Key method**: `retrieve(query, top_k) -> List[QueryResult]`

**Implementations**:
- `DenseRetriever`: Embedding-based similarity
- `BM25Retriever`: Sparse lexical matching
- `HybridRetriever`: Weighted combination

**Note**: Retriever typically composes Embedder + VectorStore

#### **Reranker** (`base.py:134-144`)
**Purpose**: Re-score retrieved results for better relevance
**Input**: `query: str, results: List[QueryResult], top_k: int`
**Output**: `List[QueryResult]` (reranked)
**Key method**: `rerank(query, results, top_k) -> List[QueryResult]`

**Implementations**:
- `CrossEncoderReranker`: Cross-encoder model for joint query-doc encoding

**Pipeline position**: Between Retriever and Generator (optional)

See: `autorag/components/rerankers/CLAUDE.md` for detailed reranker documentation

#### **Generator** (`base.py:147-157`)
**Purpose**: Generate answers from query + context
**Input**: `query: str, context: List[QueryResult]`
**Output**: `answer: str`
**Key method**: `generate(query, context) -> str`

**Implementations**:
- `OpenAIGenerator`: OpenAI chat models (GPT-3.5, GPT-4)
- `MockGenerator`: Simple template-based generation for testing

## Configuration Pattern

All components accept a `config` dict in `__init__`:

```python
# Example: Configuring a chunker
chunker_config = {
    'chunk_size': 256,
    'overlap': 50,
    'chunking_strategy': 'semantic'
}
chunker = FixedSizeChunker(chunker_config)

# Example: Configuring a generator
generator_config = {
    'model': 'gpt-3.5-turbo',
    'temperature': 0.7,
    'max_tokens': 512
}
generator = OpenAIGenerator(generator_config)
```

**Benefit**: Configs can be serialized, searched, and optimized programmatically

## Component Composition

Components are composed into pipelines in two ways:

### 1. **Manual Composition** (Full Control)
```python
chunker = FixedSizeChunker({'chunk_size': 256})
embedder = OpenAIEmbedder({'model': 'text-embedding-ada-002'})
retriever = DenseRetriever({'embedder': embedder, 'top_k': 5})
reranker = CrossEncoderReranker({'model_name': 'ms-marco-MiniLM-L-6-v2'})
generator = OpenAIGenerator({'model': 'gpt-3.5-turbo'})

# Pipeline execution
chunks = chunker.chunk(documents)
retriever.index(chunks)
results = retriever.retrieve(query, top_k=10)
reranked = reranker.rerank(query, results, top_k=5)
answer = generator.generate(query, reranked)
```

### 2. **Orchestrator-Based** (Declarative)
```python
pipeline = RAGPipeline(config_path='config.yaml')
pipeline.index(documents)
answer = pipeline.query(query)
```

## Adding New Component Implementations

### Step 1: Inherit from base class
```python
from autorag.components.base import Retriever, QueryResult

class MyCustomRetriever(Retriever):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Initialize with config

    def retrieve(self, query: str, top_k: int = 5) -> List[QueryResult]:
        # Implement retrieval logic
        pass
```

### Step 2: Register in component factory (if using orchestrator)
```python
# In evaluators.py or component registry
def build_component(component_type: str, config: dict):
    if component_type == 'retriever':
        method = config.get('retrieval_method', 'dense')
        if method == 'my_custom':
            return MyCustomRetriever(config)
```

### Step 3: Add to search space (for optimization)
```python
search_space = {
    'retriever': {
        'retrieval_method': ['dense', 'bm25', 'my_custom'],
        'top_k': [3, 5, 10]
    }
}
```

## Design Patterns

### Pattern 1: Wrapper Components
**Example**: `CachedEmbedder` wraps `OpenAIEmbedder`
**Purpose**: Add functionality (caching, logging, metrics) without modifying base

```python
class CachedEmbedder(Embedder):
    def __init__(self, base_embedder: Embedder, cache_dir: str):
        self.base = base_embedder
        self.cache = Cache(cache_dir)

    def embed(self, texts):
        cached = self.cache.get(texts)
        if cached:
            return cached
        results = self.base.embed(texts)
        self.cache.set(texts, results)
        return results
```

### Pattern 2: Composite Components
**Example**: `HybridRetriever` combines `DenseRetriever` + `BM25Retriever`
**Purpose**: Combine multiple strategies

```python
class HybridRetriever(Retriever):
    def __init__(self, dense_retriever, bm25_retriever, weight=0.5):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.weight = weight

    def retrieve(self, query, top_k):
        dense_results = self.dense.retrieve(query, top_k)
        bm25_results = self.bm25.retrieve(query, top_k)
        return self._merge_results(dense_results, bm25_results)
```

## Testing Components

Each component should be testable in isolation:

```python
# Test chunker
def test_chunker():
    chunker = FixedSizeChunker({'chunk_size': 100})
    docs = [Document(content="..." * 500)]
    chunks = chunker.chunk(docs)
    assert all(len(c.content) <= 100 for c in chunks)

# Test retriever (with mock embedder)
def test_retriever():
    embedder = MockEmbedder()
    retriever = DenseRetriever({'embedder': embedder})
    # ... test retrieval logic
```

## COSMOS Integration

Components can be wrapped with `COSMOSComponent` to add metrics collection:

```python
from autorag.cosmos.component_wrapper import COSMOSComponent
from autorag.cosmos.metrics import ComponentMetrics

# Wrap component
chunker = FixedSizeChunker({'chunk_size': 256})
metrics = ComponentMetrics()
cosmos_chunker = COSMOSComponent(chunker, 'chunker', metrics)

# Use with metrics
chunks, metrics_dict = cosmos_chunker.process_with_metrics(documents)
```

**See**: `autorag/cosmos/CLAUDE.md` for COSMOS framework details

## Component Directory Structure

```
autorag/components/
├── base.py                    # Abstract base classes + data structures
├── chunkers/
│   ├── fixed_size.py          # FixedSizeChunker
│   ├── semantic.py            # SemanticChunker
│   └── sliding_window.py      # SlidingWindowChunker
├── embedders/
│   ├── openai.py              # OpenAIEmbedder
│   ├── cached.py              # CachedEmbedder (wrapper)
│   └── mock.py                # MockEmbedder (testing)
├── retrievers/
│   ├── dense.py               # DenseRetriever
│   ├── bm25.py                # BM25Retriever
│   └── hybrid.py              # HybridRetriever
├── rerankers/
│   ├── CLAUDE.md              # Reranker architecture docs
│   └── cross_encoder.py       # CrossEncoderReranker
├── generators/
│   ├── openai.py              # OpenAIGenerator
│   └── mock.py                # MockGenerator (testing)
└── vector_stores/
    ├── simple.py              # SimpleVectorStore
    └── faiss.py               # FAISSVectorStore
```

## Key Takeaways

1. **All components** inherit from base classes in `base.py`
2. **Data flows** through well-defined types: Document → Chunk → QueryResult
3. **Reranker is optional** and sits between Retriever and Generator
4. **Configuration-driven**: All components accept config dicts
5. **Testable**: Components work in isolation
6. **Extensible**: Add new implementations by inheriting base classes

---

**Last Updated**: 2025-10-03
**Related Docs**:
- `autorag/cosmos/CLAUDE.md` - COSMOS optimization framework
- `autorag/components/rerankers/CLAUDE.md` - Reranker specifics
