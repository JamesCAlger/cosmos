# LangChain Components Integration with COSMOS

**Date**: 2025-10-01
**Status**: Implementation guide for wrapping LangChain components with COSMOS interface
**Context**: Leveraging LangChain's mature component ecosystem while maintaining COSMOS optimization capabilities

---

## Executive Summary

This guide explains how to use **LangChain components** (text splitters, retrievers, embeddings, LLMs) within the **COSMOS framework** for compositional optimization and architecture search.

### Why This Integration?

**LangChain strengths**:
- 100+ battle-tested components
- Integration with all major vector stores and LLM providers
- Active maintenance and community support
- Mature implementations with edge cases handled

**COSMOS strengths**:
- Component-intrinsic quality metrics
- Parameter and architecture optimization
- Flexible structure discovery
- Component-by-component evaluation

**Integration value**: Access LangChain's ecosystem while maintaining COSMOS's optimization capabilities.

---

## Architecture Overview

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               COSMOS Optimization Layer                      │
│  - CompositionalOptimizer                                    │
│  - Architecture Search (RL/Evolution/Grammar)                │
│  - Parameter Optimization (Bayesian/Random)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│               COSMOS Interface Layer                         │
│  - COSMOSComponent (abstract base)                          │
│  - process_with_metrics()                                   │
│  - get_config_space()                                       │
│  - get_interface_spec()                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
     ┌───────────────┴───────────────┐
     │                               │
┌────▼──────────────┐   ┌───────────▼─────────────┐
│ Custom Components │   │ LangChain Wrapped       │
│ - Full control    │   │ - Ecosystem access      │
│ - Domain-specific │   │ - Mature implementations│
└───────────────────┘   └─────────────────────────┘
```

### Design Principles

1. **Wrapper Pattern**: LangChain components wrapped with COSMOS interface, not modified
2. **Zero Breaking Changes**: Existing COSMOS code works unchanged
3. **Optional Integration**: Use LangChain components where beneficial, custom elsewhere
4. **Configuration Mapping**: Translate between COSMOS and LangChain parameter conventions
5. **Metrics Preservation**: Component-intrinsic metrics maintained via wrapper

---

## Core Wrapper Architecture

### Base Pattern

Every LangChain wrapper follows this pattern:

1. **Inherit from COSMOSComponent**: Get COSMOS interface
2. **Instantiate LangChain component**: Delegate actual work
3. **Implement process()**: Call LangChain component
4. **Implement process_with_metrics()**: Add quality metrics
5. **Implement get_config_space()**: Define optimization parameters
6. **Implement get_interface_spec()**: Define input/output contracts

### Wrapper Template

```python
from autorag.cosmos.component_wrapper import COSMOSComponent
from autorag.cosmos.metrics.component_metrics import ComponentMetrics
from langchain.some_module import SomeLangChainComponent

class COSMOSLangChain<ComponentType>(COSMOSComponent):
    """
    COSMOS wrapper for LangChain <ComponentType>

    Wraps: langchain.some_module.SomeLangChainComponent
    Purpose: <What this component does>
    """

    def __init__(self, config: Dict, metrics_collector: ComponentMetrics):
        """
        Initialize wrapper with LangChain component

        Args:
            config: COSMOS configuration dictionary
            metrics_collector: ComponentMetrics instance for quality computation
        """
        super().__init__(config)
        self.metrics = metrics_collector

        # Map COSMOS config to LangChain config
        lc_config = self._map_config_to_langchain(config)

        # Instantiate LangChain component
        self.lc_component = SomeLangChainComponent(**lc_config)

    def _map_config_to_langchain(self, cosmos_config: Dict) -> Dict:
        """
        Translate COSMOS parameter names to LangChain parameter names

        Example:
            COSMOS: {'chunk_size': 256}
            LangChain: {'chunk_size': 256}  (sometimes they match)

            COSMOS: {'overlap': 50}
            LangChain: {'chunk_overlap': 50}  (sometimes different)
        """
        return {
            # Map parameter names
            'langchain_param': cosmos_config.get('cosmos_param', default_value),
            # ...
        }

    def process(self, input_data):
        """
        Core processing logic - delegates to LangChain component

        This is a pure function: input → output
        No metrics, no side effects
        """
        return self.lc_component.some_method(input_data)

    def process_with_metrics(self, input_data, **kwargs):
        """
        Process with quality metrics (COSMOS requirement)

        Returns:
            (output, metrics) tuple where metrics include:
            - Quality metrics (accuracy, relevance, coherence)
            - Efficiency metrics (latency, throughput)
            - Cost metrics (API calls, tokens)
        """
        import time

        # Measure execution time
        start_time = time.time()

        # Execute LangChain component
        output = self.process(input_data)

        latency = time.time() - start_time

        # Compute component-specific quality metrics
        metrics = self._compute_quality_metrics(
            input_data, output, latency, **kwargs
        )

        # Store in history
        self.metrics_history.append(metrics)

        return output, metrics

    def _compute_quality_metrics(self, input_data, output, latency, **kwargs):
        """
        Compute quality metrics using ComponentMetrics

        This is component-type specific:
        - Chunker: chunk size distribution, coherence
        - Retriever: relevance, precision@k
        - Generator: answer quality, context utilization
        """
        # Delegate to ComponentMetrics for computation
        return self.metrics.compute_<component_type>_metrics(
            # appropriate arguments
        )

    def get_config_space(self) -> Dict[str, Any]:
        """
        Define optimization parameter space

        Returns:
            Dictionary mapping parameter names to:
            - Categorical: list of choices
            - Continuous/Integer: tuple (low, high)

        Example:
            {
                'chunk_size': (128, 512),           # Integer range
                'overlap': (0, 100),                # Integer range
                'separators': [                      # Categorical
                    ["\n\n", "\n", " "],
                    ["\n\n", "\n", ".", " "]
                ]
            }
        """
        return {
            # Define parameters that COSMOS can optimize
        }

    def get_interface_spec(self):
        """
        Define component interface contract

        Returns:
            InterfaceContract specifying:
            - input_type: Expected input data type
            - output_type: Produced output data type
            - required_metrics: Metrics this component reports
        """
        from autorag.cosmos.component_wrapper import InterfaceContract

        return InterfaceContract(
            input_type=<InputType>,
            output_type=<OutputType>,
            required_metrics=[<list of metric names>]
        )
```

---

## Implementation Guide by Component Type

### 1. Text Splitters (Chunkers)

#### Wrapped Components

- **RecursiveCharacterTextSplitter**: Splits on recursive separators (\n\n, \n, space)
- **TokenTextSplitter**: Splits based on token count
- **MarkdownHeaderTextSplitter**: Splits markdown by headers
- **CharacterTextSplitter**: Simple character-based splitting

#### Implementation

**File**: `autorag/cosmos/langchain_wrappers/chunkers.py`

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from autorag.cosmos.component_wrapper import COSMOSComponent, InterfaceContract
from typing import List, Dict, Any

class COSMOSLangChainChunker(COSMOSComponent):
    """
    COSMOS wrapper for LangChain text splitters

    Supports:
    - recursive: RecursiveCharacterTextSplitter
    - token: TokenTextSplitter
    - markdown: MarkdownHeaderTextSplitter
    """

    SPLITTER_TYPES = {
        'recursive': RecursiveCharacterTextSplitter,
        'token': TokenTextSplitter,
        'markdown': MarkdownHeaderTextSplitter
    }

    def __init__(self, config: Dict, metrics_collector):
        super().__init__(config)
        self.metrics = metrics_collector

        splitter_type = config.get('splitter_type', 'recursive')
        self.splitter = self._create_splitter(splitter_type, config)

    def _create_splitter(self, splitter_type: str, config: Dict):
        """Create appropriate LangChain splitter"""
        splitter_class = self.SPLITTER_TYPES[splitter_type]

        if splitter_type == 'recursive':
            return splitter_class(
                chunk_size=config.get('chunk_size', 256),
                chunk_overlap=config.get('overlap', 50),
                separators=config.get('separators', ["\n\n", "\n", " ", ""])
            )
        elif splitter_type == 'token':
            return splitter_class(
                chunk_size=config.get('chunk_size', 256),
                chunk_overlap=config.get('overlap', 50)
            )
        elif splitter_type == 'markdown':
            headers_to_split = config.get('headers_to_split', [
                ("#", "Header 1"),
                ("##", "Header 2")
            ])
            return splitter_class(headers_to_split_on=headers_to_split)

    def process(self, documents: List) -> List:
        """Split documents into chunks using LangChain splitter"""
        return self.splitter.split_documents(documents)

    def process_with_metrics(self, documents: List) -> tuple:
        """Split with quality metrics"""
        import time
        start = time.time()

        chunks = self.process(documents)
        latency = time.time() - start

        # Compute chunking quality metrics
        metrics = self.metrics.compute_chunking_metrics(
            chunks=chunks,
            latency=latency,
            compute_coherence=True
        )

        return chunks, metrics

    def get_config_space(self) -> Dict:
        """Parameter space for optimization"""
        return {
            'splitter_type': ['recursive', 'token', 'markdown'],
            'chunk_size': (128, 512),
            'overlap': (0, 100),
            'separators': [
                ["\n\n", "\n", " ", ""],
                ["\n\n", "\n", ".", "!", "?", " ", ""]
            ]
        }

    def get_interface_spec(self):
        """Interface contract"""
        return InterfaceContract(
            input_type=List,  # List of Documents
            output_type=List,  # List of Chunks
            required_metrics=['num_chunks', 'avg_chunk_size', 'semantic_coherence']
        )
```

#### Key Configuration Mappings

| COSMOS Parameter | LangChain Parameter | Type | Notes |
|------------------|---------------------|------|-------|
| `chunk_size` | `chunk_size` | int | Same name |
| `overlap` | `chunk_overlap` | int | Different name |
| `separators` | `separators` | List[str] | Same name |
| `splitter_type` | (class selection) | str | Meta-parameter |

#### Optimization Considerations

**Typical search space**:
- `chunk_size`: 128-512 (sweet spot usually 200-400)
- `overlap`: 0-100 (10-50 common)
- `splitter_type`: recursive works for most cases

**Quality metrics**:
- `avg_chunk_size`: Target ~300 words
- `semantic_coherence`: Higher is better (>0.7 good)
- `size_variance`: Lower is better (more consistent chunks)

---

### 2. Retrievers

#### Wrapped Components

- **VectorStoreRetriever**: Standard similarity search
- **ContextualCompressionRetriever**: Retrieves then compresses context
- **EnsembleRetriever**: Combines multiple retrievers (e.g., BM25 + dense)
- **MultiQueryRetriever**: Generates multiple query variations

#### Implementation

**File**: `autorag/cosmos/langchain_wrappers/retrievers.py`

```python
from langchain.retrievers import (
    VectorStoreRetriever,
    ContextualCompressionRetriever,
    EnsembleRetriever
)
from langchain.retrievers.document_compressors import CohereRerank
from autorag.cosmos.component_wrapper import COSMOSComponent, InterfaceContract

class COSMOSLangChainRetriever(COSMOSComponent):
    """
    COSMOS wrapper for LangChain retrievers

    Supports:
    - vector_store: Standard vector similarity search
    - contextual_compression: Retrieval with reranking/compression
    - ensemble: Combines multiple retrieval methods
    """

    def __init__(self, config: Dict, vector_store, metrics_collector):
        super().__init__(config)
        self.metrics = metrics_collector
        self.vector_store = vector_store

        retriever_type = config.get('retriever_type', 'vector_store')
        self.retriever = self._create_retriever(retriever_type, config)

    def _create_retriever(self, retriever_type: str, config: Dict):
        """Create appropriate LangChain retriever"""
        if retriever_type == 'vector_store':
            return VectorStoreRetriever(
                vectorstore=self.vector_store,
                search_type=config.get('search_type', 'similarity'),
                search_kwargs={
                    'k': config.get('top_k', 5),
                    'fetch_k': config.get('fetch_k', 20)  # For MMR
                }
            )

        elif retriever_type == 'contextual_compression':
            # Base retriever
            base = VectorStoreRetriever(
                vectorstore=self.vector_store,
                search_kwargs={'k': config.get('top_k', 10)}
            )

            # Compressor (reranker)
            compressor = CohereRerank(
                top_n=config.get('rerank_top_n', 5)
            )

            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base
            )

        elif retriever_type == 'ensemble':
            # Requires multiple retriever instances
            # Implementation depends on specific setup
            pass

    def process(self, query: str) -> List:
        """Retrieve relevant documents"""
        return self.retriever.get_relevant_documents(query)

    def process_with_metrics(self, query: str, ground_truth_doc_ids=None) -> tuple:
        """Retrieve with quality metrics"""
        import time
        start = time.time()

        results = self.process(query)
        latency = time.time() - start

        # Compute retrieval quality metrics
        metrics = self.metrics.compute_retrieval_metrics(
            query=query,
            results=results,
            latency=latency,
            ground_truth=ground_truth_doc_ids
        )

        return results, metrics

    def get_config_space(self) -> Dict:
        """Parameter space for optimization"""
        return {
            'retriever_type': ['vector_store', 'contextual_compression', 'ensemble'],
            'top_k': [3, 5, 10, 20],
            'search_type': ['similarity', 'mmr', 'similarity_score_threshold'],
            'fetch_k': [20, 50, 100],  # For MMR
            'rerank_top_n': [3, 5, 10]  # For compression
        }

    def get_interface_spec(self):
        """Interface contract"""
        return InterfaceContract(
            input_type=str,  # Query string
            output_type=List,  # List of Documents
            required_metrics=['num_results', 'avg_relevance', 'precision']
        )
```

#### Key Configuration Mappings

| COSMOS Parameter | LangChain Parameter | Type | Notes |
|------------------|---------------------|------|-------|
| `top_k` | `search_kwargs['k']` | int | Nested in dict |
| `search_type` | `search_type` | str | Same name |
| `fetch_k` | `search_kwargs['fetch_k']` | int | For MMR only |
| `retriever_type` | (class selection) | str | Meta-parameter |
| `rerank_top_n` | `top_n` | int | Compressor param |

#### Optimization Considerations

**Typical search space**:
- `top_k`: 5-20 (depends on downstream; reranker needs more, generator needs less)
- `search_type`: 'similarity' usually best, 'mmr' for diversity
- `retriever_type`: Try ensemble for best results (but slower)

**Quality metrics**:
- `avg_relevance`: Target >0.6 (semantic similarity to query)
- `precision`: If ground truth available, target >0.8
- `latency`: Vector search fast (<100ms), reranking slow (200-500ms)

---

### 3. Embeddings

#### Wrapped Components

- **OpenAIEmbeddings**: OpenAI text-embedding-ada-002 or text-embedding-3
- **HuggingFaceEmbeddings**: Local models (all-MiniLM-L6-v2, etc.)
- **CohereEmbeddings**: Cohere embed-english-v3.0
- **BedrockEmbeddings**: AWS Bedrock embedding models

#### Implementation

**File**: `autorag/cosmos/langchain_wrappers/embeddings.py`

```python
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    CohereEmbeddings
)
from autorag.cosmos.component_wrapper import COSMOSComponent

class COSMOSLangChainEmbeddings(COSMOSComponent):
    """
    COSMOS wrapper for LangChain embeddings

    Note: Embeddings are usually NOT optimized directly.
    They're infrastructure components used by retrievers.

    This wrapper mainly provides consistent interface for caching.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        embedding_type = config.get('embedding_type', 'openai')
        self.embeddings = self._create_embeddings(embedding_type, config)

    def _create_embeddings(self, embedding_type: str, config: Dict):
        """Create appropriate LangChain embeddings"""
        if embedding_type == 'openai':
            return OpenAIEmbeddings(
                model=config.get('model', 'text-embedding-ada-002'),
                openai_api_key=config.get('api_key')
            )

        elif embedding_type == 'huggingface':
            return HuggingFaceEmbeddings(
                model_name=config.get('model', 'all-MiniLM-L6-v2'),
                model_kwargs={'device': config.get('device', 'cpu')}
            )

        elif embedding_type == 'cohere':
            return CohereEmbeddings(
                model=config.get('model', 'embed-english-v3.0'),
                cohere_api_key=config.get('api_key')
            )

    def embed_query(self, query: str) -> List[float]:
        """Embed single query"""
        return self.embeddings.embed_query(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.embeddings.embed_documents(documents)

    def get_config_space(self) -> Dict:
        """
        Usually not optimized directly

        Embedding choice is often a fixed infrastructure decision
        """
        return {
            'embedding_type': ['openai', 'huggingface', 'cohere'],
            'model': {
                'openai': ['text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'],
                'huggingface': ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                'cohere': ['embed-english-v3.0', 'embed-multilingual-v3.0']
            }
        }
```

#### Optimization Considerations

**Usually fixed, not optimized**:
- Embeddings are infrastructure-level choice
- Changing embeddings requires re-indexing entire corpus
- COSMOS typically optimizes retrieval parameters, not embedding model

**When to optimize**:
- Initial setup: compare OpenAI vs HuggingFace vs Cohere
- Cost/quality tradeoff: larger models vs smaller models
- Domain-specific: try domain-adapted embeddings

**If optimizing**:
- Treat as architecture search decision (which embedding model?)
- Not parameter optimization (embeddings have no tunable params)

---

### 4. Language Models (LLMs)

#### Wrapped Components

- **ChatOpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **ChatAnthropic**: Claude-3 models
- **ChatCohere**: Cohere Command models
- **HuggingFaceHub**: Open source models (Llama, Mistral, etc.)

#### Implementation

**File**: `autorag/cosmos/langchain_wrappers/llms.py`

```python
from langchain.chat_models import (
    ChatOpenAI,
    ChatAnthropic,
    ChatCohere
)
from langchain.prompts import ChatPromptTemplate
from autorag.cosmos.component_wrapper import COSMOSComponent, InterfaceContract

class COSMOSLangChainGenerator(COSMOSComponent):
    """
    COSMOS wrapper for LangChain LLMs

    Handles:
    - Prompt templating
    - LLM invocation
    - Response parsing
    """

    def __init__(self, config: Dict, metrics_collector):
        super().__init__(config)
        self.metrics = metrics_collector

        llm_type = config.get('llm_type', 'openai')
        self.llm = self._create_llm(llm_type, config)
        self.prompt = self._create_prompt(config)

    def _create_llm(self, llm_type: str, config: Dict):
        """Create appropriate LangChain LLM"""
        if llm_type == 'openai':
            return ChatOpenAI(
                model=config.get('model', 'gpt-3.5-turbo'),
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 512),
                top_p=config.get('top_p', 1.0)
            )

        elif llm_type == 'anthropic':
            return ChatAnthropic(
                model=config.get('model', 'claude-3-sonnet-20240229'),
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 512)
            )

        elif llm_type == 'cohere':
            return ChatCohere(
                model=config.get('model', 'command'),
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 512)
            )

    def _create_prompt(self, config: Dict):
        """Create prompt template"""
        template = config.get('prompt_template',
            "Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return ChatPromptTemplate.from_template(template)

    def process(self, query: str, context: str) -> str:
        """Generate answer using LangChain LLM"""
        chain = self.prompt | self.llm
        response = chain.invoke({'query': query, 'context': context})
        return response.content

    def process_with_metrics(self, query: str, context: str,
                           ground_truth_answer: str = None) -> tuple:
        """Generate with quality metrics"""
        import time
        start = time.time()

        answer = self.process(query, context)
        latency = time.time() - start

        # Compute generation quality metrics
        metrics = self.metrics.compute_generation_metrics(
            query=query,
            answer=answer,
            context=context,
            latency=latency,
            ground_truth_answer=ground_truth_answer
        )

        return answer, metrics

    def get_config_space(self) -> Dict:
        """Parameter space for optimization"""
        return {
            'llm_type': ['openai', 'anthropic', 'cohere'],
            'model': {
                'openai': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
                'anthropic': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                'cohere': ['command', 'command-light']
            },
            'temperature': (0.0, 1.0),
            'max_tokens': [256, 512, 1024],
            'top_p': (0.8, 1.0),
            'prompt_template': [
                "Context: {context}\n\nQuestion: {query}\n\nAnswer:",
                "Given the context:\n{context}\n\nAnswer this question: {query}",
                "Context information:\n{context}\n\nQuery: {query}\n\nProvide a concise answer:"
            ]
        }

    def get_interface_spec(self):
        """Interface contract"""
        return InterfaceContract(
            input_type=tuple,  # (query, context)
            output_type=str,   # Generated answer
            required_metrics=['answer_length', 'answer_relevance', 'context_utilization']
        )
```

#### Key Configuration Mappings

| COSMOS Parameter | LangChain Parameter | Type | Notes |
|------------------|---------------------|------|-------|
| `temperature` | `temperature` | float | Same name |
| `max_tokens` | `max_tokens` | int | Same name |
| `model` | `model` | str | Same name |
| `llm_type` | (class selection) | str | Meta-parameter |
| `prompt_template` | (template construction) | str | Custom handling |

#### Optimization Considerations

**Typical search space**:
- `temperature`: 0.3-0.9 (lower for factual, higher for creative)
- `max_tokens`: 256-1024 (balance completeness vs cost)
- `model`: Usually fixed due to cost/latency, but can compare
- `prompt_template`: Can significantly impact quality

**Quality metrics**:
- `answer_relevance`: Semantic similarity to query (target >0.7)
- `context_utilization`: How much context is used (0.3-0.7 good)
- `accuracy`: If ground truth available (target >0.8)
- `latency`: GPT-3.5 fast (~1s), GPT-4 slow (~3-5s)

---

## Integration with COSMOS Optimization

### Component Registration

**File**: `autorag/cosmos/langchain_wrappers/__init__.py`

```python
from .chunkers import COSMOSLangChainChunker
from .retrievers import COSMOSLangChainRetriever
from .llms import COSMOSLangChainGenerator
from .embeddings import COSMOSLangChainEmbeddings

__all__ = [
    'COSMOSLangChainChunker',
    'COSMOSLangChainRetriever',
    'COSMOSLangChainGenerator',
    'COSMOSLangChainEmbeddings'
]

# Component registry for easy lookup
LANGCHAIN_COMPONENTS = {
    'chunker': COSMOSLangChainChunker,
    'retriever': COSMOSLangChainRetriever,
    'generator': COSMOSLangChainGenerator,
    'embeddings': COSMOSLangChainEmbeddings
}
```

### Using in COSMOS Optimization

```python
from autorag.cosmos.langchain_wrappers import (
    COSMOSLangChainChunker,
    COSMOSLangChainRetriever,
    COSMOSLangChainGenerator
)
from autorag.cosmos.optimization.compositional_optimizer import CompositionalOptimizer
from autorag.cosmos.metrics.component_metrics import ComponentMetrics

# Setup
metrics = ComponentMetrics()

# Create LangChain-wrapped components
chunker = COSMOSLangChainChunker(
    config={'splitter_type': 'recursive', 'chunk_size': 256},
    metrics_collector=metrics
)

retriever = COSMOSLangChainRetriever(
    config={'retriever_type': 'vector_store', 'top_k': 5},
    vector_store=vector_store,
    metrics_collector=metrics
)

generator = COSMOSLangChainGenerator(
    config={'llm_type': 'openai', 'temperature': 0.7},
    metrics_collector=metrics
)

# Optimize with COSMOS
optimizer = CompositionalOptimizer(
    components=[chunker, retriever, generator],
    strategy='bayesian'
)

results = optimizer.optimize(test_data, budget=30)
```

**Key Points**:
- LangChain components are drop-in replacements for custom components
- COSMOS optimization works identically
- Can mix custom and LangChain components in same pipeline

---

## Architecture Search with LangChain Components

### Component Library Definition

```python
from autorag.cosmos.langchain_wrappers import LANGCHAIN_COMPONENTS
from autorag.cosmos.optimization.architecture_search import ComponentLibrary

# Define available components
library = ComponentLibrary()

# Add LangChain chunkers
library.register('chunker_recursive', {
    'class': COSMOSLangChainChunker,
    'default_config': {'splitter_type': 'recursive'},
    'config_space': {
        'chunk_size': (128, 512),
        'overlap': (0, 100)
    }
})

library.register('chunker_token', {
    'class': COSMOSLangChainChunker,
    'default_config': {'splitter_type': 'token'},
    'config_space': {
        'chunk_size': (128, 512),
        'overlap': (0, 100)
    }
})

# Add LangChain retrievers
library.register('retriever_vector', {
    'class': COSMOSLangChainRetriever,
    'default_config': {'retriever_type': 'vector_store'},
    'config_space': {
        'top_k': [3, 5, 10, 20],
        'search_type': ['similarity', 'mmr']
    }
})

library.register('retriever_compressed', {
    'class': COSMOSLangChainRetriever,
    'default_config': {'retriever_type': 'contextual_compression'},
    'config_space': {
        'top_k': [10, 20, 30],
        'rerank_top_n': [3, 5, 10]
    }
})

# Add LangChain generators
library.register('generator_gpt35', {
    'class': COSMOSLangChainGenerator,
    'default_config': {'llm_type': 'openai', 'model': 'gpt-3.5-turbo'},
    'config_space': {
        'temperature': (0.3, 0.9),
        'max_tokens': [256, 512, 1024]
    }
})

library.register('generator_gpt4', {
    'class': COSMOSLangChainGenerator,
    'default_config': {'llm_type': 'openai', 'model': 'gpt-4-turbo'},
    'config_space': {
        'temperature': (0.3, 0.9),
        'max_tokens': [256, 512, 1024]
    }
})
```

### Running Architecture Search

```python
from autorag.cosmos.optimization.architecture_search import TemplateArchitectureSearch

# Define templates using LangChain components
templates = [
    {
        'name': 'simple_rag',
        'components': ['chunker_recursive', 'retriever_vector', 'generator_gpt35']
    },
    {
        'name': 'compressed_rag',
        'components': ['chunker_token', 'retriever_compressed', 'generator_gpt35']
    },
    {
        'name': 'premium_rag',
        'components': ['chunker_recursive', 'retriever_compressed', 'generator_gpt4']
    }
]

# Run template-based architecture search
search = TemplateArchitectureSearch(
    component_library=library,
    templates=templates
)

best_architecture = search.search(
    test_data=test_data,
    budget_per_template=20
)

print(f"Best architecture: {best_architecture.name}")
print(f"Components: {best_architecture.components}")
print(f"Performance: {best_architecture.score:.3f}")
```

**Result**: Discovers which combination of LangChain components works best for your specific use case.

---

## Advanced Integration Patterns

### 1. Mixing Custom and LangChain Components

```python
from autorag.components.chunkers.custom_hierarchical import HierarchicalChunker
from autorag.cosmos.langchain_wrappers import COSMOSLangChainRetriever, COSMOSLangChainGenerator

# Mix custom and LangChain components
components = [
    HierarchicalChunker(config={'levels': 2}),  # Custom
    COSMOSLangChainRetriever(config={...}),     # LangChain
    COSMOSLangChainGenerator(config={...})      # LangChain
]

optimizer = CompositionalOptimizer(components=components, strategy='bayesian')
results = optimizer.optimize(test_data, budget=30)
```

**Use case**: Custom domain-specific logic where needed, mature LangChain components elsewhere.

### 2. Caching Wrapper

```python
from autorag.optimization.cache_manager import EmbeddingCacheManager
from autorag.cosmos.langchain_wrappers import COSMOSLangChainEmbeddings

class CachedLangChainEmbeddings:
    """Add caching to LangChain embeddings"""

    def __init__(self, config, cache_dir='.embedding_cache'):
        self.embeddings = COSMOSLangChainEmbeddings(config)
        self.cache = EmbeddingCacheManager(cache_dir)

    def embed_query(self, query: str) -> List[float]:
        cache_key = self.cache.get_cache_key('query', query)

        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Compute and cache
        embedding = self.embeddings.embed_query(query)
        self.cache.put(cache_key, embedding)
        return embedding
```

**Use case**: Reduce API costs during optimization by caching embeddings.

### 3. Multi-Provider Fallback

```python
class MultiProviderGenerator(COSMOSComponent):
    """Try multiple LLM providers with fallback"""

    def __init__(self, config, metrics_collector):
        super().__init__(config)
        self.metrics = metrics_collector

        # Primary provider
        self.primary = COSMOSLangChainGenerator(
            config={'llm_type': 'openai', 'model': 'gpt-4'},
            metrics_collector=metrics_collector
        )

        # Fallback provider
        self.fallback = COSMOSLangChainGenerator(
            config={'llm_type': 'anthropic', 'model': 'claude-3-sonnet'},
            metrics_collector=metrics_collector
        )

    def process(self, query: str, context: str) -> str:
        try:
            return self.primary.process(query, context)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}, using fallback")
            return self.fallback.process(query, context)
```

**Use case**: Reliability - automatic fallback if primary provider has issues.

---

## Best Practices

### 1. Configuration Management

**Separate COSMOS and LangChain configs**:
```python
class COSMOSLangChainWrapper:
    def __init__(self, cosmos_config: Dict, metrics_collector):
        # COSMOS-level config (optimization parameters)
        self.cosmos_config = cosmos_config

        # Translate to LangChain config
        lc_config = self._cosmos_to_langchain(cosmos_config)

        # Instantiate LangChain component
        self.lc_component = SomeLangChainClass(**lc_config)
```

**Why**: Clear separation between what COSMOS optimizes vs what LangChain needs.

### 2. Error Handling

**Wrap LangChain calls with error handling**:
```python
def process_with_metrics(self, input_data):
    try:
        output = self.lc_component.some_method(input_data)
        metrics = self._compute_metrics(input_data, output)
        return output, metrics
    except Exception as e:
        logger.error(f"LangChain component error: {e}")
        # Return safe defaults or reraise
        return None, {'error': str(e)}
```

**Why**: LangChain components may fail (API limits, network issues). COSMOS optimization should handle gracefully.

### 3. Metrics Validation

**Validate metrics before returning**:
```python
def _compute_quality_metrics(self, input_data, output, latency):
    metrics = self.metrics.compute_retrieval_metrics(...)

    # Validate
    required_metrics = ['num_results', 'avg_relevance', 'precision']
    for key in required_metrics:
        if key not in metrics:
            logger.warning(f"Missing metric: {key}")
            metrics[key] = 0.0  # Safe default

    return metrics
```

**Why**: Ensure metrics are always complete for optimization algorithms.

### 4. Cost Tracking

**Track API costs in metrics**:
```python
def process_with_metrics(self, query: str, context: str):
    # Track tokens for cost estimation
    prompt_tokens = count_tokens(query + context)

    answer = self.llm.invoke(...)

    completion_tokens = count_tokens(answer)

    cost = estimate_cost(
        model=self.config['model'],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )

    metrics = {
        'answer_quality': ...,
        'cost': cost,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens
    }

    return answer, metrics
```

**Why**: Multi-objective optimization can balance quality vs cost.

### 5. Version Pinning

**Pin LangChain versions**:
```
# requirements.txt
langchain==0.1.0
langchain-openai==0.0.2
langchain-anthropic==0.0.1
```

**Why**: LangChain API changes frequently. Pin versions to avoid breaking changes during optimization runs.

---

## Testing Strategy

### Unit Tests for Wrappers

**File**: `tests/cosmos/langchain/test_chunker_wrapper.py`

```python
import pytest
from autorag.cosmos.langchain_wrappers import COSMOSLangChainChunker
from autorag.cosmos.metrics.component_metrics import ComponentMetrics

def test_langchain_chunker_basic():
    """Test basic chunking functionality"""
    config = {
        'splitter_type': 'recursive',
        'chunk_size': 256,
        'overlap': 50
    }
    metrics = ComponentMetrics()
    chunker = COSMOSLangChainChunker(config, metrics)

    # Test data
    documents = [create_test_document("Test content " * 100)]

    # Process
    chunks, metrics = chunker.process_with_metrics(documents)

    # Assertions
    assert len(chunks) > 0
    assert 'num_chunks' in metrics
    assert 'avg_chunk_size' in metrics
    assert metrics['num_chunks'] == len(chunks)

def test_config_space():
    """Test config space definition"""
    config = {'splitter_type': 'recursive'}
    chunker = COSMOSLangChainChunker(config, ComponentMetrics())

    config_space = chunker.get_config_space()

    assert 'chunk_size' in config_space
    assert 'overlap' in config_space
    assert isinstance(config_space['chunk_size'], tuple)

def test_interface_spec():
    """Test interface contract"""
    config = {'splitter_type': 'recursive'}
    chunker = COSMOSLangChainChunker(config, ComponentMetrics())

    spec = chunker.get_interface_spec()

    assert spec.input_type == List
    assert spec.output_type == List
    assert 'num_chunks' in spec.required_metrics
```

### Integration Tests

**File**: `tests/cosmos/langchain/test_full_pipeline.py`

```python
def test_langchain_full_pipeline():
    """Test full RAG pipeline with LangChain components"""
    metrics = ComponentMetrics()

    # Build pipeline
    chunker = COSMOSLangChainChunker(
        config={'splitter_type': 'recursive', 'chunk_size': 256},
        metrics_collector=metrics
    )

    retriever = COSMOSLangChainRetriever(
        config={'retriever_type': 'vector_store', 'top_k': 5},
        vector_store=setup_test_vector_store(),
        metrics_collector=metrics
    )

    generator = COSMOSLangChainGenerator(
        config={'llm_type': 'openai', 'model': 'gpt-3.5-turbo'},
        metrics_collector=metrics
    )

    # Test end-to-end
    documents = load_test_documents()
    query = "What is the capital of France?"

    # Chunk
    chunks, chunk_metrics = chunker.process_with_metrics(documents)
    assert len(chunks) > 0

    # Index (simplified)
    retriever.index(chunks)

    # Retrieve
    results, retrieval_metrics = retriever.process_with_metrics(query)
    assert len(results) > 0

    # Generate
    context = format_context(results)
    answer, generation_metrics = generator.process_with_metrics(query, context)
    assert len(answer) > 0

    # Verify metrics collected at each stage
    assert chunk_metrics['num_chunks'] > 0
    assert retrieval_metrics['num_results'] > 0
    assert generation_metrics['answer_length'] > 0
```

### Optimization Tests

**File**: `tests/cosmos/langchain/test_optimization.py`

```python
def test_optimization_with_langchain_components():
    """Test COSMOS optimization with LangChain components"""
    from autorag.cosmos.optimization.compositional_optimizer import CompositionalOptimizer

    # Setup
    metrics = ComponentMetrics()
    test_data = load_small_test_set()

    # Components
    components = [
        COSMOSLangChainChunker(config={...}, metrics_collector=metrics),
        COSMOSLangChainRetriever(config={...}, vector_store=..., metrics_collector=metrics),
        COSMOSLangChainGenerator(config={...}, metrics_collector=metrics)
    ]

    # Optimize
    optimizer = CompositionalOptimizer(
        components=components,
        strategy='random'  # Use random for faster testing
    )

    results = optimizer.optimize(test_data, budget=9)  # 3 evals per component

    # Assertions
    assert 'chunker' in results
    assert 'retriever' in results
    assert 'generator' in results
    assert results['chunker'].best_score > 0
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ImportError: cannot import name 'RecursiveCharacterTextSplitter'`

**Solution**:
```bash
# Install specific LangChain packages
pip install langchain==0.1.0
pip install langchain-openai==0.0.2
pip install langchain-community==0.0.10
```

LangChain split into multiple packages. Ensure all needed packages installed.

#### 2. API Key Not Found

**Problem**: `openai.error.AuthenticationError: No API key provided`

**Solution**:
```python
# Explicit API key passing
config = {
    'llm_type': 'openai',
    'api_key': os.getenv('OPENAI_API_KEY')
}
generator = COSMOSLangChainGenerator(config, metrics)
```

Or ensure `.env` file loaded:
```python
from dotenv import load_dotenv
load_dotenv()
```

#### 3. Metrics Not Computed

**Problem**: Metrics dictionary empty or missing keys

**Solution**:
```python
# Ensure ComponentMetrics initialized with correct model
metrics = ComponentMetrics(
    semantic_model='all-MiniLM-L6-v2'  # Specify model
)
```

Check that semantic model downloaded:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Downloads if needed
```

#### 4. Rate Limiting

**Problem**: `openai.error.RateLimitError` during optimization

**Solution**:
```python
# Add delays between evaluations
from time import sleep

class RateLimitedGenerator(COSMOSLangChainGenerator):
    def process(self, query, context):
        sleep(0.2)  # 200ms delay
        return super().process(query, context)
```

Or use exponential backoff:
```python
from tenacity import retry, wait_exponential

class RetryableGenerator(COSMOSLangChainGenerator):
    @retry(wait=wait_exponential(min=1, max=10))
    def process(self, query, context):
        return super().process(query, context)
```

#### 5. Vector Store Not Indexed

**Problem**: `Retriever returns empty results`

**Solution**:
```python
# Ensure vector store is indexed before optimization
retriever = COSMOSLangChainRetriever(config={...}, vector_store=vector_store, ...)

# Index documents first
documents = load_documents()
chunks, _ = chunker.process_with_metrics(documents)

# Add to vector store
texts = [chunk.page_content for chunk in chunks]
vector_store.add_texts(texts)

# Now retrieval works
results, _ = retriever.process_with_metrics("query")
```

---

## Migration Guide

### From Custom Components to LangChain

**Step 1**: Identify components to migrate
- Text splitters → LangChain has robust implementations
- Retrievers → If using vector stores LangChain supports
- Generators → If using OpenAI/Anthropic/Cohere

**Step 2**: Create wrappers
```python
# Before (custom)
from autorag.components.chunkers.fixed import FixedChunker
chunker = FixedChunker(config={'chunk_size': 256})

# After (LangChain)
from autorag.cosmos.langchain_wrappers import COSMOSLangChainChunker
chunker = COSMOSLangChainChunker(
    config={'splitter_type': 'recursive', 'chunk_size': 256},
    metrics_collector=metrics
)
```

**Step 3**: Update optimization scripts
```python
# No changes needed! COSMOS interface is identical
optimizer = CompositionalOptimizer(
    components=[chunker, retriever, generator],  # Wrapped LangChain components
    strategy='bayesian'
)
results = optimizer.optimize(test_data, budget=30)
```

**Step 4**: Test thoroughly
- Run unit tests for wrappers
- Run integration tests for full pipeline
- Compare optimization results (custom vs LangChain)

**Step 5**: Deploy
- Update production code to use LangChain components
- Monitor performance (should be similar or better)
- Keep custom components as fallback if needed

---

## Performance Considerations

### LangChain vs Custom Component Performance

| Aspect | Custom | LangChain | Notes |
|--------|--------|-----------|-------|
| **Latency** | Optimized | Comparable | LangChain adds minimal overhead |
| **Memory** | Minimal | Slightly higher | LangChain loads more dependencies |
| **API Costs** | Same | Same | Both use same LLM/embedding APIs |
| **Flexibility** | High | Medium | Custom gives full control |
| **Maintenance** | Manual | Automatic | LangChain updates handled by community |

### Optimization Considerations

**LangChain components suitable for**:
- Standard text splitting (recursive, token-based)
- Vector similarity search
- Standard LLM generation
- Well-supported integrations (OpenAI, Pinecone, etc.)

**Custom components better for**:
- Domain-specific chunking logic
- Custom retrieval algorithms
- Proprietary scoring functions
- Specialized preprocessing

**Recommendation**: Default to LangChain, use custom when truly needed.

---

## Conclusion

### Key Takeaways

1. **LangChain components can be wrapped** with COSMOS interface for optimization
2. **No changes to COSMOS optimization** - wrappers are drop-in replacements
3. **Best of both worlds** - LangChain's ecosystem + COSMOS's optimization
4. **Flexible architecture** - mix custom and LangChain components as needed
5. **Production path** - optionally export to LangGraph after optimization

### Next Steps

**Immediate (Week 1-2)**:
1. Implement core wrappers (chunker, retriever, generator)
2. Test with existing COSMOS optimization scripts
3. Compare results with custom components

**Near-term (Week 3-4)**:
1. Add wrappers for specialized components (rerankers, compressors)
2. Implement caching for API cost reduction
3. Create comprehensive test suite

**Long-term (Month 2+)**:
1. Architecture search with LangChain component library
2. Multi-provider fallback and error handling
3. LangGraph export for production deployment

### Resources

- **LangChain Documentation**: https://python.langchain.com/docs/
- **LangChain API Reference**: https://api.python.langchain.com/
- **COSMOS Framework**: `.kiro/specs/cosmos/cosmos_architecture_search.md`
- **Architecture Search**: `.kiro/specs/cosmos/architecture_search.md`

---

**Document Version**: 1.0
**Last Updated**: 2025-10-01
**Status**: Implementation guide for LangChain + COSMOS integration
