"""Base abstract interfaces for all RAG components"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class Document:
    """Document with content and metadata"""
    content: str
    metadata: Dict[str, Any] = None
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Chunk:
    """Chunk of a document with position information"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_id: str
    start_char: int
    end_char: int


@dataclass
class QueryResult:
    """Result from a retrieval query"""
    chunk: Chunk
    score: float
    metadata: Optional[Dict[str, Any]] = None


class Component(ABC):
    """Base class for all pipeline components"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize component with configuration"""
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.info(f"Initializing component: {self.name}")

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process input and return output"""
        pass

    def validate_config(self) -> bool:
        """Validate component configuration"""
        return True

    def get_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": self.name,
            "type": self.__class__.__base__.__name__,
            "config": self.config
        }


class Chunker(Component):
    """Abstract base class for document chunkers"""

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into chunks"""
        pass

    def process(self, documents: List[Document]) -> List[Chunk]:
        """Process wrapper for pipeline compatibility"""
        return self.chunk(documents)


class Embedder(Component):
    """Abstract base class for text embedders"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        pass

    def process(self, texts: List[str]) -> List[List[float]]:
        """Process wrapper for pipeline compatibility"""
        return self.embed(texts)


class VectorStore(Component):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add(self, embeddings: List[List[float]], chunks: List[Chunk]) -> None:
        """Add embeddings and associated chunks to store"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the vector store"""
        pass

    def process(self, query_embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        """Process wrapper for pipeline compatibility"""
        return self.search(query_embedding, top_k)


class Retriever(Component):
    """Abstract base class for retrievers"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """Retrieve relevant chunks for query"""
        pass

    def process(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """Process wrapper for pipeline compatibility"""
        return self.retrieve(query, top_k)


class Reranker(Component):
    """Abstract base class for rerankers"""

    @abstractmethod
    def rerank(self, query: str, results: List[QueryResult], top_k: int = 5) -> List[QueryResult]:
        """Rerank retrieval results"""
        pass

    def process(self, query: str, results: List[QueryResult], top_k: int = 5) -> List[QueryResult]:
        """Process wrapper for pipeline compatibility"""
        return self.rerank(query, results, top_k)


class Generator(Component):
    """Abstract base class for answer generators"""

    @abstractmethod
    def generate(self, query: str, context: List[QueryResult]) -> str:
        """Generate answer given query and context"""
        pass

    def process(self, query: str, context: List[QueryResult]) -> str:
        """Process wrapper for pipeline compatibility"""
        return self.generate(query, context)


class PostProcessor(Component):
    """Abstract base class for post-processors"""

    @abstractmethod
    def process(self, answer: str, metadata: Dict[str, Any]) -> str:
        """Post-process generated answer"""
        pass