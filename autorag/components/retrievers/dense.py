"""Dense retrieval implementation using embeddings"""

from typing import List, Dict, Any, Optional
from ...components.base import Retriever, QueryResult, Chunk, Document, VectorStore, Embedder
from loguru import logger


class DenseRetriever(Retriever):
    """Dense retriever using embeddings and vector similarity"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.embedder: Optional[Embedder] = None
        self.vector_store: Optional[VectorStore] = None
        self.chunks: List[Chunk] = []

        logger.info("DenseRetriever initialized")

    def set_components(self, embedder: Embedder, vector_store: VectorStore) -> None:
        """Set the embedder and vector store components"""
        self.embedder = embedder
        self.vector_store = vector_store
        logger.info("Components set for dense retrieval")

    def index(self, chunks: List[Chunk]) -> None:
        """Index chunks by creating embeddings and storing them"""
        if not self.embedder or not self.vector_store:
            raise ValueError("Embedder and vector store must be set before indexing")

        self.chunks = chunks

        # Extract texts from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embedder.embed(texts)

        # Store embeddings with chunks
        self.vector_store.add(embeddings, chunks)

        logger.info(f"Indexed {len(chunks)} chunks in dense retriever")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index (convert to chunks first)"""
        # Simple chunking for documents (this would normally use a chunker)
        chunks = []
        for doc_idx, doc in enumerate(documents):
            import hashlib
            doc_id = getattr(doc, 'doc_id', None) or hashlib.md5(doc.content.encode()).hexdigest()[:8]
            chunk = Chunk(
                content=doc.content,
                metadata=doc.metadata,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_0",
                start_char=0,
                end_char=len(doc.content)
            )
            chunks.append(chunk)

        self.index(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """Retrieve relevant chunks using dense retrieval"""
        if not self.embedder or not self.vector_store:
            logger.error("Embedder and vector store must be set before retrieval")
            return []

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search in vector store
        results = self.vector_store.search(query_embedding, top_k)

        # Add metadata
        for result in results:
            if result.metadata is None:
                result.metadata = {}
            result.metadata["retriever"] = "dense"

        logger.debug(f"Dense retrieval returned {len(results)} results")
        return results

    def clear(self) -> None:
        """Clear the dense retriever"""
        if self.vector_store:
            self.vector_store.clear()
        self.chunks = []
        logger.info("Dense retriever cleared")