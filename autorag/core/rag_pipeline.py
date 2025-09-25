"""Minimal RAG Pipeline for Week 1 implementation"""

from typing import List, Dict, Any
from loguru import logger
from .document_processor import Document, FixedSizeChunker
from .embedder import OpenAIEmbedder
from .retriever import FAISSRetriever
from .generator import OpenAIGenerator


class RAGPipeline:
    """Minimal RAG pipeline with hardcoded configuration for Week 1"""

    def __init__(self):
        """Initialize with fixed configuration for Week 1"""
        logger.info("Initializing minimal RAG pipeline")

        # Fixed configuration for Week 1
        self.chunker = FixedSizeChunker(chunk_size=256, overlap=0)
        self.embedder = OpenAIEmbedder(model="text-embedding-ada-002")
        self.retriever = FAISSRetriever(dimension=1536)
        self.generator = OpenAIGenerator(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=300
        )

        self.is_indexed = False

    def index(self, documents: List[Document]):
        """Index documents for retrieval"""
        logger.info(f"Indexing {len(documents)} documents")

        # Chunk documents
        chunks = self.chunker.chunk(documents)

        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed(chunk_texts)

        # Index in FAISS
        self.retriever.index_chunks(embeddings, chunks)

        self.is_indexed = True
        logger.info("Indexing complete")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using RAG"""
        if not self.is_indexed:
            raise ValueError("Pipeline not indexed. Call index() first.")

        logger.debug(f"Processing query: {question[:100]}...")

        # Generate query embedding
        query_embedding = self.embedder.embed_query(question)

        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query_embedding, top_k)

        # Generate answer
        answer = self.generator.generate(question, retrieved_chunks)

        return {
            "question": question,
            "answer": answer,
            "contexts": [
                {
                    "content": chunk.content,
                    "score": float(score),
                    "metadata": chunk.metadata
                }
                for chunk, score in retrieved_chunks
            ]
        }