"""Simple RAG pipeline that handles both indexing and querying"""

from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path

from ..components.base import Document, QueryResult
from .orchestrator import PipelineOrchestrator
from .registry import register_default_components


class SimpleRAGPipeline:
    """Simplified RAG pipeline for Week 2 that properly handles index and query operations"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration"""
        self.orchestrator = PipelineOrchestrator()

        # Register components
        register_default_components()

        # Store component references for direct access
        self.chunker = None
        self.embedder = None
        self.vectorstore = None
        self.generator = None

        if config_path:
            self.load_config(config_path)
        else:
            # Load default configuration
            default_config = Path(__file__).parent.parent.parent / "configs" / "baseline_rag.yaml"
            if default_config.exists():
                self.load_config(str(default_config))

        logger.info("SimpleRAGPipeline initialized")

    def load_config(self, config_path: str) -> None:
        """Load pipeline configuration and initialize components"""
        self.orchestrator.load_config(config_path)

        # Get direct references to components for index/query operations
        for node_id, component in self.orchestrator.components.items():
            if node_id == "chunker":
                self.chunker = component
            elif node_id == "embedder":
                self.embedder = component
            elif node_id == "vectorstore":
                self.vectorstore = component
            elif node_id == "generator":
                self.generator = component

        logger.info(f"Loaded configuration from {config_path}")

    def index(self, documents: List[Document]) -> Dict[str, Any]:
        """Index documents for retrieval"""
        logger.info(f"Indexing {len(documents)} documents")

        if not all([self.chunker, self.embedder, self.vectorstore]):
            raise ValueError("Pipeline not properly configured for indexing")

        # Chunk documents
        chunks = self.chunker.chunk(documents)
        logger.debug(f"Created {len(chunks)} chunks")

        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed(chunk_texts)
        logger.debug(f"Generated {len(embeddings)} embeddings")

        # Store in vector database
        self.vectorstore.add(embeddings, chunks)
        logger.info(f"Indexed {len(chunks)} chunks")

        return {
            "num_documents": len(documents),
            "num_chunks": len(chunks)
        }

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        logger.debug(f"Processing query: {question[:100]}...")

        if not all([self.embedder, self.vectorstore, self.generator]):
            raise ValueError("Pipeline not properly configured for querying")

        # Generate query embedding
        query_embedding = self.embedder.embed_query(question)

        # Retrieve relevant chunks
        retrieved_results = self.vectorstore.search(query_embedding, top_k)

        # Generate answer
        answer = self.generator.generate(question, retrieved_results)

        # Format response
        return {
            "question": question,
            "answer": answer,
            "contexts": [
                {
                    "content": result.chunk.content,
                    "score": float(result.score),
                    "metadata": result.chunk.metadata
                }
                for result in retrieved_results
            ]
        }

    def get_components(self) -> Dict[str, Any]:
        """Get information about pipeline components"""
        return {
            node_id: {
                "type": node.type,
                "component": node.component_name,
                "config": node.config
            }
            for node_id, node in self.orchestrator.graph.nodes.items()
        }