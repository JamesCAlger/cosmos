"""Mock chunker for testing component swapping"""

from typing import List, Dict, Any
from ...components.base import Document, Chunk, Chunker
from loguru import logger
import hashlib


class MockChunker(Chunker):
    """Mock chunker that creates single chunk per document"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.prefix = self.config.get("prefix", "[MOCK]")
        logger.info(f"MockChunker initialized with prefix: {self.prefix}")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Create one chunk per document with mock prefix"""
        chunks = []

        for doc in documents:
            doc_id = doc.doc_id or hashlib.md5(doc.content.encode()).hexdigest()[:8]

            # Create single chunk with entire document
            chunk = Chunk(
                content=f"{self.prefix} {doc.content}",
                metadata={
                    **doc.metadata,
                    "chunker": "mock",
                    "chunk_index": 0,
                    "total_chunks": 1
                },
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_0",
                start_char=0,
                end_char=len(doc.content)
            )
            chunks.append(chunk)

        logger.info(f"MockChunker created {len(chunks)} chunks from {len(documents)} documents")
        return chunks