"""Document processing and chunking for Week 1 minimal implementation"""

import tiktoken
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class Document:
    """Simple document container"""
    content: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Chunk:
    """Document chunk with metadata"""
    content: str
    doc_id: str
    chunk_id: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FixedSizeChunker:
    """Simple fixed-size text chunking (Week 1 implementation)"""

    def __init__(self, chunk_size: int = 256, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        logger.info(f"Initialized FixedSizeChunker with size={chunk_size}, overlap={overlap}")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into fixed-size chunks"""
        chunks = []

        for doc_idx, doc in enumerate(documents):
            doc_id = f"doc_{doc_idx}"
            tokens = self.encoding.encode(doc.content)

            # Calculate chunk positions
            step = self.chunk_size - self.overlap if self.overlap < self.chunk_size else self.chunk_size

            for chunk_idx, i in enumerate(range(0, len(tokens), step)):
                chunk_tokens = tokens[i:i + self.chunk_size]
                if chunk_tokens:  # Skip empty chunks
                    chunk_text = self.encoding.decode(chunk_tokens)
                    chunks.append(Chunk(
                        content=chunk_text,
                        doc_id=doc_id,
                        chunk_id=chunk_idx,
                        metadata={
                            "doc_metadata": doc.metadata,
                            "token_count": len(chunk_tokens),
                            "position": i
                        }
                    ))

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks