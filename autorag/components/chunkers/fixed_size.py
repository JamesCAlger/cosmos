"""Fixed-size chunker implementation"""

from typing import List, Dict, Any
from ...components.base import Document, Chunk, Chunker
from loguru import logger
import hashlib


class FixedSizeChunker(Chunker):
    """Split documents into fixed-size chunks"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 256)
        self.overlap = self.config.get("overlap", 0)
        self.unit = self.config.get("unit", "tokens")  # tokens or chars
        logger.info(f"FixedSizeChunker initialized: size={self.chunk_size}, overlap={self.overlap}, unit={self.unit}")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into fixed-size chunks"""
        chunks = []

        for doc in documents:
            doc_id = getattr(doc, 'doc_id', None) or hashlib.md5(doc.content.encode()).hexdigest()[:8]

            if self.unit == "tokens":
                # Simple token-based splitting (word approximation)
                tokens = doc.content.split()

                start_idx = 0
                chunk_idx = 0

                while start_idx < len(tokens):
                    end_idx = min(start_idx + self.chunk_size, len(tokens))
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = " ".join(chunk_tokens)

                    # Calculate character positions (approximate)
                    start_char = len(" ".join(tokens[:start_idx])) + (1 if start_idx > 0 else 0)
                    end_char = len(" ".join(tokens[:end_idx]))

                    chunk = Chunk(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_idx,
                            "total_chunks": None  # Will be set after all chunks created
                        },
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        start_char=start_char,
                        end_char=end_char
                    )
                    chunks.append(chunk)

                    chunk_idx += 1
                    start_idx += self.chunk_size - self.overlap

            else:  # character-based
                text = doc.content
                start_idx = 0
                chunk_idx = 0

                while start_idx < len(text):
                    end_idx = min(start_idx + self.chunk_size, len(text))
                    chunk_text = text[start_idx:end_idx]

                    chunk = Chunk(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_idx
                        },
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        start_char=start_idx,
                        end_char=end_idx
                    )
                    chunks.append(chunk)

                    chunk_idx += 1
                    start_idx += self.chunk_size - self.overlap

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks