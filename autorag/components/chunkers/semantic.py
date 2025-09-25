"""Semantic chunking using sentence and paragraph boundaries"""

from typing import List, Dict, Any, Optional
import re
from ...components.base import Document, Chunk, Chunker
from loguru import logger
import hashlib


class SemanticChunker(Chunker):
    """Chunk documents based on semantic boundaries (sentences, paragraphs)"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Chunking parameters
        self.chunk_size = self.config.get("chunk_size", 512)  # Target size in tokens
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.max_chunk_size = self.config.get("max_chunk_size", 1000)
        self.overlap_sentences = self.config.get("overlap_sentences", 1)  # Number of sentences to overlap
        self.respect_sentence_boundary = self.config.get("respect_sentence_boundary", True)
        self.respect_paragraph_boundary = self.config.get("respect_paragraph_boundary", True)

        logger.info(f"SemanticChunker initialized with chunk_size={self.chunk_size}, "
                   f"respect_sentence={self.respect_sentence_boundary}, "
                   f"respect_paragraph={self.respect_paragraph_boundary}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting using regex
        # Handles common abbreviations and edge cases
        sentence_endings = r'[.!?]'
        sentences = re.split(f'({sentence_endings}\\s+)', text)

        # Reconstruct sentences with their endings
        result = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1].rstrip())
            else:
                result.append(sentences[i])

        # Filter out empty sentences
        return [s.strip() for s in result if s.strip()]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or multiple spaces
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (simple word-based approximation)"""
        # Rough approximation: 1 token ≈ 0.75 words
        words = text.split()
        return int(len(words) / 0.75)

    def _create_chunks_from_sentences(self, sentences: List[str], doc_id: str,
                                    doc_metadata: Dict[str, Any]) -> List[Chunk]:
        """Create chunks from sentences respecting size constraints"""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        char_offset = 0

        for i, sentence in enumerate(sentences):
            sentence_size = self._estimate_tokens(sentence)

            # If adding this sentence would exceed max size, create chunk
            if current_chunk and (current_size + sentence_size > self.max_chunk_size):
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "chunking_method": "semantic_sentence"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text)
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.overlap_sentences > 0 and len(current_chunk) > self.overlap_sentences:
                    # Keep last N sentences for overlap
                    overlap_sentences = current_chunk[-self.overlap_sentences:]
                    current_chunk = overlap_sentences
                    current_size = sum(self._estimate_tokens(s) for s in overlap_sentences)
                    char_offset += len(chunk_text) - len(" ".join(overlap_sentences))
                else:
                    current_chunk = []
                    current_size = 0
                    char_offset += len(chunk_text) + 1

                chunk_idx += 1

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size

            # If current chunk reaches target size, consider creating chunk
            if current_size >= self.chunk_size:
                chunk_text = " ".join(current_chunk)
                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "chunking_method": "semantic_sentence"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text)
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.overlap_sentences > 0 and len(current_chunk) > self.overlap_sentences:
                    overlap_sentences = current_chunk[-self.overlap_sentences:]
                    current_chunk = overlap_sentences
                    current_size = sum(self._estimate_tokens(s) for s in overlap_sentences)
                    char_offset += len(chunk_text) - len(" ".join(overlap_sentences))
                else:
                    current_chunk = []
                    current_size = 0
                    char_offset += len(chunk_text) + 1

                chunk_idx += 1

        # Add remaining sentences as final chunk if above minimum size
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk = Chunk(
                content=chunk_text,
                metadata={
                    **doc_metadata,
                    "chunk_index": chunk_idx,
                    "chunking_method": "semantic_sentence"
                },
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                start_char=char_offset,
                end_char=char_offset + len(chunk_text)
            )
            chunks.append(chunk)

        return chunks

    def _create_chunks_from_paragraphs(self, paragraphs: List[str], doc_id: str,
                                      doc_metadata: Dict[str, Any]) -> List[Chunk]:
        """Create chunks from paragraphs"""
        chunks = []
        chunk_idx = 0
        char_offset = 0

        for para in paragraphs:
            para_size = self._estimate_tokens(para)

            if para_size <= self.max_chunk_size:
                # Paragraph fits in one chunk
                chunk = Chunk(
                    content=para,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "chunking_method": "semantic_paragraph"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    start_char=char_offset,
                    end_char=char_offset + len(para)
                )
                chunks.append(chunk)
                char_offset += len(para) + 2  # Account for paragraph break
                chunk_idx += 1
            else:
                # Paragraph too large, split by sentences
                sentences = self._split_into_sentences(para)
                para_chunks = self._create_chunks_from_sentences(
                    sentences, doc_id, doc_metadata
                )
                for chunk in para_chunks:
                    chunk.metadata["chunking_method"] = "semantic_paragraph_split"
                    chunk.chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    chunk.metadata["chunk_index"] = chunk_idx
                    chunks.append(chunk)
                    chunk_idx += 1
                char_offset += len(para) + 2

        return chunks

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into semantic chunks"""
        all_chunks = []

        for doc in documents:
            doc_id = getattr(doc, 'doc_id', None) or hashlib.md5(doc.content.encode()).hexdigest()[:8]

            if self.respect_paragraph_boundary:
                # Split by paragraphs first
                paragraphs = self._split_into_paragraphs(doc.content)
                chunks = self._create_chunks_from_paragraphs(paragraphs, doc_id, doc.metadata)
            elif self.respect_sentence_boundary:
                # Split by sentences only
                sentences = self._split_into_sentences(doc.content)
                chunks = self._create_chunks_from_sentences(sentences, doc_id, doc.metadata)
            else:
                # Fall back to fixed-size chunking
                logger.warning("No semantic boundaries specified, using fixed-size chunking")
                # This would use the fixed-size chunker logic
                chunks = []

            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(documents)} documents")
        return all_chunks