"""Sliding window chunking with configurable overlap"""

from typing import List, Dict, Any
from ...components.base import Document, Chunk, Chunker
from loguru import logger
import hashlib


class SlidingWindowChunker(Chunker):
    """Create overlapping chunks using a sliding window approach"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Window parameters
        self.window_size = self.config.get("window_size", 256)  # Size of each window
        self.step_size = self.config.get("step_size", 128)  # How much to slide
        self.unit = self.config.get("unit", "tokens")  # tokens or chars
        self.overlap_ratio = self.config.get("overlap_ratio", None)  # Alternative to step_size

        # Calculate step size from overlap ratio if provided
        if self.overlap_ratio is not None:
            self.step_size = int(self.window_size * (1 - self.overlap_ratio))

        # Ensure step size is valid
        if self.step_size <= 0:
            self.step_size = self.window_size // 2
        if self.step_size > self.window_size:
            self.step_size = self.window_size

        self.overlap = self.window_size - self.step_size

        logger.info(f"SlidingWindowChunker initialized: window={self.window_size}, "
                   f"step={self.step_size}, overlap={self.overlap}, unit={self.unit}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        return text.split()

    def _create_window_chunks(self, text: str, doc_id: str,
                             doc_metadata: Dict[str, Any]) -> List[Chunk]:
        """Create chunks using sliding window"""
        chunks = []

        if self.unit == "tokens":
            # Token-based windowing
            tokens = self._tokenize(text)
            total_tokens = len(tokens)

            position = 0
            chunk_idx = 0

            while position < total_tokens:
                # Get window of tokens
                end_position = min(position + self.window_size, total_tokens)
                window_tokens = tokens[position:end_position]
                chunk_text = " ".join(window_tokens)

                # Calculate character positions (approximate)
                if position == 0:
                    start_char = 0
                else:
                    start_char = len(" ".join(tokens[:position])) + 1

                end_char = len(" ".join(tokens[:end_position]))

                # Create chunk
                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "window_start": position,
                        "window_end": end_position,
                        "total_windows": None,  # Will be set after all chunks created
                        "overlap_tokens": self.overlap if position > 0 else 0,
                        "chunking_method": "sliding_window_tokens"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    start_char=start_char,
                    end_char=end_char
                )
                chunks.append(chunk)

                # Slide the window
                position += self.step_size
                chunk_idx += 1

        else:  # Character-based windowing
            text_length = len(text)
            position = 0
            chunk_idx = 0

            while position < text_length:
                # Get window of characters
                end_position = min(position + self.window_size, text_length)
                chunk_text = text[position:end_position]

                # Create chunk
                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "window_start": position,
                        "window_end": end_position,
                        "overlap_chars": self.overlap if position > 0 else 0,
                        "chunking_method": "sliding_window_chars"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    start_char=position,
                    end_char=end_position
                )
                chunks.append(chunk)

                # Slide the window
                position += self.step_size
                chunk_idx += 1

        # Update total windows count in metadata
        for chunk in chunks:
            chunk.metadata["total_windows"] = len(chunks)

        return chunks

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into overlapping chunks using sliding window"""
        all_chunks = []

        for doc in documents:
            doc_id = getattr(doc, 'doc_id', None) or hashlib.md5(doc.content.encode()).hexdigest()[:8]

            # Create sliding window chunks
            chunks = self._create_window_chunks(doc.content, doc_id, doc.metadata)

            # Add overlap information between adjacent chunks
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                curr_chunk = chunks[i]

                # Calculate actual overlap
                if self.unit == "tokens":
                    prev_tokens = set(prev_chunk.content.split())
                    curr_tokens = set(curr_chunk.content.split())
                    overlap_tokens = prev_tokens & curr_tokens
                    curr_chunk.metadata["overlap_with_previous"] = len(overlap_tokens)
                else:
                    # Find overlapping text
                    overlap_start = curr_chunk.start_char
                    overlap_end = prev_chunk.end_char
                    if overlap_end > overlap_start:
                        overlap_text = doc.content[overlap_start:overlap_end]
                        curr_chunk.metadata["overlap_with_previous"] = len(overlap_text)

            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} sliding window chunks from {len(documents)} documents "
                   f"(overlap: {self.overlap} {self.unit})")
        return all_chunks

    def calculate_optimal_step_size(self, desired_overlap_ratio: float) -> int:
        """Calculate step size for desired overlap ratio"""
        return int(self.window_size * (1 - desired_overlap_ratio))