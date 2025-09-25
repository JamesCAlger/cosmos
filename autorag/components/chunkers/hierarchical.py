"""Hierarchical chunking with multiple granularity levels"""

from typing import List, Dict, Any, Tuple
from ...components.base import Document, Chunk, Chunker
from loguru import logger
import hashlib


class HierarchicalChunker(Chunker):
    """Create chunks at multiple granularity levels for coarse-to-fine retrieval"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Hierarchy levels (from coarse to fine)
        self.levels = self.config.get("levels", [
            {"name": "coarse", "size": 1024, "overlap": 0},
            {"name": "medium", "size": 512, "overlap": 64},
            {"name": "fine", "size": 256, "overlap": 32}
        ])

        # Parent-child relationship tracking
        self.track_relationships = self.config.get("track_relationships", True)

        # Unit for chunking (tokens or chars)
        self.unit = self.config.get("unit", "tokens")

        logger.info(f"HierarchicalChunker initialized with {len(self.levels)} levels")

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        return text.split()

    def _create_chunks_at_level(self, text: str, tokens: List[str], level: Dict[str, Any],
                               level_idx: int, doc_id: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        """Create chunks at a specific hierarchy level"""
        chunks = []
        size = level["size"]
        overlap = level.get("overlap", 0)
        level_name = level.get("name", f"level_{level_idx}")

        if self.unit == "tokens":
            total_tokens = len(tokens)
            step = size - overlap
            if step <= 0:
                step = size

            position = 0
            chunk_idx = 0

            while position < total_tokens:
                end_position = min(position + size, total_tokens)
                chunk_tokens = tokens[position:end_position]
                chunk_text = " ".join(chunk_tokens)

                # Calculate character positions
                if position == 0:
                    start_char = 0
                else:
                    start_char = len(" ".join(tokens[:position])) + 1
                end_char = len(" ".join(tokens[:end_position]))

                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "hierarchy_level": level_idx,
                        "level_name": level_name,
                        "level_size": size,
                        "token_range": (position, end_position),
                        "chunking_method": "hierarchical"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_L{level_idx}_chunk_{chunk_idx}",
                    start_char=start_char,
                    end_char=end_char
                )
                chunks.append(chunk)

                position += step
                chunk_idx += 1

        else:  # Character-based
            text_length = len(text)
            step = size - overlap
            if step <= 0:
                step = size

            position = 0
            chunk_idx = 0

            while position < text_length:
                end_position = min(position + size, text_length)
                chunk_text = text[position:end_position]

                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "hierarchy_level": level_idx,
                        "level_name": level_name,
                        "level_size": size,
                        "char_range": (position, end_position),
                        "chunking_method": "hierarchical"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_L{level_idx}_chunk_{chunk_idx}",
                    start_char=position,
                    end_char=end_position
                )
                chunks.append(chunk)

                position += step
                chunk_idx += 1

        return chunks

    def _establish_relationships(self, chunks_by_level: Dict[int, List[Chunk]]) -> None:
        """Establish parent-child relationships between chunks at different levels"""
        levels_sorted = sorted(chunks_by_level.keys())

        for i in range(len(levels_sorted) - 1):
            parent_level = levels_sorted[i]
            child_level = levels_sorted[i + 1]

            parent_chunks = chunks_by_level[parent_level]
            child_chunks = chunks_by_level[child_level]

            for parent in parent_chunks:
                parent.metadata["children"] = []

                for child in child_chunks:
                    # Check if child is contained within parent based on position
                    if self._is_contained(child, parent):
                        parent.metadata["children"].append(child.chunk_id)
                        if "parents" not in child.metadata:
                            child.metadata["parents"] = []
                        child.metadata["parents"].append(parent.chunk_id)

    def _is_contained(self, child: Chunk, parent: Chunk) -> bool:
        """Check if child chunk is contained within parent chunk"""
        # Use character positions for containment check
        return (child.start_char >= parent.start_char and
                child.end_char <= parent.end_char)

    def _find_overlapping_chunks(self, chunks: List[Chunk]) -> Dict[str, List[str]]:
        """Find overlapping chunks at the same level"""
        overlaps = {}

        for i, chunk1 in enumerate(chunks):
            overlaps[chunk1.chunk_id] = []
            for j, chunk2 in enumerate(chunks):
                if i != j:
                    # Check for overlap
                    if (chunk1.start_char < chunk2.end_char and
                        chunk2.start_char < chunk1.end_char):
                        overlaps[chunk1.chunk_id].append(chunk2.chunk_id)

        return overlaps

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Create hierarchical chunks from documents"""
        all_chunks = []

        for doc in documents:
            doc_id = getattr(doc, 'doc_id', None) or hashlib.md5(doc.content.encode()).hexdigest()[:8]

            # Tokenize once for all levels
            tokens = self._tokenize(doc.content) if self.unit == "tokens" else None

            # Create chunks at each level
            chunks_by_level = {}
            for level_idx, level in enumerate(self.levels):
                level_chunks = self._create_chunks_at_level(
                    doc.content, tokens, level, level_idx, doc_id, doc.metadata
                )
                chunks_by_level[level_idx] = level_chunks

            # Establish parent-child relationships if configured
            if self.track_relationships and len(chunks_by_level) > 1:
                self._establish_relationships(chunks_by_level)

                # Also track overlaps within each level
                for level_idx, level_chunks in chunks_by_level.items():
                    overlaps = self._find_overlapping_chunks(level_chunks)
                    for chunk in level_chunks:
                        chunk.metadata["overlapping_chunks"] = overlaps.get(chunk.chunk_id, [])

            # Flatten and add all chunks
            for level_chunks in chunks_by_level.values():
                all_chunks.extend(level_chunks)

        # Add summary statistics
        level_counts = {}
        for chunk in all_chunks:
            level_name = chunk.metadata.get("level_name", "unknown")
            level_counts[level_name] = level_counts.get(level_name, 0) + 1

        logger.info(f"Created {len(all_chunks)} hierarchical chunks from {len(documents)} documents")
        logger.info(f"Chunks per level: {level_counts}")

        return all_chunks

    def get_chunks_at_level(self, chunks: List[Chunk], level: int) -> List[Chunk]:
        """Filter chunks to get only those at a specific hierarchy level"""
        return [c for c in chunks if c.metadata.get("hierarchy_level") == level]

    def get_child_chunks(self, parent_chunk: Chunk, all_chunks: List[Chunk]) -> List[Chunk]:
        """Get all child chunks of a parent chunk"""
        child_ids = parent_chunk.metadata.get("children", [])
        return [c for c in all_chunks if c.chunk_id in child_ids]

    def get_parent_chunks(self, child_chunk: Chunk, all_chunks: List[Chunk]) -> List[Chunk]:
        """Get all parent chunks of a child chunk"""
        parent_ids = child_chunk.metadata.get("parents", [])
        return [c for c in all_chunks if c.chunk_id in parent_ids]