"""Document-aware chunking that respects document structure"""

from typing import List, Dict, Any, Optional, Tuple
import re
from ...components.base import Document, Chunk, Chunker
from loguru import logger
import hashlib


class DocumentAwareChunker(Chunker):
    """Chunk documents while respecting their structure (headers, sections, lists, etc.)"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Structure detection
        self.detect_headers = self.config.get("detect_headers", True)
        self.detect_lists = self.config.get("detect_lists", True)
        self.detect_code_blocks = self.config.get("detect_code_blocks", True)
        self.detect_tables = self.config.get("detect_tables", True)

        # Chunking parameters
        self.max_chunk_size = self.config.get("max_chunk_size", 512)
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.preserve_structure = self.config.get("preserve_structure", True)
        self.include_headers_in_chunks = self.config.get("include_headers_in_chunks", True)

        # Markdown patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.list_pattern = re.compile(r'^[\s]*[-*+•]\s+(.+)$', re.MULTILINE)
        self.numbered_list_pattern = re.compile(r'^[\s]*\d+[.)]\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)

        logger.info(f"DocumentAwareChunker initialized with structure detection: "
                   f"headers={self.detect_headers}, lists={self.detect_lists}")

    def _detect_document_structure(self, text: str) -> Dict[str, Any]:
        """Detect the structure of the document"""
        structure = {
            "headers": [],
            "sections": [],
            "lists": [],
            "code_blocks": [],
            "tables": [],
            "paragraphs": []
        }

        # Find headers and their positions
        if self.detect_headers:
            for match in self.header_pattern.finditer(text):
                level = len(match.group(1))
                header_text = match.group(2)
                structure["headers"].append({
                    "level": level,
                    "text": header_text,
                    "start": match.start(),
                    "end": match.end()
                })

        # Find code blocks
        if self.detect_code_blocks:
            for match in self.code_block_pattern.finditer(text):
                structure["code_blocks"].append({
                    "content": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })

        # Find lists
        if self.detect_lists:
            # Bullet lists
            for match in self.list_pattern.finditer(text):
                structure["lists"].append({
                    "type": "bullet",
                    "content": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })

            # Numbered lists
            for match in self.numbered_list_pattern.finditer(text):
                structure["lists"].append({
                    "type": "numbered",
                    "content": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })

        # Find tables
        if self.detect_tables:
            lines = text.split('\n')
            in_table = False
            table_start = 0
            current_pos = 0

            for i, line in enumerate(lines):
                if self.table_pattern.match(line):
                    if not in_table:
                        in_table = True
                        table_start = current_pos
                elif in_table:
                    # End of table
                    structure["tables"].append({
                        "start": table_start,
                        "end": current_pos
                    })
                    in_table = False

                current_pos += len(line) + 1  # +1 for newline

        # Create sections based on headers
        if structure["headers"]:
            structure["sections"] = self._create_sections_from_headers(
                text, structure["headers"]
            )

        return structure

    def _create_sections_from_headers(self, text: str, headers: List[Dict]) -> List[Dict]:
        """Create document sections based on headers"""
        sections = []
        sorted_headers = sorted(headers, key=lambda x: x["start"])

        for i, header in enumerate(sorted_headers):
            section_start = header["start"]

            # Find the end of this section (start of next header or end of document)
            if i < len(sorted_headers) - 1:
                section_end = sorted_headers[i + 1]["start"]
            else:
                section_end = len(text)

            section_content = text[section_start:section_end].strip()

            sections.append({
                "header": header,
                "content": section_content,
                "start": section_start,
                "end": section_end,
                "level": header["level"]
            })

        return sections

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text.split())

    def _chunk_section(self, section: Dict, doc_id: str, doc_metadata: Dict[str, Any],
                      chunk_idx_start: int) -> List[Chunk]:
        """Chunk a single section respecting its structure"""
        chunks = []
        chunk_idx = chunk_idx_start

        section_content = section["content"]
        section_header = section["header"]["text"] if "header" in section else ""
        section_level = section.get("level", 0)

        # If section is small enough, keep it as one chunk
        if self._estimate_tokens(section_content) <= self.max_chunk_size:
            chunk = Chunk(
                content=section_content,
                metadata={
                    **doc_metadata,
                    "chunk_index": chunk_idx,
                    "section_header": section_header,
                    "section_level": section_level,
                    "structure_type": "section",
                    "chunking_method": "document_aware"
                },
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                start_char=section["start"],
                end_char=section["end"]
            )
            chunks.append(chunk)

        else:
            # Section too large, split it intelligently
            # Try to split by paragraphs first
            paragraphs = section_content.split('\n\n')
            current_chunk_content = []
            current_size = 0

            if self.include_headers_in_chunks and section_header:
                header_line = f"{'#' * section_level} {section_header}\n\n"
                current_chunk_content.append(header_line)
                current_size = self._estimate_tokens(header_line)

            for para in paragraphs:
                para_size = self._estimate_tokens(para)

                if current_size + para_size > self.max_chunk_size and current_chunk_content:
                    # Create chunk
                    chunk_text = '\n\n'.join(current_chunk_content)
                    chunk = Chunk(
                        content=chunk_text,
                        metadata={
                            **doc_metadata,
                            "chunk_index": chunk_idx,
                            "section_header": section_header,
                            "section_level": section_level,
                            "structure_type": "section_part",
                            "chunking_method": "document_aware"
                        },
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        start_char=section["start"],  # Approximate
                        end_char=section["start"] + len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                    # Start new chunk
                    current_chunk_content = []
                    current_size = 0

                    # Include header in continuation chunks if configured
                    if self.include_headers_in_chunks and section_header:
                        header_line = f"{'#' * section_level} {section_header} (continued)\n\n"
                        current_chunk_content.append(header_line)
                        current_size = self._estimate_tokens(header_line)

                current_chunk_content.append(para)
                current_size += para_size

            # Add remaining content
            if current_chunk_content and current_size >= self.min_chunk_size:
                chunk_text = '\n\n'.join(current_chunk_content)
                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "section_header": section_header,
                        "section_level": section_level,
                        "structure_type": "section_part",
                        "chunking_method": "document_aware"
                    },
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    start_char=section["start"],
                    end_char=section["end"]
                )
                chunks.append(chunk)

        return chunks

    def _chunk_structured_element(self, element: Dict, element_type: str,
                                 doc_id: str, doc_metadata: Dict[str, Any],
                                 chunk_idx: int) -> Chunk:
        """Create a chunk from a structured element (code block, table, etc.)"""
        content = element.get("content", "")

        chunk = Chunk(
            content=content,
            metadata={
                **doc_metadata,
                "chunk_index": chunk_idx,
                "structure_type": element_type,
                "chunking_method": "document_aware"
            },
            doc_id=doc_id,
            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
            start_char=element["start"],
            end_char=element["end"]
        )
        return chunk

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents while respecting their structure"""
        all_chunks = []

        for doc in documents:
            doc_id = getattr(doc, 'doc_id', None) or hashlib.md5(doc.content.encode()).hexdigest()[:8]

            # Detect document structure
            structure = self._detect_document_structure(doc.content)

            chunks = []
            chunk_idx = 0

            if structure["sections"]:
                # Chunk by sections
                for section in structure["sections"]:
                    section_chunks = self._chunk_section(
                        section, doc_id, doc.metadata, chunk_idx
                    )
                    chunks.extend(section_chunks)
                    chunk_idx += len(section_chunks)

            else:
                # No clear sections, fall back to paragraph-based chunking
                paragraphs = doc.content.split('\n\n')
                current_chunk = []
                current_size = 0

                for para in paragraphs:
                    para_size = self._estimate_tokens(para)

                    if current_size + para_size > self.max_chunk_size and current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        chunk = Chunk(
                            content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": chunk_idx,
                                "structure_type": "paragraph",
                                "chunking_method": "document_aware"
                            },
                            doc_id=doc_id,
                            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                            start_char=0,  # Would need proper tracking
                            end_char=len(chunk_text)
                        )
                        chunks.append(chunk)
                        chunk_idx += 1

                        current_chunk = []
                        current_size = 0

                    current_chunk.append(para)
                    current_size += para_size

                # Add remaining content
                if current_chunk and current_size >= self.min_chunk_size:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = Chunk(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk_idx,
                            "structure_type": "paragraph",
                            "chunking_method": "document_aware"
                        },
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        start_char=0,
                        end_char=len(chunk_text)
                    )
                    chunks.append(chunk)

            # Add structure summary to each chunk's metadata
            for chunk in chunks:
                chunk.metadata["document_structure"] = {
                    "total_headers": len(structure["headers"]),
                    "total_sections": len(structure["sections"]),
                    "has_code_blocks": len(structure["code_blocks"]) > 0,
                    "has_tables": len(structure["tables"]) > 0,
                    "has_lists": len(structure["lists"]) > 0
                }

            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} document-aware chunks from {len(documents)} documents")
        return all_chunks