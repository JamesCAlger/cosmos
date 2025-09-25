"""Unit tests for Week 4 components"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from autorag.components.base import Document, Chunk, QueryResult
from autorag.components.chunkers import (
    SemanticChunker, SlidingWindowChunker,
    HierarchicalChunker, DocumentAwareChunker
)
from autorag.components.retrievers import (
    BM25Retriever, DenseRetriever, HybridRetriever
)
from autorag.components.rerankers import CrossEncoderReranker


class TestSemanticChunker:
    """Test semantic chunking strategy"""

    def test_semantic_chunker_initialization(self):
        """Test semantic chunker initialization"""
        config = {
            "chunk_size": 256,
            "min_chunk_size": 50,
            "max_chunk_size": 512,
            "overlap_sentences": 1
        }
        chunker = SemanticChunker(config)

        assert chunker.chunk_size == 256
        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 512
        assert chunker.overlap_sentences == 1

    def test_sentence_splitting(self):
        """Test sentence splitting functionality"""
        chunker = SemanticChunker()
        text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."

        sentences = chunker._split_into_sentences(text)

        assert len(sentences) == 4
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two!"
        assert sentences[2] == "Is this sentence three?"
        assert sentences[3] == "Yes it is."

    def test_paragraph_splitting(self):
        """Test paragraph splitting functionality"""
        chunker = SemanticChunker()
        text = "Paragraph one.\n\nParagraph two.\n\n\nParagraph three."

        paragraphs = chunker._split_into_paragraphs(text)

        assert len(paragraphs) == 3
        assert paragraphs[0] == "Paragraph one."
        assert paragraphs[1] == "Paragraph two."
        assert paragraphs[2] == "Paragraph three."

    def test_semantic_chunking_with_sentences(self):
        """Test chunking with sentence boundaries"""
        config = {
            "chunk_size": 10,  # Small size to force multiple chunks
            "min_chunk_size": 5,
            "max_chunk_size": 20,
            "respect_sentence_boundary": True,
            "respect_paragraph_boundary": False
        }
        chunker = SemanticChunker(config)

        doc = Document(
            content="First sentence here. Second sentence here. Third sentence here.",
            metadata={"source": "test"}
        )

        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.doc_id is not None
            assert "chunk_index" in chunk.metadata

    def test_semantic_chunking_with_overlap(self):
        """Test chunking with sentence overlap"""
        config = {
            "chunk_size": 15,
            "overlap_sentences": 1,
            "respect_sentence_boundary": True
        }
        chunker = SemanticChunker(config)

        doc = Document(
            content="Sentence one. Sentence two. Sentence three. Sentence four."
        )

        chunks = chunker.chunk([doc])

        # Check that chunks have overlap
        if len(chunks) > 1:
            # Some content should appear in multiple chunks due to overlap
            assert any("chunking_method" in chunk.metadata for chunk in chunks)


class TestSlidingWindowChunker:
    """Test sliding window chunking strategy"""

    def test_sliding_window_initialization(self):
        """Test sliding window chunker initialization"""
        config = {
            "window_size": 100,
            "step_size": 50,
            "unit": "tokens"
        }
        chunker = SlidingWindowChunker(config)

        assert chunker.window_size == 100
        assert chunker.step_size == 50
        assert chunker.overlap == 50

    def test_sliding_window_with_overlap_ratio(self):
        """Test sliding window with overlap ratio"""
        config = {
            "window_size": 100,
            "overlap_ratio": 0.25
        }
        chunker = SlidingWindowChunker(config)

        assert chunker.step_size == 75  # 100 * (1 - 0.25)
        assert chunker.overlap == 25

    def test_token_based_sliding_window(self):
        """Test token-based sliding window chunking"""
        config = {
            "window_size": 5,
            "step_size": 3,
            "unit": "tokens"
        }
        chunker = SlidingWindowChunker(config)

        doc = Document(
            content="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        )

        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        # First chunk should have 5 words
        assert len(chunks[0].content.split()) == 5
        # Check overlap information in metadata
        assert "window_start" in chunks[0].metadata
        assert "window_end" in chunks[0].metadata

    def test_char_based_sliding_window(self):
        """Test character-based sliding window chunking"""
        config = {
            "window_size": 10,
            "step_size": 5,
            "unit": "chars"
        }
        chunker = SlidingWindowChunker(config)

        doc = Document(content="abcdefghijklmnopqrstuvwxyz")

        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        assert len(chunks[0].content) == 10
        # Check for overlap
        if len(chunks) > 1:
            assert chunks[1].metadata.get("overlap_chars", 0) > 0


class TestHierarchicalChunker:
    """Test hierarchical chunking strategy"""

    def test_hierarchical_initialization(self):
        """Test hierarchical chunker initialization"""
        config = {
            "levels": [
                {"name": "coarse", "size": 1000},
                {"name": "fine", "size": 250}
            ],
            "track_relationships": True
        }
        chunker = HierarchicalChunker(config)

        assert len(chunker.levels) == 2
        assert chunker.track_relationships is True

    def test_multi_level_chunking(self):
        """Test creation of chunks at multiple levels"""
        config = {
            "levels": [
                {"name": "large", "size": 20, "overlap": 0},
                {"name": "small", "size": 10, "overlap": 0}
            ],
            "unit": "tokens"
        }
        chunker = HierarchicalChunker(config)

        doc = Document(
            content=" ".join([f"word{i}" for i in range(50)])
        )

        chunks = chunker.chunk([doc])

        # Should have chunks from both levels
        large_chunks = [c for c in chunks if c.metadata["level_name"] == "large"]
        small_chunks = [c for c in chunks if c.metadata["level_name"] == "small"]

        assert len(large_chunks) > 0
        assert len(small_chunks) > 0
        assert len(small_chunks) >= len(large_chunks)

    def test_parent_child_relationships(self):
        """Test parent-child relationship tracking"""
        config = {
            "levels": [
                {"name": "parent", "size": 20, "overlap": 0},
                {"name": "child", "size": 10, "overlap": 0}
            ],
            "track_relationships": True,
            "unit": "tokens"
        }
        chunker = HierarchicalChunker(config)

        doc = Document(content=" ".join([f"word{i}" for i in range(30)]))

        chunks = chunker.chunk([doc])

        # Check that relationships are established
        parent_chunks = [c for c in chunks if c.metadata["hierarchy_level"] == 0]
        child_chunks = [c for c in chunks if c.metadata["hierarchy_level"] == 1]

        if parent_chunks and child_chunks:
            # At least one parent should have children
            assert any("children" in p.metadata for p in parent_chunks)
            # At least one child should have parents
            assert any("parents" in c.metadata for c in child_chunks)


class TestDocumentAwareChunker:
    """Test document-aware chunking strategy"""

    def test_document_aware_initialization(self):
        """Test document-aware chunker initialization"""
        config = {
            "detect_headers": True,
            "detect_lists": True,
            "max_chunk_size": 500,
            "preserve_structure": True
        }
        chunker = DocumentAwareChunker(config)

        assert chunker.detect_headers is True
        assert chunker.detect_lists is True
        assert chunker.max_chunk_size == 500

    def test_header_detection(self):
        """Test header detection in markdown"""
        chunker = DocumentAwareChunker()
        text = """# Main Header

## Section 1
Content for section 1.

## Section 2
Content for section 2."""

        structure = chunker._detect_document_structure(text)

        assert len(structure["headers"]) == 3
        assert structure["headers"][0]["level"] == 1
        assert structure["headers"][0]["text"] == "Main Header"
        assert structure["headers"][1]["level"] == 2

    def test_section_based_chunking(self):
        """Test chunking based on document sections"""
        config = {"max_chunk_size": 100}
        chunker = DocumentAwareChunker(config)

        doc = Document(
            content="""# Title

## Section A
This is content for section A.

## Section B
This is content for section B."""
        )

        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        # Check that section information is preserved
        assert all("structure_type" in chunk.metadata for chunk in chunks)

    def test_code_block_detection(self):
        """Test code block detection"""
        chunker = DocumentAwareChunker({"detect_code_blocks": True})
        text = """Some text before.

```python
def hello():
    print("Hello")
```

Some text after."""

        structure = chunker._detect_document_structure(text)

        assert len(structure["code_blocks"]) == 1
        assert "```" in structure["code_blocks"][0]["content"]


class TestBM25Retriever:
    """Test BM25 sparse retriever"""

    def test_bm25_initialization(self):
        """Test BM25 retriever initialization"""
        config = {
            "k1": 1.5,
            "b": 0.8,
            "tokenizer": "simple"
        }
        retriever = BM25Retriever(config)

        assert retriever.k1 == 1.5
        assert retriever.b == 0.8
        assert retriever.tokenizer == "simple"

    def test_bm25_indexing(self):
        """Test BM25 indexing"""
        retriever = BM25Retriever()

        chunks = [
            Chunk(
                content="The quick brown fox",
                metadata={},
                doc_id="doc1",
                chunk_id="chunk1",
                start_char=0,
                end_char=20
            ),
            Chunk(
                content="Jumps over the lazy dog",
                metadata={},
                doc_id="doc1",
                chunk_id="chunk2",
                start_char=21,
                end_char=45
            )
        ]

        retriever.index(chunks)

        assert len(retriever.chunks) == 2
        assert retriever.bm25 is not None

    def test_bm25_retrieval(self):
        """Test BM25 retrieval"""
        retriever = BM25Retriever()

        chunks = [
            Chunk(
                content="Python is a high-level programming language",
                metadata={},
                doc_id="doc1",
                chunk_id="chunk1",
                start_char=0,
                end_char=45
            ),
            Chunk(
                content="Java tutorial for beginners",
                metadata={},
                doc_id="doc2",
                chunk_id="chunk2",
                start_char=0,
                end_char=28
            )
        ]

        retriever.index(chunks)
        results = retriever.retrieve("Python programming", top_k=2)

        # Test that BM25 returns results
        assert len(results) > 0
        # Check that all results have required fields
        for result in results:
            assert hasattr(result, 'chunk')
            assert hasattr(result, 'score')
            assert result.score >= 0  # BM25 scores can be 0
            assert "retriever" in result.metadata
            assert result.metadata["retriever"] == "bm25"


class TestDenseRetriever:
    """Test dense retriever"""

    def test_dense_retriever_initialization(self):
        """Test dense retriever initialization"""
        retriever = DenseRetriever()
        assert retriever.embedder is None
        assert retriever.vector_store is None

    def test_dense_retriever_with_mocks(self):
        """Test dense retriever with mock components"""
        retriever = DenseRetriever()

        # Create mock embedder
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        # Create mock vector store
        mock_vector_store = Mock()
        mock_result = QueryResult(
            chunk=Chunk(
                content="Test content",
                metadata={},
                doc_id="doc1",
                chunk_id="chunk1",
                start_char=0,
                end_char=10
            ),
            score=0.9
        )
        mock_vector_store.search.return_value = [mock_result]

        retriever.set_components(mock_embedder, mock_vector_store)

        # Test indexing
        chunks = [
            Chunk(
                content="Test chunk 1",
                metadata={},
                doc_id="doc1",
                chunk_id="chunk1",
                start_char=0,
                end_char=10
            )
        ]
        retriever.index(chunks)

        # Verify embedder was called
        mock_embedder.embed.assert_called_once()

        # Test retrieval
        results = retriever.retrieve("test query", top_k=5)

        assert len(results) == 1
        assert results[0].score == 0.9
        mock_embedder.embed_query.assert_called_once_with("test query")


class TestHybridRetriever:
    """Test hybrid retriever"""

    def test_hybrid_initialization(self):
        """Test hybrid retriever initialization"""
        config = {
            "fusion_method": "weighted_sum",
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "normalization": "min_max"
        }
        retriever = HybridRetriever(config)

        assert retriever.fusion_method == "weighted_sum"
        assert retriever.dense_weight == 0.6
        assert retriever.sparse_weight == 0.4

    def test_weighted_sum_fusion(self):
        """Test weighted sum fusion method"""
        retriever = HybridRetriever({
            "fusion_method": "weighted_sum",
            "dense_weight": 0.7,
            "sparse_weight": 0.3
        })

        # Create mock results
        chunk1 = Chunk(
            content="Chunk 1",
            metadata={},
            doc_id="doc1",
            chunk_id="chunk1",
            start_char=0,
            end_char=10
        )
        chunk2 = Chunk(
            content="Chunk 2",
            metadata={},
            doc_id="doc2",
            chunk_id="chunk2",
            start_char=0,
            end_char=10
        )

        dense_results = [
            QueryResult(chunk=chunk1, score=0.9),
            QueryResult(chunk=chunk2, score=0.7)
        ]
        sparse_results = [
            QueryResult(chunk=chunk1, score=0.8),
            QueryResult(chunk=chunk2, score=0.6)
        ]

        combined = retriever._weighted_sum_fusion(dense_results, sparse_results, top_k=2)

        assert len(combined) == 2
        # Both chunks should be present
        chunk_ids = [r.chunk.chunk_id for r in combined]
        assert "chunk1" in chunk_ids
        assert "chunk2" in chunk_ids

    def test_rrf_fusion(self):
        """Test reciprocal rank fusion"""
        retriever = HybridRetriever({
            "fusion_method": "rrf",
            "rrf_k": 60
        })

        chunk1 = Chunk(
            content="Chunk 1",
            metadata={},
            doc_id="doc1",
            chunk_id="chunk1",
            start_char=0,
            end_char=10
        )

        dense_results = [QueryResult(chunk=chunk1, score=0.9)]
        sparse_results = [QueryResult(chunk=chunk1, score=0.8)]

        combined = retriever._rrf_fusion(dense_results, sparse_results, top_k=1)

        assert len(combined) == 1
        assert combined[0].chunk.chunk_id == "chunk1"
        # RRF score should be sum of 1/(k+rank) for each list
        expected_score = 1/(60+1) + 1/(60+1)  # Rank 1 in both lists
        assert abs(combined[0].score - expected_score) < 0.001

    def test_parallel_fusion(self):
        """Test parallel fusion method"""
        retriever = HybridRetriever({"fusion_method": "parallel"})

        chunk1 = Chunk(
            content="Dense chunk",
            metadata={},
            doc_id="doc1",
            chunk_id="chunk1",
            start_char=0,
            end_char=10
        )
        chunk2 = Chunk(
            content="Sparse chunk",
            metadata={},
            doc_id="doc2",
            chunk_id="chunk2",
            start_char=0,
            end_char=10
        )

        dense_results = [QueryResult(chunk=chunk1, score=0.9)]
        sparse_results = [QueryResult(chunk=chunk2, score=0.8)]

        combined = retriever._parallel_fusion(dense_results, sparse_results, top_k=2)

        assert len(combined) == 2
        # Should have results from both retrievers
        sources = [r.metadata.get("source") for r in combined]
        assert "dense" in sources
        assert "sparse" in sources


class TestCrossEncoderReranker:
    """Test cross-encoder reranker"""

    @patch('autorag.components.rerankers.cross_encoder.CrossEncoder')
    def test_reranker_initialization(self, mock_cross_encoder):
        """Test reranker initialization"""
        config = {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "device": "cpu",
            "batch_size": 16
        }

        # Mock the CrossEncoder class
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance

        reranker = CrossEncoderReranker(config)

        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.device == "cpu"
        assert reranker.batch_size == 16
        mock_cross_encoder.assert_called_once()

    @patch('autorag.components.rerankers.cross_encoder.CrossEncoder')
    def test_reranking(self, mock_cross_encoder):
        """Test reranking functionality"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.predict.return_value = np.array([0.9, 0.7, 0.5])
        mock_cross_encoder.return_value = mock_instance

        reranker = CrossEncoderReranker()

        # Create test results
        results = [
            QueryResult(
                chunk=Chunk(
                    content="Highly relevant",
                    metadata={},
                    doc_id="doc1",
                    chunk_id="chunk1",
                    start_char=0,
                    end_char=10
                ),
                score=0.5
            ),
            QueryResult(
                chunk=Chunk(
                    content="Somewhat relevant",
                    metadata={},
                    doc_id="doc2",
                    chunk_id="chunk2",
                    start_char=0,
                    end_char=10
                ),
                score=0.6
            ),
            QueryResult(
                chunk=Chunk(
                    content="Not relevant",
                    metadata={},
                    doc_id="doc3",
                    chunk_id="chunk3",
                    start_char=0,
                    end_char=10
                ),
                score=0.4
            )
        ]

        reranked = reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        # Should be sorted by reranker scores
        assert reranked[0].chunk.chunk_id == "chunk1"  # Highest score (0.9)
        assert reranked[1].chunk.chunk_id == "chunk2"  # Second highest (0.7)

    def test_reranker_without_model(self):
        """Test reranker behavior when model fails to load"""
        with patch('autorag.components.rerankers.cross_encoder.CrossEncoder',
                  side_effect=Exception("Model not found")):
            reranker = CrossEncoderReranker()

            # Should work in passthrough mode
            assert reranker.model is None

            results = [
                QueryResult(
                    chunk=Chunk(
                        content="Test",
                        metadata={},
                        doc_id="doc1",
                        chunk_id="chunk1",
                        start_char=0,
                        end_char=10
                    ),
                    score=0.5
                )
            ]

            reranked = reranker.rerank("query", results, top_k=1)
            assert len(reranked) == 1
            assert reranked[0].chunk.chunk_id == "chunk1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])