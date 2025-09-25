"""Unit tests for components"""

import pytest
from typing import List

from autorag.components.base import Document, Chunk
from autorag.components.chunkers.fixed_size import FixedSizeChunker
from autorag.components.chunkers.mock import MockChunker
from autorag.components.embedders.mock import MockEmbedder
from autorag.components.generators.mock import MockGenerator
from autorag.components.retrievers.faiss_store import FAISSVectorStore


class TestChunkers:
    """Test chunker implementations"""

    def test_fixed_size_chunker(self):
        """Test fixed-size chunking"""
        chunker = FixedSizeChunker({"chunk_size": 10, "overlap": 0, "unit": "tokens"})

        documents = [
            Document(content="This is a test document with multiple words for chunking.", metadata={"id": 1})
        ]

        chunks = chunker.chunk(documents)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert chunks[0].doc_id is not None
        assert chunks[0].chunk_id is not None

    def test_fixed_size_chunker_with_overlap(self):
        """Test chunking with overlap"""
        chunker = FixedSizeChunker({"chunk_size": 5, "overlap": 2, "unit": "tokens"})

        documents = [
            Document(content="word1 word2 word3 word4 word5 word6 word7 word8", metadata={})
        ]

        chunks = chunker.chunk(documents)

        # With overlap, chunks should share some words
        assert len(chunks) >= 2
        # Check that chunks have expected size
        for chunk in chunks[:-1]:  # All but last chunk
            words = chunk.content.split()
            assert len(words) <= 5

    def test_mock_chunker(self):
        """Test mock chunker"""
        chunker = MockChunker({"prefix": "[TEST]"})

        documents = [
            Document(content="Test document", metadata={"id": 1}),
            Document(content="Another document", metadata={"id": 2})
        ]

        chunks = chunker.chunk(documents)

        assert len(chunks) == 2
        assert all(chunk.content.startswith("[TEST]") for chunk in chunks)
        assert chunks[0].metadata["chunker"] == "mock"


class TestEmbedders:
    """Test embedder implementations"""

    def test_mock_embedder_consistency(self):
        """Test that mock embedder produces consistent embeddings"""
        embedder = MockEmbedder({"dimension": 128, "seed": 42})

        texts = ["test text", "another text"]
        embeddings1 = embedder.embed(texts)
        embeddings2 = embedder.embed(texts)

        assert len(embeddings1) == 2
        assert len(embeddings1[0]) == 128
        # Same text should produce same embedding
        assert embeddings1[0] == embeddings2[0]

    def test_mock_embedder_query(self):
        """Test query embedding"""
        embedder = MockEmbedder({"dimension": 64})

        query = "test query"
        embedding = embedder.embed_query(query)

        assert len(embedding) == 64
        assert all(0 <= val <= 1 for val in embedding)


class TestVectorStore:
    """Test vector store implementation"""

    def test_faiss_store_add_and_search(self):
        """Test adding and searching in FAISS store"""
        store = FAISSVectorStore({"dimension": 128})

        # Create mock embeddings and chunks
        embeddings = [[0.1] * 128, [0.2] * 128, [0.3] * 128]
        chunks = [
            Chunk(content=f"Chunk {i}", metadata={}, doc_id=f"doc_{i}",
                 chunk_id=f"chunk_{i}", start_char=0, end_char=10)
            for i in range(3)
        ]

        store.add(embeddings, chunks)

        # Search with a query embedding
        query_embedding = [0.15] * 128
        results = store.search(query_embedding, top_k=2)

        assert len(results) == 2
        assert results[0].score > 0
        assert results[0].chunk.content in ["Chunk 0", "Chunk 1", "Chunk 2"]

    def test_faiss_store_clear(self):
        """Test clearing the store"""
        store = FAISSVectorStore({"dimension": 64})

        embeddings = [[0.1] * 64]
        chunks = [Chunk(content="Test", metadata={}, doc_id="1",
                       chunk_id="1", start_char=0, end_char=4)]

        store.add(embeddings, chunks)
        assert len(store.chunks) == 1

        store.clear()
        assert len(store.chunks) == 0

        results = store.search([0.1] * 64, top_k=1)
        assert len(results) == 0


class TestGenerators:
    """Test generator implementations"""

    def test_mock_generator(self):
        """Test mock generator"""
        generator = MockGenerator({
            "template": "Answer: {query} | Context: {context_summary}"
        })

        from autorag.components.base import QueryResult
        query = "What is the weather?"
        contexts = [
            QueryResult(
                chunk=Chunk(content="It's sunny", metadata={}, doc_id="1",
                           chunk_id="1", start_char=0, end_char=10),
                score=0.9
            )
        ]

        answer = generator.generate(query, contexts)

        assert "What is the weather?" in answer
        assert "1 contexts" in answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])