"""BM25 sparse retrieval implementation"""

from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from ...components.base import Retriever, QueryResult, Chunk, Document
from loguru import logger
import pickle
import hashlib


class BM25Retriever(Retriever):
    """BM25-based sparse retriever for keyword matching"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k1 = self.config.get("k1", 1.2)  # BM25 k1 parameter
        self.b = self.config.get("b", 0.75)  # BM25 b parameter
        self.epsilon = self.config.get("epsilon", 0.25)  # BM25 epsilon parameter
        self.tokenizer = self.config.get("tokenizer", "simple")  # simple, nltk, spacy

        self.bm25 = None
        self.chunks: List[Chunk] = []
        self.corpus: List[List[str]] = []

        logger.info(f"BM25Retriever initialized with k1={self.k1}, b={self.b}, tokenizer={self.tokenizer}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text based on configured tokenizer"""
        if self.tokenizer == "simple":
            # Simple whitespace + lowercase tokenization
            return text.lower().split()
        elif self.tokenizer == "nltk":
            try:
                import nltk
                # Download punkt if not available
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                tokens = nltk.word_tokenize(text.lower())
                # Remove punctuation-only tokens
                return [t for t in tokens if any(c.isalnum() for c in t)]
            except ImportError:
                logger.warning("NLTK not available, falling back to simple tokenizer")
                return text.lower().split()
        else:
            # Default to simple tokenizer
            return text.lower().split()

    def index(self, chunks: List[Chunk]) -> None:
        """Index chunks for BM25 retrieval"""
        self.chunks = chunks
        self.corpus = [self._tokenize(chunk.content) for chunk in chunks]

        # Initialize BM25 with the corpus
        self.bm25 = BM25Okapi(
            self.corpus,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon
        )

        logger.info(f"Indexed {len(chunks)} chunks for BM25 retrieval")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index (convert to chunks first)"""
        # Simple chunking for documents (this would normally use a chunker)
        chunks = []
        for doc_idx, doc in enumerate(documents):
            doc_id = getattr(doc, 'doc_id', None) or hashlib.md5(doc.content.encode()).hexdigest()[:8]
            chunk = Chunk(
                content=doc.content,
                metadata=doc.metadata,
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_0",
                start_char=0,
                end_char=len(doc.content)
            )
            chunks.append(chunk)

        self.index(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """Retrieve relevant chunks using BM25"""
        if self.bm25 is None or not self.chunks:
            logger.warning("BM25 index is empty")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[-k:][::-1]

        # Create QueryResult objects
        results = []
        for idx in top_indices:
            # Include all results, even with zero scores (BM25 can return 0)
            result = QueryResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                metadata={
                    "retriever": "bm25",
                    "raw_score": float(scores[idx])
                }
            )
            results.append(result)

        logger.debug(f"BM25 retrieved {len(results)} results for query")
        return results

    def save_index(self, path: str) -> None:
        """Save BM25 index to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'corpus': self.corpus
            }, f)
        logger.info(f"Saved BM25 index to {path}")

    def load_index(self, path: str) -> None:
        """Load BM25 index from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
            self.corpus = data['corpus']
        logger.info(f"Loaded BM25 index from {path}")

    def clear(self) -> None:
        """Clear the BM25 index"""
        self.bm25 = None
        self.chunks = []
        self.corpus = []
        logger.info("BM25 index cleared")