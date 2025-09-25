"""OpenAI embedder implementation"""

from typing import List, Dict, Any
import openai
from ...components.base import Embedder
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIEmbedder(Embedder):
    """OpenAI text embedding implementation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "text-embedding-ada-002")
        self.batch_size = self.config.get("batch_size", 100)

        # Initialize OpenAI client
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")

        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"OpenAIEmbedder initialized with model: {self.model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Generated embeddings for batch {i//self.batch_size + 1}")

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise