"""Embedding generation for Week 1 minimal implementation"""

import os
import numpy as np
from typing import List, Union
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
import time
from tqdm import tqdm

load_dotenv()


class OpenAIEmbedder:
    """OpenAI text-embedding-ada-002 embedder"""

    def __init__(self, model: str = "text-embedding-ada-002", batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"Initialized OpenAI Embedder with model={model}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Retry once with exponential backoff
                time.sleep(2)
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    logger.error(f"Retry failed: {e}")
                    raise

        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Generated {len(embeddings_array)} embeddings")
        return embeddings_array

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.embed(query)[0]