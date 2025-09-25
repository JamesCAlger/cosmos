"""Generation component for Week 1 minimal implementation"""

import os
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from .document_processor import Chunk

load_dotenv()


class OpenAIGenerator:
    """Simple OpenAI GPT-3.5-turbo generator"""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0, max_tokens: int = 300):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"Initialized OpenAI Generator with model={model}, temp={temperature}")

    def generate(self, query: str, retrieved_chunks: List[Tuple[Chunk, float]]) -> str:
        """Generate answer using retrieved contexts"""

        # Format contexts
        contexts = "\n\n".join([
            f"Context {i+1} (score: {score:.3f}):\n{chunk.content}"
            for i, (chunk, score) in enumerate(retrieved_chunks)
        ])

        # Simple prompt template
        prompt = f"""Answer the following question based on the provided contexts. If the answer cannot be found in the contexts, say "I cannot find the answer in the provided contexts."

Contexts:
{contexts}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided contexts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated answer of length {len(answer)}")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise