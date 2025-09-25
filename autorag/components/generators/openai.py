"""OpenAI generator implementation"""

from typing import List, Dict, Any
import openai
from ...components.base import Generator, QueryResult
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIGenerator(Generator):
    """OpenAI text generation implementation"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.temperature = self.config.get("temperature", 0)
        self.max_tokens = self.config.get("max_tokens", 300)
        self.system_prompt = self.config.get("system_prompt",
            "You are a helpful assistant that answers questions based on the provided context.")

        # Initialize OpenAI client
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")

        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"OpenAIGenerator initialized with model: {self.model}")

    def generate(self, query: str, context: List[QueryResult]) -> str:
        """Generate answer based on query and context"""
        if not context:
            logger.warning("No context provided for generation")
            context_text = "No relevant context found."
        else:
            # Format context
            context_text = "\n\n".join([
                f"Context {i+1} (score: {result.score:.3f}):\n{result.chunk.content}"
                for i, result in enumerate(context)
            ])

        # Create prompt
        user_prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
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