"""OpenAI generator implementation"""

from typing import List, Dict, Any
import openai
import time
from ...components.base import Generator, QueryResult
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIGenerator(Generator):
    """OpenAI text generation implementation with rate limiting"""

    # Class-level variable to track last API call across all instances
    _last_api_call = 0
    _rate_limit_lock = None  # Will be initialized on first use

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.temperature = self.config.get("temperature", 0)
        self.max_tokens = self.config.get("max_tokens", 300)
        self.system_prompt = self.config.get("system_prompt",
            "You are a helpful assistant that answers questions based on the provided context.")

        # Rate limiting configuration:
        # - Free tier (Tier 0): 3 RPM = 20s delay
        # - Tier 1: 60 RPM = 1s delay
        # - Tier 2+: 3500+ RPM = 0.02s delay
        # Default to 1.5s for Tier 1 (with safety margin)
        self.rate_limit_delay = self.config.get("rate_limit_delay", 1.5)

        # Initialize OpenAI client
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")

        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"OpenAIGenerator initialized with model: {self.model}, rate_limit: {self.rate_limit_delay}s")

    def _enforce_rate_limit(self):
        """Enforce rate limiting by waiting if necessary"""
        current_time = time.time()
        time_since_last_call = current_time - OpenAIGenerator._last_api_call

        if time_since_last_call < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last_call
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s before next API call")
            time.sleep(wait_time)

        # Update last call time
        OpenAIGenerator._last_api_call = time.time()

    def generate(self, query: str, context: List[QueryResult]) -> str:
        """Generate answer based on query and context"""
        if not context:
            logger.warning("No context provided for generation")
            context_text = "No relevant context found."
        else:
            # Format context - handle both string and object inputs
            context_parts = []
            for i, result in enumerate(context):
                if isinstance(result, str):
                    # Plain string input
                    context_parts.append(f"Context {i+1}:\n{result}")
                elif hasattr(result, 'score') and hasattr(result, 'chunk'):
                    # Structured object with score and chunk
                    context_parts.append(
                        f"Context {i+1} (score: {result.score:.3f}):\n{result.chunk.content}"
                    )
                elif hasattr(result, 'content'):
                    # Object with content attribute
                    if hasattr(result, 'score'):
                        context_parts.append(
                            f"Context {i+1} (score: {result.score:.3f}):\n{result.content}"
                        )
                    else:
                        context_parts.append(f"Context {i+1}:\n{result.content}")
                else:
                    # Fallback to string representation
                    context_parts.append(f"Context {i+1}:\n{str(result)}")

            context_text = "\n\n".join(context_parts)

        # Create prompt
        user_prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            # Enforce rate limiting before API call
            self._enforce_rate_limit()

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