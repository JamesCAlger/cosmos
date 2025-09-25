"""Mock generator for testing"""

from typing import List, Dict, Any
from ...components.base import Generator, QueryResult
from loguru import logger


class MockGenerator(Generator):
    """Mock generator that returns templated answers"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.template = self.config.get("template", "Mock answer for: {query}")
        logger.info("MockGenerator initialized")

    def generate(self, query: str, context: List[QueryResult]) -> str:
        """Generate mock answer"""
        context_summary = f"Based on {len(context)} contexts" if context else "No context"
        answer = self.template.format(query=query, context_summary=context_summary)
        logger.debug(f"Generated mock answer: {answer[:50]}...")
        return answer