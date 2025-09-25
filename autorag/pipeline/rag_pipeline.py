"""Modular RAG pipeline using the new architecture"""

from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path

from .orchestrator import PipelineOrchestrator
from ..components.base import Document


class ModularRAGPipeline:
    """High-level RAG pipeline interface using modular architecture"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration"""
        self.orchestrator = PipelineOrchestrator()
        self.is_indexed = False

        if config_path:
            self.load_config(config_path)
        else:
            # Load default configuration
            default_config = Path(__file__).parent.parent.parent / "configs" / "baseline_linear.yaml"
            if default_config.exists():
                self.load_config(str(default_config))

        logger.info("ModularRAGPipeline initialized")

    def load_config(self, config_path: str) -> None:
        """Load pipeline configuration from file"""
        self.orchestrator.load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

    def index(self, documents: List[Document]) -> None:
        """Index documents for retrieval"""
        logger.info(f"Indexing {len(documents)} documents")

        try:
            # For simple pipelines, use the convenience method
            self.orchestrator.index_documents(documents)
            self.is_indexed = True
            logger.info("Indexing complete")

        except ValueError:
            # For complex pipelines, execute with documents as input
            result = self.orchestrator.execute(documents)
            self.is_indexed = True
            logger.info(f"Indexing complete via full pipeline execution")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        if not self.is_indexed:
            logger.warning("Pipeline not indexed. Results may be empty.")

        logger.debug(f"Processing query: {question[:100]}...")

        # Execute pipeline with query
        result = self.orchestrator.execute(question)

        # Format output
        output = result.get("output")
        if isinstance(output, str):
            # Simple output - just the answer
            return {
                "question": question,
                "answer": output,
                "execution_trace": result.get("execution_trace", []),
                "total_time": result.get("total_time", 0)
            }
        elif isinstance(output, dict):
            # Complex output - multiple components
            return {
                "question": question,
                **output,
                "execution_trace": result.get("execution_trace", []),
                "total_time": result.get("total_time", 0)
            }
        else:
            # Return raw result
            return result

    def get_components(self) -> Dict[str, Any]:
        """Get information about pipeline components"""
        return {
            node_id: {
                "type": node.type,
                "component": node.component_name,
                "config": node.config
            }
            for node_id, node in self.orchestrator.graph.nodes.items()
        }

    def get_execution_order(self) -> List[str]:
        """Get the execution order of components"""
        return self.orchestrator.graph.get_execution_order()