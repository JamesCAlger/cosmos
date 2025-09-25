"""Pipeline orchestrator for executing RAG pipelines"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger
import time

from .graph import PipelineGraph, Node, Edge
from .registry import get_registry, register_default_components
from ..config.loader import ConfigLoader
from ..components.base import Document, Chunk, QueryResult


@dataclass
class PipelineContext:
    """Context passed through pipeline execution"""
    input_data: Any
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)


class PipelineOrchestrator:
    """Orchestrate execution of RAG pipelines"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize orchestrator with optional configuration"""
        self.graph = PipelineGraph()
        self.registry = get_registry()
        self.config_loader = ConfigLoader()
        self.components: Dict[str, Any] = {}

        # Register default components
        register_default_components()

        if config:
            self.load_config(config)

        logger.info("PipelineOrchestrator initialized")

    def load_config(self, config: Dict[str, Any]) -> None:
        """Load pipeline configuration"""
        if isinstance(config, str):
            # Load from file
            config = self.config_loader.load(config)

        pipeline_config = config.get("pipeline", {})

        # Clear existing graph
        self.graph.clear()
        self.components.clear()

        # Build graph from configuration
        if "nodes" in pipeline_config:
            self._build_dag_pipeline(pipeline_config)
        elif "components" in pipeline_config:
            self._build_linear_pipeline(pipeline_config)
        else:
            raise ValueError("Configuration must contain 'nodes' (DAG) or 'components' (linear)")

        # Validate graph
        if not self.graph.validate():
            raise ValueError("Invalid pipeline graph")

        logger.info("Pipeline configuration loaded successfully")

    def _build_dag_pipeline(self, config: Dict[str, Any]) -> None:
        """Build DAG-based pipeline from configuration"""
        nodes = config.get("nodes", [])
        edges = config.get("edges", [])

        # Create nodes
        for node_config in nodes:
            node = Node(
                id=node_config["id"],
                type=node_config["type"],
                component_name=node_config.get("component", node_config["type"]),
                config=node_config.get("config", {})
            )

            # Create component instance
            component = self.registry.create_component(
                node.type,
                node.component_name,
                node.config
            )
            node.component_instance = component
            self.components[node.id] = component

            self.graph.add_node(node)

        # Create edges
        for edge_config in edges:
            edge = Edge(
                from_nodes=edge_config["from"] if isinstance(edge_config["from"], list)
                          else [edge_config["from"]],
                to_nodes=edge_config["to"] if isinstance(edge_config["to"], list)
                        else [edge_config["to"]],
                metadata=edge_config.get("metadata", {})
            )
            self.graph.add_edge(edge)

    def _build_linear_pipeline(self, config: Dict[str, Any]) -> None:
        """Build linear pipeline from configuration (backward compatibility)"""
        components = config.get("components", [])

        prev_node_id = "input"

        for i, comp_config in enumerate(components):
            node_id = comp_config.get("id", f"component_{i}")

            node = Node(
                id=node_id,
                type=comp_config["type"],
                component_name=comp_config["name"],
                config=comp_config.get("config", {})
            )

            # Create component instance
            component = self.registry.create_component(
                node.type,
                node.component_name,
                node.config
            )
            node.component_instance = component
            self.components[node_id] = component

            self.graph.add_node(node)

            # Add edge from previous node
            edge = Edge(from_nodes=[prev_node_id], to_nodes=[node_id])
            self.graph.add_edge(edge)

            prev_node_id = node_id

        # Add final edge to output
        edge = Edge(from_nodes=[prev_node_id], to_nodes=["output"])
        self.graph.add_edge(edge)

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute the pipeline with given input"""
        logger.info("Starting pipeline execution")
        start_time = time.time()

        # Create execution context
        context = PipelineContext(input_data=input_data)

        # Get execution order
        execution_order = self.graph.get_execution_order()

        # Execute nodes in order
        for node_id in execution_order:
            node = self.graph.nodes[node_id]
            component = self.components[node_id]

            # Determine input for this node
            node_input = self._get_node_input(node_id, context)

            # Execute component
            logger.debug(f"Executing node: {node_id} (type: {node.type})")
            exec_start = time.time()

            try:
                result = self._execute_component(component, node.type, node_input, context)
                exec_time = time.time() - exec_start

                # Store result
                context.intermediate_results[node_id] = result

                # Add to execution trace
                context.execution_trace.append({
                    "node_id": node_id,
                    "type": node.type,
                    "execution_time": exec_time,
                    "status": "success"
                })

                logger.debug(f"Node {node_id} executed in {exec_time:.3f}s")

            except Exception as e:
                logger.error(f"Error executing node {node_id}: {e}")
                context.execution_trace.append({
                    "node_id": node_id,
                    "type": node.type,
                    "error": str(e),
                    "status": "failed"
                })
                raise

        # Get final output
        output_nodes = self.graph.get_output_nodes()
        if len(output_nodes) == 1:
            final_output = context.intermediate_results.get(output_nodes[0])
        else:
            # Multiple outputs - return all
            final_output = {
                node_id: context.intermediate_results.get(node_id)
                for node_id in output_nodes
            }

        total_time = time.time() - start_time
        logger.info(f"Pipeline execution completed in {total_time:.3f}s")

        return {
            "output": final_output,
            "execution_trace": context.execution_trace,
            "total_time": total_time
        }

    def _get_node_input(self, node_id: str, context: PipelineContext) -> Any:
        """Get input for a node based on its dependencies"""
        dependencies = self.graph.reverse_adjacency.get(node_id, set())

        if not dependencies or dependencies == {"input"}:
            # This is an input node
            return context.input_data

        # Get outputs from dependencies
        dep_results = []
        for dep_id in dependencies:
            if dep_id == "input":
                dep_results.append(context.input_data)
            elif dep_id in context.intermediate_results:
                dep_results.append(context.intermediate_results[dep_id])

        # If single dependency, return it directly
        if len(dep_results) == 1:
            return dep_results[0]

        # Multiple dependencies - return as list
        return dep_results

    def _execute_component(self, component: Any, component_type: str,
                          input_data: Any, context: PipelineContext) -> Any:
        """Execute a component with appropriate input format"""
        # Handle different component types
        if component_type == "chunker":
            if not isinstance(input_data, list):
                input_data = [input_data]
            # Ensure input is Document objects
            documents = []
            for item in input_data:
                if isinstance(item, Document):
                    documents.append(item)
                elif isinstance(item, dict):
                    documents.append(Document(**item))
                elif isinstance(item, str):
                    documents.append(Document(content=item, metadata={}))
            return component.chunk(documents)

        elif component_type == "embedder":
            if isinstance(input_data, list) and len(input_data) > 0:
                if isinstance(input_data[0], Chunk):
                    texts = [chunk.content for chunk in input_data]
                elif isinstance(input_data[0], str):
                    texts = input_data
                else:
                    texts = [str(item) for item in input_data]
                return component.embed(texts)
            elif isinstance(input_data, str):
                return component.embed_query(input_data)

        elif component_type == "vectorstore":
            # Expect embeddings and chunks
            if isinstance(input_data, list) and len(input_data) == 2:
                embeddings, chunks = input_data
                component.add(embeddings, chunks)
                return {"status": "indexed", "count": len(chunks)}
            else:
                raise ValueError(f"VectorStore expects [embeddings, chunks], got {type(input_data)}")

        elif component_type == "retriever" or component_type == "vectorstore_search":
            # For retrieval, we need query embedding
            if isinstance(input_data, list):
                query_embedding = input_data
            else:
                query_embedding = input_data
            return component.search(query_embedding, top_k=5)

        elif component_type == "generator":
            # Expect query and contexts
            if isinstance(input_data, list) and len(input_data) == 2:
                query, contexts = input_data
            elif isinstance(input_data, dict):
                query = input_data.get("query", "")
                contexts = input_data.get("contexts", [])
            else:
                query = str(input_data)
                contexts = []
            return component.generate(query, contexts)

        else:
            # Default - call process method
            return component.process(input_data)

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for retrieval (for simple pipelines)"""
        # This is a convenience method for simple RAG pipelines
        # Find chunker, embedder, and vector store nodes
        chunker_node = None
        embedder_node = None
        vectorstore_node = None

        for node_id, node in self.graph.nodes.items():
            if node.type == "chunker":
                chunker_node = node_id
            elif node.type == "embedder":
                embedder_node = node_id
            elif node.type == "vectorstore":
                vectorstore_node = node_id

        if not all([chunker_node, embedder_node, vectorstore_node]):
            raise ValueError("Pipeline must have chunker, embedder, and vectorstore for indexing")

        # Execute indexing pipeline
        chunks = self.components[chunker_node].chunk(documents)
        texts = [chunk.content for chunk in chunks]
        embeddings = self.components[embedder_node].embed(texts)
        self.components[vectorstore_node].add(embeddings, chunks)

        logger.info(f"Indexed {len(documents)} documents ({len(chunks)} chunks)")

    def query(self, query: str) -> Dict[str, Any]:
        """Process a query through the pipeline"""
        return self.execute(query)