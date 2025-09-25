"""Graph-based pipeline architecture"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from loguru import logger


@dataclass
class Node:
    """Pipeline node representing a component"""
    id: str
    type: str
    component_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    component_instance: Optional[Any] = None


@dataclass
class Edge:
    """Pipeline edge connecting nodes"""
    from_nodes: List[str]
    to_nodes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineGraph:
    """Directed Acyclic Graph for pipeline execution"""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        logger.info("PipelineGraph initialized")

    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        if node.id in self.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists")

        self.nodes[node.id] = node
        logger.debug(f"Added node: {node.id} (type: {node.type})")

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        # Validate nodes exist
        for from_node in edge.from_nodes:
            if from_node != "input" and from_node not in self.nodes:
                raise ValueError(f"From node '{from_node}' not found")

        for to_node in edge.to_nodes:
            if to_node != "output" and to_node not in self.nodes:
                raise ValueError(f"To node '{to_node}' not found")

        self.edges.append(edge)

        # Update adjacency lists
        for from_node in edge.from_nodes:
            for to_node in edge.to_nodes:
                if from_node != "input":
                    self.adjacency[from_node].add(to_node)
                if to_node != "output":
                    self.reverse_adjacency[to_node].add(from_node)

        logger.debug(f"Added edge: {edge.from_nodes} -> {edge.to_nodes}")

    def get_execution_order(self) -> List[str]:
        """Get topological sort of nodes for execution order"""
        # Find nodes with no dependencies (or only 'input' dependency)
        in_degree = {}
        for node_id in self.nodes:
            dependencies = self.reverse_adjacency.get(node_id, set())
            # Filter out 'input' as it's not a real node
            real_deps = [d for d in dependencies if d != "input" and d in self.nodes]
            in_degree[node_id] = len(real_deps)

        # Start with nodes that have no dependencies
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # Reduce in-degree for neighbors
            for neighbor in self.adjacency.get(node_id, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Check for cycles
        if len(result) != len(self.nodes):
            raise ValueError("Pipeline graph contains cycles")

        logger.debug(f"Execution order: {result}")
        return result

    def get_input_nodes(self) -> List[str]:
        """Get nodes that receive input directly"""
        input_nodes = []
        for edge in self.edges:
            if "input" in edge.from_nodes:
                input_nodes.extend(edge.to_nodes)
        return [n for n in input_nodes if n != "output"]

    def get_output_nodes(self) -> List[str]:
        """Get nodes that produce final output"""
        output_nodes = []
        for edge in self.edges:
            if "output" in edge.to_nodes:
                output_nodes.extend(edge.from_nodes)
        return [n for n in output_nodes if n != "input"]

    def validate(self) -> bool:
        """Validate the graph structure"""
        try:
            # Check for cycles
            self.get_execution_order()

            # Check all nodes are reachable from input
            input_nodes = self.get_input_nodes()
            if not input_nodes:
                logger.warning("No input nodes defined")
                return False

            # Check path to output exists
            output_nodes = self.get_output_nodes()
            if not output_nodes:
                logger.warning("No output nodes defined")
                return False

            logger.info("Pipeline graph validation successful")
            return True

        except Exception as e:
            logger.error(f"Pipeline graph validation failed: {e}")
            return False

    def clear(self) -> None:
        """Clear the graph"""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self.reverse_adjacency.clear()
        logger.debug("Pipeline graph cleared")