"""Unit tests for pipeline graph"""

import pytest
from autorag.pipeline.graph import PipelineGraph, Node, Edge


class TestPipelineGraph:
    """Test pipeline graph functionality"""

    def test_add_nodes(self):
        """Test adding nodes to graph"""
        graph = PipelineGraph()

        node1 = Node(id="node1", type="chunker", component_name="fixed_size")
        node2 = Node(id="node2", type="embedder", component_name="openai")

        graph.add_node(node1)
        graph.add_node(node2)

        assert "node1" in graph.nodes
        assert "node2" in graph.nodes
        assert graph.nodes["node1"].type == "chunker"

    def test_add_edges(self):
        """Test adding edges to graph"""
        graph = PipelineGraph()

        node1 = Node(id="node1", type="chunker", component_name="fixed_size")
        node2 = Node(id="node2", type="embedder", component_name="openai")

        graph.add_node(node1)
        graph.add_node(node2)

        edge = Edge(from_nodes=["node1"], to_nodes=["node2"])
        graph.add_edge(edge)

        assert "node2" in graph.adjacency["node1"]
        assert "node1" in graph.reverse_adjacency["node2"]

    def test_topological_sort(self):
        """Test getting execution order"""
        graph = PipelineGraph()

        # Create a simple linear pipeline
        nodes = [
            Node(id="n1", type="chunker", component_name="test"),
            Node(id="n2", type="embedder", component_name="test"),
            Node(id="n3", type="generator", component_name="test")
        ]

        for node in nodes:
            graph.add_node(node)

        graph.add_edge(Edge(from_nodes=["input"], to_nodes=["n1"]))
        graph.add_edge(Edge(from_nodes=["n1"], to_nodes=["n2"]))
        graph.add_edge(Edge(from_nodes=["n2"], to_nodes=["n3"]))
        graph.add_edge(Edge(from_nodes=["n3"], to_nodes=["output"]))

        order = graph.get_execution_order()
        assert order == ["n1", "n2", "n3"]

    def test_parallel_paths(self):
        """Test graph with parallel paths"""
        graph = PipelineGraph()

        # Create parallel paths
        nodes = [
            Node(id="split", type="splitter", component_name="test"),
            Node(id="path1", type="processor", component_name="test1"),
            Node(id="path2", type="processor", component_name="test2"),
            Node(id="merge", type="merger", component_name="test")
        ]

        for node in nodes:
            graph.add_node(node)

        graph.add_edge(Edge(from_nodes=["input"], to_nodes=["split"]))
        graph.add_edge(Edge(from_nodes=["split"], to_nodes=["path1", "path2"]))
        graph.add_edge(Edge(from_nodes=["path1", "path2"], to_nodes=["merge"]))
        graph.add_edge(Edge(from_nodes=["merge"], to_nodes=["output"]))

        order = graph.get_execution_order()

        # Split should come first
        assert order.index("split") < order.index("path1")
        assert order.index("split") < order.index("path2")

        # Merge should come after both paths
        assert order.index("path1") < order.index("merge")
        assert order.index("path2") < order.index("merge")

    def test_cycle_detection(self):
        """Test detection of cycles in graph"""
        graph = PipelineGraph()

        nodes = [
            Node(id="n1", type="test", component_name="test"),
            Node(id="n2", type="test", component_name="test"),
            Node(id="n3", type="test", component_name="test")
        ]

        for node in nodes:
            graph.add_node(node)

        # Create a cycle
        graph.add_edge(Edge(from_nodes=["n1"], to_nodes=["n2"]))
        graph.add_edge(Edge(from_nodes=["n2"], to_nodes=["n3"]))
        graph.add_edge(Edge(from_nodes=["n3"], to_nodes=["n1"]))

        with pytest.raises(ValueError, match="cycles"):
            graph.get_execution_order()

    def test_input_output_nodes(self):
        """Test getting input and output nodes"""
        graph = PipelineGraph()

        nodes = [
            Node(id="n1", type="test", component_name="test"),
            Node(id="n2", type="test", component_name="test"),
            Node(id="n3", type="test", component_name="test")
        ]

        for node in nodes:
            graph.add_node(node)

        graph.add_edge(Edge(from_nodes=["input"], to_nodes=["n1", "n2"]))
        graph.add_edge(Edge(from_nodes=["n1"], to_nodes=["n3"]))
        graph.add_edge(Edge(from_nodes=["n2"], to_nodes=["n3"]))
        graph.add_edge(Edge(from_nodes=["n3"], to_nodes=["output"]))

        input_nodes = graph.get_input_nodes()
        assert set(input_nodes) == {"n1", "n2"}

        output_nodes = graph.get_output_nodes()
        assert output_nodes == ["n3"]

    def test_graph_validation(self):
        """Test graph validation"""
        graph = PipelineGraph()

        # Empty graph should fail validation
        assert not graph.validate()

        # Add minimal valid pipeline
        node = Node(id="n1", type="test", component_name="test")
        graph.add_node(node)
        graph.add_edge(Edge(from_nodes=["input"], to_nodes=["n1"]))
        graph.add_edge(Edge(from_nodes=["n1"], to_nodes=["output"]))

        assert graph.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])