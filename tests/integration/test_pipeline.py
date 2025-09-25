"""Integration tests for modular pipeline"""

import pytest
from pathlib import Path
import tempfile
import yaml

from autorag.pipeline.orchestrator import PipelineOrchestrator
from autorag.pipeline.rag_pipeline import ModularRAGPipeline
from autorag.components.base import Document
from autorag.config.loader import ConfigLoader


class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""

    def test_load_yaml_config(self):
        """Test loading pipeline from YAML configuration"""
        config = {
            "pipeline": {
                "components": [
                    {"id": "chunker", "type": "chunker", "name": "mock"},
                    {"id": "embedder", "type": "embedder", "name": "mock", "config": {"dimension": 128}},
                    {"id": "vectorstore", "type": "vectorstore", "name": "faiss", "config": {"dimension": 128}},
                    {"id": "generator", "type": "generator", "name": "mock"}
                ]
            }
        }

        # Save to temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Load pipeline
            pipeline = ModularRAGPipeline(config_path)
            components = pipeline.get_components()

            assert "chunker" in components
            assert components["chunker"]["component"] == "mock"
            assert "embedder" in components
            assert components["embedder"]["config"]["dimension"] == 128

        finally:
            Path(config_path).unlink()

    def test_component_swapping(self):
        """Test swapping components via configuration"""
        # Create two different configurations
        config1 = {
            "pipeline": {
                "components": [
                    {"id": "chunker", "type": "chunker", "name": "fixed_size",
                     "config": {"chunk_size": 100}},
                    {"id": "embedder", "type": "embedder", "name": "mock"},
                    {"id": "vectorstore", "type": "vectorstore", "name": "faiss", "config": {"dimension": 384}},
                    {"id": "generator", "type": "generator", "name": "mock"}
                ]
            }
        }

        config2 = {
            "pipeline": {
                "components": [
                    {"id": "chunker", "type": "chunker", "name": "mock"},
                    {"id": "embedder", "type": "embedder", "name": "mock"},
                    {"id": "vectorstore", "type": "vectorstore", "name": "faiss", "config": {"dimension": 384}},
                    {"id": "generator", "type": "generator", "name": "mock"}
                ]
            }
        }

        orchestrator = PipelineOrchestrator()

        # Load first configuration
        orchestrator.load_config(config1)
        components1 = orchestrator.graph.nodes

        # Load second configuration
        orchestrator.load_config(config2)
        components2 = orchestrator.graph.nodes

        # Check that components were swapped
        assert components1["chunker"].component_name == "fixed_size"
        assert components2["chunker"].component_name == "mock"

    def test_end_to_end_mock_pipeline(self):
        """Test complete pipeline execution with mock components"""
        config = {
            "pipeline": {
                "components": [
                    {"id": "chunker", "type": "chunker", "name": "mock"},
                    {"id": "embedder", "type": "embedder", "name": "mock", "config": {"dimension": 64}},
                    {"id": "vectorstore", "type": "vectorstore", "name": "faiss", "config": {"dimension": 64}},
                    {"id": "generator", "type": "generator", "name": "mock"}
                ]
            }
        }

        pipeline = ModularRAGPipeline()
        pipeline.orchestrator.load_config(config)

        # Index documents
        documents = [
            Document(content="Test document 1", metadata={"id": 1}),
            Document(content="Test document 2", metadata={"id": 2})
        ]
        pipeline.index(documents)

        # Query
        result = pipeline.query("Test query")

        assert "question" in result
        assert "answer" in result
        assert result["question"] == "Test query"
        assert "Mock answer" in result["answer"]

    def test_dag_pipeline_execution(self):
        """Test DAG-based pipeline configuration"""
        config = {
            "pipeline": {
                "nodes": [
                    {"id": "chunker", "type": "chunker", "component": "mock"},
                    {"id": "embedder", "type": "embedder", "component": "mock"},
                    {"id": "generator", "type": "generator", "component": "mock"}
                ],
                "edges": [
                    {"from": "input", "to": "chunker"},
                    {"from": "chunker", "to": "embedder"},
                    {"from": "embedder", "to": "generator"},
                    {"from": "generator", "to": "output"}
                ]
            }
        }

        orchestrator = PipelineOrchestrator(config)

        # Get execution order
        order = orchestrator.graph.get_execution_order()
        assert order == ["chunker", "embedder", "generator"]

    def test_config_inheritance(self):
        """Test configuration inheritance"""
        base_config = {
            "pipeline": {
                "components": [
                    {"id": "chunker", "type": "chunker", "name": "fixed_size",
                     "config": {"chunk_size": 256}},
                    {"id": "embedder", "type": "embedder", "name": "mock"},
                    {"id": "vectorstore", "type": "vectorstore", "name": "faiss", "config": {"dimension": 384}},
                    {"id": "generator", "type": "generator", "name": "mock"}
                ]
            }
        }

        override_config = {
            "extends": "base",
            "pipeline": {
                "components": [
                    {"id": "chunker", "type": "chunker", "name": "fixed_size",
                     "config": {"chunk_size": 512}}  # Override chunk size
                ]
            }
        }

        loader = ConfigLoader()
        loader.register_base_config("base", base_config)

        # Save override config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(override_config, f)
            config_path = f.name

        try:
            merged_config = loader.load(config_path)

            # Check that chunk size was overridden
            chunker_config = merged_config["pipeline"]["components"][0]
            assert chunker_config["config"]["chunk_size"] == 512

        finally:
            Path(config_path).unlink()

    def test_parallel_execution_dag(self):
        """Test DAG with parallel paths"""
        config = {
            "pipeline": {
                "nodes": [
                    {"id": "embedder1", "type": "embedder", "component": "mock",
                     "config": {"dimension": 64}},
                    {"id": "embedder2", "type": "embedder", "component": "mock",
                     "config": {"dimension": 128}},
                    {"id": "generator", "type": "generator", "component": "mock"}
                ],
                "edges": [
                    {"from": "input", "to": ["embedder1", "embedder2"]},  # Parallel
                    {"from": ["embedder1", "embedder2"], "to": "generator"},  # Merge
                    {"from": "generator", "to": "output"}
                ]
            }
        }

        orchestrator = PipelineOrchestrator(config)

        # Execute with test input
        result = orchestrator.execute("test input")

        assert "output" in result
        assert "execution_trace" in result
        # Check that both embedders were executed
        node_ids = [trace["node_id"] for trace in result["execution_trace"]]
        assert "embedder1" in node_ids
        assert "embedder2" in node_ids

    def test_error_handling(self):
        """Test error handling in pipeline"""
        config = {
            "pipeline": {
                "components": [
                    {"id": "invalid", "type": "chunker", "name": "nonexistent"}
                ]
            }
        }

        orchestrator = PipelineOrchestrator()

        with pytest.raises(ValueError, match="Component not found"):
            orchestrator.load_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])