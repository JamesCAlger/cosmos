"""
Unit tests for COSMOSComponent wrapper

Tests that components can be wrapped and produce metrics correctly.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from autorag.cosmos.component_wrapper import COSMOSComponent
from autorag.cosmos.metrics import ComponentMetrics
from autorag.components.base import Document, Chunk, QueryResult


class TestCOSMOSComponentInitialization:
    """Test wrapper initialization"""

    def test_init_with_config(self):
        """Test initialization with component that has config"""
        mock_component = Mock()
        mock_component.config = {'param': 'value'}
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)

        assert wrapper.base == mock_component
        assert wrapper.type == 'chunker'
        assert wrapper.metric_collector == mock_metrics
        assert wrapper.config == {'param': 'value'}
        assert wrapper.metrics_history == []

    def test_init_without_config(self):
        """Test initialization with component without config"""
        mock_component = Mock(spec=[])  # No config attribute
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'retriever', mock_metrics)

        assert wrapper.config == {}


class TestChunkerWrapper:
    """Test chunker wrapper functionality"""

    def test_process_chunker_basic(self):
        """Test wrapping a chunker and collecting metrics"""
        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.config = {'chunk_size': 256}
        mock_chunks = [
            Mock(content="chunk 1"),
            Mock(content="chunk 2")
        ]
        mock_chunker.chunk = Mock(return_value=mock_chunks)

        # Mock metrics
        mock_metrics = Mock(spec=ComponentMetrics)
        expected_metrics = {
            'time': 0.1,
            'chunks_created': 2,
            'avg_chunk_size': 5.0,
            'semantic_coherence': 0.7
        }
        mock_metrics.compute_chunking_metrics = Mock(return_value=expected_metrics)

        # Wrap and execute
        wrapper = COSMOSComponent(mock_chunker, 'chunker', mock_metrics)
        documents = [Mock(content="test doc")]

        chunks, metrics = wrapper.process_with_metrics(documents)

        # Verify
        assert chunks == mock_chunks
        assert metrics == expected_metrics
        assert len(wrapper.metrics_history) == 1
        assert wrapper.metrics_history[0] == expected_metrics
        mock_chunker.chunk.assert_called_once_with(documents)

    def test_process_chunker_multiple_calls(self):
        """Test multiple calls accumulate metrics history"""
        mock_chunker = Mock()
        mock_chunker.chunk = Mock(return_value=[Mock(content="chunk")])
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_chunking_metrics = Mock(return_value={'time': 0.1, 'chunks_created': 1})

        wrapper = COSMOSComponent(mock_chunker, 'chunker', mock_metrics)

        # Call three times
        for _ in range(3):
            wrapper.process_with_metrics([Mock()])

        assert len(wrapper.metrics_history) == 3


class TestRetrieverWrapper:
    """Test retriever wrapper functionality"""

    def test_process_retriever_basic(self):
        """Test wrapping a retriever and collecting metrics"""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.config = {'top_k': 5}
        mock_results = [
            Mock(content="result 1"),
            Mock(content="result 2")
        ]
        mock_retriever.retrieve = Mock(return_value=mock_results)

        # Mock metrics
        mock_metrics = Mock(spec=ComponentMetrics)
        expected_metrics = {
            'time': 0.2,
            'docs_retrieved': 2,
            'avg_relevance': 0.85,
            'precision': 0.8
        }
        mock_metrics.compute_retrieval_metrics = Mock(return_value=expected_metrics)

        # Wrap and execute
        wrapper = COSMOSComponent(mock_retriever, 'retriever', mock_metrics)
        query = "test query"

        results, metrics = wrapper.process_with_metrics(query, top_k=5)

        # Verify
        assert results == mock_results
        assert metrics == expected_metrics
        assert len(wrapper.metrics_history) == 1
        mock_retriever.retrieve.assert_called_once_with(query, top_k=5)

    def test_process_retriever_with_ground_truth(self):
        """Test retriever with ground truth data"""
        mock_retriever = Mock()
        mock_retriever.retrieve = Mock(return_value=[Mock()])
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_retrieval_metrics = Mock(return_value={'precision': 1.0})

        wrapper = COSMOSComponent(mock_retriever, 'retriever', mock_metrics)

        ground_truth = {'relevant_chunks': ['1', '2']}
        results, metrics = wrapper.process_with_metrics("query", top_k=5, ground_truth=ground_truth)

        # Check that ground truth was passed to metrics
        call_args = mock_metrics.compute_retrieval_metrics.call_args
        assert call_args[1]['ground_truth'] == ground_truth


class TestGeneratorWrapper:
    """Test generator wrapper functionality"""

    def test_process_generator_basic(self):
        """Test wrapping a generator and collecting metrics"""
        # Mock generator
        mock_generator = Mock()
        mock_generator.config = {'temperature': 0.7}
        mock_answer = "This is the generated answer"
        mock_generator.generate = Mock(return_value=mock_answer)

        # Mock metrics
        mock_metrics = Mock(spec=ComponentMetrics)
        expected_metrics = {
            'time': 1.5,
            'answer_length': 5,
            'answer_relevance': 0.9,
            'context_utilization': 0.6,
            'accuracy': 0.85
        }
        mock_metrics.compute_generation_metrics = Mock(return_value=expected_metrics)

        # Wrap and execute
        wrapper = COSMOSComponent(mock_generator, 'generator', mock_metrics)
        query = "What is AI?"
        context = [Mock(content="AI context")]

        answer, metrics = wrapper.process_with_metrics(query, context)

        # Verify
        assert answer == mock_answer
        assert metrics == expected_metrics
        assert len(wrapper.metrics_history) == 1
        mock_generator.generate.assert_called_once_with(query, context)

    def test_process_generator_with_ground_truth(self):
        """Test generator with ground truth answer"""
        mock_generator = Mock()
        mock_generator.generate = Mock(return_value="answer")
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_generation_metrics = Mock(return_value={'accuracy': 0.95})

        wrapper = COSMOSComponent(mock_generator, 'generator', mock_metrics)

        gt_answer = "ground truth answer"
        answer, metrics = wrapper.process_with_metrics(
            "query", [Mock()], ground_truth_answer=gt_answer
        )

        # Check that ground truth was passed
        call_args = mock_metrics.compute_generation_metrics.call_args
        assert call_args[1]['ground_truth_answer'] == gt_answer


class TestMetricsHistory:
    """Test metrics history tracking"""

    def test_get_average_metrics(self):
        """Test computing average metrics"""
        mock_component = Mock()
        mock_component.chunk = Mock(return_value=[Mock()])
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)

        # Add some metrics to history
        wrapper.metrics_history = [
            {'time': 0.1, 'chunks_created': 5, 'avg_chunk_size': 100.0},
            {'time': 0.2, 'chunks_created': 3, 'avg_chunk_size': 150.0},
            {'time': 0.15, 'chunks_created': 4, 'avg_chunk_size': 125.0}
        ]

        avg = wrapper.get_average_metrics()

        assert 'time' in avg
        assert 'chunks_created' in avg
        assert 'avg_chunk_size' in avg
        assert abs(avg['time'] - 0.15) < 0.01  # (0.1 + 0.2 + 0.15) / 3
        assert abs(avg['chunks_created'] - 4.0) < 0.01  # (5 + 3 + 4) / 3
        assert abs(avg['avg_chunk_size'] - 125.0) < 0.01  # (100 + 150 + 125) / 3

    def test_get_average_metrics_empty(self):
        """Test average metrics with no history"""
        mock_component = Mock()
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)

        avg = wrapper.get_average_metrics()
        assert avg == {}

    def test_get_metrics_summary(self):
        """Test comprehensive metrics summary"""
        mock_component = Mock()
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'retriever', mock_metrics)
        wrapper.metrics_history = [
            {'time': 0.1, 'avg_relevance': 0.8},
            {'time': 0.2, 'avg_relevance': 0.7},
            {'time': 0.15, 'avg_relevance': 0.9}
        ]

        summary = wrapper.get_metrics_summary()

        assert summary['count'] == 3
        assert 'mean' in summary
        assert 'std' in summary
        assert 'min' in summary
        assert 'max' in summary

        assert abs(summary['mean']['time'] - 0.15) < 0.01
        assert abs(summary['min']['time'] - 0.1) < 0.01
        assert abs(summary['max']['time'] - 0.2) < 0.01

    def test_clear_metrics(self):
        """Test clearing metrics history"""
        mock_component = Mock()
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)
        wrapper.metrics_history = [{'time': 0.1}, {'time': 0.2}]

        assert len(wrapper.metrics_history) == 2

        wrapper.clear_metrics()

        assert len(wrapper.metrics_history) == 0


class TestQualityScores:
    """Test quality score computation"""

    def test_get_quality_score(self):
        """Test getting quality score for latest execution"""
        mock_component = Mock()
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_quality_score = Mock(return_value=0.85)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)
        wrapper.metrics_history = [
            {'time': 0.1, 'chunks_created': 5},
            {'time': 0.2, 'chunks_created': 3}
        ]

        score = wrapper.get_quality_score()

        assert score == 0.85
        # Should compute score on latest metrics
        mock_metrics.compute_quality_score.assert_called_once_with('chunker', {'time': 0.2, 'chunks_created': 3})

    def test_get_quality_score_empty(self):
        """Test quality score with no history"""
        mock_component = Mock()
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)

        score = wrapper.get_quality_score()
        assert score == 0.0

    def test_get_average_quality_score(self):
        """Test computing average quality score"""
        mock_component = Mock()
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_quality_score = Mock(side_effect=[0.8, 0.9, 0.7])

        wrapper = COSMOSComponent(mock_component, 'retriever', mock_metrics)
        wrapper.metrics_history = [
            {'time': 0.1},
            {'time': 0.2},
            {'time': 0.15}
        ]

        avg_score = wrapper.get_average_quality_score()

        assert abs(avg_score - 0.8) < 0.01  # (0.8 + 0.9 + 0.7) / 3
        assert mock_metrics.compute_quality_score.call_count == 3


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_component_type(self):
        """Test that invalid component type raises error"""
        mock_component = Mock()
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'invalid_type', mock_metrics)

        with pytest.raises(ValueError, match="Unsupported component type"):
            wrapper.process_with_metrics("input")


class TestAttributeDelegation:
    """Test attribute delegation to base component"""

    def test_delegate_to_base(self):
        """Test that unknown attributes are delegated to base component"""
        mock_component = Mock()
        mock_component.custom_method = Mock(return_value="custom_result")
        mock_component.custom_attribute = "custom_value"
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)

        # Access custom attribute
        assert wrapper.custom_attribute == "custom_value"

        # Call custom method
        result = wrapper.custom_method("arg")
        assert result == "custom_result"
        mock_component.custom_method.assert_called_once_with("arg")


class TestIntegration:
    """Integration tests with real components"""

    def test_wrap_real_chunker(self):
        """Test wrapping a real FixedSizeChunker"""
        from autorag.components.chunkers.fixed_size import FixedSizeChunker
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Create real components
        chunker = FixedSizeChunker({'chunk_size': 50, 'overlap': 10, 'unit': 'tokens'})
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        # Wrap
        wrapper = COSMOSComponent(chunker, 'chunker', metric_collector)

        # Execute
        documents = [Document(content="Test document. " * 20, doc_id="1")]
        chunks, metrics = wrapper.process_with_metrics(documents, compute_coherence=True)

        # Verify
        assert len(chunks) > 0
        assert 'time' in metrics
        assert 'chunks_created' in metrics
        assert metrics['chunks_created'] == len(chunks)
        assert len(wrapper.metrics_history) == 1

        # Check quality score
        quality = wrapper.get_quality_score()
        assert 0.0 <= quality <= 1.0

    def test_wrap_real_generator(self):
        """Test wrapping a real MockGenerator"""
        from autorag.components.generators.mock import MockGenerator
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Create real components
        generator = MockGenerator({'temperature': 0.7})
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        # Wrap
        wrapper = COSMOSComponent(generator, 'generator', metric_collector)

        # Execute
        query = "What is machine learning?"
        context = [Mock(content="ML is a subset of AI")]
        answer, metrics = wrapper.process_with_metrics(query, context)

        # Verify
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert 'time' in metrics
        assert 'answer_length' in metrics
        assert 'answer_relevance' in metrics
        assert len(wrapper.metrics_history) == 1

    def test_metrics_consistency_across_calls(self):
        """Test that metrics are consistent across multiple calls"""
        from autorag.components.chunkers.fixed_size import FixedSizeChunker
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        chunker = FixedSizeChunker({'chunk_size': 100, 'overlap': 0, 'unit': 'tokens'})
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        wrapper = COSMOSComponent(chunker, 'chunker', metric_collector)

        # Same input should give same chunk count
        documents = [Document(content="Test " * 50, doc_id="1")]

        chunks1, metrics1 = wrapper.process_with_metrics(documents, compute_coherence=False)
        chunks2, metrics2 = wrapper.process_with_metrics(documents, compute_coherence=False)

        assert metrics1['chunks_created'] == metrics2['chunks_created']
        assert len(chunks1) == len(chunks2)


class TestRepr:
    """Test string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        mock_component = Mock()
        mock_component.__class__.__name__ = "TestComponent"
        mock_metrics = Mock(spec=ComponentMetrics)

        wrapper = COSMOSComponent(mock_component, 'chunker', mock_metrics)
        wrapper.metrics_history = [{'time': 0.1}, {'time': 0.2}]

        repr_str = repr(wrapper)

        assert 'COSMOSComponent' in repr_str
        assert 'chunker' in repr_str
        assert 'TestComponent' in repr_str
        assert 'executions=2' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])