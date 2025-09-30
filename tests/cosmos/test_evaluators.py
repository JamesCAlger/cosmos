"""
Unit tests for Component Evaluators

Tests that components can be evaluated in isolation or with fixed upstream components.
"""

import pytest
from unittest.mock import Mock, MagicMock
from autorag.cosmos.optimization.evaluators import ComponentEvaluator, build_component
from autorag.cosmos.metrics import ComponentMetrics
from autorag.components.base import Document


class TestBuildComponent:
    """Test component building from configuration"""

    def test_build_fixed_chunker(self):
        """Test building fixed size chunker"""
        config = {
            'chunking_strategy': 'fixed',
            'chunk_size': 256,
            'overlap': 50
        }

        component = build_component('chunker', config)

        assert component is not None
        assert hasattr(component, 'chunk')
        assert component.config['chunk_size'] == 256

    def test_build_semantic_chunker(self):
        """Test building semantic chunker"""
        config = {
            'chunking_strategy': 'semantic',
            'chunk_size': 512,
            'threshold': 0.6
        }

        component = build_component('chunker', config)

        assert component is not None
        assert hasattr(component, 'chunk')
        assert component.config['chunk_size'] == 512

    def test_build_dense_retriever(self):
        """Test building dense retriever"""
        config = {
            'retrieval_method': 'dense',
            'retrieval_top_k': 5
        }

        component = build_component('retriever', config)

        assert component is not None
        assert hasattr(component, 'retrieve')

    def test_build_sparse_retriever(self):
        """Test building sparse retriever"""
        config = {
            'retrieval_method': 'sparse',
            'k1': 1.5,
            'b': 0.8
        }

        component = build_component('retriever', config)

        assert component is not None
        assert hasattr(component, 'retrieve')

    def test_build_mock_generator(self):
        """Test building mock generator"""
        config = {
            'use_real_api': False,
            'temperature': 0.7
        }

        component = build_component('generator', config)

        assert component is not None
        assert hasattr(component, 'generate')

    def test_build_invalid_type(self):
        """Test that invalid type raises error"""
        with pytest.raises(ValueError):
            build_component('invalid_type', {})


class TestChunkerEvaluator:
    """Test chunker evaluation"""

    def test_evaluate_chunker_basic(self):
        """Test basic chunker evaluation"""
        # Mock metric collector
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_chunking_metrics = Mock(return_value={
            'time': 0.1,
            'chunks_created': 5,
            'avg_chunk_size': 250,
            'semantic_coherence': 0.7
        })
        mock_metrics.compute_quality_score = Mock(return_value=0.8)

        # Test data
        test_data = {
            'documents': [
                "This is test document one. " * 20,
                "This is test document two. " * 20
            ],
            'queries': []
        }

        # Create evaluator
        evaluator = ComponentEvaluator(
            component_type='chunker',
            test_data=test_data,
            metric_collector=mock_metrics
        )

        # Evaluate config
        config = {
            'chunking_strategy': 'fixed',
            'chunk_size': 256,
            'overlap': 50
        }

        score = evaluator.evaluate(config)

        # Verify
        assert 0.0 <= score <= 1.0
        assert score == 0.8
        mock_metrics.compute_chunking_metrics.assert_called_once()
        mock_metrics.compute_quality_score.assert_called_once()

    def test_evaluate_chunker_handles_errors(self):
        """Test chunker evaluation handles errors gracefully"""
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_chunking_metrics = Mock(side_effect=Exception("Test error"))

        test_data = {'documents': ["test doc"], 'queries': []}

        evaluator = ComponentEvaluator('chunker', test_data, mock_metrics)

        score = evaluator.evaluate({'chunking_strategy': 'fixed', 'chunk_size': 256})

        # Should return 0.0 on error
        assert score == 0.0


class TestRetrieverEvaluator:
    """Test retriever evaluation"""

    def test_evaluate_retriever_with_upstream_chunker(self):
        """Test retriever evaluation with provided upstream chunker"""
        from autorag.components.chunkers.fixed_size import FixedSizeChunker

        # Mock metric collector
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_retrieval_metrics = Mock(return_value={
            'time': 0.2,
            'docs_retrieved': 5,
            'avg_relevance': 0.85
        })
        mock_metrics.compute_quality_score = Mock(return_value=0.8)

        # Upstream chunker
        chunker = FixedSizeChunker({'chunk_size': 256, 'overlap': 50})

        # Test data
        test_data = {
            'documents': ["Document about machine learning. " * 10] * 5,
            'queries': [
                {'query': 'What is machine learning?'},
                {'query': 'How does AI work?'}
            ]
        }

        # Create evaluator
        evaluator = ComponentEvaluator(
            component_type='retriever',
            test_data=test_data,
            metric_collector=mock_metrics,
            upstream_components={'chunker': chunker}
        )

        # Evaluate config
        config = {
            'retrieval_method': 'sparse',  # Use sparse for simplicity (no embedder needed)
            'retrieval_top_k': 5
        }

        score = evaluator.evaluate(config)

        # Verify
        assert 0.0 <= score <= 1.0

    def test_evaluate_retriever_without_upstream(self):
        """Test retriever falls back to default chunker when none provided"""
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_retrieval_metrics = Mock(return_value={
            'time': 0.2,
            'avg_relevance': 0.7
        })
        mock_metrics.compute_quality_score = Mock(return_value=0.7)

        test_data = {
            'documents': ["Test document. " * 10],
            'queries': [{'query': 'test query'}]
        }

        # No upstream components provided
        evaluator = ComponentEvaluator('retriever', test_data, mock_metrics)

        config = {'retrieval_method': 'sparse', 'retrieval_top_k': 3}
        score = evaluator.evaluate(config)

        assert 0.0 <= score <= 1.0

    def test_evaluate_retriever_handles_errors(self):
        """Test retriever evaluation handles errors gracefully"""
        mock_metrics = Mock(spec=ComponentMetrics)

        test_data = {'documents': [], 'queries': []}

        evaluator = ComponentEvaluator('retriever', test_data, mock_metrics)

        score = evaluator.evaluate({'retrieval_method': 'dense'})

        # Should return 0.0 on error (no documents to index)
        assert score == 0.0


class TestGeneratorEvaluator:
    """Test generator evaluation"""

    def test_evaluate_generator_with_upstream_components(self):
        """Test generator evaluation with chunker and retriever"""
        from autorag.components.chunkers.fixed_size import FixedSizeChunker
        from autorag.components.retrievers.bm25 import BM25Retriever

        # Mock metric collector
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_generation_metrics = Mock(return_value={
            'time': 1.0,
            'answer_length': 20,
            'answer_relevance': 0.9,
            'accuracy': 0.85
        })
        mock_metrics.compute_quality_score = Mock(return_value=0.85)

        # Upstream components
        chunker = FixedSizeChunker({'chunk_size': 256, 'overlap': 50})
        retriever = BM25Retriever({'k1': 1.2, 'b': 0.75})

        # Test data with ground truth
        test_data = {
            'documents': [
                "Machine learning is a subset of AI. " * 5,
                "Deep learning uses neural networks. " * 5
            ],
            'queries': [
                {
                    'query': 'What is machine learning?',
                    'answer': 'Machine learning is a subset of artificial intelligence'
                }
            ]
        }

        # Create evaluator
        evaluator = ComponentEvaluator(
            component_type='generator',
            test_data=test_data,
            metric_collector=mock_metrics,
            upstream_components={
                'chunker': chunker,
                'retriever': retriever
            }
        )

        # Evaluate config
        config = {
            'use_real_api': False,  # Use mock generator
            'temperature': 0.7
        }

        score = evaluator.evaluate(config)

        # Verify
        assert 0.0 <= score <= 1.0
        assert score == 0.85

    def test_evaluate_generator_without_upstream(self):
        """Test generator falls back to defaults when no upstream provided"""
        mock_metrics = Mock(spec=ComponentMetrics)
        mock_metrics.compute_generation_metrics = Mock(return_value={
            'time': 1.0,
            'accuracy': 0.7
        })
        mock_metrics.compute_quality_score = Mock(return_value=0.7)

        test_data = {
            'documents': ["Test doc. " * 10],
            'queries': [{'query': 'test', 'answer': 'expected answer'}]
        }

        evaluator = ComponentEvaluator('generator', test_data, mock_metrics)

        config = {'use_real_api': False, 'temperature': 0.5}
        score = evaluator.evaluate(config)

        assert 0.0 <= score <= 1.0

    def test_evaluate_generator_handles_errors(self):
        """Test generator evaluation handles errors gracefully"""
        mock_metrics = Mock(spec=ComponentMetrics)

        test_data = {'documents': [], 'queries': []}

        evaluator = ComponentEvaluator('generator', test_data, mock_metrics)

        score = evaluator.evaluate({'use_real_api': False})

        assert score == 0.0


class TestEvaluatorIntegration:
    """Integration tests with real components"""

    def test_chunker_evaluation_end_to_end(self):
        """Test complete chunker evaluation with real components"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics

        # Real metric collector
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        # Test data
        test_data = {
            'documents': [
                "Machine learning is a fascinating field. " * 30,
                "Artificial intelligence powers modern applications. " * 30
            ],
            'queries': []
        }

        # Create evaluator
        evaluator = ComponentEvaluator('chunker', test_data, metric_collector)

        # Evaluate different configs
        config1 = {'chunking_strategy': 'fixed', 'chunk_size': 128, 'overlap': 20}
        config2 = {'chunking_strategy': 'fixed', 'chunk_size': 512, 'overlap': 50}

        score1 = evaluator.evaluate(config1)
        score2 = evaluator.evaluate(config2)

        # Both should return valid scores
        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0

        # Scores should be different (different configurations)
        # (May be same by chance, but unlikely)
        assert score1 > 0.0 or score2 > 0.0

    def test_retriever_evaluation_end_to_end(self):
        """Test complete retriever evaluation with real components"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics
        from autorag.components.chunkers.fixed_size import FixedSizeChunker

        # Real metric collector
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        # Fixed upstream chunker
        chunker = FixedSizeChunker({'chunk_size': 256, 'overlap': 50})

        # Test data
        test_data = {
            'documents': [
                "Python is a programming language. " * 10,
                "JavaScript is used for web development. " * 10,
                "Machine learning requires data. " * 10
            ],
            'queries': [
                {'query': 'What is Python?'},
                {'query': 'Tell me about machine learning'}
            ]
        }

        # Create evaluator
        evaluator = ComponentEvaluator(
            'retriever',
            test_data,
            metric_collector,
            upstream_components={'chunker': chunker}
        )

        # Evaluate config (use sparse for speed)
        config = {'retrieval_method': 'sparse', 'retrieval_top_k': 3}
        score = evaluator.evaluate(config)

        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should have some quality

    def test_generator_evaluation_end_to_end(self):
        """Test complete generator evaluation with real components"""
        from autorag.evaluation.semantic_metrics import SemanticMetrics
        from autorag.components.chunkers.fixed_size import FixedSizeChunker
        from autorag.components.retrievers.bm25 import BM25Retriever

        # Real metric collector
        semantic_eval = SemanticMetrics(model_name='all-MiniLM-L6-v2')
        metric_collector = ComponentMetrics(semantic_evaluator=semantic_eval)

        # Fixed upstream components
        chunker = FixedSizeChunker({'chunk_size': 256, 'overlap': 50})
        retriever = BM25Retriever({'k1': 1.2, 'b': 0.75})

        # Test data with ground truth
        test_data = {
            'documents': [
                "Python is a high-level programming language known for readability. " * 5,
                "Machine learning is a branch of AI that learns from data. " * 5
            ],
            'queries': [
                {
                    'query': 'What is Python?',
                    'answer': 'Python is a programming language'
                }
            ]
        }

        # Create evaluator
        evaluator = ComponentEvaluator(
            'generator',
            test_data,
            metric_collector,
            upstream_components={
                'chunker': chunker,
                'retriever': retriever
            }
        )

        # Evaluate config (mock generator)
        config = {'use_real_api': False, 'temperature': 0.5}
        score = evaluator.evaluate(config)

        assert 0.0 <= score <= 1.0
        assert score > 0.0


class TestInvalidComponentType:
    """Test handling of invalid component types"""

    def test_invalid_component_type_raises_error(self):
        """Test that invalid component type raises error"""
        mock_metrics = Mock(spec=ComponentMetrics)
        test_data = {'documents': [], 'queries': []}

        evaluator = ComponentEvaluator('invalid_type', test_data, mock_metrics)

        with pytest.raises(ValueError):
            evaluator.evaluate({})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])