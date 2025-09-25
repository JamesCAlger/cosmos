"""RAGAS evaluation for Week 1 minimal implementation"""

import os
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance
)
from loguru import logger
import pandas as pd
from .traditional_metrics import TraditionalMetrics
from .semantic_metrics import SemanticMetrics


class RAGASEvaluator:
    """Minimal RAGAS evaluator with basic metrics that don't require ground truth"""

    def __init__(self, use_semantic_metrics: bool = True):
        """Initialize with basic RAGAS metrics"""
        self.metrics = [
            Faithfulness(),       # No ground truth needed
            AnswerRelevancy(),   # No ground truth needed
            ContextRelevance()   # No ground truth needed
        ]
        self.traditional_metrics = TraditionalMetrics()
        self.use_semantic_metrics = use_semantic_metrics
        if use_semantic_metrics:
            self.semantic_metrics = SemanticMetrics()
        logger.info(f"Initialized RAGAS evaluator with {len(self.metrics)} metrics (semantic={use_semantic_metrics})")

    def evaluate(self, results: List[Dict[str, Any]],
                ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline results using RAGAS and optionally traditional metrics

        Args:
            results: List of query results from RAG pipeline, each containing:
                - question: The input question
                - answer: Generated answer
                - contexts: List of retrieved contexts
            ground_truths: Optional list of ground truth answers for traditional metrics

        Returns:
            Dictionary with evaluation metrics (RAGAS + traditional if ground truth provided)
        """
        logger.info(f"Evaluating {len(results)} results with RAGAS{' and traditional metrics' if ground_truths else ''}")

        # Prepare data for RAGAS
        questions = []
        answers = []
        contexts_list = []

        for result in results:
            questions.append(result["question"])
            answers.append(result["answer"])

            # Extract context contents
            contexts = [ctx["content"] for ctx in result["contexts"]]
            contexts_list.append(contexts)

        # Create dataset for RAGAS
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list
        })

        try:
            # Run RAGAS evaluation
            ragas_results = evaluate(
                eval_dataset,
                metrics=self.metrics,
                raise_exceptions=False
            )

            # Convert to dictionary
            metrics_dict = ragas_results.to_pandas().to_dict('records')[0] if hasattr(ragas_results, 'to_pandas') else dict(ragas_results)

            logger.info("RAGAS evaluation complete")
            logger.info(f"Metrics: {metrics_dict}")

            evaluation_results = {
                "ragas_metrics": metrics_dict,
                "num_samples": len(results)
            }

            # Add traditional and semantic metrics if ground truth is provided
            if ground_truths:
                # Traditional token-level metrics
                try:
                    traditional_results = self.traditional_metrics.evaluate(answers, ground_truths)
                    evaluation_results["traditional_metrics"] = traditional_results
                    logger.info(f"Traditional metrics: {traditional_results}")
                except Exception as e:
                    logger.error(f"Error calculating traditional metrics: {e}")
                    evaluation_results["traditional_metrics_error"] = str(e)

                # Semantic similarity metrics
                if self.use_semantic_metrics:
                    try:
                        semantic_results = self.semantic_metrics.evaluate(answers, ground_truths)
                        # Keep per-sample details for analysis
                        evaluation_results["semantic_metrics"] = semantic_results
                        logger.info(f"Semantic metrics: Accuracy={semantic_results['semantic_accuracy']:.3f}, "
                                  f"F1={semantic_results['semantic_f1']:.3f}")
                    except Exception as e:
                        logger.error(f"Error calculating semantic metrics: {e}")
                        evaluation_results["semantic_metrics_error"] = str(e)

            return evaluation_results

        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {e}")
            # Try to at least compute traditional metrics if RAGAS fails
            evaluation_results = {
                "ragas_error": str(e),
                "num_samples": len(results)
            }

            if ground_truths:
                try:
                    traditional_results = self.traditional_metrics.evaluate(answers, ground_truths)
                    evaluation_results["traditional_metrics"] = traditional_results
                    logger.info(f"Traditional metrics computed despite RAGAS error: {traditional_results}")
                except Exception as e:
                    logger.error(f"Error calculating traditional metrics: {e}")
                    evaluation_results["traditional_metrics_error"] = str(e)

                if self.use_semantic_metrics:
                    try:
                        semantic_results = self.semantic_metrics.evaluate(answers, ground_truths)
                        semantic_summary = {k: v for k, v in semantic_results.items()
                                          if not k.startswith("per_sample")}
                        evaluation_results["semantic_metrics"] = semantic_summary
                        logger.info(f"Semantic metrics computed despite RAGAS error")
                    except Exception as e:
                        logger.error(f"Error calculating semantic metrics: {e}")
                        evaluation_results["semantic_metrics_error"] = str(e)

            return evaluation_results