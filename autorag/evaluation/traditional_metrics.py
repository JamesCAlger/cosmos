"""Traditional evaluation metrics for RAG systems"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from loguru import logger
import re
from collections import Counter


class TraditionalMetrics:
    """Calculate traditional metrics like accuracy, precision, recall, F1"""

    def __init__(self):
        logger.info("Initialized Traditional Metrics evaluator")

    def normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison"""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def exact_match(self, pred: str, truth: str) -> bool:
        """Check if prediction exactly matches ground truth (after normalization)"""
        return self.normalize_answer(pred) == self.normalize_answer(truth)

    def token_f1(self, pred: str, truth: str) -> Dict[str, float]:
        """Calculate token-level precision, recall, and F1"""
        pred_tokens = self.normalize_answer(pred).split()
        truth_tokens = self.normalize_answer(truth).split()

        if not truth_tokens:
            # If ground truth is empty, return 1.0 if prediction is also empty
            if not pred_tokens:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            else:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if not pred_tokens:
            # If prediction is empty but ground truth is not
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Calculate token overlap
        pred_counter = Counter(pred_tokens)
        truth_counter = Counter(truth_tokens)

        # Calculate intersection
        common = sum((pred_counter & truth_counter).values())

        # Calculate precision, recall, F1
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(truth_tokens) if truth_tokens else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth answers

        Args:
            predictions: List of generated answers
            ground_truths: List of ground truth answers

        Returns:
            Dictionary with various metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(f"Number of predictions ({len(predictions)}) != ground truths ({len(ground_truths)})")

        # Calculate exact match accuracy
        exact_matches = [self.exact_match(pred, truth)
                        for pred, truth in zip(predictions, ground_truths)]
        accuracy = sum(exact_matches) / len(exact_matches)

        # Calculate token-level metrics for each pair
        token_metrics = [self.token_f1(pred, truth)
                        for pred, truth in zip(predictions, ground_truths)]

        # Aggregate token-level metrics
        avg_precision = np.mean([m["precision"] for m in token_metrics])
        avg_recall = np.mean([m["recall"] for m in token_metrics])
        avg_f1 = np.mean([m["f1"] for m in token_metrics])

        results = {
            "exact_match_accuracy": accuracy,
            "token_precision": avg_precision,
            "token_recall": avg_recall,
            "token_f1": avg_f1,
            "num_samples": len(predictions)
        }

        logger.info(f"Traditional metrics - Accuracy: {accuracy:.3f}, F1: {avg_f1:.3f}")

        return results