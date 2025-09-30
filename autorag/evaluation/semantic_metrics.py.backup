"""Semantic similarity-based evaluation metrics for RAG systems"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time


class SemanticMetrics:
    """Calculate semantic similarity-based metrics instead of token-level metrics"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.7,
                 batch_size: int = 32):
        """
        Initialize semantic metrics evaluator

        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Threshold for considering answers semantically equivalent
            batch_size: Batch size for encoding
        """
        logger.info(f"Initializing Semantic Metrics with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.cache = {}  # Cache for embeddings
        logger.info(f"Semantic similarity threshold: {similarity_threshold}")

    def encode_with_cache(self, texts: List[str], prefix: str = "") -> np.ndarray:
        """Encode texts with caching to avoid re-computation"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache
        for i, text in enumerate(texts):
            cache_key = f"{prefix}:{text}"
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)

        # Encode uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(uncached_texts) > 100
            )

            # Update cache and results
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                cache_key = f"{prefix}:{text}"
                self.cache[cache_key] = emb
                embeddings[idx] = emb

        return np.array(embeddings)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def semantic_similarity_batch(self, predictions: List[str],
                                 ground_truths: List[str]) -> List[float]:
        """
        Compute semantic similarity for multiple prediction-truth pairs

        Returns:
            List of similarity scores (0-1)
        """
        start_time = time.time()

        # Batch encode all texts
        logger.info(f"Encoding {len(predictions)} predictions and ground truths")
        pred_embeddings = self.encode_with_cache(predictions, prefix="pred")
        truth_embeddings = self.encode_with_cache(ground_truths, prefix="truth")

        # Compute similarities
        similarities = []
        for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
            similarity = self.compute_similarity(pred_emb, truth_emb)
            similarities.append(float(similarity))

        encoding_time = time.time() - start_time
        logger.info(f"Semantic encoding completed in {encoding_time:.2f} seconds")

        return similarities

    def classify_semantic_match(self, similarities: List[float]) -> List[bool]:
        """
        Classify predictions as correct/incorrect based on semantic similarity

        Args:
            similarities: List of similarity scores

        Returns:
            List of boolean values (True = semantically equivalent)
        """
        return [sim >= self.similarity_threshold for sim in similarities]

    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truths using semantic similarity

        Args:
            predictions: List of generated answers
            ground_truths: List of ground truth answers

        Returns:
            Dictionary with semantic-based metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(f"Number of predictions ({len(predictions)}) != ground truths ({len(ground_truths)})")

        logger.info(f"Evaluating {len(predictions)} samples with semantic metrics")

        # Compute semantic similarities
        similarities = self.semantic_similarity_batch(predictions, ground_truths)

        # Classify as correct/incorrect based on threshold
        semantic_matches = self.classify_semantic_match(similarities)

        # Calculate accuracy (what percentage are semantically equivalent)
        semantic_accuracy = sum(semantic_matches) / len(semantic_matches)

        # For precision/recall/F1, we need to define positive and negative classes
        # Let's consider: positive = answer is semantically correct
        # For this, we'll treat each answer as a binary classification

        # Create binary labels (all ground truths are "positive" examples)
        y_true = [1] * len(ground_truths)  # All ground truths are correct
        y_pred = [1 if match else 0 for match in semantic_matches]  # Predicted as correct if similar

        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        # Additional statistics
        similarities_array = np.array(similarities)

        results = {
            # Main metrics
            "semantic_accuracy": semantic_accuracy,
            "semantic_precision": float(precision),
            "semantic_recall": float(recall),
            "semantic_f1": float(f1),

            # Similarity statistics
            "similarity_mean": float(similarities_array.mean()),
            "similarity_std": float(similarities_array.std()),
            "similarity_min": float(similarities_array.min()),
            "similarity_max": float(similarities_array.max()),
            "similarity_median": float(np.median(similarities_array)),

            # Threshold info
            "similarity_threshold": self.similarity_threshold,
            "num_semantic_matches": sum(semantic_matches),
            "num_samples": len(predictions),

            # Per-sample details (for analysis)
            "per_sample_similarities": similarities,
            "per_sample_matches": semantic_matches
        }

        logger.info(f"Semantic evaluation complete:")
        logger.info(f"  Accuracy: {semantic_accuracy:.3f} ({sum(semantic_matches)}/{len(semantic_matches)} matches)")
        logger.info(f"  Mean similarity: {similarities_array.mean():.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")

        return results

    def evaluate_with_thresholds(self, predictions: List[str],
                                ground_truths: List[str],
                                thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate at multiple similarity thresholds to find optimal threshold

        Args:
            predictions: List of generated answers
            ground_truths: List of ground truth answers
            thresholds: List of thresholds to test (default: 0.5 to 0.9)

        Returns:
            Dictionary with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

        # Compute similarities once
        similarities = self.semantic_similarity_batch(predictions, ground_truths)

        results = {}
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            # Classify at this threshold
            semantic_matches = [sim >= threshold for sim in similarities]

            # Calculate metrics
            accuracy = sum(semantic_matches) / len(semantic_matches)

            y_true = [1] * len(ground_truths)
            y_pred = [1 if match else 0 for match in semantic_matches]

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )

            results[f"threshold_{threshold}"] = {
                "accuracy": accuracy,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "num_matches": sum(semantic_matches)
            }

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        results["best_threshold"] = best_threshold
        results["best_f1"] = best_f1
        results["similarities"] = similarities

        logger.info(f"Best threshold: {best_threshold} with F1: {best_f1:.3f}")

        return results