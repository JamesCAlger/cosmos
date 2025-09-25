"""Enhanced dataset loader with stratified sampling and train/dev/test splits"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import hashlib
from ..core.document_processor import Document


class EnhancedDatasetLoader:
    """Enhanced dataset loader with stratified sampling and split management"""

    SUPPORTED_DATASETS = {
        "ms_marco": {
            "hf_name": "ms_marco",
            "config": "v2.1",
            "splits": ["train", "validation"],
            "query_field": "query",
            "passage_field": "passages",
            "answer_field": "answers"
        },
        "beir": {
            "hf_name": "BeIR/msmarco",  # Starting with MS MARCO from BEIR
            "config": None,
            "splits": ["corpus", "queries"],
            "query_field": "text",
            "passage_field": "text",
            "answer_field": None
        }
    }

    def __init__(self,
                 dataset_name: str = "ms_marco",
                 cache_dir: str = "cache/datasets",
                 seed: int = 42):
        """
        Initialize enhanced dataset loader

        Args:
            dataset_name: Name of dataset to load
            cache_dir: Directory for caching processed data
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        np.random.seed(seed)

        if dataset_name not in self.SUPPORTED_DATASETS:
            logger.warning(f"Dataset {dataset_name} not officially supported, using MS MARCO config")
            self.dataset_config = self.SUPPORTED_DATASETS["ms_marco"]
        else:
            self.dataset_config = self.SUPPORTED_DATASETS[dataset_name]

    def load_with_splits(self,
                         num_docs: int = 1000,
                         num_queries: int = 500,
                         train_ratio: float = 0.7,
                         dev_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         stratify_by: Optional[str] = None,
                         use_cache: bool = True) -> Dict[str, Any]:
        """
        Load dataset with train/dev/test splits

        Args:
            num_docs: Total number of documents to load
            num_queries: Total number of queries to load
            train_ratio: Proportion for training set
            dev_ratio: Proportion for development set
            test_ratio: Proportion for test set
            stratify_by: Field to stratify by (e.g., "query_length", "has_answer")
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary with train/dev/test splits
        """
        # Validate ratios
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"

        # Check cache
        cache_key = self._generate_cache_key(num_docs, num_queries, train_ratio, dev_ratio, test_ratio)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached dataset from {cache_file}")
            with open(cache_file, "r") as f:
                return json.load(f)

        # Load full dataset
        logger.info(f"Loading {self.dataset_name} dataset with splits")
        documents, queries = self._load_raw_data(num_docs, num_queries)

        # Create stratified splits
        splits = self._create_splits(
            documents,
            queries,
            train_ratio,
            dev_ratio,
            test_ratio,
            stratify_by
        )

        # Add metadata
        result = {
            "dataset_name": self.dataset_name,
            "num_documents": len(documents),
            "num_queries": len(queries),
            "splits": splits,
            "split_ratios": {
                "train": train_ratio,
                "dev": dev_ratio,
                "test": test_ratio
            },
            "metadata": self._compute_dataset_statistics(documents, queries)
        }

        # Cache result
        if use_cache:
            self._save_cache(cache_file, result)

        return result

    def load_stratified_sample(self,
                                num_samples: int,
                                stratify_by: str = "query_length",
                                strata_proportions: Optional[Dict[str, float]] = None) -> Tuple[List[Document], List[Dict]]:
        """
        Load stratified sample of data

        Args:
            num_samples: Number of samples to load
            stratify_by: Stratification criterion
            strata_proportions: Desired proportions for each stratum

        Returns:
            Tuple of documents and queries
        """
        # Load initial larger sample
        documents, queries = self._load_raw_data(num_samples * 2, num_samples * 2)

        # Compute strata
        strata = self._compute_strata(queries, stratify_by)

        # Sample from each stratum
        sampled_queries = self._stratified_sample(
            queries,
            strata,
            num_samples,
            strata_proportions
        )

        # Get corresponding documents
        doc_ids = set()
        for query in sampled_queries:
            if "relevant_docs" in query:
                doc_ids.update(query["relevant_docs"])

        sampled_docs = [doc for doc in documents if doc.metadata.get("id") in doc_ids]

        # Ensure we have enough documents
        if len(sampled_docs) < len(documents) // 2:
            sampled_docs = documents[:len(documents) // 2]

        return sampled_docs, sampled_queries

    def _load_raw_data(self, num_docs: int, num_queries: int) -> Tuple[List[Document], List[Dict]]:
        """Load raw data from the dataset"""
        try:
            config = self.dataset_config
            dataset = load_dataset(
                config["hf_name"],
                config.get("config"),
                split=config["splits"][0],
                streaming=True
            )

            documents = []
            queries = []
            seen_docs = set()

            for idx, sample in enumerate(dataset):
                if len(queries) >= num_queries and len(documents) >= num_docs:
                    break

                # Extract query
                query_text = sample.get(config["query_field"])
                if query_text and len(queries) < num_queries:
                    query_data = {
                        "id": f"q_{len(queries)}",
                        "question": query_text,
                        "metadata": {
                            "source": self.dataset_name,
                            "idx": len(queries)
                        }
                    }

                    # Add answer if available
                    if config["answer_field"] and sample.get(config["answer_field"]):
                        answers = sample[config["answer_field"]]
                        if answers:
                            query_data["ground_truth_answer"] = answers[0] if isinstance(answers, list) else answers

                    # Extract passages
                    if config["passage_field"] in sample:
                        passages = sample[config["passage_field"]]
                        if isinstance(passages, dict) and "passage_text" in passages:
                            for passage in passages["passage_text"]:
                                if passage and passage not in seen_docs and len(documents) < num_docs:
                                    doc_id = f"d_{len(documents)}"
                                    documents.append(Document(
                                        content=passage,
                                        metadata={
                                            "id": doc_id,
                                            "source": self.dataset_name,
                                            "idx": len(documents)
                                        }
                                    ))
                                    seen_docs.add(passage)

                                    # Track relevant docs for query
                                    if "relevant_docs" not in query_data:
                                        query_data["relevant_docs"] = []
                                    query_data["relevant_docs"].append(doc_id)

                    queries.append(query_data)

                if idx > 50000:  # Safety limit
                    break

            logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
            return documents, queries

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Return synthetic data as fallback
            return self._generate_synthetic_data(num_docs, num_queries)

    def _create_splits(self,
                       documents: List[Document],
                       queries: List[Dict],
                       train_ratio: float,
                       dev_ratio: float,
                       test_ratio: float,
                       stratify_by: Optional[str]) -> Dict[str, Dict]:
        """Create train/dev/test splits"""

        # Prepare stratification labels if needed
        stratify_labels = None
        if stratify_by:
            stratify_labels = self._compute_strata(queries, stratify_by)

        # First split: train vs (dev+test)
        if stratify_labels is not None:
            train_queries, temp_queries, train_labels, temp_labels = train_test_split(
                queries,
                stratify_labels,
                test_size=(dev_ratio + test_ratio),
                stratify=stratify_labels,
                random_state=self.seed
            )
        else:
            train_queries, temp_queries = train_test_split(
                queries,
                test_size=(dev_ratio + test_ratio),
                random_state=self.seed
            )
            train_labels = None
            temp_labels = None

        # Second split: dev vs test
        dev_test_ratio = test_ratio / (dev_ratio + test_ratio)
        dev_queries, test_queries = train_test_split(
            temp_queries,
            test_size=dev_test_ratio,
            stratify=temp_labels if stratify_by else None,
            random_state=self.seed + 1
        )

        # Split documents based on which are referenced in each query set
        train_docs, dev_docs, test_docs = self._split_documents(
            documents,
            train_queries,
            dev_queries,
            test_queries
        )

        return {
            "train": {
                "documents": [self._doc_to_dict(d) for d in train_docs],
                "queries": train_queries,
                "num_docs": len(train_docs),
                "num_queries": len(train_queries)
            },
            "dev": {
                "documents": [self._doc_to_dict(d) for d in dev_docs],
                "queries": dev_queries,
                "num_docs": len(dev_docs),
                "num_queries": len(dev_queries)
            },
            "test": {
                "documents": [self._doc_to_dict(d) for d in test_docs],
                "queries": test_queries,
                "num_docs": len(test_docs),
                "num_queries": len(test_queries)
            }
        }

    def _split_documents(self,
                         documents: List[Document],
                         train_queries: List[Dict],
                         dev_queries: List[Dict],
                         test_queries: List[Dict]) -> Tuple[List[Document], List[Document], List[Document]]:
        """Split documents based on query splits"""

        # Get document IDs for each split
        train_doc_ids = set()
        dev_doc_ids = set()
        test_doc_ids = set()

        for q in train_queries:
            if "relevant_docs" in q:
                train_doc_ids.update(q["relevant_docs"])

        for q in dev_queries:
            if "relevant_docs" in q:
                dev_doc_ids.update(q["relevant_docs"])

        for q in test_queries:
            if "relevant_docs" in q:
                test_doc_ids.update(q["relevant_docs"])

        # Assign documents to splits
        train_docs = []
        dev_docs = []
        test_docs = []
        unassigned_docs = []

        for doc in documents:
            doc_id = doc.metadata.get("id")
            if doc_id in train_doc_ids:
                train_docs.append(doc)
            elif doc_id in dev_doc_ids:
                dev_docs.append(doc)
            elif doc_id in test_doc_ids:
                test_docs.append(doc)
            else:
                unassigned_docs.append(doc)

        # Distribute unassigned documents proportionally
        if unassigned_docs:
            n_train = int(len(unassigned_docs) * 0.7)
            n_dev = int(len(unassigned_docs) * 0.15)

            train_docs.extend(unassigned_docs[:n_train])
            dev_docs.extend(unassigned_docs[n_train:n_train + n_dev])
            test_docs.extend(unassigned_docs[n_train + n_dev:])

        return train_docs, dev_docs, test_docs

    def _compute_strata(self, queries: List[Dict], stratify_by: str) -> List[str]:
        """Compute stratification labels for queries"""
        labels = []

        for query in queries:
            if stratify_by == "query_length":
                # Stratify by query length
                length = len(query["question"].split())
                if length < 10:
                    label = "short"
                elif length < 20:
                    label = "medium"
                else:
                    label = "long"

            elif stratify_by == "has_answer":
                # Stratify by whether query has ground truth answer
                label = "has_answer" if query.get("ground_truth_answer") else "no_answer"

            elif stratify_by == "num_relevant_docs":
                # Stratify by number of relevant documents
                num_docs = len(query.get("relevant_docs", []))
                if num_docs == 0:
                    label = "no_docs"
                elif num_docs < 3:
                    label = "few_docs"
                else:
                    label = "many_docs"

            else:
                # Default stratification
                label = "default"

            labels.append(label)

        return labels

    def _stratified_sample(self,
                            items: List[Any],
                            strata: List[str],
                            num_samples: int,
                            strata_proportions: Optional[Dict[str, float]] = None) -> List[Any]:
        """Perform stratified sampling"""

        # Group items by stratum
        strata_groups = {}
        for item, stratum in zip(items, strata):
            if stratum not in strata_groups:
                strata_groups[stratum] = []
            strata_groups[stratum].append(item)

        # Calculate samples per stratum
        if strata_proportions:
            samples_per_stratum = {
                s: int(num_samples * p)
                for s, p in strata_proportions.items()
            }
        else:
            # Proportional to stratum size
            total_items = len(items)
            samples_per_stratum = {
                s: int(num_samples * len(group) / total_items)
                for s, group in strata_groups.items()
            }

        # Sample from each stratum
        sampled = []
        for stratum, n_samples in samples_per_stratum.items():
            if stratum in strata_groups:
                group = strata_groups[stratum]
                n_samples = min(n_samples, len(group))
                sampled.extend(np.random.choice(group, n_samples, replace=False))

        return sampled

    def _compute_dataset_statistics(self, documents: List[Document], queries: List[Dict]) -> Dict:
        """Compute statistics about the dataset"""
        doc_lengths = [len(d.content.split()) for d in documents]
        query_lengths = [len(q["question"].split()) for q in queries]

        stats = {
            "documents": {
                "count": len(documents),
                "avg_length": np.mean(doc_lengths),
                "std_length": np.std(doc_lengths),
                "min_length": np.min(doc_lengths),
                "max_length": np.max(doc_lengths)
            },
            "queries": {
                "count": len(queries),
                "avg_length": np.mean(query_lengths),
                "std_length": np.std(query_lengths),
                "min_length": np.min(query_lengths),
                "max_length": np.max(query_lengths),
                "with_answers": sum(1 for q in queries if q.get("ground_truth_answer"))
            }
        }

        return stats

    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = f"{self.dataset_name}_{args}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _save_cache(self, cache_file: Path, data: Dict):
        """Save data to cache"""
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Cached dataset to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache dataset: {e}")

    def _doc_to_dict(self, doc: Document) -> Dict:
        """Convert Document to dictionary"""
        return {
            "content": doc.content,
            "metadata": doc.metadata
        }

    def _generate_synthetic_data(self, num_docs: int, num_queries: int) -> Tuple[List[Document], List[Dict]]:
        """Generate synthetic data for testing"""
        logger.warning("Generating synthetic data")

        documents = []
        for i in range(num_docs):
            content = f"This is synthetic document {i}. It contains information about topic {i % 10}."
            documents.append(Document(
                content=content,
                metadata={"id": f"d_{i}", "source": "synthetic", "idx": i}
            ))

        queries = []
        for i in range(num_queries):
            queries.append({
                "id": f"q_{i}",
                "question": f"What is topic {i % 10}?",
                "ground_truth_answer": f"Topic {i % 10} is described in document {i}.",
                "relevant_docs": [f"d_{i}"],
                "metadata": {"source": "synthetic", "idx": i}
            })

        return documents, queries