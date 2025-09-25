"""MS MARCO dataset loader for Week 1 minimal implementation"""

from typing import List, Dict, Any, Tuple
from datasets import load_dataset
from loguru import logger
import random
from ..core.document_processor import Document


class MSMARCOLoader:
    """Load and prepare MS MARCO dataset for evaluation"""

    def __init__(self):
        logger.info("Initializing MS MARCO loader")

    def load_subset(self, num_docs: int = 100, num_queries: int = 20,
                    include_answers: bool = False) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Load a small subset of MS MARCO for testing

        Args:
            num_docs: Number of documents to load
            num_queries: Number of queries to load
            include_answers: Whether to include ground truth answers (for traditional metrics)

        Returns:
            Tuple of (documents, queries)
        """
        logger.info(f"Loading MS MARCO subset: {num_docs} docs, {num_queries} queries, answers={include_answers}")

        try:
            # Load MS MARCO from HuggingFace datasets
            # Using the passage ranking dataset which is smaller
            dataset = load_dataset("ms_marco", "v2.1", split="train", streaming=True)

            documents = []
            queries = []
            seen_passages = set()

            # Collect samples
            for idx, sample in enumerate(dataset):
                if len(queries) >= num_queries and len(documents) >= num_docs:
                    break

                # Extract query
                if len(queries) < num_queries and sample.get("query"):
                    query_data = {
                        "question": sample["query"],
                        "passages": sample.get("passages", {})
                    }

                    # Add ground truth answer if requested
                    if include_answers and sample.get("answers"):
                        # MS MARCO has a list of answers, we'll take the first one
                        query_data["ground_truth_answer"] = sample["answers"][0] if sample["answers"] else None

                    # Add passages as documents
                    if "passage_text" in sample["passages"]:
                        for passage in sample["passages"]["passage_text"]:
                            if passage and passage not in seen_passages and len(documents) < num_docs:
                                documents.append(Document(
                                    content=passage,
                                    metadata={"source": "ms_marco", "idx": len(documents)}
                                ))
                                seen_passages.add(passage)

                    queries.append(query_data)

                # Safety limit
                if idx > 10000:
                    break

            # If we don't have enough, generate synthetic data
            if len(documents) < num_docs:
                logger.warning(f"Only found {len(documents)} documents, adding synthetic data")
                synthetic_docs = [
                    "Python is a high-level programming language known for its simplicity and readability.",
                    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
                    "The human genome contains approximately 3 billion base pairs of DNA.",
                    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
                    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                    "The Great Wall of China is over 13,000 miles long.",
                    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                    "The Pacific Ocean is the largest and deepest ocean on Earth."
                ]
                while len(documents) < num_docs:
                    doc_content = random.choice(synthetic_docs)
                    documents.append(Document(
                        content=doc_content,
                        metadata={"source": "synthetic", "idx": len(documents)}
                    ))

            if len(queries) < num_queries:
                logger.warning(f"Only found {len(queries)} queries, adding synthetic queries")
                synthetic_queries = [
                    {"question": "What is Python programming language?", "passages": {},
                     "ground_truth_answer": "Python is a high-level programming language known for its simplicity and readability."},
                    {"question": "How does machine learning work?", "passages": {},
                     "ground_truth_answer": "Machine learning works by training algorithms on data to make predictions or decisions."},
                    {"question": "Where is the Eiffel Tower located?", "passages": {},
                     "ground_truth_answer": "The Eiffel Tower is located in Paris, France."},
                    {"question": "What causes climate change?", "passages": {},
                     "ground_truth_answer": "Climate change is primarily caused by greenhouse gas emissions from human activities."},
                    {"question": "How many base pairs are in human DNA?", "passages": {},
                     "ground_truth_answer": "Human DNA contains approximately 3 billion base pairs."},
                    {"question": "How many plays did Shakespeare write?", "passages": {},
                     "ground_truth_answer": "Shakespeare wrote 37 plays."},
                    {"question": "What is the speed of light?", "passages": {},
                     "ground_truth_answer": "The speed of light is approximately 299,792,458 meters per second."},
                    {"question": "How long is the Great Wall of China?", "passages": {},
                     "ground_truth_answer": "The Great Wall of China is over 13,000 miles long."},
                    {"question": "What is photosynthesis?", "passages": {},
                     "ground_truth_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy."},
                    {"question": "Which ocean is the largest?", "passages": {},
                     "ground_truth_answer": "The Pacific Ocean is the largest ocean on Earth."}
                ]
                while len(queries) < num_queries:
                    query = random.choice(synthetic_queries).copy()
                    if not include_answers:
                        query.pop("ground_truth_answer", None)
                    queries.append(query)

            # Ensure we have the requested number
            documents = documents[:num_docs]
            queries = queries[:num_queries]

            logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
            return documents, queries

        except Exception as e:
            logger.error(f"Error loading MS MARCO: {e}")
            logger.info("Falling back to synthetic dataset")

            # Fallback: Create synthetic dataset
            documents = []
            for i in range(num_docs):
                documents.append(Document(
                    content=f"This is document {i}. It contains information about topic {i % 10}.",
                    metadata={"source": "synthetic", "idx": i}
                ))

            queries = []
            for i in range(num_queries):
                queries.append({
                    "question": f"What is topic {i % 10}?",
                    "passages": {}
                })

            return documents, queries