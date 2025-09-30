# MLflow Integration for RAG Pipeline Bayesian Optimization

## Executive Summary

This document provides a comprehensive implementation guide for integrating MLflow with the auto-RAG system's Bayesian optimization framework. The integration will provide experiment tracking, model versioning, and reproducibility for RAG pipeline optimization.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RAGExperimentRunner                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   MLflow    │  │   Bayesian   │  │     Pipeline     │  │
│  │   Tracker   │◄─┤  Optimizer   ├─►│   Orchestrator   │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│         ▲                                      │            │
│         │                                      ▼            │
│  ┌──────┴──────┐                    ┌──────────────────┐  │
│  │   Metrics   │                    │    Component     │  │
│  │   Registry  │                    │     Registry     │  │
│  └─────────────┘                    └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Install Dependencies

```bash
pip install mlflow>=2.9.0
pip install optuna>=3.4.0  # For advanced Bayesian optimization
pip install scikit-optimize>=0.9.0
pip install plotly>=5.18.0  # For visualization
```

#### 1.2 Create Base Experiment Runner

**File: `autorag/optimization/experiment_runner.py`**

```python
"""
Centralized experiment runner for RAG pipeline optimization.
Integrates MLflow for tracking and Bayesian optimization for parameter search.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Type
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import numpy as np
from loguru import logger
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from ..pipeline.orchestrator import PipelineOrchestrator
from ..optimization.cache_manager import EmbeddingCacheManager
from ..components.base import Document


@dataclass
class ExperimentConfig:
    """Configuration for an optimization experiment"""
    name: str
    description: str
    base_pipeline_config: str
    search_space: Dict[str, Any]
    n_trials: int
    metrics: List[str]
    dataset_config: Dict[str, Any]
    optimization_config: Dict[str, Any] = None
    tracking_config: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load experiment configuration from YAML"""
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


class MLflowTracker:
    """MLflow integration for experiment tracking"""

    def __init__(self,
                 experiment_name: str,
                 tracking_uri: str = "file:./mlruns",
                 artifact_location: str = None,
                 tags: Dict[str, str] = None):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
            artifact_location: Location to store artifacts
            tags: Default tags for all runs
        """
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
                tags=tags or {}
            )
        else:
            experiment_id = experiment.experiment_id

        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri)
        self.default_tags = tags or {}

        logger.info(f"MLflow tracker initialized for experiment: {experiment_name}")

    def start_run(self, run_name: str = None, nested: bool = False) -> str:
        """Start a new MLflow run"""
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested,
            tags=self.default_tags
        )
        return run.info.run_id

    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters with optional prefix"""
        flat_params = self._flatten_dict(params, prefix)
        # MLflow has a limit of 500 params per call
        for i in range(0, len(flat_params), 500):
            batch = dict(list(flat_params.items())[i:i+500])
            mlflow.log_params(batch)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics for current run"""
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact file or directory"""
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict_as_json(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dictionary, f, indent=2, default=str)
            temp_path = f.name

        mlflow.log_artifact(temp_path, "configs")
        os.unlink(temp_path)

    def log_pipeline_diagram(self, pipeline_config: Dict):
        """Create and log pipeline architecture diagram"""
        try:
            import plotly.graph_objects as go

            # Create simple pipeline flow diagram
            fig = go.Figure()

            components = pipeline_config.get('pipeline', {}).get('components', [])
            for i, comp in enumerate(components):
                fig.add_trace(go.Scatter(
                    x=[i], y=[0],
                    mode='markers+text',
                    marker=dict(size=30),
                    text=comp['id'],
                    textposition="top center"
                ))

            fig.update_layout(
                title="Pipeline Architecture",
                showlegend=False,
                height=400
            )

            # Save and log
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                fig.write_html(f.name)
                mlflow.log_artifact(f.name, "visualizations")
                os.unlink(f.name)
        except ImportError:
            logger.warning("Plotly not available, skipping pipeline diagram")

    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run"""
        mlflow.end_run(status=status)

    def get_best_run(self, metric_name: str = "accuracy", ascending: bool = False) -> Dict:
        """Get best run from experiment"""
        order = "ASC" if ascending else "DESC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1
        )

        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            }
        return None

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for MLflow params"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class MetricsRegistry:
    """Registry for evaluation metrics"""

    def __init__(self):
        self.metrics = {}
        self._register_default_metrics()

    def _register_default_metrics(self):
        """Register default evaluation metrics"""
        from ..evaluation.semantic_metrics import SemanticMetrics

        # Semantic similarity metric
        semantic_eval = SemanticMetrics()
        self.register("accuracy", semantic_eval.similarity_score)

        # Latency metric
        self.register("latency", lambda start, end: end - start)

        # Token usage metric
        self.register("tokens", self._count_tokens)

    def register(self, name: str, metric_fn: Callable):
        """Register a new metric"""
        self.metrics[name] = metric_fn
        logger.debug(f"Registered metric: {name}")

    def compute(self, name: str, *args, **kwargs) -> float:
        """Compute a metric"""
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not registered")
        return self.metrics[name](*args, **kwargs)

    def compute_all(self, metric_names: List[str], *args, **kwargs) -> Dict[str, float]:
        """Compute multiple metrics"""
        results = {}
        for name in metric_names:
            try:
                results[name] = self.compute(name, *args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to compute metric {name}: {e}")
                results[name] = None
        return results

    def _count_tokens(self, text: str) -> int:
        """Simple token counting"""
        return len(text.split())


class SearchSpaceBuilder:
    """Build search spaces for Bayesian optimization"""

    @staticmethod
    def build(config: Dict[str, Any]) -> List:
        """
        Build scikit-optimize search space from configuration

        Config format:
        {
            "param_name": {
                "type": "categorical|integer|real",
                "values": [...] for categorical,
                "range": [min, max] for integer/real,
                "prior": "uniform|log-uniform" (optional)
            }
        }
        """
        space = []
        param_names = []

        for param_name, param_config in config.items():
            param_type = param_config['type']

            if param_type == 'categorical':
                dimension = Categorical(
                    param_config['values'],
                    name=param_name
                )
            elif param_type == 'integer':
                dimension = Integer(
                    *param_config['range'],
                    name=param_name,
                    prior=param_config.get('prior', 'uniform')
                )
            elif param_type == 'real':
                dimension = Real(
                    *param_config['range'],
                    name=param_name,
                    prior=param_config.get('prior', 'uniform')
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            space.append(dimension)
            param_names.append(param_name)

        return space, param_names


class RAGExperimentRunner:
    """Main experiment runner with MLflow integration"""

    def __init__(self,
                 experiment_config: ExperimentConfig,
                 use_cache: bool = True,
                 cache_dir: str = ".embedding_cache"):
        """
        Initialize experiment runner

        Args:
            experiment_config: Experiment configuration
            use_cache: Whether to use embedding cache
            cache_dir: Directory for cache storage
        """
        self.config = experiment_config

        # Initialize components
        self.metrics_registry = MetricsRegistry()
        self.search_space_builder = SearchSpaceBuilder()

        # Initialize MLflow tracking
        tracking_config = experiment_config.tracking_config or {}
        self.tracker = MLflowTracker(
            experiment_name=experiment_config.name,
            **tracking_config
        )

        # Initialize cache if enabled
        self.cache_manager = None
        if use_cache:
            self.cache_manager = EmbeddingCacheManager(
                cache_dir=cache_dir,
                max_memory_mb=1024
            )

        # Load base pipeline configuration
        self.base_pipeline_config = self._load_pipeline_config(
            experiment_config.base_pipeline_config
        )

        logger.info(f"Experiment runner initialized: {experiment_config.name}")

    def run(self) -> Dict[str, Any]:
        """
        Run the optimization experiment

        Returns:
            Dictionary with best configuration and results
        """
        logger.info(f"Starting experiment: {self.config.name}")

        # Start MLflow run for entire experiment
        run_id = self.tracker.start_run(run_name=f"{self.config.name}_main")

        # Log experiment configuration
        self.tracker.log_params({"experiment": self.config.name})
        self.tracker.log_dict_as_json(self.config.to_dict(), "experiment_config.json")

        # Build search space
        search_space, param_names = self.search_space_builder.build(
            self.config.search_space
        )

        # Load dataset
        train_data, eval_data = self._load_dataset(self.config.dataset_config)

        # Log dataset info
        self.tracker.log_params({
            "dataset.num_train_docs": len(train_data['documents']),
            "dataset.num_eval_queries": len(eval_data['queries'])
        })

        # Optimization tracking
        self.trial_count = 0
        self.best_score = float('-inf')
        self.best_config = None

        # Define objective function for Bayesian optimization
        @use_named_args(search_space)
        def objective(**params):
            return self._evaluate_trial(params, train_data, eval_data)

        # Run Bayesian optimization
        optimization_config = self.config.optimization_config or {}
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=self.config.n_trials,
            n_initial_points=optimization_config.get('n_initial_points', 10),
            acq_func=optimization_config.get('acquisition_function', 'EI'),
            random_state=optimization_config.get('random_state', 42),
            verbose=True
        )

        # Log final results
        self.tracker.log_metrics({
            "best_score": float(result.fun),
            "total_trials": self.config.n_trials
        })

        # Save optimization history
        self._save_optimization_history(result)

        # End main run
        self.tracker.end_run()

        # Get best configuration
        best_params = dict(zip(param_names, result.x))

        return {
            "best_params": best_params,
            "best_score": float(result.fun),
            "optimization_result": result,
            "best_run_id": self.best_run_id if hasattr(self, 'best_run_id') else None
        }

    def _evaluate_trial(self,
                       params: Dict[str, Any],
                       train_data: Dict,
                       eval_data: Dict) -> float:
        """
        Evaluate a single trial configuration

        Args:
            params: Trial parameters
            train_data: Training dataset
            eval_data: Evaluation dataset

        Returns:
            Objective value (lower is better for minimization)
        """
        self.trial_count += 1
        trial_name = f"trial_{self.trial_count:03d}"

        # Start nested MLflow run for this trial
        trial_run_id = self.tracker.start_run(
            run_name=trial_name,
            nested=True
        )

        try:
            # Log trial parameters
            self.tracker.log_params(params)
            self.tracker.log_params({"trial_id": self.trial_count})

            # Build pipeline with trial configuration
            pipeline = self._build_pipeline(params)

            # Index training documents
            start_time = time.time()
            self._index_documents(pipeline, train_data['documents'])
            indexing_time = time.time() - start_time

            # Evaluate on queries
            metrics = self._evaluate_queries(
                pipeline,
                eval_data['queries'],
                eval_data.get('ground_truth', {})
            )

            # Add system metrics
            metrics['indexing_time'] = indexing_time
            metrics['total_time'] = time.time() - start_time

            # Log all metrics
            self.tracker.log_metrics(metrics)

            # Calculate objective (minimize negative accuracy)
            objective_value = -metrics.get('accuracy', 0.0)

            # Update best if improved
            if objective_value < -self.best_score:
                self.best_score = -objective_value
                self.best_config = params
                self.best_run_id = trial_run_id

                # Save best pipeline configuration
                self._save_best_pipeline(pipeline, params)

            # Log trial status
            self.tracker.log_metrics({
                "trial_objective": objective_value,
                "is_best": objective_value == -self.best_score
            })

            logger.info(f"Trial {self.trial_count}: accuracy={-objective_value:.4f}")

            return objective_value

        except Exception as e:
            logger.error(f"Trial {self.trial_count} failed: {e}")
            self.tracker.log_params({"error": str(e)})
            return 0.0  # Return worst possible score

        finally:
            self.tracker.end_run()

    def _build_pipeline(self, params: Dict[str, Any]) -> PipelineOrchestrator:
        """Build pipeline with given parameters"""
        # Merge base config with trial parameters
        pipeline_config = self._merge_configs(
            self.base_pipeline_config,
            params
        )

        # Create pipeline orchestrator
        orchestrator = PipelineOrchestrator()
        orchestrator.load_config(pipeline_config)

        # Wrap with cache if enabled
        if self.cache_manager:
            self._wrap_with_cache(orchestrator)

        return orchestrator

    def _index_documents(self, pipeline: PipelineOrchestrator, documents: List[str]):
        """Index documents into pipeline"""
        # Convert to Document objects
        doc_objects = [
            Document(content=doc, doc_id=str(i))
            for i, doc in enumerate(documents)
        ]

        # Execute indexing flow
        # This depends on your pipeline implementation
        # For now, using a simplified approach
        if hasattr(pipeline, 'index'):
            pipeline.index(doc_objects)
        else:
            # Manual indexing through components
            chunker = pipeline.components.get('chunker')
            embedder = pipeline.components.get('embedder')
            vectorstore = pipeline.components.get('vectorstore')

            if all([chunker, embedder, vectorstore]):
                chunks = chunker.chunk(doc_objects)
                embeddings = embedder.embed([c.content for c in chunks])
                vectorstore.add(embeddings, chunks)

    def _evaluate_queries(self,
                         pipeline: PipelineOrchestrator,
                         queries: List[Dict[str, str]],
                         ground_truth: Dict) -> Dict[str, float]:
        """Evaluate pipeline on queries"""
        metrics_to_compute = self.config.metrics
        aggregated_metrics = {metric: [] for metric in metrics_to_compute}

        for query_data in queries:
            query_text = query_data['query']
            query_id = query_data.get('query_id', '')
            expected_answer = ground_truth.get(query_id, query_data.get('answer', ''))

            # Get pipeline response
            start_time = time.time()

            if hasattr(pipeline, 'query'):
                response = pipeline.query(query_text)
            else:
                # Manual query through components
                response = self._execute_query(pipeline, query_text)

            query_time = time.time() - start_time

            # Compute metrics for this query
            if 'accuracy' in metrics_to_compute and expected_answer:
                accuracy = self.metrics_registry.compute(
                    'accuracy',
                    response.get('answer', ''),
                    expected_answer
                )
                aggregated_metrics['accuracy'].append(accuracy)

            if 'latency' in metrics_to_compute:
                aggregated_metrics['latency'].append(query_time)

            if 'tokens' in metrics_to_compute:
                tokens = self.metrics_registry.compute(
                    'tokens',
                    response.get('answer', '')
                )
                aggregated_metrics['tokens'].append(tokens)

        # Aggregate metrics
        final_metrics = {}
        for metric_name, values in aggregated_metrics.items():
            if values:
                final_metrics[metric_name] = np.mean(values)
                final_metrics[f"{metric_name}_std"] = np.std(values)
                final_metrics[f"{metric_name}_min"] = np.min(values)
                final_metrics[f"{metric_name}_max"] = np.max(values)

        return final_metrics

    def _execute_query(self, pipeline: PipelineOrchestrator, query: str) -> Dict:
        """Execute query through pipeline components"""
        # This is a simplified implementation
        # Should be replaced with proper pipeline execution

        embedder = pipeline.components.get('embedder')
        vectorstore = pipeline.components.get('vectorstore')
        generator = pipeline.components.get('generator')

        if all([embedder, vectorstore, generator]):
            # Embed query
            query_embedding = embedder.embed_query(query)

            # Retrieve relevant chunks
            results = vectorstore.search(query_embedding, top_k=5)

            # Generate answer
            answer = generator.generate(query, results)

            return {"answer": answer, "contexts": results}

        return {"answer": "Pipeline execution failed", "contexts": []}

    def _load_dataset(self, dataset_config: Dict) -> Tuple[Dict, Dict]:
        """Load dataset for training and evaluation"""
        from datasets import load_dataset

        dataset_type = dataset_config.get('type', 'ms_marco')

        if dataset_type == 'ms_marco':
            # Load MS MARCO dataset
            dataset = load_dataset(
                'ms_marco',
                'v2.1',
                split='train',
                streaming=True
            )

            documents = []
            queries = []
            ground_truth = {}

            num_docs = dataset_config.get('num_documents', 100)
            num_queries = dataset_config.get('num_queries', 20)

            for i, item in enumerate(dataset):
                if len(documents) < num_docs:
                    for passage in item.get('passages', {}).get('passage_text', []):
                        if passage and len(passage) > 50:
                            documents.append(passage)
                            if len(documents) >= num_docs:
                                break

                if len(queries) < num_queries and item.get('query'):
                    query_id = f"q_{i}"
                    queries.append({
                        'query': item['query'],
                        'query_id': query_id,
                        'answer': item['answers'][0] if item.get('answers') else ''
                    })

                    if item.get('answers'):
                        ground_truth[query_id] = item['answers'][0]

                if len(documents) >= num_docs and len(queries) >= num_queries:
                    break

            return (
                {'documents': documents},
                {'queries': queries, 'ground_truth': ground_truth}
            )

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _load_pipeline_config(self, config_path: str) -> Dict:
        """Load base pipeline configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _merge_configs(self, base_config: Dict, params: Dict) -> Dict:
        """Merge base configuration with trial parameters"""
        import copy
        merged = copy.deepcopy(base_config)

        # Apply parameter overrides
        # This needs to map flat parameter names to nested config structure
        # Example: "chunking.chunk_size" -> config['pipeline']['components'][0]['config']['chunk_size']

        for param_name, param_value in params.items():
            self._set_nested_value(merged, param_name, param_value)

        return merged

    def _set_nested_value(self, config: Dict, path: str, value: Any):
        """Set value in nested dictionary using dot notation path"""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _wrap_with_cache(self, orchestrator: PipelineOrchestrator):
        """Wrap embedder component with cache"""
        from ..components.embedders.cached import CachedEmbedder

        if 'embedder' in orchestrator.components:
            original_embedder = orchestrator.components['embedder']
            cached_embedder = CachedEmbedder(
                embedder=original_embedder,
                cache_manager=self.cache_manager
            )
            orchestrator.components['embedder'] = cached_embedder

    def _save_best_pipeline(self, pipeline: PipelineOrchestrator, params: Dict):
        """Save best pipeline configuration"""
        import pickle
        import tempfile

        # Save pipeline configuration
        config = {
            'params': params,
            'pipeline_config': pipeline.graph.to_dict() if hasattr(pipeline, 'graph') else {}
        }

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2, default=str)
            self.tracker.log_artifact(f.name, "best_pipeline")
            os.unlink(f.name)

    def _save_optimization_history(self, result):
        """Save optimization history and visualizations"""
        try:
            from skopt.plots import plot_convergence, plot_objective
            import matplotlib.pyplot as plt
            import tempfile

            # Convergence plot
            fig = plt.figure(figsize=(10, 6))
            plot_convergence(result)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name)
                self.tracker.log_artifact(f.name, "visualizations")
                os.unlink(f.name)
            plt.close()

            # Save optimization history as JSON
            history = {
                'func_vals': result.func_vals.tolist(),
                'x_iters': [x.tolist() if hasattr(x, 'tolist') else x
                           for x in result.x_iters],
                'models': str(result.models) if hasattr(result, 'models') else None
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(history, f, indent=2)
                self.tracker.log_artifact(f.name, "optimization_history")
                os.unlink(f.name)

        except Exception as e:
            logger.warning(f"Failed to save optimization history: {e}")
```

### Phase 2: Configuration System (Week 1-2)

#### 2.1 Experiment Configuration Format

**File: `experiments/configs/example_experiment.yaml`**

```yaml
# Experiment configuration for RAG optimization
name: "rag_optimization_v1"
description: "Optimize RAG pipeline for MS MARCO dataset"

# Base pipeline configuration
base_pipeline_config: "configs/baseline_rag.yaml"

# Search space definition
search_space:
  # Chunking parameters
  chunking.strategy:
    type: categorical
    values: [fixed, semantic, sliding_window]

  chunking.chunk_size:
    type: integer
    range: [128, 512]
    prior: uniform

  chunking.overlap:
    type: integer
    range: [0, 100]

  # Retrieval parameters
  retrieval.method:
    type: categorical
    values: [bm25, dense, hybrid]

  retrieval.top_k:
    type: integer
    range: [3, 20]

  retrieval.hybrid_weight:  # Only used when method=hybrid
    type: real
    range: [0.0, 1.0]

  # Generation parameters
  generation.temperature:
    type: real
    range: [0.0, 1.0]
    prior: uniform

  generation.max_tokens:
    type: integer
    range: [50, 500]

# Number of trials for optimization
n_trials: 100

# Metrics to evaluate
metrics:
  - accuracy
  - latency
  - tokens
  - coherence

# Dataset configuration
dataset_config:
  type: ms_marco
  num_documents: 1000
  num_queries: 100
  train_test_split: 0.8

# Optimization configuration
optimization_config:
  n_initial_points: 10
  acquisition_function: EI  # Expected Improvement
  random_state: 42

# MLflow tracking configuration
tracking_config:
  tracking_uri: "http://localhost:5000"  # Or "file:./mlruns" for local
  artifact_location: "s3://my-bucket/mlflow-artifacts"  # Optional
  tags:
    team: "ml-team"
    project: "rag-optimization"
    version: "1.0.0"
```

#### 2.2 Custom Metrics Implementation

**File: `autorag/optimization/custom_metrics.py`**

```python
"""Custom metrics for RAG evaluation"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer


class RAGMetrics:
    """Collection of RAG-specific metrics"""

    def __init__(self):
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def coherence_score(self, answer: str, contexts: List[str]) -> float:
        """
        Measure how well the answer aligns with retrieved contexts
        """
        if not contexts or not answer:
            return 0.0

        # Encode answer and contexts
        answer_emb = self.semantic_model.encode(answer)
        context_embs = self.semantic_model.encode(contexts)

        # Calculate average similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([answer_emb], context_embs)[0]

        return float(np.mean(similarities))

    def diversity_score(self, contexts: List[str]) -> float:
        """
        Measure diversity of retrieved contexts (higher is better)
        """
        if len(contexts) < 2:
            return 1.0

        embeddings = self.semantic_model.encode(contexts)
        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate pairwise similarities
        sim_matrix = cosine_similarity(embeddings)

        # Get upper triangle (excluding diagonal)
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        # Diversity is inverse of average similarity
        avg_similarity = np.mean(upper_triangle)
        diversity = 1.0 - avg_similarity

        return float(diversity)

    def faithfulness_score(self, answer: str, contexts: List[str]) -> float:
        """
        Measure how faithful the answer is to the contexts (no hallucination)
        """
        if not answer or not contexts:
            return 0.0

        # Tokenize answer into sentences
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        answer_sentences = nltk.sent_tokenize(answer)
        combined_context = ' '.join(contexts)

        # Check each sentence against contexts
        sentence_scores = []
        for sentence in answer_sentences:
            sent_emb = self.semantic_model.encode(sentence)
            context_emb = self.semantic_model.encode(combined_context)

            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([sent_emb], [context_emb])[0][0]
            sentence_scores.append(similarity)

        return float(np.mean(sentence_scores)) if sentence_scores else 0.0

    def rouge_score(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, generated)

        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
        }


def register_custom_metrics(metrics_registry):
    """Register all custom metrics with the registry"""

    rag_metrics = RAGMetrics()

    # Register individual metrics
    metrics_registry.register('coherence', rag_metrics.coherence_score)
    metrics_registry.register('diversity', rag_metrics.diversity_score)
    metrics_registry.register('faithfulness', rag_metrics.faithfulness_score)

    # Register ROUGE as a composite metric
    def rouge_wrapper(generated, reference):
        scores = rag_metrics.rouge_score(generated, reference)
        return scores['rougeL_f1']  # Return F1 as main score

    metrics_registry.register('rouge', rouge_wrapper)
```

### Phase 3: MLflow UI and Analysis (Week 2)

#### 3.1 Starting MLflow Server

```bash
# Start MLflow tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

#### 3.2 Experiment Analysis Tools

**File: `autorag/optimization/experiment_analysis.py`**

```python
"""Tools for analyzing MLflow experiment results"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional


class ExperimentAnalyzer:
    """Analyze and visualize MLflow experiments"""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.client = MlflowClient(tracking_uri)

    def get_experiment_df(self, experiment_name: str) -> pd.DataFrame:
        """Get all runs from experiment as DataFrame"""
        experiment = self.client.get_experiment_by_name(experiment_name)

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        data = []
        for run in runs:
            row = {
                'run_id': run.info.run_id,
                'start_time': run.info.start_time,
                'duration': (run.info.end_time - run.info.start_time) / 1000,
                **run.data.params,
                **run.data.metrics
            }
            data.append(row)

        return pd.DataFrame(data)

    def plot_optimization_history(self, experiment_name: str, metric: str = 'accuracy'):
        """Plot optimization history"""
        df = self.get_experiment_df(experiment_name)

        # Sort by start time
        df = df.sort_values('start_time')
        df['trial'] = range(len(df))

        # Calculate cumulative best
        df['cumulative_best'] = df[metric].cummax()

        # Create plot
        fig = go.Figure()

        # Add trial scores
        fig.add_trace(go.Scatter(
            x=df['trial'],
            y=df[metric],
            mode='markers',
            name='Trial Score',
            marker=dict(size=8)
        ))

        # Add cumulative best line
        fig.add_trace(go.Scatter(
            x=df['trial'],
            y=df['cumulative_best'],
            mode='lines',
            name='Best So Far',
            line=dict(width=2, color='red')
        ))

        fig.update_layout(
            title=f'Optimization History: {metric}',
            xaxis_title='Trial',
            yaxis_title=metric.capitalize(),
            height=500
        )

        return fig

    def plot_parameter_importance(self, experiment_name: str, metric: str = 'accuracy'):
        """Plot parameter importance using correlation analysis"""
        df = self.get_experiment_df(experiment_name)

        # Get parameter columns
        param_cols = [col for col in df.columns
                     if not col.startswith('metric')
                     and col not in ['run_id', 'start_time', 'duration', metric]]

        # Calculate correlations
        correlations = {}
        for col in param_cols:
            # Handle categorical variables
            if df[col].dtype == 'object':
                # Use one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                for dummy_col in dummies.columns:
                    corr = dummies[dummy_col].corr(df[metric])
                    correlations[dummy_col] = abs(corr)
            else:
                corr = df[col].corr(df[metric])
                correlations[col] = abs(corr)

        # Sort by importance
        importance_df = pd.DataFrame(
            list(correlations.items()),
            columns=['Parameter', 'Importance']
        ).sort_values('Importance', ascending=True)

        # Create horizontal bar plot
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Parameter',
            orientation='h',
            title='Parameter Importance'
        )

        fig.update_layout(height=max(400, len(importance_df) * 30))

        return fig

    def plot_parallel_coordinates(self, experiment_name: str, metric: str = 'accuracy'):
        """Create parallel coordinates plot for parameter exploration"""
        df = self.get_experiment_df(experiment_name)

        # Select top N runs
        top_n = min(50, len(df))
        df_top = df.nlargest(top_n, metric)

        # Prepare dimensions
        dimensions = []

        for col in df_top.columns:
            if col in ['run_id', 'start_time', 'duration']:
                continue

            if df_top[col].dtype == 'object':
                # Categorical dimension
                dimensions.append(
                    dict(
                        label=col,
                        values=pd.Categorical(df_top[col]).codes,
                        ticktext=df_top[col].unique(),
                        tickvals=list(range(len(df_top[col].unique())))
                    )
                )
            else:
                # Numerical dimension
                dimensions.append(
                    dict(
                        label=col,
                        values=df_top[col],
                        range=[df_top[col].min(), df_top[col].max()]
                    )
                )

        # Create parallel coordinates plot
        fig = go.Figure(data=go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color=df_top[metric],
                colorscale='Viridis',
                showscale=True,
                cmin=df_top[metric].min(),
                cmax=df_top[metric].max()
            )
        ))

        fig.update_layout(
            title=f'Parameter Exploration (Top {top_n} Runs)',
            height=600
        )

        return fig

    def generate_report(self, experiment_name: str, output_path: str = 'report.html'):
        """Generate comprehensive HTML report"""

        # Get experiment data
        df = self.get_experiment_df(experiment_name)

        # Create subplots
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Optimization History',
                'Parameter Importance',
                'Metric Correlations',
                'Best Configurations'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'heatmap'}, {'type': 'table'}]
            ]
        )

        # Add optimization history
        df_sorted = df.sort_values('start_time')
        df_sorted['trial'] = range(len(df_sorted))

        fig.add_trace(
            go.Scatter(
                x=df_sorted['trial'],
                y=df_sorted.get('accuracy', df_sorted.columns[-1]),
                mode='markers+lines',
                name='Accuracy'
            ),
            row=1, col=1
        )

        # Add parameter importance (simplified)
        # ... (implement based on previous method)

        # Add correlation heatmap
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df[numeric_cols].corr()

        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu'
            ),
            row=2, col=1
        )

        # Add best configurations table
        best_runs = df.nlargest(5, 'accuracy')[['run_id', 'accuracy'] +
                                               [col for col in df.columns
                                                if col.startswith('param')]]

        fig.add_trace(
            go.Table(
                header=dict(values=list(best_runs.columns)),
                cells=dict(values=[best_runs[col] for col in best_runs.columns])
            ),
            row=2, col=2
        )

        fig.update_layout(height=1000, showlegend=False, title_text="Experiment Report")

        # Save report
        fig.write_html(output_path)

        return fig
```

### Phase 4: Running Experiments (Week 2)

#### 4.1 Main Execution Script

**File: `run_experiment.py`**

```python
#!/usr/bin/env python
"""
Main script to run RAG optimization experiments with MLflow tracking
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from autorag.optimization.experiment_runner import (
    RAGExperimentRunner,
    ExperimentConfig
)
from autorag.optimization.custom_metrics import register_custom_metrics
from autorag.optimization.experiment_analysis import ExperimentAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Run RAG optimization experiment')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML'
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Enable embedding cache'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='.embedding_cache',
        help='Directory for cache storage'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Generate analysis report after experiment'
    )

    args = parser.parse_args()

    # Load experiment configuration
    config = ExperimentConfig.from_yaml(args.config)

    # Initialize runner
    runner = RAGExperimentRunner(
        experiment_config=config,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir
    )

    # Register custom metrics
    register_custom_metrics(runner.metrics_registry)

    # Run experiment
    print(f"Starting experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Number of trials: {config.n_trials}")
    print("-" * 50)

    results = runner.run()

    print("-" * 50)
    print(f"Experiment completed!")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")

    # Generate analysis report if requested
    if args.analyze:
        print("\nGenerating analysis report...")
        analyzer = ExperimentAnalyzer()
        analyzer.generate_report(
            config.name,
            output_path=f"{config.name}_report.html"
        )
        print(f"Report saved to {config.name}_report.html")

        # Also generate individual plots
        fig = analyzer.plot_optimization_history(config.name)
        fig.write_html(f"{config.name}_history.html")

        fig = analyzer.plot_parameter_importance(config.name)
        fig.write_html(f"{config.name}_importance.html")

        print("Analysis complete!")


if __name__ == "__main__":
    main()
```

#### 4.2 Running the Experiment

```bash
# Start MLflow server (in separate terminal)
mlflow server --host 0.0.0.0 --port 5000

# Run experiment
python run_experiment.py \
    --config experiments/configs/rag_optimization_v1.yaml \
    --use-cache \
    --analyze

# View results in MLflow UI
# Open browser to http://localhost:5000
```

## Best Practices

### 1. Experiment Organization

- Use descriptive experiment names with versioning
- Tag experiments with team, project, and purpose
- Document search space rationale in config comments
- Keep experiment configs in version control

### 2. Metric Selection

- Always include accuracy/performance metric
- Add cost metrics (tokens, API calls)
- Include latency metrics for production readiness
- Consider domain-specific metrics

### 3. Search Space Design

- Start with wider ranges, narrow based on results
- Use log-uniform prior for parameters with large ranges
- Consider parameter interactions (conditional parameters)
- Set reasonable bounds based on domain knowledge

### 4. Optimization Strategy

- Use sufficient initial random points (10-20)
- Run multiple seeds for robustness
- Save intermediate results frequently
- Monitor convergence and stop early if plateaued

### 5. Production Deployment

```python
# Load best model from MLflow
import mlflow

# Get best run
client = MlflowClient()
experiment = client.get_experiment_by_name("rag_optimization_v1")
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)[0]

# Load configuration
best_config = best_run.data.params

# Or load registered model
model_name = "rag_pipeline"
model_version = 1
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
```

## Monitoring and Maintenance

### 1. Set Up Alerts

```python
# mlflow_alerts.py
def check_experiment_degradation(experiment_name: str, threshold: float = 0.1):
    """Alert if recent runs show performance degradation"""
    analyzer = ExperimentAnalyzer()
    df = analyzer.get_experiment_df(experiment_name)

    # Get last 10 runs
    recent = df.nlargest(10, 'start_time')
    older = df.nlargest(50, 'start_time').iloc[10:]

    recent_mean = recent['accuracy'].mean()
    older_mean = older['accuracy'].mean()

    if recent_mean < older_mean * (1 - threshold):
        send_alert(f"Performance degradation detected: {recent_mean:.3f} vs {older_mean:.3f}")
```

### 2. Regular Retraining

```yaml
# .github/workflows/retrain.yml
name: Weekly RAG Optimization
on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday
jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run optimization
        run: |
          python run_experiment.py \
            --config experiments/configs/weekly_optimization.yaml \
            --use-cache
      - name: Deploy if improved
        run: |
          python deploy_best_model.py --threshold 0.02
```

## Troubleshooting

### Common Issues

1. **MLflow connection errors**
   - Check if server is running: `curl http://localhost:5000/health`
   - Verify firewall settings
   - Check SQLite permissions for backend store

2. **Out of memory during optimization**
   - Reduce batch sizes
   - Enable gradient checkpointing
   - Use disk-based caching for embeddings

3. **Slow optimization**
   - Reduce search space dimensionality
   - Use more aggressive acquisition function
   - Parallelize trials with Ray Tune

## Conclusion

This implementation provides a robust, scalable framework for optimizing RAG pipelines with comprehensive tracking and analysis capabilities. The modular design allows easy extension with new components, metrics, and optimization strategies while maintaining reproducibility through MLflow integration.

Key benefits:
- Complete experiment tracking and versioning
- Modular and extensible architecture
- Production-ready model deployment
- Comprehensive analysis and visualization
- Cost and performance optimization

Next steps:
1. Implement the core framework
2. Run initial experiments
3. Analyze results and refine search spaces
4. Deploy best configurations to production
5. Set up continuous optimization pipeline