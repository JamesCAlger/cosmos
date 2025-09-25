"""Result management and tracking for optimization experiments"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from loguru import logger
import numpy as np
from scipy import stats


class ResultManager:
    """Manage and analyze optimization results"""

    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize result manager

        Args:
            results_dir: Directory to store results (default: "optimization_results")
        """
        self.results_dir = Path(results_dir) if results_dir else Path("optimization_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.baseline_result: Optional[Dict[str, Any]] = None

        logger.info(f"ResultManager initialized with directory: {self.results_dir}")

    def add_result(self, result: Dict[str, Any]):
        """Add a new result to the manager"""
        self.results.append(result)

        # Update best result
        if self.best_result is None or result["score"] > self.best_result["score"]:
            self.best_result = result
            logger.info(f"New best configuration found: {result['config_id']} "
                       f"with score {result['score']:.4f}")

        # Set first result as baseline
        if len(self.results) == 1:
            self.baseline_result = result

    def get_best_configuration(self) -> Dict[str, Any]:
        """Get the best performing configuration"""
        if not self.best_result:
            raise ValueError("No results available")
        return self.best_result

    def get_top_configurations(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N performing configurations"""
        sorted_results = sorted(self.results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:n]

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all results"""
        return self.results

    def get_recent_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get N most recent results"""
        return self.results[-n:] if len(self.results) >= n else self.results

    def get_evaluated_config_ids(self) -> List[str]:
        """Get list of already evaluated configuration IDs"""
        return [result["config_id"] for result in self.results]

    def compare_configurations(self, config_a_id: str, config_b_id: str) -> Dict[str, Any]:
        """
        Statistically compare two configurations

        Args:
            config_a_id: ID of first configuration
            config_b_id: ID of second configuration

        Returns:
            Comparison results with statistical tests
        """
        result_a = self._find_result_by_id(config_a_id)
        result_b = self._find_result_by_id(config_b_id)

        if not result_a or not result_b:
            raise ValueError(f"Configuration not found")

        comparison = {
            "config_a": {
                "id": config_a_id,
                "score": result_a["score"],
                "metrics": result_a["metrics"]
            },
            "config_b": {
                "id": config_b_id,
                "score": result_b["score"],
                "metrics": result_b["metrics"]
            },
            "score_difference": result_b["score"] - result_a["score"],
            "relative_improvement": ((result_b["score"] - result_a["score"]) /
                                   max(0.001, result_a["score"])) * 100,
            "cost_difference": result_b["cost"] - result_a["cost"]
        }

        # Add statistical significance if we have multiple samples
        if "sample_scores" in result_a and "sample_scores" in result_b:
            t_stat, p_value = stats.ttest_ind(
                result_a["sample_scores"],
                result_b["sample_scores"]
            )
            comparison["statistical_test"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": p_value < 0.05,
                "cohens_d": self._calculate_cohens_d(
                    result_a["sample_scores"],
                    result_b["sample_scores"]
                )
            }

        return comparison

    def _find_result_by_id(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Find result by configuration ID"""
        for result in self.results:
            if result["config_id"] == config_id:
                return result
        return None

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        if pooled_std > 0:
            return (np.mean(group2) - np.mean(group1)) / pooled_std
        return 0.0

    def save_results(self, filename: Optional[str] = None):
        """Save all results to file"""
        if not filename:
            filename = f"results_{datetime.now():%Y%m%d_%H%M%S}.json"

        filepath = self.results_dir / filename

        data = {
            "results": self.results,
            "best_result": self.best_result,
            "baseline_result": self.baseline_result,
            "summary": self.get_summary(),
            "timestamp": datetime.now().isoformat()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def load_results(self, filename: str):
        """Load results from file"""
        filepath = self.results_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.results = data["results"]
        self.best_result = data.get("best_result")
        self.baseline_result = data.get("baseline_result")

        logger.info(f"Loaded {len(self.results)} results from {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all results"""
        if not self.results:
            return {}

        scores = [r["score"] for r in self.results]
        costs = [r["cost"] for r in self.results]
        times = [r["evaluation_time"] for r in self.results]

        summary = {
            "total_configurations": len(self.results),
            "best_score": max(scores),
            "worst_score": min(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "median_score": np.median(scores),
            "total_cost": sum(costs),
            "mean_cost_per_config": np.mean(costs),
            "total_time": sum(times),
            "mean_time_per_config": np.mean(times),
            "improvement_over_baseline": None
        }

        if self.baseline_result:
            improvement = ((self.best_result["score"] - self.baseline_result["score"]) /
                         max(0.001, self.baseline_result["score"])) * 100
            summary["improvement_over_baseline"] = improvement

        return summary

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export results to pandas DataFrame for analysis"""
        if not self.results:
            return pd.DataFrame()

        # Flatten results for DataFrame
        flattened = []
        for result in self.results:
            row = {
                "config_id": result["config_id"],
                "score": result["score"],
                "cost": result["cost"],
                "evaluation_time": result["evaluation_time"],
                "timestamp": result["timestamp"]
            }

            # Flatten parameters
            for component, params in result["parameters"].items():
                for param_name, param_value in params.items():
                    row[f"{component}_{param_name}"] = param_value

            # Add key metrics
            if "metrics" in result:
                if "ragas_metrics" in result["metrics"]:
                    for metric, value in result["metrics"]["ragas_metrics"].items():
                        row[f"ragas_{metric}"] = value
                else:
                    for metric, value in result["metrics"].items():
                        if isinstance(value, (int, float)):
                            row[metric] = value

            flattened.append(row)

        return pd.DataFrame(flattened)

    def generate_comparison_matrix(self) -> pd.DataFrame:
        """Generate pairwise comparison matrix of all configurations"""
        n = len(self.results)
        if n < 2:
            return pd.DataFrame()

        config_ids = [r["config_id"] for r in self.results]
        scores = [r["score"] for r in self.results]

        # Create comparison matrix
        matrix = pd.DataFrame(index=config_ids, columns=config_ids)

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix.iloc[i, j] = 0
                else:
                    # Relative improvement of j over i
                    improvement = ((scores[j] - scores[i]) / max(0.001, scores[i])) * 100
                    matrix.iloc[i, j] = improvement

        return matrix

    def analyze_parameter_impact(self) -> Dict[str, Dict[str, float]]:
        """Analyze the impact of each parameter on performance"""
        df = self.export_to_dataframe()

        if df.empty:
            return {}

        impact_analysis = {}

        # Get parameter columns (exclude metrics and metadata)
        param_columns = [col for col in df.columns
                        if not col.startswith(("config_id", "score", "cost",
                                              "evaluation_time", "timestamp", "ragas_"))]

        for param_col in param_columns:
            # Group by parameter value and calculate mean score
            grouped = df.groupby(param_col)["score"].agg(["mean", "std", "count"])

            impact_analysis[param_col] = {
                "values": grouped.to_dict("index"),
                "variance_explained": self._calculate_variance_explained(df, param_col, "score")
            }

        # Sort by variance explained
        impact_analysis = dict(sorted(
            impact_analysis.items(),
            key=lambda x: x[1]["variance_explained"],
            reverse=True
        ))

        return impact_analysis

    def _calculate_variance_explained(self, df: pd.DataFrame, feature: str, target: str) -> float:
        """Calculate variance explained by a feature"""
        try:
            # Convert categorical to numeric if needed
            if df[feature].dtype == 'object':
                df[feature] = pd.Categorical(df[feature]).codes

            # Calculate correlation
            correlation = df[feature].corr(df[target])
            return correlation ** 2 if not pd.isna(correlation) else 0.0
        except:
            return 0.0

    def find_pareto_optimal(self) -> List[Dict[str, Any]]:
        """Find Pareto optimal configurations (best trade-off between score and cost)"""
        if not self.results:
            return []

        scores = np.array([r["score"] for r in self.results])
        costs = np.array([r["cost"] for r in self.results])

        # We want to maximize score and minimize cost
        # Convert to minimization problem: minimize (-score, cost)
        objectives = np.column_stack((-scores, costs))

        pareto_optimal = []
        for i, point in enumerate(objectives):
            # Check if this point is dominated by any other point
            is_dominated = False
            for j, other in enumerate(objectives):
                if i != j:
                    # Check if other dominates point
                    if all(other <= point) and any(other < point):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_optimal.append(self.results[i])

        logger.info(f"Found {len(pareto_optimal)} Pareto optimal configurations")
        return pareto_optimal

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive markdown report"""
        report = []
        report.append("# Configuration Optimization Report\n")
        report.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

        # Summary
        summary = self.get_summary()
        report.append("## Summary\n")
        report.append(f"- **Total Configurations Evaluated**: {summary.get('total_configurations', 0)}\n")
        report.append(f"- **Best Score**: {summary.get('best_score', 0):.4f}\n")
        report.append(f"- **Mean Score**: {summary.get('mean_score', 0):.4f} ± {summary.get('std_score', 0):.4f}\n")
        report.append(f"- **Total Cost**: ${summary.get('total_cost', 0):.2f}\n")
        report.append(f"- **Total Time**: {summary.get('total_time', 0):.1f}s\n")

        if summary.get('improvement_over_baseline') is not None:
            report.append(f"- **Improvement over Baseline**: {summary['improvement_over_baseline']:.1f}%\n")

        # Best Configuration
        if self.best_result:
            report.append("\n## Best Configuration\n")
            report.append(f"**Config ID**: {self.best_result['config_id']}\n")
            report.append(f"**Score**: {self.best_result['score']:.4f}\n")
            report.append("\n### Parameters:\n")
            for component, params in self.best_result['parameters'].items():
                report.append(f"- **{component}**:\n")
                for param, value in params.items():
                    report.append(f"  - {param}: {value}\n")

        # Top 5 Configurations
        top_configs = self.get_top_configurations(5)
        if top_configs:
            report.append("\n## Top 5 Configurations\n")
            report.append("| Rank | Config ID | Score | Cost | Parameters |\n")
            report.append("|------|-----------|-------|------|------------|\n")
            for i, config in enumerate(top_configs, 1):
                params_str = ", ".join([
                    f"{comp}:{p.get('method', p.get('strategy', 'default'))}"
                    for comp, p in config['parameters'].items()
                ])
                report.append(f"| {i} | {config['config_id']} | "
                            f"{config['score']:.4f} | ${config['cost']:.3f} | "
                            f"{params_str} |\n")

        # Parameter Impact
        impact = self.analyze_parameter_impact()
        if impact:
            report.append("\n## Parameter Impact Analysis\n")
            report.append("| Parameter | Variance Explained | Best Value |\n")
            report.append("|-----------|-------------------|------------|\n")
            for param, data in list(impact.items())[:10]:
                best_value = max(data['values'].items(), key=lambda x: x[1]['mean'])[0]
                report.append(f"| {param} | {data['variance_explained']:.3f} | {best_value} |\n")

        # Pareto Optimal Configurations
        pareto = self.find_pareto_optimal()
        if pareto:
            report.append(f"\n## Pareto Optimal Configurations\n")
            report.append(f"Found {len(pareto)} configurations on the Pareto frontier:\n\n")
            report.append("| Config ID | Score | Cost | Score/Cost Ratio |\n")
            report.append("|-----------|-------|------|------------------|\n")
            for config in pareto[:10]:
                ratio = config['score'] / max(0.001, config['cost'])
                report.append(f"| {config['config_id']} | {config['score']:.4f} | "
                            f"${config['cost']:.3f} | {ratio:.2f} |\n")

        report_text = "".join(report)

        # Save to file if specified
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text