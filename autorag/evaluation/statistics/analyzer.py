"""Statistical analysis framework for RAG evaluation"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)


@dataclass
class StatisticalTest:
    """Results from a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None


@dataclass
class ComparisonResult:
    """Results from comparing two configurations"""
    config_a: str
    config_b: str
    metrics: Dict[str, StatisticalTest]
    winner: Optional[str] = None
    confidence: float = 0.0
    summary: Optional[str] = None


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for RAG evaluation"""

    def __init__(self, confidence_level: float = 0.95, min_samples: int = 5):
        """
        Initialize statistical analyzer

        Args:
            confidence_level: Confidence level for significance tests (default 0.95)
            min_samples: Minimum samples required for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.min_samples = min_samples

    def compare_configurations(self,
                                results_a: List[Dict],
                                results_b: List[Dict],
                                config_a_name: str = "Config A",
                                config_b_name: str = "Config B",
                                metrics_to_compare: Optional[List[str]] = None) -> ComparisonResult:
        """
        Statistically compare two configurations

        Args:
            results_a: Results from configuration A
            results_b: Results from configuration B
            config_a_name: Name of configuration A
            config_b_name: Name of configuration B
            metrics_to_compare: Specific metrics to compare (None = all)

        Returns:
            Comprehensive comparison results
        """
        if len(results_a) < self.min_samples or len(results_b) < self.min_samples:
            logger.warning(f"Insufficient samples for statistical comparison "
                           f"(A: {len(results_a)}, B: {len(results_b)})")

        # Extract metrics
        metrics_a = self._extract_metrics(results_a)
        metrics_b = self._extract_metrics(results_b)

        # Determine which metrics to compare
        if metrics_to_compare:
            metric_names = metrics_to_compare
        else:
            metric_names = set(metrics_a.keys()) & set(metrics_b.keys())

        # Run statistical tests for each metric
        test_results = {}
        for metric in metric_names:
            if metric in metrics_a and metric in metrics_b:
                values_a = metrics_a[metric]
                values_b = metrics_b[metric]

                if len(values_a) >= 2 and len(values_b) >= 2:
                    test_results[metric] = self._run_comparison_tests(
                        values_a, values_b, metric
                    )

        # Determine overall winner
        winner, confidence = self._determine_winner(test_results, config_a_name, config_b_name)

        # Generate summary
        summary = self._generate_comparison_summary(
            test_results, config_a_name, config_b_name, winner, confidence
        )

        return ComparisonResult(
            config_a=config_a_name,
            config_b=config_b_name,
            metrics=test_results,
            winner=winner,
            confidence=confidence,
            summary=summary
        )

    def _run_comparison_tests(self,
                               values_a: np.ndarray,
                               values_b: np.ndarray,
                               metric_name: str) -> StatisticalTest:
        """Run comprehensive statistical tests for a metric"""

        # Paired t-test if same size, otherwise independent t-test
        if len(values_a) == len(values_b):
            statistic, p_value = stats.ttest_rel(values_a, values_b)
            test_name = "Paired t-test"
        else:
            statistic, p_value = stats.ttest_ind(values_a, values_b)
            test_name = "Independent t-test"

        # Calculate effect size (Cohen's d)
        effect_size = self.calculate_cohens_d(values_a, values_b)

        # Calculate confidence interval for the difference
        ci = self.calculate_confidence_interval_difference(values_a, values_b)

        # Determine if significant
        significant = p_value < self.alpha

        # Generate interpretation
        interpretation = self._interpret_results(
            metric_name, statistic, p_value, effect_size, significant
        )

        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std

        return d

    def interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def calculate_confidence_interval_difference(self,
                                                  group1: np.ndarray,
                                                  group2: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for the difference between groups"""
        diff = np.mean(group1) - np.mean(group2)

        # Standard error of the difference
        se = np.sqrt(np.var(group1, ddof=1) / len(group1) +
                     np.var(group2, ddof=1) / len(group2))

        # Critical value
        df = len(group1) + len(group2) - 2
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)

        # Confidence interval
        margin = t_critical * se
        ci_lower = diff - margin
        ci_upper = diff + margin

        return (ci_lower, ci_upper)

    def calculate_required_sample_size(self,
                                        effect_size: float = 0.5,
                                        power: float = 0.8) -> int:
        """Calculate required sample size for desired power"""
        from statsmodels.stats.power import TTestPower

        analysis = TTestPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=self.alpha
        )

        return int(np.ceil(sample_size))

    def run_anova(self, groups: Dict[str, List[float]], metric_name: str = "metric") -> Dict:
        """Run one-way ANOVA for multiple groups"""
        group_values = list(groups.values())

        if len(group_values) < 2:
            return {"error": "Need at least 2 groups for ANOVA"}

        # Run ANOVA
        f_stat, p_value = stats.f_oneway(*group_values)

        result = {
            "test": "One-way ANOVA",
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "groups": list(groups.keys()),
            "interpretation": self._interpret_anova(p_value, len(groups))
        }

        # If significant, run post-hoc tests
        if result["significant"] and len(groups) > 2:
            result["post_hoc"] = self._run_post_hoc_tests(groups)

        return result

    def _run_post_hoc_tests(self, groups: Dict[str, List[float]]) -> Dict:
        """Run Bonferroni-corrected pairwise comparisons"""
        group_names = list(groups.keys())
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        adjusted_alpha = self.alpha / n_comparisons  # Bonferroni correction

        results = {}
        for i, name1 in enumerate(group_names):
            for name2 in group_names[i + 1:]:
                _, p_value = stats.ttest_ind(groups[name1], groups[name2])
                comparison_key = f"{name1} vs {name2}"
                results[comparison_key] = {
                    "p_value": p_value,
                    "adjusted_alpha": adjusted_alpha,
                    "significant": p_value < adjusted_alpha
                }

        return results

    def calculate_variance_components(self, results: List[Dict]) -> Dict:
        """Analyze variance components in results"""
        metrics = self._extract_metrics(results)
        variance_analysis = {}

        for metric_name, values in metrics.items():
            if len(values) > 1:
                variance_analysis[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1),
                    "variance": np.var(values, ddof=1),
                    "cv": np.std(values, ddof=1) / np.mean(values) if np.mean(values) != 0 else 0,
                    "min": np.min(values),
                    "max": np.max(values),
                    "q25": np.percentile(values, 25),
                    "median": np.median(values),
                    "q75": np.percentile(values, 75),
                    "iqr": np.percentile(values, 75) - np.percentile(values, 25)
                }

        return variance_analysis

    def bootstrap_confidence_interval(self,
                                       data: List[float],
                                       n_bootstrap: int = 1000,
                                       statistic_func: Optional[callable] = None) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if statistic_func is None:
            statistic_func = np.mean

        data_array = np.array(data)
        bootstrap_samples = []

        # Generate bootstrap samples
        np.random.seed(42)  # For reproducibility
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_samples.append(statistic_func(sample))

        # Calculate confidence interval
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100

        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)

        return (ci_lower, ci_upper)

    def _extract_metrics(self, results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract numeric metrics from results"""
        metrics = {}

        if not results:
            return metrics

        # Find all numeric fields
        sample = results[0]
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                values = [r.get(key) for r in results if isinstance(r.get(key), (int, float))]
                if values:
                    metrics[key] = np.array(values)

        return metrics

    def _determine_winner(self,
                          test_results: Dict[str, StatisticalTest],
                          config_a: str,
                          config_b: str) -> Tuple[Optional[str], float]:
        """Determine overall winner from test results"""
        if not test_results:
            return None, 0.0

        # Count significant wins for each config
        a_wins = 0
        b_wins = 0
        total_tests = 0

        for metric, test in test_results.items():
            if test.significant:
                total_tests += 1
                # Positive statistic means A > B
                if test.statistic > 0:
                    a_wins += 1
                else:
                    b_wins += 1

        if total_tests == 0:
            return None, 0.0

        # Determine winner and confidence
        if a_wins > b_wins:
            winner = config_a
            confidence = a_wins / total_tests
        elif b_wins > a_wins:
            winner = config_b
            confidence = b_wins / total_tests
        else:
            winner = None
            confidence = 0.0

        return winner, confidence

    def _interpret_results(self,
                            metric_name: str,
                            statistic: float,
                            p_value: float,
                            effect_size: float,
                            significant: bool) -> str:
        """Generate human-readable interpretation of results"""
        effect_interpretation = self.interpret_effect_size(effect_size)

        if significant:
            direction = "higher" if statistic > 0 else "lower"
            interpretation = (
                f"The difference in {metric_name} is statistically significant "
                f"(p={p_value:.4f}). The first configuration has {direction} values "
                f"with a {effect_interpretation} effect size (d={effect_size:.3f})."
            )
        else:
            interpretation = (
                f"No significant difference in {metric_name} was found "
                f"(p={p_value:.4f}). The effect size is {effect_interpretation} "
                f"(d={effect_size:.3f})."
            )

        return interpretation

    def _interpret_anova(self, p_value: float, n_groups: int) -> str:
        """Interpret ANOVA results"""
        if p_value < self.alpha:
            return (f"Significant differences exist among the {n_groups} groups "
                    f"(p={p_value:.4f}). Post-hoc tests can identify specific differences.")
        else:
            return (f"No significant differences found among the {n_groups} groups "
                    f"(p={p_value:.4f}).")

    def _generate_comparison_summary(self,
                                      test_results: Dict[str, StatisticalTest],
                                      config_a: str,
                                      config_b: str,
                                      winner: Optional[str],
                                      confidence: float) -> str:
        """Generate summary of comparison results"""
        lines = [f"Statistical Comparison: {config_a} vs {config_b}"]
        lines.append("=" * 60)

        if winner:
            lines.append(f"\nOverall Winner: {winner} (Confidence: {confidence:.1%})")
        else:
            lines.append("\nNo clear winner identified")

        lines.append("\nDetailed Results:")
        for metric, test in test_results.items():
            lines.append(f"\n{metric}:")
            lines.append(f"  Test: {test.test_name}")
            lines.append(f"  P-value: {test.p_value:.4f}")
            lines.append(f"  Significant: {'Yes' if test.significant else 'No'}")
            if test.effect_size is not None:
                lines.append(f"  Effect size: {test.effect_size:.3f} ({self.interpret_effect_size(test.effect_size)})")
            if test.confidence_interval:
                lines.append(f"  CI: [{test.confidence_interval[0]:.3f}, {test.confidence_interval[1]:.3f}]")

        return "\n".join(lines)