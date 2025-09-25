"""Statistical comparison and analysis for configuration optimization"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.stats import bootstrap
import pandas as pd
from loguru import logger
from dataclasses import dataclass


@dataclass
class StatisticalTest:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float = 0.95
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


class StatisticalComparison:
    """Statistical comparison utilities for configuration evaluation"""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical comparison

        Args:
            confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        logger.info(f"StatisticalComparison initialized with confidence level {confidence_level}")

    def compare_configurations(self,
                              scores_a: List[float],
                              scores_b: List[float],
                              test_type: str = "auto") -> StatisticalTest:
        """
        Compare two configurations statistically

        Args:
            scores_a: Scores from configuration A
            scores_b: Scores from configuration B
            test_type: Type of test ("paired", "independent", "auto")

        Returns:
            Statistical test result
        """
        if test_type == "auto":
            # Determine test type based on data
            test_type = "paired" if len(scores_a) == len(scores_b) else "independent"

        if test_type == "paired":
            return self.paired_t_test(scores_a, scores_b)
        else:
            return self.independent_t_test(scores_a, scores_b)

    def paired_t_test(self, scores_a: List[float], scores_b: List[float]) -> StatisticalTest:
        """
        Perform paired t-test for same test set

        Args:
            scores_a: Scores from configuration A
            scores_b: Scores from configuration B

        Returns:
            Statistical test result
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Paired t-test requires equal length score lists")

        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(scores_a, scores_b)

        # Calculate effect size (Cohen's d for paired samples)
        differences = np.array(scores_b) - np.array(scores_a)
        effect_size = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0

        # Calculate confidence interval for mean difference
        ci = self.calculate_confidence_interval(differences)

        # Interpretation
        interpretation = self._interpret_results(p_value, effect_size)

        return StatisticalTest(
            test_name="Paired t-test",
            statistic=t_statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def independent_t_test(self, scores_a: List[float], scores_b: List[float]) -> StatisticalTest:
        """
        Perform independent t-test for different test sets

        Args:
            scores_a: Scores from configuration A
            scores_b: Scores from configuration B

        Returns:
            Statistical test result
        """
        # Check for equal variances
        _, levene_p = stats.levene(scores_a, scores_b)
        equal_var = levene_p > 0.05

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=equal_var)

        # Calculate effect size (Cohen's d)
        effect_size = self.calculate_cohens_d(scores_a, scores_b)

        # Calculate confidence interval for difference in means
        mean_diff = np.mean(scores_b) - np.mean(scores_a)
        se_diff = np.sqrt(np.var(scores_a, ddof=1)/len(scores_a) +
                         np.var(scores_b, ddof=1)/len(scores_b))
        t_crit = stats.t.ppf(1 - self.alpha/2, len(scores_a) + len(scores_b) - 2)
        ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

        # Interpretation
        test_name = f"Independent t-test (equal_var={equal_var})"
        interpretation = self._interpret_results(p_value, effect_size)

        return StatisticalTest(
            test_name=test_name,
            statistic=t_statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def mann_whitney_u_test(self, scores_a: List[float], scores_b: List[float]) -> StatisticalTest:
        """
        Perform Mann-Whitney U test (non-parametric alternative)

        Args:
            scores_a: Scores from configuration A
            scores_b: Scores from configuration B

        Returns:
            Statistical test result
        """
        u_statistic, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')

        # Calculate effect size (rank biserial correlation)
        n1, n2 = len(scores_a), len(scores_b)
        effect_size = 1 - (2 * u_statistic) / (n1 * n2)

        # Bootstrap confidence interval
        ci = self.bootstrap_confidence_interval(scores_a, scores_b)

        interpretation = self._interpret_results(p_value, effect_size)

        return StatisticalTest(
            test_name="Mann-Whitney U test",
            statistic=u_statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def wilcoxon_signed_rank_test(self, scores_a: List[float], scores_b: List[float]) -> StatisticalTest:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test)

        Args:
            scores_a: Scores from configuration A
            scores_b: Scores from configuration B

        Returns:
            Statistical test result
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Wilcoxon test requires equal length score lists")

        w_statistic, p_value = stats.wilcoxon(scores_a, scores_b)

        # Calculate effect size (matched pairs rank biserial correlation)
        differences = np.array(scores_b) - np.array(scores_a)
        effect_size = self._calculate_rank_biserial(differences)

        # Bootstrap confidence interval
        ci = self.calculate_confidence_interval(differences)

        interpretation = self._interpret_results(p_value, effect_size)

        return StatisticalTest(
            test_name="Wilcoxon signed-rank test",
            statistic=w_statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def multiple_comparison_correction(self,
                                      p_values: List[float],
                                      method: str = "bonferroni") -> List[bool]:
        """
        Apply multiple comparison correction

        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ("bonferroni", "holm", "fdr")

        Returns:
            List of boolean values indicating significance after correction
        """
        n = len(p_values)

        if method == "bonferroni":
            # Bonferroni correction
            adjusted_alpha = self.alpha / n
            return [p < adjusted_alpha for p in p_values]

        elif method == "holm":
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            is_significant = np.zeros(n, dtype=bool)

            for i, p in enumerate(sorted_p):
                if p < self.alpha / (n - i):
                    is_significant[sorted_indices[i]] = True
                else:
                    break  # Stop at first non-significant

            return is_significant.tolist()

        elif method == "fdr":
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            is_significant = np.zeros(n, dtype=bool)

            for i in range(n-1, -1, -1):
                if sorted_p[i] <= (i + 1) * self.alpha / n:
                    is_significant[sorted_indices[:i+1]] = True
                    break

            return is_significant.tolist()

        else:
            raise ValueError(f"Unknown correction method: {method}")

    def calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size

        Args:
            group1: First group of scores
            group2: Second group of scores

        Returns:
            Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std > 0:
            return (mean2 - mean1) / pooled_std
        return 0.0

    def calculate_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean

        Args:
            data: List of values

        Returns:
            (lower_bound, upper_bound)
        """
        mean = np.mean(data)
        se = stats.sem(data)
        margin = se * stats.t.ppf((1 + self.confidence_level) / 2, len(data) - 1)
        return (mean - margin, mean + margin)

    def bootstrap_confidence_interval(self,
                                     scores_a: List[float],
                                     scores_b: List[float],
                                     n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for difference in means

        Args:
            scores_a: Scores from configuration A
            scores_b: Scores from configuration B
            n_bootstrap: Number of bootstrap samples

        Returns:
            (lower_bound, upper_bound)
        """
        def mean_diff(a, b):
            return np.mean(b) - np.mean(a)

        # Bootstrap resampling
        rng = np.random.RandomState(42)
        bootstrap_diffs = []

        for _ in range(n_bootstrap):
            sample_a = rng.choice(scores_a, size=len(scores_a), replace=True)
            sample_b = rng.choice(scores_b, size=len(scores_b), replace=True)
            bootstrap_diffs.append(mean_diff(sample_a, sample_b))

        # Calculate percentile confidence interval
        lower = np.percentile(bootstrap_diffs, (1 - self.confidence_level) * 50)
        upper = np.percentile(bootstrap_diffs, 100 - (1 - self.confidence_level) * 50)

        return (lower, upper)

    def _calculate_rank_biserial(self, differences: np.ndarray) -> float:
        """Calculate rank biserial correlation for paired data"""
        positive = np.sum(differences > 0)
        negative = np.sum(differences < 0)
        total = positive + negative

        if total > 0:
            return (positive - negative) / total
        return 0.0

    def _interpret_results(self, p_value: float, effect_size: float) -> str:
        """
        Interpret statistical test results

        Args:
            p_value: P-value from test
            effect_size: Effect size measure

        Returns:
            Human-readable interpretation
        """
        significance = "statistically significant" if p_value < self.alpha else "not statistically significant"

        # Interpret effect size (Cohen's d)
        if abs(effect_size) < 0.2:
            effect_magnitude = "negligible"
        elif abs(effect_size) < 0.5:
            effect_magnitude = "small"
        elif abs(effect_size) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

        direction = "improvement" if effect_size > 0 else "degradation"

        return (f"The difference is {significance} (p={p_value:.4f}) with a "
               f"{effect_magnitude} effect size ({effect_size:.3f}), "
               f"indicating {direction} in performance.")

    def calculate_required_sample_size(self,
                                      effect_size: float,
                                      power: float = 0.8) -> int:
        """
        Calculate required sample size for desired power

        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power (default: 0.8)

        Returns:
            Required sample size per group
        """
        from statsmodels.stats.power import tt_solve_power

        try:
            n = tt_solve_power(effect_size=effect_size,
                             alpha=self.alpha,
                             power=power,
                             alternative='two-sided')
            return int(np.ceil(n))
        except:
            # Fallback approximation
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(power)
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))

    def analyze_variance(self, results_df: pd.DataFrame, target_col: str = "score") -> Dict[str, Any]:
        """
        Perform variance analysis on results

        Args:
            results_df: DataFrame with results
            target_col: Target column for analysis

        Returns:
            Variance analysis results
        """
        analysis = {
            "total_variance": results_df[target_col].var(),
            "mean": results_df[target_col].mean(),
            "std": results_df[target_col].std(),
            "cv": results_df[target_col].std() / results_df[target_col].mean()
                  if results_df[target_col].mean() > 0 else 0
        }

        # Within-configuration variance (if multiple runs per config)
        if "config_id" in results_df.columns:
            within_var = results_df.groupby("config_id")[target_col].var().mean()
            between_var = results_df.groupby("config_id")[target_col].mean().var()

            analysis["within_config_variance"] = within_var
            analysis["between_config_variance"] = between_var
            analysis["variance_ratio"] = between_var / within_var if within_var > 0 else np.inf

        return analysis

    def perform_anova(self, groups: List[List[float]]) -> StatisticalTest:
        """
        Perform one-way ANOVA for multiple groups

        Args:
            groups: List of score lists for each configuration

        Returns:
            Statistical test result
        """
        f_statistic, p_value = stats.f_oneway(*groups)

        # Calculate eta squared (effect size)
        all_scores = np.concatenate(groups)
        grand_mean = np.mean(all_scores)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_scores - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        interpretation = (f"ANOVA shows {'significant' if p_value < self.alpha else 'no significant'} "
                        f"differences between groups (F={f_statistic:.3f}, p={p_value:.4f}, "
                        f"η²={eta_squared:.3f})")

        return StatisticalTest(
            test_name="One-way ANOVA",
            statistic=f_statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=eta_squared,
            interpretation=interpretation
        )