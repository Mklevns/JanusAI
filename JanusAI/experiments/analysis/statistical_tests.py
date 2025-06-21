"""
Statistical Tests for Experiment Analysis
=========================================

Provides common statistical tests for analyzing experiment results,
comparing performance across different runs, or assessing significance.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
from unittest.mock import patch


class StatisticalAnalyzer:
    """
    A utility class for performing common statistical tests on experiment data.
    """

    def __init__(self):
        pass

    def t_test_independent(self,
                           data1: Union[List[float], np.ndarray],
                           data2: Union[List[float], np.ndarray],
                           equal_var: bool = False # Welch's t-test if False
                          ) -> Dict[str, float]:
        """
        Performs an independent two-sample t-test (Student's or Welch's).

        Args:
            data1: Numerical data from the first group.
            data2: Numerical data from the second group.
            equal_var: If True, assume equal population variances (Student's t-test).
                       If False (default), perform Welch's t-test (does not assume equal variances).

        Returns:
            A dictionary containing 't_statistic' and 'p_value'.
        """
        data1_arr = np.asarray(data1)
        data2_arr = np.asarray(data2)
        if data1_arr.size == 0 or data2_arr.size == 0:
            raise ValueError("Both data sets must not be empty.")

        t_stat, p_val = stats.ttest_ind(data1_arr, data2_arr, equal_var=equal_var)
        return {"t_statistic": float(t_stat), "p_value": float(p_val)}

    def wilcoxon_signed_rank_test(self,
                                  data1: Union[List[float], np.ndarray],
                                  data2: Union[List[float], np.ndarray]
                                 ) -> Dict[str, float]:
        """
        Performs the Wilcoxon signed-rank test for paired samples.
        This is a non-parametric alternative to the paired t-test.

        Args:
            data1: Numerical data from the first measurement.
            data2: Numerical data from the second measurement (paired with data1).

        Returns:
            A dictionary containing 'statistic' and 'p_value'.
        """
        data1_arr = np.asarray(data1)
        data2_arr = np.asarray(data2)
        if len(data1_arr) != len(data2_arr):
            raise ValueError("Paired data sets must have the same length.")
        if data1_arr.size == 0:
            raise ValueError("Data sets must not be empty.")

        statistic, p_val = stats.wilcoxon(data1_arr, data2_arr)
        return {"statistic": float(statistic), "p_value": float(p_val)}

    def kruskal_wallis_h_test(self, *data_groups: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        Performs the Kruskal-Wallis H-test for independent samples from three or more groups.
        This is a non-parametric alternative to one-way ANOVA.

        Args:
            *data_groups: Variable number of arguments, each being a list or array of numerical data for a group.

        Returns:
            A dictionary containing 'h_statistic' and 'p_value'.
        """
        if len(data_groups) < 2:
            raise ValueError("Kruskal-Wallis test requires at least two groups.")

        processed_groups = []
        for group in data_groups:
            arr_group = np.asarray(group)
            if arr_group.size == 0:
                raise ValueError("All data groups must not be empty.")
            processed_groups.append(arr_group)

        h_stat, p_val = stats.kruskal(*processed_groups)
        return {"h_statistic": float(h_stat), "p_value": float(p_val)}

    def calculate_effect_size(self,
                              mean1: float, std1: float, n1: int,
                              mean2: float, std2: float, n2: int,
                              method: str = "cohens_d"
                             ) -> Optional[float]:
        """
        Calculates effect size (e.g., Cohen's d).

        Args:
            mean1, std1, n1: Mean, standard deviation, and sample size for group 1.
            mean2, std2, n2: Mean, standard deviation, and sample size for group 2.
            method: The effect size method ('cohens_d').

        Returns:
            The calculated effect size, or None if method is not supported.
        """
        if method == "cohens_d":
            if n1 < 2 or n2 < 2: # Avoid division by zero if n1+n2-2 = 0
                 # Handle cases with insufficient data points for pooled_std calculation
                if std1 == 0 and std2 == 0 : return 0.0 # Or handle as appropriate (e.g. raise error, return NaN)
                # If one std is zero, use the other? Or require both to be non-zero?
                # This depends on desired behavior for edge cases.
                # For now, if pooled_std would be zero due to n1/n2 < 2, and stds are non-zero, this might be an issue
                # Let's assume if n1 or n2 is 1, pooled_std is not well-defined in the typical sense.
                # A simple approach: if pooled_std would be zero or based on too little data,
                # fall back to a simpler difference or indicate issue.
                # For now, returning 0.0 if pooled_std is zero.
                pass


            # Pooled standard deviation
            # Ensure n1 + n2 - 2 is not zero
            if (n1 + n2 - 2) == 0:
                if std1 == std2 : # if both stds are same (e.g. 0), then cohens_d is 0 or undefined.
                    return 0.0 # Or np.nan, or raise error
                else: # Cannot calculate pooled_std if denominator is 0 and stds differ
                    # This case (e.g. n1=1, n2=1) should be handled based on statistical best practices.
                    # For now, let's return np.nan to indicate it's problematic.
                    return np.nan


            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            if pooled_std == 0: return 0.0 # Avoid division by zero
            cohens_d = (mean1 - mean2) / pooled_std
            return float(cohens_d)
        return None


if __name__ == "__main__":
    print("--- Testing StatisticalAnalyzer ---")
    analyzer = StatisticalAnalyzer()

    # --- Test T-test (Independent Samples) ---
    print("\n--- Independent Samples T-test (Welch's) ---")
    np.random.seed(42)
    group_a = np.random.normal(loc=10, scale=2, size=30)
    group_b = np.random.normal(loc=11, scale=3, size=25)

    try:
        ttest_results = analyzer.t_test_independent(group_a, group_b, equal_var=False)
        print(f"Group A Mean: {np.mean(group_a):.2f}, Std: {np.std(group_a):.2f}")
        print(f"Group B Mean: {np.mean(group_b):.2f}, Std: {np.std(group_b):.2f}")
        print("T-test results:", ttest_results)
        assert isinstance(ttest_results['t_statistic'], float)
        assert isinstance(ttest_results['p_value'], float)
        # Expected p-value to be moderately low if means are different
        assert ttest_results['p_value'] < 0.2
    except ValueError as e:
        print(f"Error during t-test: {e}")

    # --- Test Wilcoxon Signed-Rank Test (Paired Samples) ---
    print("\n--- Wilcoxon Signed-Rank Test ---")
    before_treatment = np.random.normal(loc=50, scale=5, size=20)
    after_treatment = before_treatment + np.random.normal(loc=2, scale=1, size=20) # Expected improvement

    try:
        wilcoxon_results = analyzer.wilcoxon_signed_rank_test(before_treatment, after_treatment)
        print(f"Before Mean: {np.mean(before_treatment):.2f}, After Mean: {np.mean(after_treatment):.2f}")
        print("Wilcoxon results:", wilcoxon_results)
        assert isinstance(wilcoxon_results['statistic'], float)
        assert isinstance(wilcoxon_results['p_value'], float)
        # Expected low p-value if there's a significant difference
        assert wilcoxon_results['p_value'] < 0.05 # Assuming a real effect
    except ValueError as e:
        print(f"Error during Wilcoxon test: {e}")


    # --- Test Kruskal-Wallis H-test (Multiple Independent Groups) ---
    print("\n--- Kruskal-Wallis H-test ---")
    group1 = np.random.normal(loc=10, scale=2, size=20)
    group2 = np.random.normal(loc=12, scale=2, size=20)
    group3 = np.random.normal(loc=10, scale=2, size=20)

    try:
        kruskal_results = analyzer.kruskal_wallis_h_test(group1, group2, group3)
        print(f"Group 1 Mean: {np.mean(group1):.2f}")
        print(f"Group 2 Mean: {np.mean(group2):.2f}")
        print(f"Group 3 Mean: {np.mean(group3):.2f}")
        print("Kruskal-Wallis results:", kruskal_results)
        assert isinstance(kruskal_results['h_statistic'], float)
        assert isinstance(kruskal_results['p_value'], float)
        # p-value might be low if group2 is significantly different
        assert kruskal_results['p_value'] < 0.1 # Some difference expected
    except ValueError as e:
        print(f"Error during Kruskal-Wallis test: {e}")

    # --- Test Effect Size (Cohen's d) ---
    print("\n--- Effect Size (Cohen's d) ---")
    mean1, std1, n1 = 10, 2, 30
    mean2, std2, n2 = 11.5, 2.5, 25
    cohens_d = analyzer.calculate_effect_size(mean1, std1, n1, mean2, std2, n2)
    print(f"Cohen's d: {cohens_d:.4f}")
    assert isinstance(cohens_d, float)
    assert cohens_d is not None and cohens_d < 0 # mean1 < mean2, so d should be negative

    # Test edge case for effect size: n1+n2-2 = 0
    mean1, std1, n1 = 10, 2, 1
    mean2, std2, n2 = 12, 2, 1
    cohens_d_edge = analyzer.calculate_effect_size(mean1, std1, n1, mean2, std2, n2)
    print(f"Cohen's d (edge case n1=1, n2=1): {cohens_d_edge}")
    assert cohens_d_edge is np.nan # Or other appropriate handling for this undefined case

    # Test edge case: pooled_std = 0
    mean1, std1, n1 = 10, 0, 5
    mean2, std2, n2 = 10, 0, 5
    cohens_d_zero_std = analyzer.calculate_effect_size(mean1, std1, n1, mean2, std2, n2)
    print(f"Cohen's d (zero std): {cohens_d_zero_std}")
    assert cohens_d_zero_std == 0.0


    print("\nAll StatisticalAnalyzer tests completed.")
