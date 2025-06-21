"""
Tests for experiments/analysis/statistical_tests.py
"""
import pytest
import numpy as np
from scipy import stats # For comparing results if needed, or for direct use in StatisticalAnalyzer

from experiments.analysis.statistical_tests import StatisticalAnalyzer

@pytest.fixture
def analyzer():
    return StatisticalAnalyzer()

class TestStatisticalAnalyzer:

    def test_init(self, analyzer):
        assert isinstance(analyzer, StatisticalAnalyzer)

    # --- t_test_independent Tests ---
    def test_t_test_independent_welchs_significant(self, analyzer):
        data1 = np.random.normal(loc=10, scale=2, size=30)
        data2 = np.random.normal(loc=12, scale=2, size=30) # Different means
        results = analyzer.t_test_independent(data1, data2, equal_var=False)
        assert 't_statistic' in results
        assert 'p_value' in results
        assert isinstance(results['t_statistic'], float)
        assert isinstance(results['p_value'], float)
        # For different means, p-value should ideally be low
        # This depends on variance and sample size, so check might be loose
        assert results['p_value'] < 0.05

    def test_t_test_independent_students_significant(self, analyzer):
        data1 = np.random.normal(loc=10, scale=2, size=30)
        data2 = np.random.normal(loc=12, scale=2, size=30) # Different means, same scale
        results = analyzer.t_test_independent(data1, data2, equal_var=True)
        assert results['p_value'] < 0.05

    def test_t_test_independent_no_difference(self, analyzer):
        data1 = np.random.normal(loc=10, scale=2, size=100)
        data2 = np.random.normal(loc=10, scale=2, size=100) # Same distribution
        results_welch = analyzer.t_test_independent(data1, data2, equal_var=False)
        results_student = analyzer.t_test_independent(data1, data2, equal_var=True)
        # p-value should be high, indicating no significant difference
        assert results_welch['p_value'] > 0.05
        assert results_student['p_value'] > 0.05

    def test_t_test_independent_empty_data(self, analyzer):
        with pytest.raises(ValueError, match="Both data sets must not be empty."):
            analyzer.t_test_independent([], [1,2,3])
        with pytest.raises(ValueError, match="Both data sets must not be empty."):
            analyzer.t_test_independent([1,2,3], [])

    # --- wilcoxon_signed_rank_test Tests ---
    def test_wilcoxon_significant_difference(self, analyzer):
        data1 = np.random.normal(loc=20, scale=3, size=25)
        data2 = data1 + np.random.normal(loc=1.5, scale=0.5, size=25) # data2 is generally higher
        results = analyzer.wilcoxon_signed_rank_test(data1, data2)
        assert 'statistic' in results
        assert 'p_value' in results
        assert isinstance(results['statistic'], float)
        assert isinstance(results['p_value'], float)
        assert results['p_value'] < 0.05 # Expect significance

    def test_wilcoxon_no_difference(self, analyzer):
        data1 = np.random.normal(loc=20, scale=3, size=50)
        data2 = np.random.normal(loc=20, scale=3, size=50) # No systematic difference
        # Note: Wilcoxon expects paired data. If data1 and data2 are independent here,
        # the test might not be appropriate unless we consider their differences from a common point or each other.
        # For this test, let's make data2 derived from data1 but with no systematic shift.
        data2_paired_no_diff = data1 + np.random.normal(loc=0, scale=0.1, size=50)
        results = analyzer.wilcoxon_signed_rank_test(data1, data2_paired_no_diff)
        assert results['p_value'] > 0.05

    def test_wilcoxon_mismatched_lengths(self, analyzer):
        with pytest.raises(ValueError, match="Paired data sets must have the same length."):
            analyzer.wilcoxon_signed_rank_test([1,2,3], [1,2,3,4])

    def test_wilcoxon_empty_data(self, analyzer):
        with pytest.raises(ValueError, match="Data sets must not be empty."):
            analyzer.wilcoxon_signed_rank_test([], [])

    # --- kruskal_wallis_h_test Tests ---
    def test_kruskal_wallis_significant_difference(self, analyzer):
        group1 = np.random.normal(loc=5, scale=1, size=20)
        group2 = np.random.normal(loc=7, scale=1, size=20) # Different median
        group3 = np.random.normal(loc=5, scale=1, size=20)
        results = analyzer.kruskal_wallis_h_test(group1, group2, group3)
        assert 'h_statistic' in results
        assert 'p_value' in results
        assert isinstance(results['h_statistic'], float)
        assert isinstance(results['p_value'], float)
        assert results['p_value'] < 0.05 # Expect significance due to group2

    def test_kruskal_wallis_no_difference(self, analyzer):
        group1 = np.random.normal(loc=5, scale=1, size=30)
        group2 = np.random.normal(loc=5, scale=1, size=30)
        group3 = np.random.normal(loc=5, scale=1, size=30)
        results = analyzer.kruskal_wallis_h_test(group1, group2, group3)
        assert results['p_value'] > 0.05

    def test_kruskal_wallis_too_few_groups(self, analyzer):
        with pytest.raises(ValueError, match="Kruskal-Wallis test requires at least two groups."):
            analyzer.kruskal_wallis_h_test([1,2,3])
        # Scipy's kruskal actually works with 2 groups, but the check in StatisticalAnalyzer is stricter.
        # Test with 2 groups should pass the analyzer's check but might be better with ANOVA/t-test.
        # The code says len(data_groups) < 2, so 2 groups should be fine.
        # Let's test if the wrapper allows 2 groups as per scipy.
        # The error is if len < 2, so 2 groups should pass.
        # Test if `stats.kruskal` is called.
        with patch('scipy.stats.kruskal') as mock_scipy_kruskal:
            mock_scipy_kruskal.return_value = (0.0, 1.0) # Dummy stat, pval
            analyzer.kruskal_wallis_h_test([1,2,3], [4,5,6])
            mock_scipy_kruskal.assert_called_once()


    def test_kruskal_wallis_empty_group(self, analyzer):
        with pytest.raises(ValueError, match="All data groups must not be empty."):
            analyzer.kruskal_wallis_h_test([1,2,3], [], [4,5,6])

    # --- calculate_effect_size (Cohen's d) Tests ---
    def test_cohens_d_calculation(self, analyzer):
        # Example from Wikipedia or standard source for Cohen's d
        # Mean1=20, SD1=5, N1=50
        # Mean2=25, SD2=5, N2=50
        # Pooled SD = sqrt( ((49*25) + (49*25)) / (50+50-2) ) = sqrt( (1225 + 1225) / 98 ) = sqrt(2450/98) = sqrt(25) = 5
        # Cohen's d = (20 - 25) / 5 = -5 / 5 = -1.0
        effect_size = analyzer.calculate_effect_size(mean1=20, std1=5, n1=50, mean2=25, std2=5, n2=50)
        assert abs(effect_size - (-1.0)) < 1e-6

        # Test with different stds
        # Mean1=10, SD1=2, N1=30
        # Mean2=12, SD2=3, N2=30
        # Pooled SD = sqrt( ((29*4) + (29*9)) / (30+30-2) ) = sqrt( (116 + 261) / 58 ) = sqrt(377/58) ~ sqrt(6.5) ~ 2.5495
        # Cohen's d = (10-12) / 2.5495 = -2 / 2.5495 ~ -0.784
        effect_size_diff_std = analyzer.calculate_effect_size(mean1=10, std1=2, n1=30, mean2=12, std2=3, n2=30)

        pooled_std_calc = np.sqrt(((29 * 2**2) + (29 * 3**2)) / (30 + 30 - 2))
        expected_d = (10 - 12) / pooled_std_calc
        assert abs(effect_size_diff_std - expected_d) < 1e-6


    def test_cohens_d_zero_pooled_std(self, analyzer):
        # This happens if std1 and std2 are both zero.
        effect_size = analyzer.calculate_effect_size(mean1=10, std1=0, n1=30, mean2=10, std2=0, n2=30)
        assert effect_size == 0.0 # Should return 0.0 to avoid division by zero if means are same

        effect_size_diff_mean = analyzer.calculate_effect_size(mean1=10, std1=0, n1=30, mean2=12, std2=0, n2=30)
        # If means are different but stds are 0, pooled_std is 0.
        # The current code returns 0.0 if pooled_std is 0.
        # This might hide a potentially infinite effect size if means differ.
        # Depending on interpretation, this could be np.inf or an error.
        # For now, test current behavior.
        assert effect_size_diff_mean == 0.0


    def test_calculate_effect_size_unsupported_method(self, analyzer):
        result = analyzer.calculate_effect_size(1,1,1,1,1,1, method="unknown_method")
        assert result is None
