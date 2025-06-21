"""
Tests for experiments/analysis/report_generation.py
"""
import pytest
import os
import json
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

from experiments.analysis.report_generation import ReportGenerator

@pytest.fixture
def report_generator(tmp_path):
    # Create a temporary directory for reports for each test
    report_dir = tmp_path / "reports"
    return ReportGenerator(output_dir=str(report_dir))

@pytest.fixture
def sample_experiment_results():
    return {
        'experiment_name': 'TestExp',
        'total_timesteps': 5000,
        'overall_average_fidelity': 0.92,
        'overall_correct_discovery_rate': 0.75,
        'total_elapsed_time_seconds': 123.45,
        'config': {'alpha': 0.1, 'beta': 'xyz'},
        'task_results': [
            {'task_name': 'TaskA', 'difficulty': 0.3, 'avg_fidelity': 0.95, 'correct_discovery_rate': 0.8},
            {'task_name': 'TaskB', 'difficulty': 0.7, 'avg_fidelity': 0.89, 'correct_discovery_rate': 0.7}
        ]
    }

class TestReportGenerator:

    def test_init(self, tmp_path):
        report_dir_path = tmp_path / "custom_reports"
        assert not os.path.exists(report_dir_path) # Ensure it doesn't exist before
        ReportGenerator(output_dir=str(report_dir_path))
        assert os.path.exists(report_dir_path) # Check directory is created

    def test_generate_experiment_summary_report(self, report_generator, sample_experiment_results):
        report_filename = "test_summary.md"
        report_generator.generate_experiment_summary_report(sample_experiment_results, report_filename)

        expected_report_path = os.path.join(report_generator.output_dir, report_filename)
        assert os.path.exists(expected_report_path)

        with open(expected_report_path, 'r') as f:
            content = f.read()

        assert f"# Experiment Summary Report: {sample_experiment_results['experiment_name']}" in content
        assert f"- **Total Timesteps Run:** {sample_experiment_results['total_timesteps']:,}" in content
        assert f"- **Overall Average Fidelity:** {sample_experiment_results['overall_average_fidelity']:.4f}" in content
        assert "## Configuration" in content
        assert json.dumps(sample_experiment_results['config'], indent=2) in content
        assert "## Per-Task Results" in content
        assert f"| {sample_experiment_results['task_results'][0]['task_name']} " in content

    def test_generate_experiment_summary_report_missing_keys(self, report_generator):
        minimal_results = {'experiment_name': 'MinimalExp'}
        report_generator.generate_experiment_summary_report(minimal_results, "minimal_summary.md")
        expected_report_path = os.path.join(report_generator.output_dir, "minimal_summary.md")
        assert os.path.exists(expected_report_path)
        with open(expected_report_path, 'r') as f:
            content = f.read()
        assert "Overall Average Fidelity:** N/A" in content # Check N/A for missing
        assert "## Configuration" not in content # Config section skipped
        assert "## Per-Task Results" not in content # Task results section skipped


    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    def test_plot_metric_over_time(self, mock_grid, mock_legend, mock_ylabel, mock_xlabel, mock_title, mock_plot, mock_figure, mock_close, mock_savefig, report_generator):
        metric_hist = {'metric1': [1,2,3], 'metric2': [4,5,6]}
        plot_filename = "test_metric_plot.png"

        report_generator.plot_metric_over_time(metric_hist, title="Test Plot", xlabel="X", ylabel="Y", filename=plot_filename)

        mock_figure.assert_called_once()
        assert mock_plot.call_count == 2 # Called for each metric
        mock_plot.assert_any_call([1,2,3], label='metric1')
        mock_plot.assert_any_call([4,5,6], label='metric2')
        mock_title.assert_called_once_with("Test Plot")
        mock_xlabel.assert_called_once_with("X")
        mock_ylabel.assert_called_once_with("Y")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once_with(True)
        expected_save_path = os.path.join(report_generator.output_dir, plot_filename)
        mock_savefig.assert_called_once_with(expected_save_path)
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig') # Ensure all plotting calls are properly managed
    def test_plot_metric_over_time_empty_metric(self, mock_savefig, report_generator):
        # Test that it doesn't error if a metric list is empty or history is empty
        report_generator.plot_metric_over_time({'empty_metric': []}, filename="empty_plot.png")
        # Plot should still be saved (e.g. empty plot with title/labels)
        mock_savefig.assert_called_once()

        report_generator.plot_metric_over_time({}, filename="no_metrics_plot.png")
        # mock_savefig should be called twice now in total for this test
        assert mock_savefig.call_count == 2


    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.boxplot')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.grid')
    def test_plot_comparison_boxplot(self, mock_grid, mock_ylabel, mock_title, mock_xticks, mock_boxplot, mock_figure, mock_close, mock_savefig, report_generator):
        data_groups = {'GroupA': [1,2,2,3], 'GroupB': [3,4,4,5]}
        plot_filename = "test_boxplot.png"

        report_generator.plot_comparison_boxplot(data_groups, title="Boxplot Test", ylabel="Value", filename=plot_filename)

        mock_figure.assert_called_once()
        # Check that sns.boxplot was called with the correct data format
        # sns.boxplot(data=...) expects a list of arrays/lists
        called_data = mock_boxplot.call_args[1]['data']
        assert len(called_data) == 2
        assert list(called_data[0]) == [1,2,2,3]
        assert list(called_data[1]) == [3,4,4,5]

        mock_xticks.assert_called_once_with(ticks=range(2), labels=['GroupA', 'GroupB'])
        mock_title.assert_called_once_with("Boxplot Test")
        mock_ylabel.assert_called_once_with("Value")
        mock_grid.assert_called_once_with(axis='y', linestyle='--', alpha=0.7)
        expected_save_path = os.path.join(report_generator.output_dir, plot_filename)
        mock_savefig.assert_called_once_with(expected_save_path)
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_comparison_boxplot_empty(self, mock_savefig, report_generator):
        report_generator.plot_comparison_boxplot({}, filename="empty_boxplot.png")
        mock_savefig.assert_not_called() # Should skip if no data


    @patch.object(ReportGenerator, 'generate_experiment_summary_report')
    @patch.object(ReportGenerator, 'plot_metric_over_time')
    @patch.object(ReportGenerator, 'plot_comparison_boxplot')
    def test_generate_full_report(self, mock_plot_boxplot, mock_plot_metric, mock_gen_summary, report_generator, sample_experiment_results, tmp_path):
        metric_ts_data = {'loss': [0.1, 0.05], 'reward': [0.8, 0.9]}
        comparison_data = {'Run1': [0.9, 0.92], 'Run2': [0.85, 0.88]}
        report_name = "my_full_report"

        original_output_dir = report_generator.output_dir # Should be tmp_path / "reports"

        report_generator.generate_full_report(sample_experiment_results, metric_ts_data, comparison_data, report_name)

        expected_report_specific_dir = os.path.join(original_output_dir, report_name)
        assert os.path.exists(expected_report_specific_dir) # Subdirectory created

        # Check that output_dir was changed and restored
        # generate_experiment_summary_report is called with the new dir
        mock_gen_summary.assert_called_once_with(sample_experiment_results, f"{report_name}_summary.md")

        # Check plot_metric_over_time calls (for loss and reward)
        assert mock_plot_metric.call_count == 2
        mock_plot_metric.assert_any_call(
            {'Loss': metric_ts_data['loss']}, title="Training Loss Over Time", ylabel="Loss", filename="training_loss.png"
        )
        mock_plot_metric.assert_any_call(
            {'Reward': metric_ts_data['reward']}, title="Training Reward Over Time", ylabel="Reward", filename="training_reward.png"
        )

        # Check plot_comparison_boxplot call
        mock_plot_boxplot.assert_called_once_with(
            comparison_data, title="Experiment Performance Comparison", ylabel="Performance Metric", filename="performance_comparison_boxplot.png"
        )

        # Verify self.output_dir was restored
        assert report_generator.output_dir == original_output_dir

        # Test with no comparison data
        mock_gen_summary.reset_mock()
        mock_plot_metric.reset_mock()
        mock_plot_boxplot.reset_mock()
        report_generator.generate_full_report(sample_experiment_results, metric_ts_data, None, "report_no_compare")
        mock_plot_boxplot.assert_not_called()
        assert mock_plot_metric.call_count == 2 # Still called for time series

        # Test with no loss/reward in time series data
        mock_plot_metric.reset_mock()
        report_generator.generate_full_report(sample_experiment_results, {}, None, "report_no_ts")
        mock_plot_metric.assert_not_called()
