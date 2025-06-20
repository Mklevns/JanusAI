"""
Report Generation for Experiment Analysis
=========================================

Provides utilities for generating comprehensive reports and visualizations
from experiment results. This aids in summarizing findings, comparing runs,
and communicating insights effectively.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class ReportGenerator:
    """
    Generates structured reports from experiment results, including
    textual summaries, tables, and visualizations.
    """

    def __init__(self, output_dir: str = "./reports"):
        """
        Initializes the ReportGenerator.

        Args:
            output_dir: The directory where generated reports will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Reports will be saved to: {self.output_dir}")

    def generate_experiment_summary_report(self, 
                                           experiment_results: Dict[str, Any], 
                                           report_filename: str = "summary_report.md"):
        """
        Generates a markdown-formatted summary report for a single experiment.

        Args:
            experiment_results: A dictionary containing the results of an experiment.
                                Expected keys might include 'overall_average_fidelity',
                                'overall_correct_discovery_rate', 'task_results', etc.
            report_filename: The name of the markdown file to save the report.
        """
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(f"# Experiment Summary Report: {experiment_results.get('experiment_name', 'Unnamed Experiment')}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Overview\n\n")
            f.write(f"- **Total Timesteps Run:** {experiment_results.get('total_timesteps', 'N/A'):,}\n")
            f.write(f"- **Overall Average Fidelity:** {experiment_results.get('overall_average_fidelity', 'N/A'):.4f}\n")
            f.write(f"- **Overall Correct Discovery Rate:** {experiment_results.get('overall_correct_discovery_rate', 'N/A'):.2f}\n")
            f.write(f"- **Experiment Duration:** {experiment_results.get('total_elapsed_time_seconds', 'N/A'):.2f} seconds\n\n")

            if 'config' in experiment_results:
                f.write("## Configuration\n\n")
                f.write("```json\n")
                f.write(json.dumps(experiment_results['config'], indent=2))
                f.write("\n```\n\n")

            if 'task_results' in experiment_results and experiment_results['task_results']:
                f.write("## Per-Task Results\n\n")
                f.write("| Task Name | Difficulty | Avg Fidelity | Correct Rate |\n")
                f.write("|---|---|---|---|\n")
                for task_res in experiment_results['task_results']:
                    f.write(f"| {task_res.get('task_name', 'N/A')} "
                            f"| {task_res.get('difficulty', 'N/A'):.1f} "
                            f"| {task_res.get('avg_fidelity', 'N/A'):.4f} "
                            f"| {task_res.get('correct_discovery_rate', 'N/A'):.2f} |\n")
                f.write("\n")
            
            f.write("## Conclusion\n\n")
            f.write("This report summarizes the key outcomes of the experiment. Further detailed "
                    "analysis and visualizations are available in supplementary files.\n")
        
        print(f"Summary report generated: {report_path}")

    def plot_metric_over_time(self, 
                              metric_history: Dict[str, List[float]], 
                              title: str = "Training Metric Over Time",
                              xlabel: str = "Steps",
                              ylabel: str = "Metric Value",
                              filename: str = "metric_plot.png"):
        """
        Generates a line plot for one or more metrics over time.

        Args:
            metric_history: A dictionary where keys are metric names (str) and values are
                            lists of historical data points (float).
            title: The title of the plot.
            xlabel: Label for the X-axis.
            ylabel: Label for the Y-axis.
            filename: The filename to save the plot (e.g., "loss_history.png").
        """
        plt.figure(figsize=(10, 6))
        for metric_name, values in metric_history.items():
            if values: # Only plot if data exists
                plt.plot(values, label=metric_name)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path)
        plt.close() # Close plot to free memory
        print(f"Plot saved to: {plot_path}")

    def plot_comparison_boxplot(self, 
                                data_groups: Dict[str, Union[List[float], np.ndarray]],
                                title: str = "Performance Comparison",
                                ylabel: str = "Metric Value",
                                filename: str = "comparison_boxplot.png"):
        """
        Generates a box plot to compare a metric across different groups/experiments.

        Args:
            data_groups: A dictionary where keys are group names (str) and values are
                         lists or arrays of numerical data for that group.
            title: The title of the plot.
            ylabel: Label for the Y-axis.
            filename: The filename to save the plot.
        """
        if not data_groups:
            print("No data groups provided for box plot. Skipping.")
            return

        group_names = list(data_groups.keys())
        data_to_plot = [data_groups[name] for name in group_names]

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data_to_plot)
        plt.xticks(ticks=range(len(group_names)), labels=group_names)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Box plot saved to: {plot_path}")

    def generate_full_report(self, 
                             experiment_results: Dict[str, Any], 
                             metric_time_series_data: Dict[str, List[float]], 
                             comparison_data: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
                             report_name: str = "full_report"):
        """
        Generates a comprehensive report including summary, time-series plots,
        and comparison plots.
        """
        # Ensure output directory for this specific report
        report_specific_dir = os.path.join(self.output_dir, report_name)
        os.makedirs(report_specific_dir, exist_ok=True)
        
        # Temporarily change output_dir for plots
        original_output_dir = self.output_dir
        self.output_dir = report_specific_dir

        # Generate summary markdown
        self.generate_experiment_summary_report(experiment_results, f"{report_name}_summary.md")

        # Generate time-series plots
        if 'loss' in metric_time_series_data:
            self.plot_metric_over_time(
                {'Loss': metric_time_series_data['loss']},
                title="Training Loss Over Time",
                ylabel="Loss",
                filename="training_loss.png"
            )
        if 'reward' in metric_time_series_data:
            self.plot_metric_over_time(
                {'Reward': metric_time_series_data['reward']},
                title="Training Reward Over Time",
                ylabel="Reward",
                filename="training_reward.png"
            )
        
        # Generate comparison box plot if data provided
        if comparison_data:
            self.plot_comparison_boxplot(
                comparison_data,
                title="Experiment Performance Comparison",
                ylabel="Performance Metric",
                filename="performance_comparison_boxplot.png"
            )
        
        self.output_dir = original_output_dir # Restore original output dir
        print(f"Full report assets saved in: {report_specific_dir}")


if __name__ == "__main__":
    print("--- Testing ReportGenerator ---")
    report_gen = ReportGenerator(output_dir="./test_reports")

    # Mock experiment results
    mock_results = {
        'experiment_name': 'SampleExperiment_v1',
        'total_timesteps': 10000,
        'overall_average_fidelity': 0.85,
        'overall_correct_discovery_rate': 0.60,
        'total_elapsed_time_seconds': 360.5,
        'config': {'param_a': 10, 'param_b': 'value'},
        'task_results': [
            {'task_name': 'HO', 'difficulty': 0.2, 'avg_fidelity': 0.95, 'correct_discovery_rate': 1.0},
            {'task_name': 'Pendulum', 'difficulty': 0.5, 'avg_fidelity': 0.70, 'correct_discovery_rate': 0.3}
        ]
    }

    # Mock time-series data
    mock_loss_history = [1.5 - i * 0.01 for i in range(100)]
    mock_reward_history = [0.1 + i * 0.005 for i in range(100)]
    mock_time_series_data = {
        'loss': mock_loss_history,
        'reward': mock_reward_history
    }

    # Mock comparison data (e.g., from multiple runs)
    mock_comparison_data = {
        'Run A': np.random.normal(loc=0.8, scale=0.1, size=20),
        'Run B': np.random.normal(loc=0.7, scale=0.15, size=20),
        'Run C': np.random.normal(loc=0.9, scale=0.05, size=20)
    }

    # Generate full report
    report_gen.generate_full_report(
        experiment_results=mock_results,
        metric_time_series_data=mock_time_series_data,
        comparison_data=mock_comparison_data,
        report_name="sample_experiment_report"
    )

    # Verify that files are created
    report_path = os.path.join("./test_reports/sample_experiment_report", "sample_experiment_report_summary.md")
    assert os.path.exists(report_path)
    assert os.path.exists(os.path.join("./test_reports/sample_experiment_report", "training_loss.png"))
    assert os.path.exists(os.path.join("./test_reports/sample_experiment_report", "performance_comparison_boxplot.png"))

    # Clean up test reports directory
    import shutil
    shutil.rmtree("./test_reports")
    print(f"\nCleaned up test reports directory: {'./test_reports'}")

    print("\nAll ReportGenerator tests completed.")

