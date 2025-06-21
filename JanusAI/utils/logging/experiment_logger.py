"""
Experiment Logger
=================

Provides utilities for logging experiment metrics, progress, and events.
Integrates core logging functionalities for training runs.
"""

import os
import json
import time
from datetime import datetime
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Union, Deque
import numpy as np # Added import


class TrainingLogger:
    """
    Logs and tracks key metrics and events during a training run.
    Can save logs to file and provide real-time monitoring insights.
    Absorbs core logging logic from `live_monitor.py`.
    """

    def __init__(self, log_dir: str = "./logs", experiment_name: str = "default_experiment"):
        """
        Initializes the TrainingLogger.

        Args:
            log_dir: The base directory where log files will be saved.
            experiment_name: A unique name for the current experiment.
        """
        self.log_dir = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file_path = os.path.join(self.log_dir, "training_log.jsonl")
        self.summary_file_path = os.path.join(self.log_dir, "summary.json")
        self.start_time = time.time()
        
        self.metrics_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100)) # Stores recent metrics
        self.step_counter = 0 # Tracks global steps or iterations

        print(f"Logger initialized. Logs will be saved to: {self.log_dir}")

    def log_step(self, step_metrics: Dict[str, Union[float, int]], global_step: Optional[int] = None):
        """
        Logs metrics for a single training step/iteration.

        Args:
            step_metrics: A dictionary of metrics for the current step
                          (e.g., {'loss': 0.1, 'reward': 1.5}).
            global_step: The global step or iteration number. If None, `self.step_counter` is used.
        """
        current_step = global_step if global_step is not None else self.step_counter
        log_entry = {
            "step": current_step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "metrics": step_metrics
        }
        
        # Append to line-delimited JSON log file
        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Update in-memory metrics history for summaries/trends
        for key, value in step_metrics.items():
            self.metrics_history[key].append(float(value)) # Ensure float type

        self.step_counter += 1

    def get_recent_average(self, metric_name: str, window_size: Optional[int] = None) -> Optional[float]:
        """
        Calculates the average of a metric over a recent window.

        Args:
            metric_name: The name of the metric.
            window_size: The number of recent values to average. If None, averages all history.

        Returns:
            The average value, or None if metric history is empty.
        """
        history = self.metrics_history.get(metric_name)
        if history:
            data_to_avg = list(history)[-window_size:] if window_size else list(history)
            if data_to_avg:
                return float(np.mean(data_to_avg))
        return None

    def save_summary(self, final_metrics: Dict[str, Any]):
        """
        Saves a final summary of the experiment, typically at the end of a run.

        Args:
            final_metrics: A dictionary of overall summary metrics.
        """
        summary_data = {
            "experiment_name": os.path.basename(self.log_dir),
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_steps": self.step_counter,
            "total_elapsed_time_seconds": time.time() - self.start_time,
            "final_metrics": final_metrics,
            "recent_averages": {
                name: self.get_recent_average(name, window_size=50) # Average over last 50 steps
                for name in self.metrics_history.keys()
            }
        }
        with open(self.summary_file_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Experiment summary saved to: {self.summary_file_path}")

    def get_live_metrics_report(self) -> Dict[str, Any]:
        """
        Provides a real-time snapshot of current training progress and recent metrics.
        """
        report = {
            "current_step": self.step_counter,
            "elapsed_time_seconds": round(time.time() - self.start_time, 2),
            "recent_metrics": {}
        }
        for metric_name in self.metrics_history.keys():
            avg_val = self.get_recent_average(metric_name, window_size=10)
            if avg_val is not None:
                report["recent_metrics"][metric_name] = round(avg_val, 4)
        return report

    def reset(self):
        """Resets the logger's internal state (but not disk logs)."""
        self.metrics_history.clear()
        self.step_counter = 0
        self.start_time = time.time()
        print("Logger state reset.")


if __name__ == "__main__":
    print("--- Testing TrainingLogger ---")

    # Create a logger instance
    logger = TrainingLogger(log_dir="./test_logs", experiment_name="my_test_run")

    # Simulate a training loop
    num_steps = 50
    for i in range(num_steps):
        # Generate some dummy metrics
        step_metrics = {
            "loss": 1.0 / (i + 1) + np.random.rand() * 0.05,
            "reward": 0.1 * i + np.random.rand() * 0.1,
            "episode_length": np.random.randint(20, 50)
        }
        logger.log_step(step_metrics, global_step=i)

        if i % 10 == 0:
            report = logger.get_live_metrics_report()
            print(f"\nLive Report (Step {report['current_step']}):")
            for metric, value in report['recent_metrics'].items():
                print(f"  {metric}: {value}")

        time.sleep(0.01) # Simulate some work

    # Save final summary
    final_metrics = {
        "final_loss_avg_last_50": logger.get_recent_average("loss"),
        "final_reward_avg_last_50": logger.get_recent_average("reward"),
        "best_reward": max(logger.metrics_history.get("reward", [0]))
    }
    logger.save_summary(final_metrics)

    # Test reset
    logger.reset()
    print("\nLogger reset. Current step:", logger.step_counter)
    assert logger.step_counter == 0
    assert not logger.metrics_history

    # Clean up test logs directory
    import shutil
    shutil.rmtree("./test_logs")
    print("\nCleaned up test logs directory.")

    print("\nAll TrainingLogger tests completed.")
