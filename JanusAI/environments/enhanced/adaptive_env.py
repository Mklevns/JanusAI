"""
Adaptive Environment Components
===============================

Provides components for adaptive training within the environment,
specifically the `AdaptiveTrainingController` which dynamically adjusts
training parameters based on emergent behaviors and performance.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Deque


class AdaptiveTrainingController:
    """
    Dynamically adjusts training parameters based on observed performance
    and emergent behaviors within the environment. This controller aims
    to optimize the learning process by adapting learning rates, exploration
    bonuses, and curriculum difficulty.
    """

    def __init__(self,
                 base_lr: float = 3e-4,
                 base_exploration_coef: float = 0.1,
                 base_complexity_penalty: float = 0.01,
                 stagnation_threshold: float = 0.01, # Threshold for performance trend to detect stagnation
                 breakthrough_threshold: float = 0.05, # Threshold for performance trend to detect breakthrough
                 history_length: int = 50 # Length of metric history to consider for trends
                 ):
        """
        Initializes the AdaptiveTrainingController.

        Args:
            base_lr: Initial (or baseline) learning rate for the optimizer.
            base_exploration_coef: Initial coefficient for exploration (e.g., entropy bonus weight).
            base_complexity_penalty: Initial weight for complexity penalties in rewards.
            stagnation_threshold: If performance trend falls below this, stagnation is detected.
            breakthrough_threshold: If performance trend rises above this, a breakthrough is detected.
            history_length: How many past metric values to keep for trend analysis.
        """
        self.base_lr = base_lr
        self.base_exploration_coef = base_exploration_coef
        self.base_complexity_penalty = base_complexity_penalty

        self.stagnation_threshold = stagnation_threshold
        self.breakthrough_threshold = breakthrough_threshold
        self.history_length = history_length

        # Histories for tracking training metrics over time
        self.performance_history: Deque[float] = deque(maxlen=history_length) # E.g., mean episode reward
        self.discovery_rate_history: Deque[float] = deque(maxlen=history_length) # E.g., unique discoveries / total steps
        self.complexity_history: Deque[float] = deque(maxlen=history_length) # E.g., mean complexity of discoveries

        # Track detected training phases
        self.current_phase: str = "initial" # "initial", "exploration", "refinement", "stagnation", "breakthrough"
        self.phase_history: Deque[str] = deque(maxlen=20) # Recent phases

    def update_metrics(self, metrics: Dict[str, float]):
        """
        Updates the controller's internal history with new training metrics.

        Args:
            metrics: A dictionary of current training metrics, typically from the environment
                     or trainer (e.g., `mean_reward_episode`, `discovery_rate`, `mean_complexity_episode`).
        """
        if 'mean_reward_episode' in metrics:
            self.performance_history.append(metrics['mean_reward_episode'])
        if 'discovery_rate' in metrics:
            self.discovery_rate_history.append(metrics['discovery_rate'])
        if 'mean_complexity_episode' in metrics:
            self.complexity_history.append(metrics['mean_complexity_episode'])

        # Detect current training phase after updating histories
        self.current_phase = self._detect_training_phase()
        self.phase_history.append(self.current_phase)

    def _detect_training_phase(self) -> str:
        """
        Analyzes recent metric trends to determine the current training phase.
        This logic is derived from the `AdaptiveTrainingController` in `enhanced_feedback.py`.
        """
        # Determine the minimum number of points needed for a trend.
        # np.polyfit (deg=1) needs at least 2 points. For a more stable trend, 3-5 points are better.
        # The history_length determines the max number of points available.
        # We use at least 3 points if available, up to a cap of 5, respecting history_length.

        # Effective number of points to consider for trend detection based on history_length
        # If history_length is small (e.g., 3), use all available points (i.e., history_length).
        # If history_length is large (e.g., 50), use a window (e.g., last 5 points) for recent trend.
        # The test case implies that if history_length=3, 3 points are enough.
        # Let's set min_points to be min(self.history_length, 5), but ensure it's at least 2 for polyfit.

        min_points_for_meaningful_trend = 3 # General minimum for a somewhat reliable trend

        # Number of points to use for trend calculation: min of history_length, a practical cap (e.g. 5),
        # but not less than min_points_for_meaningful_trend if history_length allows.
        # If history_length is very small (e.g. < 3), this logic might need refinement,
        # but for the test case (history_length=3), this should work.

        # Effective minimum points based on how many data points are actually IN the history deque
        num_perf_points = len(self.performance_history)
        num_discovery_points = len(self.discovery_rate_history)

        # If not enough points have been collected yet (less than what history_length allows,
        # or less than our defined minimum for a trend), stay in "initial".
        if num_perf_points < min_points_for_meaningful_trend or \
           num_discovery_points < min_points_for_meaningful_trend:
            # This check also covers cases where history_length is less than min_points_for_meaningful_trend,
            # as num_perf_points will be <= history_length.
            return "initial"

        # Calculate recent trends using all available points up to history_length
        recent_perf = list(self.performance_history) # uses all points in deque (max history_length)
        recent_discovery_rate = list(self.discovery_rate_history)

        # Performance trend: slope of a linear fit
        # Ensure there are enough points for polyfit (at least 2 for degree 1)
        if len(recent_perf) >= 2:
            perf_trend = np.polyfit(range(len(recent_perf)), recent_perf, 1)[0]
        else:
            perf_trend = 0.0 # Not enough data for a trend line, assume flat

        # Average recent discovery rate
        avg_discovery_rate = np.mean(recent_discovery_rate) if recent_discovery_rate else 0.0

        # Heuristic rules for phase detection
        if perf_trend > self.breakthrough_threshold and avg_discovery_rate > 0.5:
            return "breakthrough" # Significant improvement with high discovery
        elif perf_trend < -self.stagnation_threshold and avg_discovery_rate < 0.2:
            return "stagnation" # Performance declining, low discovery
        elif avg_discovery_rate > 0.6 and perf_trend >= 0: # Condition from test case
            return "exploration" # High discovery rate, even if performance is stable/improving slowly
        elif perf_trend > 0 and avg_discovery_rate < 0.3:
            return "refinement" # Performance improving, but discovery is low (focus on current discoveries)
        else:
            return "standard" # Default or stable phase

    def adapt_parameters(self) -> Dict[str, float]:
        """
        Calculates and returns adapted training parameters based on the detected phase.

        Returns:
            A dictionary of adapted parameters (e.g., 'learning_rate', 'entropy_coeff', 'complexity_penalty').
        """
        lr = self.base_lr
        exploration_coef = self.base_exploration_coef
        complexity_penalty = self.base_complexity_penalty

        # Adapt based on current phase
        if self.current_phase == "breakthrough":
            lr *= 0.5    # Slower learning to stabilize
            exploration_coef *= 0.5 # Less exploration
            complexity_penalty *= 0.5 # Allow more complexity (exploit discovered pattern)
        elif self.current_phase == "simplification": # Hypothetical phase for explicit simplification
            lr *= 1.2
            exploration_coef *= 0.8
            complexity_penalty *= 2.0 # Stronger penalty for complexity
        elif self.current_phase == "exploration":
            lr *= 1.5
            exploration_coef *= 2.0 # Boost exploration
            complexity_penalty *= 0.8 # Slightly reduce penalty
        elif self.current_phase == "refinement":
            lr *= 0.3
            exploration_coef *= 0.3 # Less exploration, more exploitation
            complexity_penalty *= 1.0 # Maintain penalty
        elif self.current_phase == "stagnation":
            lr *= 2.0
            exploration_coef *= 3.0 # High exploration to escape local optima
            complexity_penalty *= 0.5 # Loosen complexity constraint
        else: # "initial" or "standard"
            pass # Use base parameters

        # Apply smoothing to adapted parameters (e.g., exponential moving average)
        # This prevents jerky changes in parameters.
        if len(self.performance_history) > 1:
            # Use a simple weighted average with previous output if previous output is stored.
            # For simplicity, just return the calculated values here; smoothing is external or handled by caller.
            pass

        return {
            'learning_rate': lr,
            'entropy_coeff': exploration_coef,
            'complexity_penalty': complexity_penalty
        }

    def suggest_intervention(self) -> Optional[str]:
        """
        Suggests high-level interventions based on the current training phase
        and metrics, for a meta-controller to act upon (e.g., adjusting curriculum).
        """
        # Use last 5 points for these suggestions if available, else all history
        complexity_window = list(self.complexity_history)[-5:] if len(self.complexity_history) >= 5 else list(self.complexity_history)
        discovery_window = list(self.discovery_rate_history)[-5:] if len(self.discovery_rate_history) >= 5 else list(self.discovery_rate_history)
        performance_window = list(self.performance_history)[-5:] if len(self.performance_history) >= 5 else list(self.performance_history)

        if self.current_phase == "stagnation":
            if complexity_window and np.mean(complexity_window) > 20:
                return "reduce_max_complexity"
            elif discovery_window and np.mean(discovery_window) < 0.1:
                return "increase_exploration_bonus"

        if self.current_phase == "breakthrough":
            if performance_window and np.mean(performance_window) > 0.8:
                return "advance_curriculum"

        return None


if __name__ == "__main__":
    # This __main__ block demonstrates the usage of AdaptiveTrainingController.

    print("--- Testing AdaptiveTrainingController ---")

    controller = AdaptiveTrainingController(
        base_lr=1e-4,
        base_exploration_coef=0.05,
        stagnation_threshold=0.005,
        breakthrough_threshold=0.02,
        history_length=10 # Test with history_length=10
    )

    print(f"Initial parameters (Phase: {controller.current_phase}):", controller.adapt_parameters())

    # --- Simulate different training phases ---

    print("\n--- Phase 1: Initial/Exploration ---")
    # Simulate a phase where performance is flat, but discovery rate is okay
    # Needs enough steps to get out of "initial" based on min_points_for_meaningful_trend = 3
    for i in range(controller.history_length): # Fill up history
        mock_metrics = {
            'mean_reward_episode': 0.1 + i * 0.005 + np.random.rand() * 0.01,
            'discovery_rate': 0.7 + np.random.rand() * 0.1, # High discovery to trigger exploration
            'mean_complexity_episode': 5 + np.random.randint(0, 3)
        }
        controller.update_metrics(mock_metrics)
        adapted_params = controller.adapt_parameters()
        print(f"Step {i+1}: Phase='{controller.current_phase}', LR={adapted_params['learning_rate']:.2e}, Exploration={adapted_params['entropy_coeff']:.3f}, PerfTrend={np.polyfit(range(len(controller.performance_history)), list(controller.performance_history),1)[0] if len(controller.performance_history) >=2 else 0:.3f}, AvgDisc={np.mean(list(controller.discovery_rate_history)):.3f}")

    print("Suggested intervention:", controller.suggest_intervention())

    print("\n--- Phase 2: Breakthrough ---")
    # Simulate a sudden jump in performance and high discovery rate
    for i in range(5): # Add 5 more points
        mock_metrics = {
            'mean_reward_episode': 0.8 + np.random.rand() * 0.1,
            'discovery_rate': 0.8 + np.random.rand() * 0.1,
            'mean_complexity_episode': 10 + np.random.randint(0, 5)
        }
        controller.update_metrics(mock_metrics)
        adapted_params = controller.adapt_parameters()
        print(f"Step {i+1}: Phase='{controller.current_phase}', LR={adapted_params['learning_rate']:.2e}, Exploration={adapted_params['entropy_coeff']:.3f}, PerfTrend={np.polyfit(range(len(controller.performance_history)), list(controller.performance_history),1)[0] if len(controller.performance_history) >=2 else 0:.3f}, AvgDisc={np.mean(list(controller.discovery_rate_history)):.3f}")

    print("Suggested intervention:", controller.suggest_intervention())


    print("\n--- Phase 3: Stagnation ---")
    # Simulate performance decline and low discovery
    for i in range(controller.history_length): # Fill history with stagnation data
        mock_metrics = {
            'mean_reward_episode': 0.5 - i * 0.02 - np.random.rand() * 0.01,
            'discovery_rate': 0.05 + np.random.rand() * 0.05,
            'mean_complexity_episode': 25 + np.random.randint(0, 5)
        }
        controller.update_metrics(mock_metrics)
        adapted_params = controller.adapt_parameters()
        print(f"Step {i+1}: Phase='{controller.current_phase}', LR={adapted_params['learning_rate']:.2e}, Exploration={adapted_params['entropy_coeff']:.3f}, PerfTrend={np.polyfit(range(len(controller.performance_history)), list(controller.performance_history),1)[0] if len(controller.performance_history) >=2 else 0:.3f}, AvgDisc={np.mean(list(controller.discovery_rate_history)):.3f}")

    print("Suggested intervention:", controller.suggest_intervention())

    # Test with history_length = 3 as in the unit test
    print("\n--- Test with history_length = 3 (like unit test) ---")
    controller_short_hist = AdaptiveTrainingController(history_length=3)
    metrics_seq = [
        {'mean_reward_episode': 0.1, 'discovery_rate': 0.5, 'mean_complexity_episode': 10},
        {'mean_reward_episode': 0.2, 'discovery_rate': 0.6},
        {'mean_reward_episode': 0.3, 'discovery_rate': 0.7, 'mean_complexity_episode': 12},
        {'mean_reward_episode': 0.4, 'discovery_rate': 0.8, 'mean_complexity_episode': 13},
        {'mean_reward_episode': 0.5, 'discovery_rate': 0.9, 'mean_complexity_episode': 14}
    ]
    for i, metrics in enumerate(metrics_seq):
        controller_short_hist.update_metrics(metrics)
        print(f"Step {i+1}: Phase='{controller_short_hist.current_phase}', PerfHist={list(controller_short_hist.performance_history)}, DiscHist={list(controller_short_hist.discovery_rate_history)}")
        if i == 4: # After 5th update, as in test
             # perf_trend for [0.3,0.4,0.5] is 0.1. avg_discovery for [0.7,0.8,0.9] is 0.8.
             # exploration: avg_discovery_rate > 0.6 and perf_trend >= 0. (0.8 > 0.6 and 0.1 >=0) -> True.
            assert controller_short_hist.current_phase == "exploration"
            print("Assertion for exploration phase after 5 updates with history_length=3 PASSED.")


    print("\nAdaptiveTrainingController demonstration complete.")
