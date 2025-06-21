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
        if len(self.performance_history) < 3 or len(self.discovery_rate_history) < 3: # Changed from 5 to 3
            return "initial" # Not enough data to detect phase

        # Calculate recent trends
        recent_perf = list(self.performance_history)
        recent_discovery_rate = list(self.discovery_rate_history)

        # Performance trend: slope of a linear fit
        # Ensure there are enough points for polyfit to avoid RankWarning or errors
        if len(recent_perf) >= 2:
            perf_trend = np.polyfit(range(len(recent_perf)), recent_perf, 1)[0]
        else:
            perf_trend = 0.0

        # Average recent discovery rate
        if len(recent_discovery_rate) > 0:
            avg_discovery_rate = np.mean(recent_discovery_rate)
        else:
            avg_discovery_rate = 0.0



        # Heuristic rules for phase detection
        if perf_trend > self.breakthrough_threshold and avg_discovery_rate > 0.5:
            return "breakthrough" # Significant improvement with high discovery
        elif perf_trend < -self.stagnation_threshold and avg_discovery_rate < 0.2:
            return "stagnation" # Performance declining, low discovery
        elif avg_discovery_rate > 0.6 and perf_trend >= 0:

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
        # Ensure histories are not empty before calculating mean, check for at least 5 for consistency with original intent
        min_len_for_suggestion = 5

        if self.current_phase == "stagnation":
            if len(self.complexity_history) >= min_len_for_suggestion and \
               np.mean(list(self.complexity_history)[-min_len_for_suggestion:]) > 20:
                return "reduce_max_complexity"
            elif len(self.discovery_rate_history) >= min_len_for_suggestion and \
                 np.mean(list(self.discovery_rate_history)[-min_len_for_suggestion:]) < 0.1:
                return "increase_exploration_bonus"

        if self.current_phase == "breakthrough":
            if len(self.performance_history) >= min_len_for_suggestion and \
               np.mean(list(self.performance_history)[-min_len_for_suggestion:]) > 0.8:
                return "advance_curriculum"

        return None # No specific intervention suggested



if __name__ == "__main__":
    # This __main__ block demonstrates the usage of AdaptiveTrainingController.

    print("--- Testing AdaptiveTrainingController ---")

    controller = AdaptiveTrainingController(
        base_lr=1e-4,
        base_exploration_coef=0.05,
        stagnation_threshold=0.005,
        breakthrough_threshold=0.02,
        history_length=10 # Keep history length for demo shorter
    )

    print("Initial parameters:", controller.adapt_parameters())

    # --- Simulate different training phases ---

    print("\n--- Phase 1: Initial/Exploration (with new <3 rule) ---")
    # Simulate a phase where performance is flat, but discovery rate is okay
    # Phase should change from 'initial' after 3 updates
    for i in range(5): # Run for 5 steps to see transition
        mock_metrics = {
            'mean_reward_episode': 0.1 + i * 0.01 + np.random.rand() * 0.005, # Slowly increasing
            'discovery_rate': 0.65 + np.random.rand() * 0.1, # High discovery for exploration

            'mean_complexity_episode': 5 + np.random.randint(0, 3)
        }
        controller.update_metrics(mock_metrics)
        adapted_params = controller.adapt_parameters()
        print(f"Step {i+1}: Phase='{controller.current_phase}', LR={adapted_params['learning_rate']:.2e}, Exploration={adapted_params['entropy_coeff']:.3f}")

    print("Suggested intervention:", controller.suggest_intervention()) # Might be None if history not long enough for suggestion logic

    print("\n--- Phase 2: Breakthrough ---")
    # Simulate a sudden jump in performance and high discovery rate
    # Fill history to ensure phase detection
    for _ in range(controller.history_length - len(controller.performance_history)): # Fill up history if needed
         controller.update_metrics({'mean_reward_episode': 0.1, 'discovery_rate': 0.1})

    for i in range(5):
        mock_metrics = {
            'mean_reward_episode': 0.8 + np.random.rand() * 0.1, # High reward
            'discovery_rate': 0.8 + np.random.rand() * 0.1, # High discovery

            'mean_complexity_episode': 10 + np.random.randint(0, 5)
        }
        controller.update_metrics(mock_metrics)
        adapted_params = controller.adapt_parameters()
        print(f"Step {i+1}: Phase='{controller.current_phase}', LR={adapted_params['learning_rate']:.2e}, Exploration={adapted_params['entropy_coeff']:.3f}")


    print("Suggested intervention:", controller.suggest_intervention())


    print("\n--- Phase 3: Stagnation ---")
    # Simulate performance decline and low discovery
    for _ in range(controller.history_length - len(controller.performance_history)): # Fill up history
         controller.update_metrics({'mean_reward_episode': 0.8, 'discovery_rate': 0.8})

    for i in range(10): # Stagnation might take a few steps to detect with trend
        mock_metrics = {
            'mean_reward_episode': 0.5 - i * 0.02 - np.random.rand() * 0.01, # Declining reward
            'discovery_rate': 0.05 + np.random.rand() * 0.05, # Very low discovery
            'mean_complexity_episode': 25 + np.random.randint(0, 5) # High complexity
        }
        controller.update_metrics(mock_metrics)
        adapted_params = controller.adapt_parameters()
        print(f"Step {i+1}: Phase='{controller.current_phase}', LR={adapted_params['learning_rate']:.2e}, Exploration={adapted_params['entropy_coeff']:.3f}")

    print("Suggested intervention:", controller.suggest_intervention())


    print("\nAdaptiveTrainingController demonstration complete.")
