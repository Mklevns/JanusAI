"""
Tests for environments/enhanced/adaptive_env.py
"""
import pytest
import numpy as np
from collections import deque

from environments.enhanced.adaptive_env import AdaptiveTrainingController

class TestAdaptiveTrainingController:

    def test_init_defaults(self):
        controller = AdaptiveTrainingController()
        assert controller.base_lr == 3e-4
        assert controller.base_exploration_coef == 0.1
        assert controller.base_complexity_penalty == 0.01
        assert controller.stagnation_threshold == 0.01
        assert controller.breakthrough_threshold == 0.05
        assert controller.history_length == 50
        assert isinstance(controller.performance_history, deque)
        assert controller.performance_history.maxlen == 50
        assert controller.current_phase == "initial"
        assert isinstance(controller.phase_history, deque)
        assert controller.phase_history.maxlen == 20

    def test_init_custom_params(self):
        controller = AdaptiveTrainingController(
            base_lr=1e-3, base_exploration_coef=0.05, base_complexity_penalty=0.005,
            stagnation_threshold=0.02, breakthrough_threshold=0.08, history_length=30
        )
        assert controller.base_lr == 1e-3
        assert controller.base_exploration_coef == 0.05
        assert controller.base_complexity_penalty == 0.005
        assert controller.stagnation_threshold == 0.02
        assert controller.breakthrough_threshold == 0.08
        assert controller.history_length == 30
        assert controller.performance_history.maxlen == 30

    def test_update_metrics(self):
        controller = AdaptiveTrainingController(history_length=3)

        metrics1 = {'mean_reward_episode': 0.1, 'discovery_rate': 0.5, 'mean_complexity_episode': 10}
        controller.update_metrics(metrics1)
        assert list(controller.performance_history) == [0.1]
        assert list(controller.discovery_rate_history) == [0.5]
        assert list(controller.complexity_history) == [10]
        assert controller.current_phase == "initial" # Not enough data yet
        assert list(controller.phase_history) == ["initial"]

        metrics2 = {'mean_reward_episode': 0.2, 'discovery_rate': 0.6} # Missing complexity
        controller.update_metrics(metrics2)
        assert list(controller.performance_history) == [0.1, 0.2]
        assert list(controller.discovery_rate_history) == [0.5, 0.6]
        assert list(controller.complexity_history) == [10] # Unchanged
        assert list(controller.phase_history) == ["initial", "initial"]


        metrics3 = {'mean_reward_episode': 0.3, 'discovery_rate': 0.7, 'mean_complexity_episode': 12}
        controller.update_metrics(metrics3)
        metrics4 = {'mean_reward_episode': 0.4, 'discovery_rate': 0.8, 'mean_complexity_episode': 13} # This will make history_length = 3
        controller.update_metrics(metrics4)

        assert list(controller.performance_history) == [0.2, 0.3, 0.4] # Maxlen is 3
        assert list(controller.discovery_rate_history) == [0.6, 0.7, 0.8]
        assert list(controller.complexity_history) == [10, 12, 13] # 10 was from metrics1, then 12, 13

        # Phase detection will run with these 3 points, but still might be initial if len < 5 check is strict
        # The code has `if len(self.performance_history) < 5 ... return "initial"`
        # So, even with 3-4 points, it should still be "initial"
        assert controller.current_phase == "initial"
        assert len(controller.phase_history) == 4 # initial, initial, initial, initial

        # Add enough metrics to pass the < 5 check
        controller.update_metrics({'mean_reward_episode': 0.5, 'discovery_rate': 0.9, 'mean_complexity_episode': 14})
        # Now performance_history has [0.3, 0.4, 0.5]. Phase detection runs.
        # Assuming it goes to "standard" or "exploration" based on values.
        # perf_trend for [0.3,0.4,0.5] is 0.1. avg_discovery for [0.7,0.8,0.9] is 0.8.
        # exploration: avg_discovery_rate > 0.6 and perf_trend >= 0. (0.8 > 0.6 and 0.1 >=0) -> True.
        assert controller.current_phase == "exploration"


    @pytest.mark.parametrize("perf_hist, disc_rate_hist, expected_phase", [
        # Initial (not enough data)
        ([0.1, 0.2], [0.5, 0.6], "initial"),
        # Breakthrough
        ([0.1, 0.2, 0.3, 0.8, 0.9], [0.6, 0.7, 0.7, 0.8, 0.9], "breakthrough"), # perf_trend > 0.05, disc_rate > 0.5
        # Stagnation
        ([0.5, 0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.05, 0.1, 0.15], "stagnation"), # perf_trend < -0.01, disc_rate < 0.2
        # Exploration
        ([0.1, 0.1, 0.1, 0.1, 0.12], [0.7, 0.8, 0.7, 0.8, 0.9], "exploration"), # disc_rate > 0.6, perf_trend >=0
        # Refinement
        ([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.15, 0.2, 0.25], "refinement"), # perf_trend > 0, disc_rate < 0.3
        # Standard (default case)
        ([0.1, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3, 0.3], "standard"), # Flat perf, moderate discovery
    ])
    def test_detect_training_phase(self, perf_hist, disc_rate_hist, expected_phase):
        controller = AdaptiveTrainingController(history_length=5, stagnation_threshold=0.01, breakthrough_threshold=0.05)
        controller.performance_history.extend(perf_hist)
        controller.discovery_rate_history.extend(disc_rate_hist)

        detected_phase = controller._detect_training_phase()
        assert detected_phase == expected_phase

    def test_detect_training_phase_edge_cases_polyfit(self):
        controller = AdaptiveTrainingController(history_length=5)
        # Polyfit needs at least 2 points for degree 1.
        # Case: 1 point in history (though main check is < 5)
        controller.performance_history.append(0.1)
        controller.discovery_rate_history.append(0.1)
        assert controller._detect_training_phase() == "initial" # Due to len < 5

        # Case: 0 points (empty history)
        controller.performance_history.clear()
        controller.discovery_rate_history.clear()
        assert controller._detect_training_phase() == "initial" # Due to len < 5

    @pytest.mark.parametrize("phase, expected_lr_factor, expected_exp_factor, expected_comp_factor", [
        ("initial", 1.0, 1.0, 1.0),
        ("standard", 1.0, 1.0, 1.0),
        ("breakthrough", 0.5, 0.5, 0.5),
        ("exploration", 1.5, 2.0, 0.8),
        ("refinement", 0.3, 0.3, 1.0),
        ("stagnation", 2.0, 3.0, 0.5),
        ("simplification", 1.2, 0.8, 2.0), # Hypothetical, but in code
    ])
    def test_adapt_parameters(self, phase, expected_lr_factor, expected_exp_factor, expected_comp_factor):
        base_lr = 1e-4
        base_exp = 0.1
        base_comp = 0.01
        controller = AdaptiveTrainingController(base_lr=base_lr, base_exploration_coef=base_exp, base_complexity_penalty=base_comp)
        controller.current_phase = phase # Manually set phase for test

        # Add some history so that the `if len(self.performance_history) > 1:` check passes (though it's for smoothing, not base adaptation)
        controller.performance_history.extend([0.1, 0.2])


        params = controller.adapt_parameters()

        assert abs(params['learning_rate'] - base_lr * expected_lr_factor) < 1e-9
        assert abs(params['entropy_coeff'] - base_exp * expected_exp_factor) < 1e-9
        assert abs(params['complexity_penalty'] - base_comp * expected_comp_factor) < 1e-9


    @pytest.mark.parametrize("current_phase, complexity_hist, discovery_hist, perf_hist, expected_suggestion", [
        # Stagnation cases
        ("stagnation", [25, 26, 28], [0.05, 0.06], [0.1, 0.09], "reduce_max_complexity"), # High complexity
        ("stagnation", [5, 6, 7], [0.05, 0.06], [0.1, 0.09], "increase_exploration_bonus"), # Low complexity, low discovery
        ("stagnation", [5, 6, 7], [0.3, 0.4], [0.1, 0.09], None), # Stagnation but discovery rate not that low
        # Breakthrough cases
        ("breakthrough", [10,11,12], [0.7,0.8], [0.85, 0.9, 0.95], "advance_curriculum"), # High perf
        ("breakthrough", [10,11,12], [0.7,0.8], [0.5, 0.6, 0.55], None), # Breakthrough but perf not consistently high
        # Other phases
        ("exploration", [10,11,12], [0.7,0.8], [0.5, 0.6], None),
        ("initial", [], [], [], None),
    ])
    def test_suggest_intervention(self, current_phase, complexity_hist, discovery_hist, perf_hist, expected_suggestion):
        controller = AdaptiveTrainingController()
        controller.current_phase = current_phase
        if complexity_hist: controller.complexity_history.extend(complexity_hist)
        if discovery_hist: controller.discovery_rate_history.extend(discovery_hist)
        if perf_hist: controller.performance_history.extend(perf_hist)

        suggestion = controller.suggest_intervention()
        assert suggestion == expected_suggestion

    def test_suggest_intervention_empty_histories(self):
        controller = AdaptiveTrainingController()
        controller.current_phase = "stagnation"
        # Histories are empty
        assert controller.suggest_intervention() is None # Should not error, return None

        controller.current_phase = "breakthrough"
        assert controller.suggest_intervention() is None
