"""
Intrinsic Rewards
=================

Provides various intrinsic reward functions and a consolidated calculator
that supplements extrinsic rewards, encouraging desired agent behaviors
like exploration, novelty, or adherence to physical principles.
"""

import numpy as np
import sympy as sp
from typing import Any, Dict, Optional, List, Deque
from collections import deque

# Import BaseReward
from janus_ai.ml.rewards.base_reward import BaseReward

# Import utilities from new structure
from janus_ai.core.expressions.expression import Variable
from janus_ai.core.expressions.symbolic_math import get_variables_from_expression, evaluate_expression_on_data
from janus_ai.utils.math.operations import calculate_symbolic_accuracy, calculate_expression_complexity

# Import the ConservationDetector from physics laws
from janus_ai.physics.laws.conservation import ConservationDetector


class NoveltyReward(BaseReward):
    """
    Rewards the agent for discovering novel expressions or states.
    Utilized internally by IntrinsicRewardCalculator or directly.
    """
    def __init__(self, weight: float = 1.0, history_size: int = 100, novelty_threshold: float = 0.1):
        super().__init__(weight)
        self.history_size = history_size
        self.expression_history: deque[str] = deque(maxlen=history_size)
        self.novelty_threshold = novelty_threshold
        self.expression_cache: Dict[str, int] = {} # Tracks frequency for simple novelty

    def calculate_reward(self,
                         current_observation: Any, # Not directly used for novelty, but part of BaseReward API
                         action: Any,              # Not directly used
                         next_observation: Any,    # Not directly used
                         reward_from_env: float,   # Not directly used
                         done: bool,               # Not directly used
                         info: Dict[str, Any]) -> float:
        """Calculates a novelty reward based on the expression in info."""
        if 'expression' not in info or not info['expression']:
            return 0.0

        current_expr_str = info['expression']

        # Frequency-based novelty: reward for seeing a new expression, decaying over time
        if current_expr_str not in self.expression_cache:
            self.expression_cache[current_expr_str] = 1
            base_novelty = 1.0
        else:
            self.expression_cache[current_expr_str] += 1
            base_novelty = 1.0 / self.expression_cache[current_expr_str] # Decay inversely proportional to frequency

        # Semantic novelty: bonus if the expression is semantically new compared to recent history
        semantic_novelty_score = 0.0
        variables = info.get('variables', [])
        if variables and self.expression_history: # Only check if history exists
             # Check against history for semantic similarity
            is_semantically_new = True
            for historical_expr_str in self.expression_history:
                if self._is_similar(current_expr_str, historical_expr_str, variables):
                    is_semantically_new = False
                    break
            if is_semantically_new:
                semantic_novelty_score = 1.0 # Significant bonus for truly new semantic discovery

        # Add to history if semantically new or for general tracking
        self.expression_history.append(current_expr_str)

        # Simple operator novelty: bonus for using "rare" or complex operators
        op_novelty_bonus = 0.0
        rare_ops = ['sin', 'cos', 'log', 'exp', '**', 'sqrt', 'inv', 'diff', 'int'] # Extended list of operators
        for op in rare_ops:
            if op in current_expr_str: # Simple string check for operator presence
                op_novelty_bonus += 0.05 # Small bonus for each rare operator encountered

        # Combine different novelty aspects. Normalize by the number of components for a reasonable scale.
        total_novelty = (base_novelty + semantic_novelty_score + op_novelty_bonus)
        return total_novelty / (3.0 if semantic_novelty_score > 0 else 2.0) # Average, or adjust if semantic check not done


    def _is_similar(self, expr1_str: str, expr2_str: str, variables: List[Variable]) -> bool:
        """Compares two expressions for semantic similarity using symbolic accuracy."""
        try:
            # Create a dummy ground_truth_dict for calculate_symbolic_accuracy
            expr1_sympy = sp.sympify(expr1_str, evaluate=False)
            ground_truth_dict = {'true_law': expr1_sympy}

            # `calculate_symbolic_accuracy` expects its first argument to be the discovered expression string
            # and its second to be a dictionary containing the true law (sympy expression)
            similarity = calculate_symbolic_accuracy(expr2_str, ground_truth_dict, var_symbols=[v.name for v in variables])
            return similarity > (1.0 - self.novelty_threshold)
        except (sp.SympifyError, TypeError, AttributeError):
            # Log any issues if needed
            return False


class ComplexityReward(BaseReward):
    """
    Rewards for finding expressions within a desired complexity range,
    or penalizes for excessively simple/complex expressions.
    Utilized internally by IntrinsicRewardCalculator or directly.
    """
    def __init__(self, weight: float = 1.0, target_complexity: int = 10, tolerance: int = 5):
        super().__init__(weight)
        self.target_complexity = target_complexity
        self.tolerance = tolerance

    def calculate_reward(self,
                         current_observation: Any, # Not directly used
                         action: Any,              # Not directly used
                         next_observation: Any,    # Not directly used
                         reward_from_env: float,   # Not directly used
                         done: bool,               # Not directly used
                         info: Dict[str, Any]) -> float:
        """Calculates a reward based on the expression's complexity in info."""
        if 'expression' not in info or not info['expression']:
            return 0.0

        try:
            complexity = calculate_expression_complexity(info['expression'])

            reward = 0.0
            if abs(complexity - self.target_complexity) <= self.tolerance:
                reward = 1.0 # Reward for being exactly in the target range
            elif complexity < self.target_complexity:
                # Linearly scale penalty for being too simple
                reward = -0.5 * (1 - (complexity / self.target_complexity))
            else:
                # Linearly scale penalty for being too complex
                reward = -1.0 * ((complexity - self.target_complexity) / (self.target_complexity * 2)) # Heavier penalty
                reward = max(-1.0, reward) # Cap penalty

            return reward

        except (sp.SympifyError, ValueError):
            return -0.1 # Small penalty if expression is malformed or complexity cannot be calculated


class ConservationLawReward(BaseReward):
    """
    Rewards the agent for discovering expressions that adhere to known
    conservation laws (e.g., conservation of energy, momentum).
    This class *uses* a ConservationDetector to perform the checks.
    """
    def __init__(self,
                 weight: float = 1.0,
                 law_type: str = 'energy',
                 apply_entropy_penalty: bool = False,
                 entropy_penalty_weight: float = 0.1):
        super().__init__(weight)
        self.law_type = law_type.lower()
        if self.law_type not in ['energy', 'momentum', 'angular_momentum']:
            raise ValueError(f"Unsupported conservation law type: {law_type}")

        # Instantiate the ConservationDetector to perform actual conservation checks
        self.detector = ConservationDetector()
        self.apply_entropy_penalty = apply_entropy_penalty
        self.entropy_penalty_weight = entropy_penalty_weight

    def calculate_reward(self,
                         current_observation: Any,
                         action: Any,
                         next_observation: Any,
                         reward_from_env: float,
                         done: bool,
                         info: Dict[str, Any]) -> float:
        """
        Calculates a reward based on adherence to a specified conservation law.

        Args:
            info: Must contain 'expression' (hypothesized expression string),
                  'trajectory_data' (actual trajectory to test against),
                  'variables' (list of Variable objects used in environment).
                  May optionally contain 'predicted_forward_traj' and 'predicted_backward_traj'
                  for entropy production penalty.
        """
        if 'expression' not in info or not info['expression'] or \
           'trajectory_data' not in info or not info['trajectory_data'] or \
           'variables' not in info or not info['variables']:
            return 0.0 # Cannot evaluate conservation if critical info is missing

        expr_str = info['expression']
        trajectory_data = info['trajectory_data']
        variables = info['variables']

        is_conserved = False
        try:
            # Call the appropriate check method on the detector
            if self.law_type == 'energy':
                is_conserved = self.detector.check_energy_conservation(expr_str, trajectory_data, variables)
            elif self.law_type == 'momentum':
                is_conserved = self.detector.check_momentum_conservation(expr_str, trajectory_data, variables)
            elif self.law_type == 'angular_momentum':
                is_conserved = self.detector.check_angular_momentum_conservation(expr_str, trajectory_data, variables)
            # Add other conservation checks here if supported by detector
        except Exception: # Catch any error during the check to avoid crashing
            return -0.1 # Small penalty if expression is uncheckable or causes error

        conservation_reward = 1.0 if is_conserved else -0.5 # Reward for conservation, penalty otherwise

        entropy_penalty = 0.0
        if self.apply_entropy_penalty:
            predicted_forward_traj = info.get('predicted_forward_traj')
            predicted_backward_traj = info.get('predicted_backward_traj')
            if predicted_forward_traj and predicted_backward_traj:
                entropy_penalty = self.detector.get_entropy_production_penalty(
                    predicted_forward_traj, predicted_backward_traj
                )
                conservation_reward += entropy_penalty * self.entropy_penalty_weight # Apply scaled penalty

        return conservation_reward


class IntrinsicRewardCalculator:
    """
    Calculates a combined intrinsic reward based on multiple factors like novelty,
    diversity, complexity growth, and adherence to conservation laws.

    This class consolidates the logic from the original `IntrinsicRewardCalculator`
    in `enhanced_feedback.py`. It uses instances of `BaseReward` subclasses
    (like `NoveltyReward`, `ComplexityReward`, `ConservationLawReward`) internally.
    """

    def __init__(self,
                 novelty_weight: float = 0.3,
                 diversity_weight: float = 0.2,
                 complexity_growth_weight: float = 0.1,
                 conservation_weight: float = 0.4,
                 conservation_types: Optional[List[str]] = None, # Specify which laws to check for conservation
                 apply_entropy_penalty: bool = False,
                 entropy_penalty_weight: float = 0.1,
                 history_size: int = 1000 # Max size for expression/embedding history
                ):

        self.novelty_weight = novelty_weight
        self.diversity_weight = diversity_weight
        self.complexity_growth_weight = complexity_growth_weight
        self.conservation_weight = conservation_weight

        # Initialize internal reward components using BaseReward subclasses.
        self.novelty_reward_comp = NoveltyReward(weight=1.0, history_size=history_size) # Internal weight is 1, scaled by self.novelty_weight later
        self.complexity_reward_comp = ComplexityReward(weight=1.0)

        # Default conservation types if not provided
        if conservation_types is None:
            conservation_types = ['energy', 'momentum', 'mass']

        # Create a ConservationLawReward instance for each specified conservation type
        # Or, if only one type is expected, pass the first one. For simplicity,
        # the previous `ConservationBiasedReward` and `ConservationLawReward` were single-type focused.
        # Let's adapt this to use the general `ConservationLawReward` by iterating or assuming one type.
        # For simplicity, we'll create one `ConservationLawReward` for the primary type.
        primary_conservation_type = conservation_types[0] if conservation_types else 'energy'
        self.conservation_reward_comp = ConservationLawReward(
            weight=1.0,
            law_type=primary_conservation_type,
            apply_entropy_penalty=apply_entropy_penalty,
            entropy_penalty_weight=entropy_penalty_weight
        )
        # If multiple conservation types are needed, this would need to be a list of ConservationLawReward instances.
        # For now, it matches the original `conservation_calculator` which was a single instance.

        # History tracking for diversity and complexity growth
        self.expression_history: Deque[str] = deque(maxlen=history_size)
        self.complexity_history: Deque[int] = deque(maxlen=history_size)
        self.discovery_embeddings: Deque[np.ndarray] = deque(maxlen=history_size)

    def calculate_intrinsic_reward(self,
                                 expression: str,
                                 complexity: int,
                                 extrinsic_reward: float, # Original extrinsic reward from environment
                                 embedding: Optional[np.ndarray], # Embedding of the discovered expression
                                 data: np.ndarray, # Full trajectory data for conservation checks
                                 variables: List[Variable], # List of Variable objects for expression evaluation
                                 predicted_forward_traj: Optional[Dict[str, Any]] = None, # For entropy penalty
                                 predicted_backward_traj: Optional[Dict[str, Any]] = None, # For entropy penalty
                                 ) -> float:
        """
        Calculates the combined intrinsic reward for a given discovery step.

        Args:
            expression: The string representation of the currently hypothesized expression.
            complexity: The calculated complexity of the expression.
            extrinsic_reward: The reward directly received from the environment.
            embedding: A numerical embedding of the expression, if available, for diversity.
            data: The full raw trajectory data (e.g., from `env.target_data`) needed
                  for evaluating expressions and checking conservation laws.
            variables: A list of `Variable` objects relevant to the environment,
                       needed for symbolic expression evaluation.
            predicted_forward_traj: Dictionary containing predicted conserved quantities
                                    from a forward pass (for entropy penalty).
            predicted_backward_traj: Dictionary containing predicted conserved quantities
                                     from a backward pass (for entropy penalty).

        Returns:
            The total calculated intrinsic reward (unweighted by `self.weight` in individual `BaseReward` components,
            but scaled by the `self.<type>_weight` attributes of this calculator).
            This is meant to be added to the extrinsic reward.
        """

        # Prepare a unified `info` dictionary to pass to individual BaseReward components
        reward_info = {
            'expression': expression,
            'complexity': complexity,
            'variables': variables,
            'trajectory_data': data,
            'predicted_forward_traj': predicted_forward_traj,
            'predicted_backward_traj': predicted_backward_traj
        }

        # Calculate individual intrinsic reward components
        # Note: `calculate_reward` methods of BaseReward subclasses take many args,
        # but only use the ones relevant to them.
        novelty_rew = self.novelty_reward_comp.calculate_reward(None, None, None, extrinsic_reward, False, reward_info)
        complexity_rew = self.complexity_reward_comp.calculate_reward(None, None, None, extrinsic_reward, False, reward_info)
        conservation_rew = self.conservation_reward_comp.calculate_reward(None, None, None, extrinsic_reward, False, reward_info)

        # Diversity reward (depends on internal history of this calculator)
        diversity_rew = self._calculate_diversity_reward(expression, embedding)

        # Complexity growth reward (depends on internal history of this calculator)
        complexity_growth_rew = self._calculate_complexity_growth_reward(complexity)

        # Combine intrinsic rewards with their specific weights defined in this calculator's init
        intrinsic_total = (
            self.novelty_weight * novelty_rew +
            self.diversity_weight * diversity_rew +
            self.complexity_growth_weight * complexity_growth_rew +
            self.conservation_weight * conservation_rew
        )

        # Update history for subsequent steps' calculations
        self.expression_history.append(expression)
        self.complexity_history.append(complexity)
        if embedding is not None:
            self.discovery_embeddings.append(embedding)

        return intrinsic_total # Return only the intrinsic part to be added to extrinsic reward

    def _calculate_diversity_reward(self,
                                  expression: str, # Expression string for history tracking
                                  embedding: Optional[np.ndarray]) -> float:
        """Internal method to calculate diversity reward based on embeddings."""
        if embedding is None or len(self.discovery_embeddings) < 2:
            return 0.0

        # Calculate distance to nearest neighbors from recent history
        distances = []
        # Consider only a subset of recent embeddings to keep computation fast
        recent_embeddings_sample = list(self.discovery_embeddings)[-min(20, len(self.discovery_embeddings)):]
        for past_embedding in recent_embeddings_sample:
            dist = np.linalg.norm(embedding - past_embedding) # Euclidean distance
            distances.append(dist)

        if distances:
            min_dist = np.min(distances)
            mean_dist = np.mean(distances)

            # Reward for being far from closest neighbors
            # Tanh squashes the value between 0 and 1, min_dist/mean_dist emphasizes uniqueness.
            diversity_score = np.tanh(min_dist / (mean_dist + 1e-6)) # Avoid division by zero
            return diversity_score

        return 0.0

    def _calculate_complexity_growth_reward(self, complexity: int) -> float:
        """Internal method to calculate reward for appropriate complexity growth."""
        if len(self.complexity_history) < 2:
            return 0.0

        recent_complexities = list(self.complexity_history)[-min(10, len(self.complexity_history)):]
        mean_complexity = np.mean(recent_complexities)

        if complexity > mean_complexity:
            growth_rate = (complexity - mean_complexity) / (mean_complexity + 1e-6)
            # Reward moderate growth, penalize excessive growth
            if 0 < growth_rate < 0.5: # Moderate growth
                return 0.5 * growth_rate
            elif growth_rate >= 0.5: # Too fast growth
                return -0.2 # Penalty for very rapid growth
            return 0.0
        else:
            # Reward finding simpler expressions or maintaining reasonable complexity
            if complexity < mean_complexity * 0.9: # Significant reduction in complexity
                return 0.3

        return 0.0

    def get_exploration_bonus(self) -> float:
        """Calculates an exploration bonus based on recent stagnation of discoveries."""
        if len(self.expression_history) < 50: # Need sufficient history to detect stagnation
            return 0.0

        recent = list(self.expression_history)[-min(50, len(self.expression_history)):]
        unique_recent = len(set(recent))
        stagnation_ratio = unique_recent / len(recent) # Ratio of unique expressions to total in recent history

        # Higher bonus when stagnating (low unique ratio)
        if stagnation_ratio < 0.3:
            return 0.5 # High stagnation
        elif stagnation_ratio < 0.5:
            return 0.2 # Moderate stagnation
        else:
            return 0.0 # Good diversity, no extra bonus


if __name__ == "__main__":
    # Using mock environment and dependent classes for testing

    class MockEnv(object): # Simplified mock env for testing intrinsic rewards
        def __init__(self, grammar_mock, target_data_mock, variables_mock, max_depth, max_complexity, reward_config=None):
            self.grammar = grammar_mock
            self.target_data = target_data_mock
            self.variables = variables_mock
            self.max_depth = max_depth
            self.max_complexity = max_complexity
            self.current_expr = "x"
            self.episode_num = 0
            self.action_space_n = 10
            self.observation_space = type('obs_space', (object,), {'shape': (5,)})()
            self.current_observation_state = np.zeros(self.observation_space.shape)

        def step(self, action):
            reward = 0.0 # Extrinsic reward from env
            done = False
            truncated = False
            info = {}

            if action % 2 == 0:
                self.current_expr += " + 1"
            else:
                self.current_expr += " * y"

            next_obs = np.random.rand(self.observation_space.shape[0])
            self.current_observation_state = next_obs

            info['expression'] = self.current_expr
            info['variables'] = self.variables

            # Mock trajectory data for ConservationLawReward
            dummy_trajectory_data = {
                'x': np.linspace(0, 10, 100), 'y': np.sin(np.linspace(0, 10, 100)), 'v': np.cos(np.linspace(0, 10, 100)),
                'energy_val': np.full(100, 10.0), 'momentum_val': np.full(100, 5.0)
            }
            info['trajectory_data'] = dummy_trajectory_data

            # Dummy data for entropy penalty (for ConservationLawReward)
            info['predicted_forward_traj'] = {'conserved_energy': np.array([10.0, 9.9, 9.8])}
            info['predicted_backward_traj'] = {'conserved_energy_reversed_path': np.array([9.8, 9.9, 10.0])} # Mock ideal reversal

            if len(self.current_expr) > 20 or self.episode_num >= 5:
                done = True
                self.episode_num = 0
                self.current_expr = "x"
            self.episode_num += 1
            return next_obs, reward, done, truncated, info

        def reset(self, **kwargs):
            self.current_expr = "x"
            self.episode_num = 0
            self.current_observation_state = np.zeros(self.observation_space.shape)
            return self.current_observation_state, {}

        def get_action_mask(self):
            return np.ones(self.action_space_n, dtype=bool)

    class MockGrammar: pass

    # Mock for symbolic_math utilities (get_variables_from_expression, evaluate_expression_on_data)
    try:
        pass
        # from janus.core.expressions.symbolic_math import (
        #     get_variables_from_expression as real_get_vars,
        #     evaluate_expression_on_data as real_eval_on_data
        # )
    except ImportError:
        print("Using mock symbolic_math utilities for intrinsic_rewards.py test.")
        def get_variables_from_expression(expr_str: str, all_variables: List[Variable]) -> List[Variable]:
            return [var for var in all_variables if var.name in expr_str]

        def evaluate_expression_on_data(expr_str: str, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
            if 'x' in expr_str and 'x' in data_dict:
                return data_dict['x'] * 2.0
            return np.full_like(data_dict.get(list(data_dict.keys())[0], np.array([0.0])), 0.0)

    # Mock for math.operations utilities (calculate_symbolic_accuracy, calculate_expression_complexity)
    try:
        pass
        # from janus.utils.math.operations import (
        #     calculate_symbolic_accuracy as real_calc_sym_acc,
        #     calculate_expression_complexity as real_calc_expr_comp
        # )
    except ImportError:
        print("Using mock math.operations utilities for intrinsic_rewards.py test.")
        def calculate_symbolic_accuracy(expr_str: str, ground_truth_dict: Dict[str, Any], var_symbols: List[str]) -> float:
            return 1.0 if len(expr_str) < 10 else 0.5
        def calculate_expression_complexity(expr_str: str) -> int:
            return len(expr_str)

    # --- Test Case Setup for IntrinsicRewardCalculator ---
    print("--- Testing IntrinsicRewardCalculator ---")
    grammar = MockGrammar()
    variables = [Variable("x", 0), Variable("y", 1)] # Use the real Variable class
    target_data = np.random.rand(100, 3)

    env_test = MockEnv(grammar, target_data, variables, max_depth=5, max_complexity=20)

    # Instantiate the IntrinsicRewardCalculator
    intrinsic_calculator = IntrinsicRewardCalculator(
        novelty_weight=0.2,
        diversity_weight=0.1,
        complexity_growth_weight=0.05,
        conservation_weight=0.3,
        conservation_types=['energy'],
        apply_entropy_penalty=True,
        entropy_penalty_weight=0.05,
        history_size=5 # Small history for testing
    )

    # Simulate interaction with the environment, calling the calculator
    obs, info = env_test.reset()
    print("\nStarting environment interaction with IntrinsicRewardCalculator...")
    total_steps_simulated = 10
    for i in range(total_steps_simulated):
        action = np.random.randint(env_test.action_space_n)
        next_obs, extrinsic_reward, done, truncated, info = env_test.step(action)

        expression_embedding = np.random.rand(128) # Dummy embedding for diversity calculation

        total_intrinsic_reward_value = intrinsic_calculator.calculate_intrinsic_reward(
            expression=info.get('expression', 'N/A'),
            complexity=calculate_expression_complexity(info.get('expression', 'N/A')),
            extrinsic_reward=extrinsic_reward,
            embedding=expression_embedding,
            data=info['trajectory_data'], # Pass the trajectory data for conservation check
            variables=info['variables'], # Pass variables for symbolic evaluation
            predicted_forward_traj=info.get('predicted_forward_traj'),
            predicted_backward_traj=info.get('predicted_backward_traj')
        )

        final_reward = extrinsic_reward + total_intrinsic_reward_value

        print(f"\n--- Step {i+1} ---")
        print(f"  Hypothesis: {info.get('expression', 'N/A')}")
        print(f"  Extrinsic Reward: {extrinsic_reward:.4f}")
        print(f"  Intrinsic Reward: {total_intrinsic_reward_value:.4f}")
        print(f"  Final Reward: {final_reward:.4f}")

        if done:
            print("  Episode Done. Resetting environment.")
            obs, info = env_test.reset()
        else:
            obs = next_obs

    print("\nIntrinsicRewardCalculator demonstration complete.")
