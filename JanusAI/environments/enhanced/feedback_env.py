"""
Feedback Environment (EnhancedSymbolicDiscoveryEnv)
==================================================

Implements an enhanced symbolic discovery environment that integrates
richer observations and hooks for intrinsic rewards and adaptive training.

Includes the `EnhancedObservationEncoder` for detailed observation processing.
"""

import numpy as np
import torch # For potential expression embedding
import torch.nn as nn # For dummy model or general neural network structures
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass # For Variable, if ExpressionNode is dataclass

# Import base symbolic environment
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv, TreeState, ExpressionNode # Ensure ExpressionNode is accessible if needed

# Import core expression and grammar (for _get_expression_embedding, _extract_grammar_features)
from janus_ai.core.expressions.expression import Variable # For variable processing
from janus_ai.core.grammar.base_grammar import ProgressiveGrammar # For grammar features

# Import intrinsic reward calculator
from janus_ai.ml.rewards.intrinsic_rewards import IntrinsicRewardCalculator

# Placeholder for HypothesisNet if not directly imported or for type hinting
try:
    from janus_ai.ml.networks.hypothesis_net import HypothesisNet # For _get_expression_embedding
except ImportError:
    print("Warning: HypothesisNet not found. _get_expression_embedding might be limited.")
    HypothesisNet = type('DummyHypothesisNet', (object,), {}) # Dummy class


class EnhancedObservationEncoder:
    """
    Encodes observations with rich contextual and historical information
    to provide a more comprehensive state representation to the RL agent.
    """

    def __init__(self,
                 base_dim: int = 128, # Expected feature dim of a base observation (e.g., encoded tree node)
                 history_length: int = 10): # How many past steps to track history for

        self.base_dim = base_dim
        self.history_length = history_length

        # History tracking for actions, rewards, and expressions (for novelty/diversity context)
        self.action_history: Deque[int] = deque(maxlen=history_length)
        self.reward_history: Deque[float] = deque(maxlen=history_length)
        self.expression_history: Deque[str] = deque(maxlen=history_length)

        # A simple linear encoder for historical context (can be more complex, e.g., LSTM)
        # Input size: history_length for actions + history_length for rewards + history_length for expression (e.g., complexity/embedding if available)
        # For simplicity, let's assume raw action/reward history is enough, or it aggregates features.
        # Here we'll make its input dynamic based on the concatenated features.
        # The `context_encoder` might be better placed in the HypothesisNet's observation processing.
        # For now, it's illustrative.
        self.context_encoder = nn.Sequential(
            nn.Linear(history_length * 2 + 4, 64), # Example: 2 for action/reward, 4 for basic expression stats
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def enhance_observation(self,
                          base_obs: np.ndarray,
                          current_state: TreeState, # The internal tree state object from SymbolicDiscoveryEnv
                          grammar: ProgressiveGrammar # The grammar instance
                          ) -> np.ndarray:
        """
        Combines a base observation with rich contextual features derived from
        tree structure, historical data, and grammar state.

        Args:
            base_obs: The raw observation from `SymbolicDiscoveryEnv`.
            current_state: The current `TreeState` object (from symbolic_env).
            grammar: The `ProgressiveGrammar` instance.

        Returns:
            A concatenated NumPy array representing the enhanced observation.
        """

        enhanced_features_list = []

        # 1. Base observation
        enhanced_features_list.append(base_obs.flatten()) # Ensure it's 1D

        # 2. Tree structure features
        tree_features = self._extract_tree_features(current_state)
        enhanced_features_list.append(tree_features)

        # 3. Historical context features
        history_features = self._encode_history()
        # Pass through context encoder
        history_features_encoded = self.context_encoder(torch.FloatTensor(history_features).unsqueeze(0)).squeeze(0).cpu().numpy()
        enhanced_features_list.append(history_features_encoded)

        # 4. Grammar state features
        grammar_features = self._extract_grammar_features(grammar)
        enhanced_features_list.append(grammar_features)

        # 5. Current complexity budget features
        complexity_features = self._extract_complexity_features(current_state)
        enhanced_features_list.append(complexity_features)

        # Concatenate all features into a single flat array
        enhanced_obs = np.concatenate(enhanced_features_list)

        return enhanced_obs

    def _extract_tree_features(self, state: TreeState) -> np.ndarray:
        """Extract structural features from the current expression tree."""

        features = []

        # Depth statistics: mean, max, std of node depths
        depths = self._get_node_depths(state.root)
        features.extend([
            np.mean(depths) if depths else 0,
            np.max(depths) if depths else 0,
            np.std(depths) if depths and len(depths) > 1 else 0 # Ensure std calculation for >1 element
        ])

        # Node type distribution: counts of 'operator', 'variable', 'constant', 'empty'
        node_types_counts = self._count_node_types(state.root)
        total_nodes = sum(node_types_counts.values()) + 1e-6 # Avoid division by zero
        features.extend([
            node_types_counts.get('operator', 0) / total_nodes,
            node_types_counts.get('variable', 0) / total_nodes,
            node_types_counts.get('constant', 0) / total_nodes,
            node_types_counts.get('empty', 0) / total_nodes # Count of empty placeholders
        ])

        # Completion status (is the tree a valid, complete expression?)
        features.append(1.0 if state.is_complete() else 0.0)

        # Tree balance (e.g., difference in size of left/right subtrees for binary ops)
        balance = self._calculate_tree_balance(state.root)
        features.append(balance)

        return np.array(features, dtype=np.float32)

    def _get_node_depths(self, node: ExpressionNode, depth: int = 0) -> List[int]:
        """Recursively get depths of all nodes in the expression tree."""
        if node.node_type.value == "empty": # Base case for empty nodes
            return []

        depths = [depth]
        for child in node.children: # Assuming ExpressionNode has 'children' attribute
            depths.extend(self._get_node_depths(child, depth + 1))

        return depths

    def _count_node_types(self, node: ExpressionNode) -> Dict[str, int]:
        """Recursively count node types (operator, variable, constant, empty) in tree."""
        if node.node_type.value == "empty":
            return {"empty": 1}

        counts = {node.node_type.value: 1} # Count current node
        for child in node.children:
            child_counts = self._count_node_types(child)
            for node_type, count in child_counts.items():
                counts[node_type] = counts.get(node_type, 0) + count

        return counts

    def _calculate_tree_balance(self, node: ExpressionNode) -> float:
        """Calculates a simple tree balance metric (e.g., for binary trees)."""
        if not node.children or len(node.children) < 2: # Not a branching node or not binary
            return 0.0

        # For binary nodes, compare size of first two children's subtrees
        size1 = self._get_subtree_size(node.children[0])
        size2 = self._get_subtree_size(node.children[1])

        total_size = size1 + size2 + 1e-6 # Avoid division by zero
        balance = abs(size1 - size2) / total_size # Normalized difference

        return balance

    def _get_subtree_size(self, node: ExpressionNode) -> int:
        """Recursively calculate the number of nodes in a subtree."""
        if node.node_type.value == "empty":
            return 0
        return 1 + sum(self._get_subtree_size(child) for child in node.children)

    def _encode_history(self) -> np.ndarray:
        """Encodes action, reward, and expression history into a fixed-size vector."""

        # Pad histories to `history_length` if they are shorter
        action_vec = list(self.action_history) + [0] * (self.history_length - len(self.action_history))
        reward_vec = list(self.reward_history) + [0] * (self.history_length - len(self.reward_history))

        # Simple statistics for expression history (e.g., mean complexity, mean length)
        expr_lengths = [len(e) for e in self.expression_history if e] or [0]


        expr_features = np.array([
            np.mean(expr_lengths),
            np.max(expr_lengths),
            len(set(self.expression_history)) / self.history_length # Uniqueness ratio
        ], dtype=np.float32)

        # Concatenate numerical history vectors
        numerical_history = np.concatenate([
            np.array(action_vec, dtype=np.float32),
            np.array(reward_vec, dtype=np.float32)
        ])

        return np.concatenate([numerical_history, expr_features])

    def _extract_grammar_features(self, grammar: ProgressiveGrammar) -> np.ndarray:
        """Extract features about the current state of the grammar."""

        features = []

        # Number of learned functions (if grammar supports this, e.g., in a progressive grammar)
        n_learned_functions = len(getattr(grammar, 'learned_functions', {}))
        features.append(n_learned_functions / 10.0)  # Normalize by a typical max

        # Number of current variables available in the grammar's context
        n_variables_in_grammar = len(getattr(grammar, 'variables', {})) # Accessing .variables from grammar directly
        features.append(n_variables_in_grammar / 10.0)  # Normalize

        # Grammar complexity (e.g., total number of primitive rules/operators)
        # This assumes `grammar.primitives` is a dict of lists of operators.
        n_primitives = sum(len(ops) for ops in grammar.primitives.values())
        features.append(n_primitives / 20.0) # Normalize by a typical max

        # Current grammar depth or stage if it's progressive
        current_grammar_stage = getattr(grammar, 'current_stage', 0)
        features.append(current_grammar_stage / 5.0) # Normalize by max stages

        return np.array(features, dtype=np.float32)

    def _extract_complexity_features(self, state: TreeState) -> np.ndarray:
        """Extract features about the complexity budget and current tree complexity."""

        features = []

        current_complexity = state.count_nodes() # Assuming TreeState has count_nodes
        # Access max_complexity from the environment if state doesn't hold it
        max_complexity = getattr(state, 'max_complexity', 30) # Default if not found in state

        # Complexity usage ratio
        features.append(current_complexity / max_complexity)

        # Remaining complexity budget
        remaining_complexity = max_complexity - current_complexity
        features.append(remaining_complexity / max_complexity)

        # Complexity per depth: current complexity divided by current max depth
        current_max_depth = self._get_max_depth(state.root)
        if current_max_depth > 0:
            features.append(current_complexity / current_max_depth)
        else:
            features.append(0.0) # Avoid division by zero

        return np.array(features, dtype=np.float32)

    def _get_max_depth(self, node: ExpressionNode, depth: int = 0) -> int:
        """Recursively get maximum depth of the expression tree."""
        if node.node_type.value == "empty" or not node.children:
            return depth

        return max(self._get_max_depth(child, depth + 1) for child in node.children)

    def update_history(self, action: int, reward: float, expression: str):
        """Updates internal history trackers after each environment step."""
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.expression_history.append(expression) # Expression string is tracked for novelty


class EnhancedSymbolicDiscoveryEnv(SymbolicDiscoveryEnv):
    """
    An enhanced version of SymbolicDiscoveryEnv that integrates:
    - Richer observation encoding via `EnhancedObservationEncoder`.
    - Calculation of intrinsic rewards via `IntrinsicRewardCalculator`.
    - Tracking of episode-specific metrics for adaptive training (e.g., by an external controller).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Initialize the intrinsic reward calculator and observation encoder
        self.intrinsic_calculator = IntrinsicRewardCalculator(
            novelty_weight=kwargs.get('novelty_weight', 0.3),
            diversity_weight=kwargs.get('diversity_weight', 0.2),
            complexity_growth_weight=kwargs.get('complexity_growth_weight', 0.1),
            conservation_weight=kwargs.get('conservation_weight', 0.4),
            conservation_types=kwargs.get('conservation_types', ['energy', 'momentum']),
            apply_entropy_penalty=kwargs.get('apply_entropy_penalty', False),
            entropy_penalty_weight=kwargs.get('entropy_penalty_weight', 0.1),
            history_size=kwargs.get('history_size', 1000)
        )
        self.observation_encoder = EnhancedObservationEncoder()

        # Metrics to be tracked per episode for external adaptive controllers
        self.episode_discoveries: List[str] = [] # All expressions generated in current episode
        self.episode_complexities: List[int] = [] # Complexities of generated expressions
        self.episode_rewards_list: List[float] = [] # List of rewards within current episode

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Performs one step in the environment, calculates enhanced rewards,
        and produces an enhanced observation.
        """
        # Call the standard SymbolicDiscoveryEnv's step method
        base_obs, extrinsic_reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        # --- Intrinsic Reward Calculation ---
        # The intrinsic calculator needs the *current* state of the expression being built.
        # This is typically found in `self.current_state` and info['expression'].

        # Extract relevant info for intrinsic rewards
        current_expression_str = info.get('expression', '')
        current_expression_complexity = info.get('complexity', 0)

        # If the environment itself doesn't generate data for 'trajectory_data' or 'variables' in info,
        # it needs to be provided by the environment's `reset` or other means.
        # Assuming `info['trajectory_data']` and `info['variables']` are populated by the environment.
        trajectory_data = info.get('trajectory_data', np.array([]))
        variables = info.get('variables', [])

        # Get expression embedding for diversity (simple dummy for now if HypothesisNet not used)
        expression_embedding = self._get_expression_embedding(current_expression_str)

        total_intrinsic_reward_value = self.intrinsic_calculator.calculate_intrinsic_reward(
            expression=current_expression_str,
            complexity=current_expression_complexity,
            extrinsic_reward=extrinsic_reward, # Pass the direct reward from env
            embedding=expression_embedding,
            data=trajectory_data,
            variables=variables,
            predicted_forward_traj=info.get('predicted_forward_traj'),
            predicted_backward_traj=info.get('predicted_backward_traj')
        )

        # Combine extrinsic and intrinsic rewards
        combined_reward = extrinsic_reward + total_intrinsic_reward_value

        # --- Observation Enhancement ---
        # The observation encoder needs the raw obs, current tree state, and grammar
        enhanced_obs = self.observation_encoder.enhance_observation(
            base_obs,
            self.current_state,
            self.grammar
        )

        # --- Update Histories and Metrics ---
        # Update observation encoder's internal history
        self.observation_encoder.update_history(
            action,
            combined_reward, # Update history with the *combined* reward
            current_expression_str
        )

        # Track episode-specific metrics for external adaptive controllers
        self.episode_rewards_list.append(combined_reward) # Track all rewards
        self.episode_discoveries.append(current_expression_str)
        self.episode_complexities.append(current_expression_complexity)

        # Update info dictionary with enhanced reward details
        info['intrinsic_reward_components_sum'] = total_intrinsic_reward_value
        info['combined_reward'] = combined_reward # The new reward to be used by the agent
        # The main 'reward' return value is also the combined one

        return enhanced_obs, combined_reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment for a new episode.
        Also resets episode-specific metrics and intrinsic reward calculator's state if needed.
        """
        # Call base class reset
        base_obs, info = super().reset(seed=seed, options=options)

        # Reset episode-specific metrics
        self.reset_episode_metrics()

        # Optionally, reset internal state of intrinsic_calculator here if its history needs to be episode-bound.
        # For now, intrinsic_calculator histories (like novelty) are long-term, not per-episode.

        # Ensure observation encoder history is reset or initialized for the new episode
        self.observation_encoder.update_history(0, 0.0, "") # Reset or initial state

        # Enhance the initial observation
        enhanced_initial_obs = self.observation_encoder.enhance_observation(
            base_obs,
            self.current_state,
            self.grammar
        )

        return enhanced_initial_obs, info

    def _get_expression_embedding(self, expr: str) -> np.ndarray:
        """
        Generates a simplified embedding for an expression.
        In a real system, this might use a dedicated ExpressionEncoder.
        """
        # Simple feature extraction based on string properties
        features = []

        # Operator counts
        ops = ['+', '-', '*', '/', '**', 'sin', 'cos', 'log', 'exp']
        for op in ops:
            features.append(expr.count(op))

        # Variable usage: need to iterate through environment's actual variables
        # For this mock, assume variables from the grammar/env are simple strings
        # In a real scenario, `self.variables` (list of Variable objects) would be used.
        mock_variables_names = [v.name for v in self.variables] if hasattr(self, 'variables') else ['x','y','z']
        for var_name in mock_variables_names:
            features.append(expr.count(var_name))

        # Length and depth approximation
        features.append(len(expr))
        features.append(expr.count('('))  # Parentheses as depth proxy

        return np.array(features, dtype=np.float32)

    def get_adaptation_metrics(self) -> Dict[str, float]:
        """
        Provides episode-level metrics relevant for adaptive training controllers.
        These metrics summarize the agent's performance and discovery patterns
        within the current episode.
        """
        discovery_rate = len(set(self.episode_discoveries)) / (len(self.episode_discoveries) + 1e-6) # Avoid div by zero

        if len(self.episode_complexities) > 1:
            # Simple linear trend of complexity over the episode
            complexity_trend = np.polyfit(
                range(len(self.episode_complexities)),
                self.episode_complexities,
                1
            )[0]
        else:
            complexity_trend = 0.0

        return {
            'discovery_rate': discovery_rate,
            'complexity_trend': complexity_trend,
            'unique_discoveries_episode': len(set(self.episode_discoveries)),
            'mean_complexity_episode': np.mean(self.episode_complexities) if self.episode_complexities else 0,
            'mean_reward_episode': np.mean(self.episode_rewards_list) if self.episode_rewards_list else 0.0,
            'total_steps_episode': len(self.episode_rewards_list)
        }

    def reset_episode_metrics(self):
        """Resets episode-specific tracking metrics."""
        self.episode_discoveries = []
        self.episode_complexities = []
        self.episode_rewards_list = []


if __name__ == "__main__":
    # This __main__ block demonstrates the usage of EnhancedSymbolicDiscoveryEnv.
    # It requires mock versions of SymbolicDiscoveryEnv, Grammar, and Variable if
    # the full janus modules are not available in the test environment.

    # Mock SymbolicDiscoveryEnv for testing EnhancedSymbolicDiscoveryEnv
    # This is a simplified version of what would be in symbolic_env.py
    class MockBaseSymbolicDiscoveryEnv:
        def __init__(self, grammar, target_data, variables, max_depth, max_complexity, reward_config, action_space_size, provide_tree_structure):
            self.grammar = grammar
            self.target_data = target_data # Can be ignored or used for mock rewards
            self.variables = variables # List of Variable objects
            self.max_depth = max_depth
            self.max_complexity = max_complexity
            self.reward_config = reward_config
            self.action_space = type('ActionSpace', (object,), {'n': action_space_size or 5})()
            self.observation_space = type('ObsSpace', (object,), {'shape': (64,)})() # Dummy base obs dim
            self.provide_tree_structure = provide_tree_structure
            self.current_state = type('TreeState', (object,), { # Mock TreeState
                'root': type('Node', (object,), {
                    'node_type': type('NT', (object,), {'value': 'operator'})(),
                    'children': [type('Node', (object,), {'node_type': type('NT', (object,), {'value': 'empty'})(), 'children': []})() for _ in range(2)]
                })(),
                'count_nodes': lambda self_node: 3, # Example complexity
                'is_complete': lambda : True # For mock
            })()
            self.current_observation_state = np.zeros(self.observation_space.shape)


        def reset(self, seed=None, options=None):
            self.current_state.root.node_type.value = "operator" # Reset root
            self.current_state.root.children = [type('Node', (object,), {'node_type': type('NT', (object,), {'value': 'empty'})(), 'children': []})() for _ in range(2)]
            initial_obs = np.random.rand(self.observation_space.shape[0])
            info = {'expression': 'x + y', 'complexity': 5} # Mock initial expression info

            # Mock trajectory data for conservation law checks
            info['trajectory_data'] = {
                'x': np.linspace(0, 1, 10), 'y': np.linspace(0, 1, 10), 'v': np.linspace(0, 1, 10),
                'energy_true': np.full(10, 10.0), 'momentum_true': np.full(10, 5.0)
            }
            info['variables'] = self.variables # Pass environment's variables
            info['predicted_forward_traj'] = {'conserved_energy': np.array([10.0, 9.9, 9.8])}
            info['predicted_backward_traj'] = {'conserved_energy_reversed_path': np.array([9.8, 9.9, 10.0])}

            return initial_obs, info

        def step(self, action):
            next_obs = np.random.rand(self.observation_space.shape[0])
            reward = 0.1 # Example extrinsic reward
            done = False
            truncated = False
            # Simulate building an expression
            new_expr = self.current_expr + (f" + {action}" if action % 2 == 0 else f" * {self.variables[0].name}")
            new_complexity = len(new_expr) # Simple complexity mock
            info = {
                'expression': new_expr,
                'complexity': new_complexity
            }
            # Add trajectory_data and variables to info for intrinsic rewards
            info['trajectory_data'] = {
                'x': np.linspace(0, 1, 10), 'y': np.linspace(0, 1, 10), 'v': np.linspace(0, 1, 10),
                'energy_true': np.full(10, 10.0 + np.random.rand()*0.1), # Slight variation
                'momentum_true': np.full(10, 5.0 + np.random.rand()*0.1)
            }
            info['variables'] = self.variables
            info['predicted_forward_traj'] = {'conserved_energy': np.array([10.0, 9.9, 9.8])}
            info['predicted_backward_traj'] = {'conserved_energy_reversed_path': np.array([9.8, 9.9, 10.0])}

            self.current_expr = new_expr # Update for next step's info
            return next_obs, reward, done, truncated, info

        def get_action_mask(self): return np.ones(self.action_space.n, dtype=bool)

    # Mock components for intrinsic_rewards calculations
    # These are needed if intrinsic_rewards.py cannot fully resolve its imports in this test context
    try:
        from janus_ai.core.expressions.expression import Variable as RealVariable
    except ImportError:
        print("Using mock Variable for feedback_env.py test.")
        @dataclass(eq=True, frozen=False)
        class RealVariable:
            name: str
            index: int
            properties: Dict[str, Any] = field(default_factory=dict)
            symbolic: Any = field(init=False)
            def __post_init__(self): self.symbolic = sp.Symbol(self.name)
            def __hash__(self): return hash((self.name, self.index))
            def __str__(self): return self.name
            @property
            def complexity(self) -> int: return 1
    Variable = RealVariable

    # Mock SymbolicDiscoveryEnv for EnhancedSymbolicDiscoveryEnv's inheritance
    import sys
    # Temporarily replace SymbolicDiscoveryEnv in its module for the test
    sys.modules['janus.environments.base.symbolic_env'].SymbolicDiscoveryEnv = MockBaseSymbolicDiscoveryEnv


    # --- Test Case Setup for EnhancedSymbolicDiscoveryEnv ---
    print("--- Testing EnhancedSymbolicDiscoveryEnv ---")
    grammar_mock = MockGrammar()
    variables = [Variable("x", 0), Variable("y", 1)]
    target_data_mock = np.random.rand(100, 3)

    env = EnhancedSymbolicDiscoveryEnv(
        grammar=grammar_mock,
        target_data=target_data_mock,
        variables=variables,
        max_depth=7,
        max_complexity=15,
        action_space_size=10,
        # Pass intrinsic reward specific args to the env's init, which will pass to IntrinsicRewardCalculator
        novelty_weight=0.2,
        conservation_weight=0.3,
        apply_entropy_penalty=True
    )

    # --- Simulate interaction ---
    print("\nEnhanced Environment initialized.")

    obs, info = env.reset()
    print("Initial observation shape (enhanced):", obs.shape)
    print("Info after reset:", info.keys()) # Check for additional info

    print("\nSimulating 3 steps with enhanced environment:")
    for i in range(3):
        action = np.random.randint(env.action_space.n)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"  Step {i+1}: Action={action}, Combined Reward={reward:.4f}, Done={terminated|truncated}")
        print(f"    Enhanced Obs Shape: {next_obs.shape}")
        print(f"    Intrinsic Reward Components Sum: {step_info.get('intrinsic_reward_components_sum'):.4f}")
        print(f"    Combined Reward in Info: {step_info.get('combined_reward'):.4f}")
        print(f"    Expression: {step_info.get('expression')}")

    print("\nEpisode metrics after simulation:")
    metrics = env.get_adaptation_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nEnhancedSymbolicDiscoveryEnv test completed.")

    # Clean up mock
    del sys.modules['janus.environments.base.symbolic_env'].SymbolicDiscoveryEnv
