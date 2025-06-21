# JanusAI/environments/base/symbolic_env.py
"""
Refactored Symbolic Discovery Environment
========================================

This module contains the base environment for symbolic regression and discovery tasks.
The refactored version separates input features (X) and target values (y) for clarity.
"""

import gym
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
from enum import Enum, auto

# Import necessary modules (adjust paths as needed)
from JanusAI.core.grammar.base_grammar import ProgressiveGrammar
from JanusAI.core.expressions.expression import Expression, Variable
from JanusAI.utils.math.operations import calculate_expression_complexity
from JanusAI.core.expressions.symbolic_math import evaluate_expression_on_data
from memory.dual_memory_system import SharedMemory # Added import


class NodeType(Enum):
    """Node types in expression tree."""
    EMPTY = auto()
    CONSTANT = auto()
    VARIABLE = auto()
    OPERATOR = auto()


@dataclass
class ExpressionNode:
    """A node in the expression tree."""
    node_type: NodeType
    value: Any = None
    children: List['ExpressionNode'] = field(default_factory=list)
    depth: int = 0
    
    def is_complete(self) -> bool:
        """Check if this node and all its descendants are complete."""
        if self.node_type == NodeType.EMPTY:
            return False
        
        if self.node_type == NodeType.OPERATOR:
            # Check if operator has required number of children
            required_children = self._get_required_children()
            if len(self.children) < required_children:
                return False
            # Recursively check all children
            return all(child.is_complete() for child in self.children)
        
        # Constants and variables are always complete
        return True
    
    def _get_required_children(self) -> int:
        """Get required number of children for this operator."""
        # This should be determined based on the operator type
        # For now, assume binary operators need 2, unary need 1
        if self.value in ['+', '-', '*', '/', '**', 'max', 'min']:
            return 2
        elif self.value in ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'neg']:
            return 1
        else:
            return 2  # Default to binary


@dataclass
class TreeState:
    """State of the expression tree being built."""
    root: Optional[ExpressionNode] = None
    current_node: Optional[ExpressionNode] = None
    incomplete_nodes: List[ExpressionNode] = field(default_factory=list)
    depth: int = 0
    node_count: int = 0
    
    def is_complete(self) -> bool:
        """Check if the tree represents a complete expression."""
        if self.root is None:
            return False
        return self.root.is_complete() and len(self.incomplete_nodes) == 0
    
    def to_expression(self) -> Optional[Expression]:
        """Convert tree to Expression object."""
        if not self.is_complete():
            return None
        return self._node_to_expression(self.root)
    
    def _node_to_expression(self, node: ExpressionNode) -> Expression:
        """Recursively convert node to Expression."""
        if node.node_type == NodeType.CONSTANT:
            return Expression('const', [node.value])
        elif node.node_type == NodeType.VARIABLE:
            return node.value  # Assuming value is a Variable object
        elif node.node_type == NodeType.OPERATOR:
            child_expressions = [self._node_to_expression(child) for child in node.children]
            return Expression(node.value, child_expressions)
        else:
            raise ValueError(f"Invalid node type: {node.node_type}")


class SymbolicDiscoveryEnv(gym.Env):
    """
    Refactored environment for symbolic expression discovery.
    
    This environment separates input features (X) and target values (y)
    for clearer semantics and easier extension.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        grammar: ProgressiveGrammar,
        X_data: np.ndarray,
        y_data: np.ndarray,
        variables: List[Variable],
        max_depth: int = 10,
        max_complexity: int = 30,
        reward_config: Optional[Dict[str, float]] = None,
        max_nodes: int = 50,
        action_space_size: Optional[int] = None,
        provide_tree_structure: bool = False
    ):
        """
        Initialize the symbolic discovery environment.
        
        Args:
            grammar: Grammar defining available operations and rules
            X_data: Input features array of shape (n_samples, n_features)
            y_data: Target values array of shape (n_samples,) or (n_samples, n_outputs)
            variables: List of Variable objects corresponding to X_data columns
            max_depth: Maximum depth of expression trees
            max_complexity: Maximum complexity score for expressions
            reward_config: Configuration for reward calculation
            max_nodes: Maximum nodes in observation representation
            action_space_size: Size of discrete action space
            provide_tree_structure: Whether to include tree structure in observations
        """
        super().__init__()
        
        # Store data separately
        self.X_data = X_data
        self.y_data = y_data
        self.variables = variables
        
        # Validate inputs
        self._validate_inputs()
        
        # Configuration
        self.grammar = grammar
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.max_nodes = max_nodes
        self.provide_tree_structure = provide_tree_structure
        
        # Reward configuration with defaults
        self.reward_config = reward_config or {}
        self._set_default_reward_config()
        
        # Action and observation spaces
        self.action_space_size = action_space_size or self._calculate_action_space_size()
        self.action_space = gym.spaces.Discrete(self.action_space_size)
        
        # Observation space
        obs_dim = self._calculate_observation_dimension()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Episode state
        self.current_state: Optional[TreeState] = None
        self.episode_steps = 0
        self.max_episode_steps = max_depth * 3  # Heuristic

        # Initialize Shared Memory
        self.shared_memory = SharedMemory(capacity=10)
        
    def _validate_inputs(self):
        """Validate input data consistency."""
        # Check X_data shape
        if self.X_data.ndim != 2:
            raise ValueError(f"X_data must be 2D array, got shape {self.X_data.shape}")
        
        # Check y_data shape
        if self.y_data.ndim not in [1, 2]:
            raise ValueError(f"y_data must be 1D or 2D array, got shape {self.y_data.shape}")
        
        # Ensure y_data is 2D for consistency
        if self.y_data.ndim == 1:
            self.y_data = self.y_data.reshape(-1, 1)
        
        # Check sample count consistency
        n_samples_X = self.X_data.shape[0]
        n_samples_y = self.y_data.shape[0]
        if n_samples_X != n_samples_y:
            raise ValueError(
                f"X_data and y_data must have same number of samples. "
                f"Got X: {n_samples_X}, y: {n_samples_y}"
            )
        
        # Check variables match X_data columns
        n_features = self.X_data.shape[1]
        n_variables = len(self.variables)
        if n_features != n_variables:
            raise ValueError(
                f"Number of variables ({n_variables}) must match "
                f"number of features in X_data ({n_features})"
            )
    
    def _set_default_reward_config(self):
        """Set default reward configuration values."""
        defaults = {
            'mse_weight': -1.0,
            'complexity_penalty': -0.01,
            'depth_penalty': -0.001,
            'invalid_penalty': -10.0,
            'completion_bonus': 1.0,
            'parsimony_bonus': 0.1,
            'timeout_penalty': -5.0,
            'mse_scale_factor': 1.0,
        }
        for key, value in defaults.items():
            self.reward_config.setdefault(key, value)
    
    def _calculate_action_space_size(self) -> int:
        """Calculate the size of the action space based on grammar."""
        # Count all possible actions
        n_constants = len(self.grammar.primitives.get('constants', {}))
        n_variables = len(self.variables)
        n_operators = sum(
            len(self.grammar.primitives.get(op_type, {}))
            for op_type in ['unary_ops', 'binary_ops', 'calculus_ops']
        )
        return n_constants + n_variables + n_operators + 1  # +1 for no-op
    
    def _calculate_observation_dimension(self) -> int:
        """Calculate observation space dimension."""
        base_dim = 5  # Basic state features
        tree_dim = self.max_nodes * 4 if self.provide_tree_structure else 0
        return base_dim + tree_dim
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize new episode
        self.current_state = TreeState()
        self.episode_steps = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.episode_steps += 1
        
        # Execute action
        action_valid = self._execute_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action_valid)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        # Add high-scoring attempts to shared memory
        # Using a placeholder threshold of 0.8, adjust as needed.
        reward_threshold = 0.8
        if reward > reward_threshold and self.current_state and self.current_state.is_complete():
            current_expr_str = info.get('expression')
            if current_expr_str:
                self.shared_memory.add({
                    'expression': current_expr_str,
                    'reward': reward,
                    'metadata': {
                        'complexity': info.get('complexity', -1),
                        'depth': info.get('tree_depth', -1),
                        'steps': self.episode_steps,
                    }
                })
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> bool:
        """Execute the given action and update the tree state."""
        # Map action to grammar element
        element = self._action_to_element(action)
        if element is None:
            return False
        
        # Apply element to current tree state
        return self._apply_element_to_tree(element)
    
    def _action_to_element(self, action: int) -> Optional[Tuple[str, Any]]:
        """Map action index to grammar element."""
        # Implementation depends on how grammar elements are indexed
        # This is a simplified version
        idx = 0
        
        # Constants
        constants = self.grammar.primitives.get('constants', {})
        if action < idx + len(constants):
            const_list = list(constants.items())
            return ('constant', const_list[action - idx])
        idx += len(constants)
        
        # Variables
        if action < idx + len(self.variables):
            return ('variable', self.variables[action - idx])
        idx += len(self.variables)
        
        # Operators
        for op_type in ['unary_ops', 'binary_ops', 'calculus_ops']:
            ops = self.grammar.primitives.get(op_type, {})
            if action < idx + len(ops):
                op_list = list(ops)
                return ('operator', op_list[action - idx])
            idx += len(ops)
        
        return None  # Invalid action
    
    def _apply_element_to_tree(self, element: Tuple[str, Any]) -> bool:
        """Apply a grammar element to the current tree state."""
        elem_type, elem_value = element
        
        # If tree is empty, start with root
        if self.current_state.root is None:
            if elem_type == 'operator':
                # Start with operator as root
                node = ExpressionNode(NodeType.OPERATOR, elem_value)
                self.current_state.root = node
                self.current_state.current_node = node
                self.current_state.incomplete_nodes.append(node)
                self.current_state.node_count = 1
                return True
            else:
                # Start with constant or variable
                node_type = NodeType.CONSTANT if elem_type == 'constant' else NodeType.VARIABLE
                node = ExpressionNode(node_type, elem_value)
                self.current_state.root = node
                self.current_state.node_count = 1
                return True
        
        # Find next incomplete position
        if not self.current_state.incomplete_nodes:
            return False  # Tree is complete
        
        # Add to first incomplete node
        parent = self.current_state.incomplete_nodes[0]
        
        # Create new node
        if elem_type == 'operator':
            node = ExpressionNode(NodeType.OPERATOR, elem_value, depth=parent.depth + 1)
            self.current_state.incomplete_nodes.append(node)
        elif elem_type == 'constant':
            _, const_value = elem_value  # Unpack (name, value) tuple
            node = ExpressionNode(NodeType.CONSTANT, const_value, depth=parent.depth + 1)
        else:  # variable
            node = ExpressionNode(NodeType.VARIABLE, elem_value, depth=parent.depth + 1)
        
        # Add as child
        parent.children.append(node)
        self.current_state.node_count += 1
        
        # Check if parent is now complete
        if len(parent.children) >= parent._get_required_children():
            self.current_state.incomplete_nodes.remove(parent)
        
        # Update tree depth
        self.current_state.depth = max(self.current_state.depth, node.depth)
        
        return True
    
    def _calculate_reward(self, action_valid: bool) -> float:
        """Calculate reward for the current state."""
        # Invalid action penalty
        if not action_valid:
            return self.reward_config['invalid_penalty']
        
        # Check if expression is complete
        if not self.current_state.is_complete():
            # Small negative reward for incomplete expressions
            return -0.01
        
        # Convert to expression
        expression = self.current_state.to_expression()
        if expression is None:
            return self.reward_config['invalid_penalty']
        
        # Evaluate expression on data
        try:
            # Use the refactored evaluation that takes variables list
            predictions = evaluate_expression_on_data(
                str(expression),
                [var.name for var in self.variables],
                self.X_data
            )
            
            # Handle multi-output case
            if self.y_data.shape[1] == 1:
                y_true = self.y_data[:, 0]
            else:
                # For multi-output, use first output or implement specific logic
                y_true = self.y_data[:, 0]
                warnings.warn("Multi-output not fully supported yet, using first output")
            
            # Calculate MSE
            mse = np.mean((predictions - y_true) ** 2)
            
            # Scale MSE
            scaled_mse = mse / (self.reward_config['mse_scale_factor'] + 1e-8)
            
            # Calculate complexity
            complexity = calculate_expression_complexity(str(expression))
            
            # Combine rewards
            reward = (
                self.reward_config['mse_weight'] * scaled_mse +
                self.reward_config['complexity_penalty'] * complexity +
                self.reward_config['depth_penalty'] * self.current_state.depth +
                self.reward_config['completion_bonus']
            )
            
            # Add parsimony bonus for simple accurate expressions
            if mse < 0.1 and complexity < 10:
                reward += self.reward_config['parsimony_bonus']
            
            return float(reward)
            
        except Exception as e:
            # Expression evaluation failed
            warnings.warn(f"Expression evaluation failed: {e}")
            return self.reward_config['invalid_penalty']
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if expression is complete and valid
        if self.current_state and self.current_state.is_complete():
            expression = self.current_state.to_expression()
            if expression is not None:
                # Could add additional checks here
                return True
        
        # Terminate if tree is too deep
        if self.current_state and self.current_state.depth >= self.max_depth:
            return True
        
        # Terminate if too complex
        if self.current_state and self.current_state.node_count >= self.max_complexity:
            return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated."""
        return self.episode_steps >= self.max_episode_steps
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Basic features
        features = [
            float(self.episode_steps) / self.max_episode_steps,
            float(self.current_state.depth if self.current_state else 0) / self.max_depth,
            float(self.current_state.node_count if self.current_state else 0) / self.max_complexity,
            float(len(self.current_state.incomplete_nodes) if self.current_state else 0),
            float(self.current_state.is_complete() if self.current_state else 0),
        ]
        
        # Add tree structure if requested
        if self.provide_tree_structure and self.current_state and self.current_state.root:
            tree_features = self._tree_to_tensor(self.current_state.root)
            features.extend(tree_features.flatten())
        
        return np.array(features, dtype=np.float32)
    
    def _tree_to_tensor(self, root: ExpressionNode) -> np.ndarray:
        """Convert tree to tensor representation."""
        # Simple implementation - can be enhanced
        tensor = np.zeros((self.max_nodes, 4), dtype=np.float32)
        
        # BFS traversal
        queue = [(root, 0)]
        idx = 0
        
        while queue and idx < self.max_nodes:
            node, parent_idx = queue.pop(0)
            
            # Encode node
            tensor[idx, 0] = float(node.node_type.value)
            tensor[idx, 1] = float(hash(str(node.value)) % 1000) / 1000  # Simple hash encoding
            tensor[idx, 2] = float(node.depth)
            tensor[idx, 3] = float(parent_idx)
            
            # Add children to queue
            for child in node.children:
                queue.append((child, idx))
            
            idx += 1
        
        return tensor
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        info = {
            'episode_steps': self.episode_steps,
            'tree_depth': self.current_state.depth if self.current_state else 0,
            'node_count': self.current_state.node_count if self.current_state else 0,
            'is_complete': self.current_state.is_complete() if self.current_state else False,
        }
        
        # Add expression string if complete
        if self.current_state and self.current_state.is_complete():
            expression = self.current_state.to_expression()
            if expression:
                info['expression'] = str(expression)
                info['complexity'] = calculate_expression_complexity(str(expression))
        
        return info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the current state."""
        if mode == 'human':
            print(f"\nEpisode Step: {self.episode_steps}")
            if self.current_state:
                print(f"Tree Depth: {self.current_state.depth}")
                print(f"Node Count: {self.current_state.node_count}")
                print(f"Incomplete Nodes: {len(self.current_state.incomplete_nodes)}")
                if self.current_state.is_complete():
                    expression = self.current_state.to_expression()
                    print(f"Expression: {expression}")
            else:
                print("No tree state")
        return None