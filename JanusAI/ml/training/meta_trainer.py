# File: src/janus/ml/training/meta_trainer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import copy
import types
import sympy as sp
from pathlib import Path
import tqdm # Added import

# Updated imports based on the new structure
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus_ai.core.grammar.progressive_grammar import ProgressiveGrammar # Updated import
from janus_ai.core.expressions.expression import Variable
from janus_ai.utils.math.operations import calculate_symbolic_accuracy
from janus_ai.environments.base.symbolic_env import safe_env_reset

# Conditional imports based on the original file's structure and the new one
try:
    from janus_ai.environments.enhanced.feedback_env import IntrinsicRewardCalculator, EnhancedObservationEncoder
except ImportError:
    print("Warning: enhanced_feedback components not found, using basic feedback placeholders")
    IntrinsicRewardCalculator = None
    EnhancedObservationEncoder = None

try:
    from janus_ai.environments.enhanced.adaptive_env import add_intrinsic_rewards_to_env
except ImportError:
    print("Warning: feedback_integration module not found, intrinsic rewards disabled")
    add_intrinsic_rewards_to_env = None

from janus_ai.physics.data.generators import PhysicsTaskDistribution, PhysicsTask


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning"""
    # Meta-learning parameters
    meta_lr: float = 0.0003
    adaptation_lr: float = 0.01
    adaptation_steps: int = 5

    # Task sampling
    tasks_per_batch: int = 10
    support_episodes: int = 10
    query_episodes: int = 10

    # Environment parameters
    max_episode_steps: int = 50
    max_tree_depth: int = 7
    max_complexity: int = 20

    # Training parameters
    meta_iterations: int = 1000
    checkpoint_interval: int = 50
    eval_interval: int = 25

    # Reward configuration
    use_intrinsic_rewards: bool = True
    intrinsic_weight: float = 0.2

    # Logging
    log_dir: str = "./meta_learning_logs"
    checkpoint_dir: str = "./meta_learning_checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MetaLearningPolicy(nn.Module):
    """Enhanced HypothesisNet with meta-learning capabilities"""

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 n_layers: int = 3,
                 use_task_embedding: bool = True):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_task_embedding = use_task_embedding

        # Task encoder for conditioning
        if use_task_embedding:
            self.task_encoder = nn.LSTM(
                observation_dim,
                hidden_dim // 2,
                batch_first=True,
                bidirectional=True
            )

            # Task modulation network
            self.task_modulator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)  # Gains and biases
            )

        # Main policy network (modulated by task)
        layers = []
        input_dim = observation_dim

        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        self.feature_extractor = nn.ModuleList(layers)

        # Heads for actor-critic
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Physics-aware heads
        self.symmetry_detector = nn.Linear(hidden_dim, 10)  # Common symmetries
        self.conservation_predictor = nn.Linear(hidden_dim, 5)  # Conservation laws

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self,
              obs: torch.Tensor,
              task_embedding: Optional[torch.Tensor] = None,
              action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional task conditioning"""

        # Get task modulation if available
        if self.use_task_embedding and task_embedding is not None:
            # Encode task context
            if len(task_embedding.shape) == 2:
                task_embedding = task_embedding.unsqueeze(0)

            _, (hidden, _) = self.task_encoder(task_embedding)
            task_features = hidden.transpose(0, 1).reshape(hidden.shape[1], -1)

            # Get modulation parameters
            modulation = self.task_modulator(task_features)
            gains, biases = modulation.chunk(2, dim=-1)
            gains = 1 + 0.1 * torch.tanh(gains)
        else:
            gains = 1.0
            biases = 0.0

        # Extract features with modulation
        x = obs
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            # Apply modulation after ReLU layers
            if isinstance(layer, nn.ReLU) and i < len(self.feature_extractor) - 1:
                if isinstance(gains, torch.Tensor):
                    x = x * gains + biases

        features = x

        # Compute outputs
        policy_logits = self.policy_head(features)

        # Apply the action mask to the logits before returning
        if action_mask is not None:
            action_mask = action_mask.to(policy_logits.device)
            if action_mask.shape[-1] < policy_logits.shape[-1]:
                padding = torch.zeros(
                    *action_mask.shape[:-1],
                    policy_logits.shape[-1] - action_mask.shape[-1],
                    dtype=torch.bool,
                    device=policy_logits.device
                )
                action_mask = torch.cat([action_mask, padding], dim=-1)
            elif action_mask.shape[-1] > policy_logits.shape[-1]:
                action_mask = action_mask[..., :policy_logits.shape[-1]]

            if action_mask.shape != policy_logits.shape:
                try:
                    action_mask = action_mask.expand_as(policy_logits)
                except RuntimeError as e:
                    if len(action_mask.shape) == 1 and len(policy_logits.shape) == 2:
                        action_mask = action_mask.unsqueeze(0).expand_as(policy_logits)
                    else:
                        raise e

            policy_logits[~action_mask] = float('-inf')

        value = self.value_head(features)

        # Physics predictions (return raw logits)
        symmetry_logits = self.symmetry_detector(features)
        conservation_logits = self.conservation_predictor(features)

        return {
            'policy_logits': policy_logits,
            'value': value,
            'symmetries_logits': symmetry_logits,
            'conservations_logits': conservation_logits,
            'features': features
        }

    def act(self, obs: torch.Tensor, task_embedding: Optional[torch.Tensor] = None, action_mask: Optional[torch.Tensor] = None) -> Tuple[int, Dict]:
        """Select action using current policy"""
        with torch.no_grad():
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)

            outputs = self.forward(obs, task_embedding, action_mask)

            probs = F.softmax(outputs['policy_logits'], dim=-1)

            if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                print(f"Warning: NaN or inf detected in policy probabilities. Logits: {outputs['policy_logits']}")
                if action_mask is not None:
                    current_action_mask = action_mask
                    if current_action_mask.ndim > 1:
                        current_action_mask = current_action_mask[0]

                    valid_actions = torch.where(current_action_mask)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
                    else:
                        action = 0
                        print("Warning: No valid actions in mask during fallback. Defaulting to action 0.")
                else:
                    action = 0

                action_info = {
                    'log_prob': -np.log(probs.shape[-1] if probs.shape[-1] > 0 else 1),
                    'value': outputs['value'].item() if outputs['value'] is not None else 0.0,
                    'entropy': 0.0
                }
                return action, action_info

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            return action.item(), {
                'log_prob': dist.log_prob(action).item(),
                'value': outputs['value'].item(),
                'entropy': dist.entropy().item()
            }


class TaskEnvironmentBuilder:
    """Builds SymbolicDiscoveryEnv from PhysicsTask"""

    def __init__(self, config: MetaLearningConfig):
        self.config = config
        if EnhancedObservationEncoder is None:
            print("Warning: EnhancedObservationEncoder is not available.")
        self.observation_encoder = EnhancedObservationEncoder()

    def build_env(self, task: PhysicsTask, max_action_space: Optional[int] = None) -> SymbolicDiscoveryEnv:
        """Create environment for specific physics task"""

        data = task.generate_data(1000, noise=True)

        variables = []
        for i, var_name in enumerate(task.variables[:-1]):
            var_properties = {}

            if var_name in task.physical_parameters:
                var_properties['is_constant'] = True
                var_properties['value'] = task.physical_parameters[var_name]

            variables.append(Variable(var_name, i, var_properties))

        grammar = self._create_task_grammar(task)

        reward_config = {
            'mse_weight': -1.0,
            'complexity_penalty': -0.005,
            'parsimony_bonus': 0.1,
        }

        if task.symmetries and "none" not in task.symmetries:
            reward_config['symmetry_bonus'] = 0.2

        # Determine X_data and y_data from the combined 'data'
        # Assuming the last column of 'data' is the target (y_data)
        # and the rest are features (X_data).
        # This might need to be made more flexible or configurable per task.
        if data.shape[1] < 2:
            raise ValueError(f"Data for task {task.name} has fewer than 2 columns. Cannot split into X and y.")

        X_data = data[:, :-1]
        y_data = data[:, -1]

        # Ensure variables list matches the number of features in X_data
        # The original variables list was created based on task.variables[:-1],
        # which assumes the last variable in task.variables corresponds to the target.
        # If X_data is data[:, :-1], then the number of features is data.shape[1] - 1.
        # The number of variables in `task.variables` should be data.shape[1].

        # Adjust 'variables' list to only include those corresponding to X_data.
        # The original 'variables' list was created based on task.variables[:-1],
        # which should align if task.variables includes the target variable name.
        if len(variables) != X_data.shape[1]:
            # This case implies a mismatch between task.variables definition and data splitting.
            # For now, we'll try to proceed if task.variables[:-1] was intended for X.
            # A more robust solution might involve explicit feature/target mapping in PhysicsTask.
            print(f"Warning: Mismatch in TaskEnvironmentBuilder for task {task.name}. "
                  f"Number of X_data columns ({X_data.shape[1]}) "
                  f"does not match derived variables count ({len(variables)}). "
                  f"Original task.variables: {task.variables}. Data columns: {data.shape[1]}")
            # If task.variables was meant to list all columns including target,
            # and variables was created from task.variables[:-1], this should be fine.

        env = SymbolicDiscoveryEnv(
            grammar=grammar,
            X_data=X_data,
            y_data=y_data,
            variables=variables, # This list should correspond to columns of X_data
            max_depth=self.config.max_tree_depth,
            max_complexity=self.config.max_complexity,
            reward_config=reward_config,
            action_space_size=max_action_space
        )

        if not hasattr(env, 'get_action_mask'):
            def get_action_mask(self_env):
                if hasattr(self_env, 'current_state') and hasattr(self_env.current_state, 'get_valid_actions'):
                    valid_actions = self_env.current_state.get_valid_actions()
                    mask = np.zeros(self_env.action_space.n, dtype=bool)
                    mask[valid_actions] = True
                    return mask
                else:
                    return np.ones(self_env.action_space.n, dtype=bool)

            env.get_action_mask = types.MethodType(get_action_mask, env)

        if self.config.use_intrinsic_rewards and add_intrinsic_rewards_to_env is not None:
            try:
                add_intrinsic_rewards_to_env(env, weight=self.config.intrinsic_weight)
            except Exception as e:
                print(f"Warning: Could not add intrinsic rewards: {e}")
        elif self.config.use_intrinsic_rewards and add_intrinsic_rewards_to_env is None:
            print("Warning: Intrinsic rewards enabled in config, but feedback_integration module not found. Skipping.")

        env.task_info = {
            'name': task.name,
            'true_law': task.true_law,
            'domain': task.domain,
            'difficulty': task.difficulty,
            'symmetries': task.symmetries
        }

        return env

    def _create_task_grammar(self, task: PhysicsTask) -> ProgressiveGrammar:
        """Create grammar with operators appropriate for task"""
        grammar = ProgressiveGrammar(load_defaults=True)

        if task.domain == "mechanics":
            grammar.add_operators(['**2', 'sqrt'])
            if "pendulum" in task.name:
                grammar.add_operators(['sin', 'cos'])

        elif task.domain == "thermodynamics":
            grammar.add_operators(['log', 'exp', '**'])

        elif task.domain == "electromagnetism":
            grammar.add_operators(['**2', '1/'])

        if "**" in task.true_law:
            grammar.add_operators(['**'])

        return grammar


class MAMLTrainer:
    """Main MAML training logic for physics discovery"""

    def __init__(self,
                 config: MetaLearningConfig,
                 policy: MetaLearningPolicy,
                 task_distribution: PhysicsTaskDistribution):

        self.config = config
        self.policy = policy.to(config.device)
        self.task_distribution = task_distribution
        self.env_builder = TaskEnvironmentBuilder(config)

        self.meta_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.meta_lr
        )

        self.writer = SummaryWriter(config.log_dir)
        self.iteration = 0

        self.meta_losses = []
        self.task_metrics = defaultdict(list)
        self.discovered_laws = defaultdict(list)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def meta_train_step(self) -> Dict[str, float]:
        """Single meta-training step across multiple tasks"""

        meta_loss = 0
        meta_metrics = defaultdict(float)

        tasks = self.task_distribution.sample_task_batch(
            self.config.tasks_per_batch,
            curriculum=True
        )

        task_gradients = []

        for task_idx, task in enumerate(tasks):
            env = self.env_builder.build_env(task, max_action_space=self.policy.action_dim)

            adapted_policy = self._clone_policy()
            inner_optimizer = torch.optim.SGD(
                adapted_policy.parameters(),
                lr=self.config.adaptation_lr
            )

            support_trajectories = self._collect_trajectories(
                self.policy,
                env,
                n_episodes=self.config.support_episodes,
                task_context=None
            )

            task_embedding = self._compute_task_embedding(support_trajectories)

            for adapt_step in range(self.config.adaptation_steps):
                support_loss = self._compute_trajectory_loss(
                    adapted_policy,
                    support_trajectories,
                    task_embedding,
                    task
                )

                inner_optimizer.zero_grad()
                support_loss.backward()
                inner_optimizer.step()

            query_trajectories = self._collect_trajectories(
                adapted_policy,
                env,
                n_episodes=self.config.query_episodes,
                task_context=task_embedding
            )

            query_loss = self._compute_trajectory_loss(
                adapted_policy,
                query_trajectories,
                task_embedding,
                task
            )

            task_metrics = self._compute_task_metrics(
                query_trajectories,
                task,
                adapted_policy
            )

            for key, value in task_metrics.items():
                meta_metrics[key] += value / self.config.tasks_per_batch

            task_grad = torch.autograd.grad(
                query_loss,
                self.policy.parameters(),
                retain_graph=True,
                allow_unused=True
            )
            task_gradients.append(task_grad)

            self._log_task_performance(task, task_metrics, task_idx)

            meta_loss += query_loss

        self.meta_optimizer.zero_grad()

        for param_idx, param in enumerate(self.policy.parameters()):
            valid_grads = [g[param_idx] for g in task_gradients if g is not None and g[param_idx] is not None]
            if valid_grads:
                param.grad = sum(valid_grads) / len(valid_grads)

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

        self.meta_optimizer.step()

        avg_meta_loss = meta_loss.item() / self.config.tasks_per_batch
        self.meta_losses.append(avg_meta_loss)
        meta_metrics['meta_loss'] = avg_meta_loss

        return meta_metrics

    def _clone_policy(self) -> MetaLearningPolicy:
        """Create a functional clone of the policy for adaptation"""
        cloned = copy.deepcopy(self.policy)
        cloned.load_state_dict(self.policy.state_dict())
        return cloned

    def _collect_trajectories(self,
                            policy: MetaLearningPolicy,
                            env: SymbolicDiscoveryEnv,
                            n_episodes: int,
                            task_context: Optional[torch.Tensor] = None) -> List[Dict]:
        """Collect trajectories using given policy"""

        trajectories = []

        for episode in range(n_episodes):
            obs, _ = safe_env_reset(env)
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': [],
                'dones': [],
                'infos': []
            }

            episode_reward = 0

            for step in range(self.config.max_episode_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.config.device)

                action_mask_np = env.get_action_mask()
                action_mask = torch.BoolTensor(action_mask_np).to(self.config.device)

                action, action_info = policy.act(obs_tensor, task_context, action_mask)

                next_obs, reward, done, truncated, info = env.step(action)

                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['log_probs'].append(action_info['log_prob'])
                trajectory['values'].append(action_info['value'])
                trajectory['dones'].append(done or truncated)
                trajectory['infos'].append(info)

                episode_reward += reward
                obs = next_obs

                if done or truncated:
                    break

            trajectory['episode_reward'] = episode_reward
            trajectory['episode_length'] = len(trajectory['actions'])

            if trajectory['infos'] and 'expression' in trajectory['infos'][-1]:
                trajectory['discovered_expression'] = trajectory['infos'][-1]['expression']
                trajectory['discovery_mse'] = trajectory['infos'][-1].get('mse', float('inf'))

            trajectories.append(trajectory)

        return trajectories

    def _compute_task_embedding(self, trajectories: List[Dict]) -> torch.Tensor:
        """
        Computes the task context by collecting all observations from support trajectories.
        This sequence will be fed into the policy's task encoder (LSTM).
        """
        all_obs = []
        for traj in trajectories:
            if 'observations' in traj and traj['observations']:
                all_obs.extend(traj['observations'])

        if not all_obs:
            obs_dim = self.policy.observation_dim
            return torch.zeros((1, obs_dim), device=self.config.device)

        return torch.FloatTensor(all_obs).to(self.config.device)

    def _compute_trajectory_loss(self,
                                policy: MetaLearningPolicy,
                                trajectories: List[Dict],
                                task_embedding: torch.Tensor,
                                task: PhysicsTask) -> torch.Tensor:
        """Compute policy gradient loss for trajectories"""

        total_loss = 0

        for traj in trajectories:
            obs = torch.FloatTensor(traj['observations']).to(self.config.device)
            actions = torch.LongTensor(traj['actions']).to(self.config.device)
            rewards = torch.FloatTensor(traj['rewards']).to(self.config.device)

            outputs = policy(obs, task_embedding)

            log_probs = F.log_softmax(outputs['policy_logits'], dim=-1)

            if actions.dim() == 1:
                actions_expanded = actions.unsqueeze(1)
            else:
                actions_expanded = actions

            num_actions_from_logits = log_probs.size(1)
            actions_clamped = torch.clamp(actions_expanded, 0, num_actions_from_logits - 1)

            selected_log_probs = log_probs.gather(1, actions_clamped).squeeze(-1)

            returns = self._compute_returns(rewards)
            advantages = returns - outputs['value'].squeeze()

            policy_loss = -(selected_log_probs * advantages.detach()).mean()

            value_loss = F.mse_loss(outputs['value'].squeeze(), returns)

            entropy = -(log_probs * log_probs.exp()).sum(dim=-1).mean()

            physics_loss = 0
            if task.symmetries and "none" not in task.symmetries:
                symmetry_targets = self._create_symmetry_targets(task.symmetries)

                physics_loss = F.binary_cross_entropy_with_logits(
                    outputs['symmetries_logits'].mean(0),
                    symmetry_targets.to(self.config.device)
                )

            traj_loss = (
                policy_loss +
                0.5 * value_loss -
                0.01 * entropy +
                0.1 * physics_loss
            )

            total_loss += traj_loss

        return total_loss / len(trajectories)

    def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def _create_symmetry_targets(self, symmetries: List[str]) -> torch.Tensor:
        """Create binary targets for symmetry detection"""
        symmetry_types = [
            'time_reversal', 'spatial_translation', 'rotational',
            'galilean', 'scale_invariance', 'charge_conjugation',
            'energy_conservation', 'momentum_conservation',
            'angular_momentum_conservation', 'none'
        ]

        targets = torch.zeros(len(symmetry_types))
        for i, sym_type in enumerate(symmetry_types):
            if sym_type in symmetries:
                targets[i] = 1.0

        return targets

    def _compute_task_metrics(self,
                            trajectories: List[Dict],
                            task: PhysicsTask,
                            policy: MetaLearningPolicy) -> Dict[str, float]:
        """Compute task-specific metrics"""

        metrics = {
            'discovery_rate': 0,
            'avg_episode_reward': 0,
            'avg_episode_length': 0,
            'best_mse': float('inf'),
            'correct_discovery': 0
        }

        discoveries = []

        for traj in trajectories:
            metrics['avg_episode_reward'] += traj['episode_reward']
            metrics['avg_episode_length'] += traj['episode_length']

            if 'discovered_expression' in traj:
                expr = traj['discovered_expression']
                mse = traj['discovery_mse']

                discoveries.append(expr)
                metrics['discovery_rate'] += 1
                metrics['best_mse'] = min(metrics['best_mse'], mse)

                if self._expression_matches(expr, task.true_law):
                    metrics['correct_discovery'] += 1

        n_traj = len(trajectories)
        metrics['discovery_rate'] /= n_traj
        metrics['avg_episode_reward'] /= n_traj
        metrics['avg_episode_length'] /= n_traj
        metrics['correct_discovery'] /= n_traj

        metrics['unique_discoveries'] = len(set(discoveries))

        self.discovered_laws[task.name].extend(discoveries)

        return metrics

    def _expression_matches(self, expr: str, target: str, tol: float = 0.01) -> bool:
        """Check if discovered expression matches target using symbolic accuracy."""
        if not expr or not target:
            return False
        try:
            target_expr = sp.sympify(target)
            ground_truth_dict = {'true_law': target_expr}

            accuracy = calculate_symbolic_accuracy(expr, ground_truth_dict)

            return accuracy > 0.99
        except (sp.SympifyError, TypeError) as e:
            print(f"Error sympifying target expression: {target}. Error: {e}")
            return False

    def _log_task_performance(self,
                            task: PhysicsTask,
                            metrics: Dict[str, float],
                            task_idx: int):
        """Log individual task performance"""

        prefix = f"task_{task_idx}_{task.name}"

        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, self.iteration)

        self.writer.add_scalar(f"{prefix}/difficulty", task.difficulty, self.iteration)

    def evaluate_on_new_tasks(self, n_tasks: int = 10) -> Dict[str, float]:
        """Evaluate meta-learned policy on unseen tasks"""

        eval_metrics = defaultdict(list)

        eval_tasks = self.task_distribution.sample_task_batch(
            n_tasks,
            curriculum=False
        )

        for task in tqdm(eval_tasks, desc="Evaluating"):
            env = self.env_builder.build_env(task, max_action_space=self.policy.action_dim)

            adapted_policy = self._clone_policy()
            inner_optimizer = torch.optim.SGD(
                adapted_policy.parameters(),
                lr=self.config.adaptation_lr
            )

            adapt_trajectories = self._collect_trajectories(
                self.policy,
                env,
                n_episodes=self.config.support_episodes
            )

            task_embedding = self._compute_task_embedding(adapt_trajectories)

            for _ in range(self.config.adaptation_steps):
                loss = self._compute_trajectory_loss(
                    adapted_policy,
                    adapt_trajectories,
                    task_embedding,
                    task
                )

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            test_trajectories = self._collect_trajectories(
                adapted_policy,
                env,
                n_episodes=20,
                task_context=task_embedding
            )

            task_metrics = self._compute_task_metrics(
                test_trajectories,
                task,
                adapted_policy
            )

            for key, value in task_metrics.items():
                eval_metrics[key].append(value)
            eval_metrics['task_difficulty'].append(task.difficulty)
            eval_metrics['task_domain'].append(task.domain)

        aggregated = {}
        for key, values in eval_metrics.items():
            if key != 'task_domain':
                aggregated[f"eval/{key}_mean"] = np.mean(values)
                aggregated[f"eval/{key}_std"] = np.std(values)

        return aggregated

    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint"""
        if path is None:
            path = f"{self.config.checkpoint_dir}/checkpoint_{self.iteration}.pt"

        torch.save({
            'iteration': self.iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_losses': self.meta_losses,
            'discovered_laws': dict(self.discovered_laws),
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path)

        self.iteration = checkpoint['iteration']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.meta_losses = checkpoint['meta_losses']
        self.discovered_laws = defaultdict(list, checkpoint['discovered_laws'])

    def train(self):
        """Main training loop"""

        print(f"Starting MAML training for {self.config.meta_iterations} iterations")
        print(f"Tasks per batch: {self.config.tasks_per_batch}")
        print(f"Support episodes: {self.config.support_episodes}")
        print(f"Query episodes: {self.config.query_episodes}")
        print("-" * 50)

        for iteration in range(self.config.meta_iterations):
            self.iteration = iteration

            start_time = time.time()
            meta_metrics = self.meta_train_step()
            step_time = time.time() - start_time

            for key, value in meta_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, iteration)
            self.writer.add_scalar("train/step_time", step_time, iteration)

            if iteration % 10 == 0:
                print(f"\nIteration {iteration}/{self.config.meta_iterations}")
                print(f"  Meta loss: {meta_metrics['meta_loss']:.4f}")
                print(f"  Discovery rate: {meta_metrics['discovery_rate']:.3f}")
                print(f"  Correct discoveries: {meta_metrics['correct_discovery']:.3f}")
                print(f"  Unique discoveries: {meta_metrics['unique_discoveries']:.1f}")
                print(f"  Step time: {step_time:.2f}s")

            if iteration % self.config.eval_interval == 0 and iteration > 0:
                print("\nEvaluating on new tasks...")
                eval_metrics = self.evaluate_on_new_tasks()

                for key, value in eval_metrics.items():
                    self.writer.add_scalar(key, value, iteration)

                print(f"  Eval discovery rate: {eval_metrics['eval/discovery_rate_mean']:.3f} ± {eval_metrics['eval/discovery_rate_std']:.3f}")
                print(f"  Eval correct rate: {eval_metrics['eval/correct_discovery_mean']:.3f} ± {eval_metrics['eval/correct_discovery_std']:.3f}")

            if iteration % self.config.checkpoint_interval == 0 and iteration > 0:
                self.save_checkpoint()
                print(f"  Saved checkpoint at iteration {iteration}")

            if iteration % 50 == 0:
                self._log_discovered_laws()

        print("\nTraining complete!")
        self.save_checkpoint(f"{self.config.checkpoint_dir}/final_checkpoint.pt")

    def _log_discovered_laws(self):
        """Log summary of discovered laws"""
        print("\nDiscovered Laws Summary:")
        print("-" * 50)

        for task_name, discoveries in self.discovered_laws.items():
            if discoveries:
                unique_discoveries = list(set(discoveries))
                print(f"\n{task_name}:")
                for i, expr in enumerate(unique_discoveries[:5]):
                    count = discoveries.count(expr)
                    print(f"  {i+1}. {expr} (found {count} times)")