# File: JanusAI/ml/training/enhanced_meta_trainer.py
"""
Enhanced MAML Trainer that leverages attention-based task adaptation.

This trainer extends the basic MAMLTrainer to utilize the enhanced
MetaLearningPolicy's attention mechanisms and auxiliary outputs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import wandb
from pathlib import Path
import json

from torch.utils.tensorboard import SummaryWriter

from janus.ml.networks.attention_meta_policy import EnhancedMetaLearningPolicy
from janus.physics.data.generators import PhysicsTask, PhysicsTaskDistribution
from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv


@dataclass
class EnhancedMetaLearningConfig:
    """Extended configuration with attention-specific parameters"""
    
    # Base MAML parameters
    meta_lr: float = 0.0003
    adaptation_lr: float = 0.01
    adaptation_steps: int = 5
    tasks_per_batch: int = 10
    support_episodes: int = 10
    query_episodes: int = 10
    
    # Enhanced policy parameters
    num_attention_heads: int = 4
    fusion_type: str = "attention"  # "attention", "gating", "bilinear"
    use_hierarchical_encoding: bool = True
    
    # Multi-task learning weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    complexity_loss_weight: float = 0.1
    convergence_loss_weight: float = 0.1
    attention_entropy_weight: float = 0.01  # Encourage focused attention
    
    # Attention visualization
    save_attention_maps: bool = True
    attention_save_interval: int = 100
    
    # Training parameters
    max_episode_steps: int = 200
    meta_iterations: int = 1000
    use_intrinsic_rewards: bool = True
    intrinsic_weight: float = 0.2
    
    # Logging
    log_dir: str = "./enhanced_meta_logs"
    checkpoint_dir: str = "./enhanced_meta_checkpoints"
    use_wandb: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AttentionAnalyzer:
    """Analyzes and visualizes attention patterns during training"""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.attention_history = defaultdict(list)
        
    def record_attention(self, task_name: str, iteration: int, 
                        attention_stats: Dict[str, np.ndarray]):
        """Record attention statistics for a task"""
        self.attention_history[task_name].append({
            'iteration': iteration,
            'stats': attention_stats
        })
        
    def save_attention_heatmap(self, attention_weights: np.ndarray, 
                              task_name: str, iteration: int):
        """Save attention heatmap visualization"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention_weights, cmap='hot', aspect='auto')
        ax.set_title(f'Attention Weights - {task_name} (Iteration {iteration})')
        ax.set_xlabel('Tree Position')
        ax.set_ylabel('Batch')
        plt.colorbar(im, ax=ax)
        
        save_path = self.save_dir / f'attention_{task_name}_{iteration}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def analyze_attention_evolution(self, task_name: str) -> Dict[str, List[float]]:
        """Analyze how attention patterns evolve over training"""
        if task_name not in self.attention_history:
            return {}
        
        history = self.attention_history[task_name]
        
        # Extract metrics over time
        entropy_over_time = []
        max_attention_over_time = []
        focus_consistency = []  # How consistent is the attention focus
        
        prev_argmax = None
        for record in history:
            stats = record['stats']
            
            # Average entropy across batch
            entropy_over_time.append(stats['attention_entropy'].mean())
            max_attention_over_time.append(stats['attention_max'].mean())
            
            # Consistency: how often does attention focus on the same position
            current_argmax = stats['attention_argmax']
            if prev_argmax is not None:
                consistency = (current_argmax == prev_argmax).mean()
                focus_consistency.append(consistency)
            prev_argmax = current_argmax
        
        return {
            'entropy': entropy_over_time,
            'max_attention': max_attention_over_time,
            'focus_consistency': focus_consistency
        }


class EnhancedMAMLTrainer:
    """
    Enhanced MAML trainer with attention-based task adaptation.
    
    Key features:
    1. Leverages attention mechanisms for task-specific adaptation
    2. Multi-task learning with auxiliary predictions
    3. Attention analysis and visualization
    4. Improved gradient computation with attention regularization
    """
    
    def __init__(self, 
                 config: EnhancedMetaLearningConfig,
                 policy: Optional[EnhancedMetaLearningPolicy] = None,
                 task_distribution: Optional[PhysicsTaskDistribution] = None):
        
        self.config = config
        self.task_distribution = task_distribution or PhysicsTaskDistribution()
        
        # Initialize enhanced policy if not provided
        if policy is None:
            # Determine dimensions from a sample task
            sample_env = self._create_sample_env()
            obs_dim = sample_env.observation_space.shape[0]
            action_dim = sample_env.action_space.n
            
            self.policy = EnhancedMetaLearningPolicy(
                observation_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=256,
                num_attention_heads=config.num_attention_heads,
                fusion_type=config.fusion_type,
                use_hierarchical_encoding=config.use_hierarchical_encoding
            ).to(config.device)
        else:
            self.policy = policy.to(config.device)
        
        # Optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config.meta_lr
        )
        
        # Attention analyzer
        self.attention_analyzer = AttentionAnalyzer(
            str(Path(config.log_dir) / "attention_maps")
        )
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        if config.use_wandb:
            wandb.init(project="janus-enhanced-maml", config=config.__dict__)
        
        # Tracking
        self.iteration = 0
        self.task_performance = defaultdict(list)
        self.discovered_laws = defaultdict(list)
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_sample_env(self) -> SymbolicDiscoveryEnv:
        """Create a sample environment to determine dimensions"""
        sample_task = self.task_distribution.sample_task()
        data = sample_task.generate_data(100)
        return SymbolicDiscoveryEnv(
            data=data,
            target_expr="unknown",
            max_depth=10
        )
    
    def meta_train_step(self) -> Dict[str, float]:
        """
        Single meta-training step with enhanced task adaptation.
        """
        # Sample batch of tasks
        tasks = self.task_distribution.sample_task_batch(
            self.config.tasks_per_batch,
            curriculum=True
        )
        
        meta_loss = 0.0
        meta_metrics = defaultdict(list)
        
        for task in tasks:
            # Clone policy for inner loop adaptation
            adapted_policy = self._clone_policy()
            
            # Collect support trajectories for adaptation
            support_trajectories = self._collect_trajectories(
                adapted_policy, 
                task, 
                self.config.support_episodes,
                use_attention_stats=True
            )
            
            # Inner loop adaptation with attention
            adapted_policy, inner_metrics = self._inner_loop_adaptation(
                adapted_policy,
                support_trajectories,
                task
            )
            
            # Collect query trajectories with adapted policy
            query_trajectories = self._collect_trajectories(
                adapted_policy,
                task,
                self.config.query_episodes,
                use_attention_stats=True
            )
            
            # Compute meta-loss with attention regularization
            task_loss, task_metrics = self._compute_meta_loss(
                self.policy,
                query_trajectories,
                support_trajectories,
                task
            )
            
            meta_loss += task_loss
            
            # Record metrics
            for key, value in task_metrics.items():
                meta_metrics[key].append(value)
            
            # Analyze attention if enabled
            if self.config.save_attention_maps and self.iteration % self.config.attention_save_interval == 0:
                self._analyze_task_attention(task, support_trajectories, query_trajectories)
        
        # Meta-update
        meta_loss = meta_loss / len(tasks)
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        
        self.meta_optimizer.step()
        
        # Aggregate metrics
        aggregated_metrics = {
            'meta_loss': meta_loss.item()
        }
        for key, values in meta_metrics.items():
            aggregated_metrics[key] = np.mean(values)
        
        # Log metrics
        self._log_metrics(aggregated_metrics)
        
        self.iteration += 1
        
        return aggregated_metrics
    
    def _inner_loop_adaptation(self, policy: EnhancedMetaLearningPolicy,
                              trajectories: List[Dict],
                              task: PhysicsTask) -> Tuple[EnhancedMetaLearningPolicy, Dict]:
        """
        Inner loop adaptation with attention-based updates.
        """
        inner_optimizer = torch.optim.SGD(
            policy.parameters(),
            lr=self.config.adaptation_lr
        )
        
        inner_metrics = defaultdict(list)
        
        for step in range(self.config.adaptation_steps):
            # Prepare task context from trajectories
            task_context = self._prepare_task_context(trajectories)
            
            # Compute adaptation loss
            adaptation_loss = 0.0
            
            for traj in trajectories:
                obs = torch.FloatTensor(traj['observations']).to(self.config.device)
                actions = torch.LongTensor(traj['actions']).to(self.config.device)
                rewards = torch.FloatTensor(traj['rewards']).to(self.config.device)
                
                # Forward pass with attention
                outputs = policy(obs, task_context, return_attention=True)
                
                # Policy loss
                log_probs = F.log_softmax(outputs['policy_logits'], dim=-1)
                selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
                
                returns = self._compute_returns(rewards)
                advantages = returns - outputs['value'].squeeze()
                
                policy_loss = -(selected_log_probs * advantages.detach()).mean()
                value_loss = F.mse_loss(outputs['value'].squeeze(), returns)
                
                # Auxiliary losses (if available)
                aux_loss = 0.0
                if 'auxiliary_outputs' in outputs:
                    aux = outputs['auxiliary_outputs']
                    
                    # Complexity prediction loss
                    if 'expression' in traj['infos'][-1]:
                        true_complexity = len(str(traj['infos'][-1]['expression']))
                        complexity_target = torch.tensor([true_complexity], dtype=torch.float32).to(self.config.device)
                        complexity_loss = F.mse_loss(
                            aux['predicted_complexity'].squeeze(), 
                            complexity_target
                        )
                        aux_loss += self.config.complexity_loss_weight * complexity_loss
                    
                    # Convergence prediction loss
                    convergence_target = torch.tensor([len(traj['rewards'])], dtype=torch.float32).to(self.config.device)
                    convergence_loss = F.mse_loss(
                        aux['predicted_convergence'].squeeze(),
                        convergence_target
                    )
                    aux_loss += self.config.convergence_loss_weight * convergence_loss
                
                # Attention entropy regularization (encourage focused attention)
                if 'attention_weights' in outputs:
                    attention_entropy = -(outputs['attention_weights'] * 
                                        torch.log(outputs['attention_weights'] + 1e-10)).sum()
                    aux_loss += self.config.attention_entropy_weight * attention_entropy
                
                # Total loss
                total_loss = (self.config.policy_loss_weight * policy_loss +
                             self.config.value_loss_weight * value_loss +
                             aux_loss)
                
                adaptation_loss += total_loss
                
                # Record metrics
                inner_metrics['policy_loss'].append(policy_loss.item())
                inner_metrics['value_loss'].append(value_loss.item())
                if aux_loss > 0:
                    inner_metrics['aux_loss'].append(aux_loss.item())
            
            # Update adapted policy
            adaptation_loss = adaptation_loss / len(trajectories)
            
            inner_optimizer.zero_grad()
            adaptation_loss.backward(retain_graph=True)
            inner_optimizer.step()
            
            inner_metrics['adaptation_loss'].append(adaptation_loss.item())
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in inner_metrics.items()}
        
        return policy, avg_metrics
    
    def _compute_meta_loss(self, policy: EnhancedMetaLearningPolicy,
                          query_trajectories: List[Dict],
                          support_trajectories: List[Dict],
                          task: PhysicsTask) -> Tuple[torch.Tensor, Dict]:
        """
        Compute meta-loss with enhanced features.
        """
        # Prepare task context from support trajectories
        task_context = self._prepare_task_context(support_trajectories)
        
        total_loss = 0.0
        metrics = defaultdict(list)
        
        for traj in query_trajectories:
            obs = torch.FloatTensor(traj['observations']).to(self.config.device)
            actions = torch.LongTensor(traj['actions']).to(self.config.device)
            rewards = torch.FloatTensor(traj['rewards']).to(self.config.device)
            
            # Forward pass
            outputs = policy(obs, task_context, return_attention=True)
            
            # Compute losses (similar to inner loop but with meta-policy)
            log_probs = F.log_softmax(outputs['policy_logits'], dim=-1)
            
            # Handle action dimension mismatch
            if actions.max() >= log_probs.size(1):
                actions = torch.clamp(actions, 0, log_probs.size(1) - 1)
            
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            returns = self._compute_returns(rewards)
            advantages = returns - outputs['value'].squeeze()
            
            policy_loss = -(selected_log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(outputs['value'].squeeze(), returns)
            
            # Total trajectory loss
            traj_loss = (self.config.policy_loss_weight * policy_loss +
                        self.config.value_loss_weight * value_loss)
            
            total_loss += traj_loss
            
            # Record metrics
            metrics['policy_loss'].append(policy_loss.item())
            metrics['value_loss'].append(value_loss.item())
            metrics['episode_reward'].append(traj['episode_reward'])
            
            # Track discovery success
            if 'discovered_expression' in traj:
                metrics['discovery_success'].append(1.0)
                metrics['discovery_mse'].append(traj.get('discovery_mse', float('inf')))
            else:
                metrics['discovery_success'].append(0.0)
        
        # Average loss
        total_loss = total_loss / len(query_trajectories)
        
        # Average metrics
        avg_metrics = {f"meta/{k}": np.mean(v) for k, v in metrics.items()}
        avg_metrics['task_name'] = task.name
        
        return total_loss, avg_metrics
    
    def _prepare_task_context(self, trajectories: List[Dict]) -> torch.Tensor:
        """
        Prepare task context tensor from trajectories for the enhanced policy.
        """
        # Extract observations from all trajectories
        all_obs = []
        for traj in trajectories:
            if 'observations' in traj and traj['observations']:
                all_obs.extend(traj['observations'])
        
        if not all_obs:
            # Return dummy context
            return torch.zeros(
                (1, 1, 10, self.policy.observation_dim), 
                device=self.config.device
            )
        
        # Convert to tensor and reshape for enhanced policy
        # Shape: (batch_size=1, num_trajectories, trajectory_length, feature_dim)
        obs_array = np.array(all_obs)
        
        # Reshape into trajectories (assuming fixed length for simplicity)
        num_traj = min(len(trajectories), 5)  # Use up to 5 trajectories
        traj_len = min(len(all_obs) // num_traj, 50)  # Max 50 steps per trajectory
        
        if traj_len == 0:
            traj_len = 1
        
        # Truncate observations
        obs_array = obs_array[:num_traj * traj_len]
        
        # Reshape
        try:
            context = obs_array.reshape(1, num_traj, traj_len, -1)
        except:
            # Fallback to simple context
            context = obs_array[:traj_len].reshape(1, 1, traj_len, -1)
        
        return torch.FloatTensor(context).to(self.config.device)
    
    def _collect_trajectories(self, policy: EnhancedMetaLearningPolicy,
                            task: PhysicsTask, 
                            n_episodes: int,
                            use_attention_stats: bool = False) -> List[Dict]:
        """
        Collect trajectories with optional attention statistics.
        """
        trajectories = []
        
        for episode in range(n_episodes):
            # Create environment for task
            data = task.generate_data(200, noise=True)
            env = SymbolicDiscoveryEnv(
                data=data,
                target_expr=task.true_law if hasattr(task, 'true_law') else "unknown",
                max_depth=12,
                max_complexity=30
            )
            
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'attention_stats': [] if use_attention_stats else None
            }
            
            obs, _ = env.reset()
            episode_reward = 0
            
            # Use previous trajectories as context (if available)
            task_context = self._prepare_task_context(trajectories[-3:]) if trajectories else None
            
            for step in range(self.config.max_episode_steps):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                
                # Get action mask
                action_mask = torch.BoolTensor(env.get_action_mask()).to(self.config.device)
                
                # Act with policy
                with torch.no_grad():
                    outputs = policy(obs_tensor, task_context, return_attention=use_attention_stats)
                    action, action_info = policy.act(
                        obs_tensor.squeeze(0), 
                        task_context,
                        action_mask,
                        deterministic=False
                    )
                
                # Step environment
                next_obs, reward, done, truncated, info = env.step(action)
                
                # Record trajectory
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['values'].append(action_info['value'])
                
                # Record attention stats if requested
                if use_attention_stats and 'attention_weights' in outputs:
                    attention_stats = {
                        'weights': outputs['attention_weights'].cpu().numpy(),
                        'entropy': action_info.get('attention_entropy', 0.0)
                    }
                    trajectory['attention_stats'].append(attention_stats)
                
                episode_reward += reward
                obs = next_obs
                
                if done or truncated:
                    break
            
            trajectory['episode_reward'] = episode_reward
            trajectory['episode_length'] = len(trajectory['actions'])
            
            # Record discovered expression if any
            if info and 'expression' in info:
                trajectory['discovered_expression'] = info['expression']
                trajectory['discovery_mse'] = info.get('mse', float('inf'))
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _analyze_task_attention(self, task: PhysicsTask, 
                               support_trajectories: List[Dict],
                               query_trajectories: List[Dict]):
        """
        Analyze and visualize attention patterns for a task.
        """
        # Prepare contexts
        support_context = self._prepare_task_context(support_trajectories)
        
        # Get sample observations
        sample_obs = []
        for traj in query_trajectories[:3]:  # Use first 3 trajectories
            if traj['observations']:
                sample_obs.extend(traj['observations'][:10])  # First 10 steps
        
        if not sample_obs:
            return
        
        obs_tensor = torch.FloatTensor(sample_obs).to(self.config.device)
        
        # Get attention statistics
        attention_stats = self.policy.get_attention_stats(obs_tensor, support_context)
        
        if attention_stats:
            # Record statistics
            self.attention_analyzer.record_attention(
                task.name, 
                self.iteration, 
                attention_stats
            )
            
            # Save heatmap
            if 'attention_weights' in attention_stats:
                self.attention_analyzer.save_attention_heatmap(
                    attention_stats['attention_weights'].squeeze(),
                    task.name,
                    self.iteration
                )
    
    def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _clone_policy(self) -> EnhancedMetaLearningPolicy:
        """Create a clone of the policy for inner loop adaptation."""
        cloned = EnhancedMetaLearningPolicy(
            observation_dim=self.policy.observation_dim,
            action_dim=self.policy.action_dim,
            hidden_dim=self.policy.hidden_dim,
            num_attention_heads=self.config.num_attention_heads,
            fusion_type=self.config.fusion_type,
            use_hierarchical_encoding=self.config.use_hierarchical_encoding
        ).to(self.config.device)
        
        cloned.load_state_dict(self.policy.state_dict())
        return cloned
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tensorboard and wandb."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.iteration)
        
        if self.config.use_wandb:
            wandb.log(metrics, step=self.iteration)
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        if path is None:
            path = Path(self.config.checkpoint_dir) / f"checkpoint_{self.iteration}.pt"
        
        # Get attention evolution analysis
        attention_evolution = {}
        for task_name in ['harmonic_oscillator', 'pendulum', 'double_pendulum']:
            evolution = self.attention_analyzer.analyze_attention_evolution(task_name)
            if evolution:
                attention_evolution[task_name] = evolution
        
        checkpoint = {
            'iteration': self.iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config.__dict__,
            'task_performance': dict(self.task_performance),
            'discovered_laws': dict(self.discovered_laws),
            'attention_evolution': attention_evolution
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.iteration = checkpoint['iteration']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.task_performance = defaultdict(list, checkpoint['task_performance'])
        self.discovered_laws = defaultdict(list, checkpoint['discovered_laws'])
        
        print(f"Loaded checkpoint from {path} (iteration {self.iteration})")
    
    def train(self, n_iterations: Optional[int] = None):
        """Main training loop."""
        n_iterations = n_iterations or self.config.meta_iterations
        
        print("Starting Enhanced MAML Training")
        print(f"Config: {json.dumps(self.config.__dict__, indent=2)}")
        print("=" * 50)
        
        for i in range(n_iterations):
            metrics = self.meta_train_step()
            
            if i % 10 == 0:
                print(f"\nIteration {i}/{n_iterations}")
                print(f"Meta Loss: {metrics['meta_loss']:.4f}")
                print(f"Discovery Success: {metrics.get('meta/discovery_success', 0):.2%}")
                
                # Analyze attention evolution periodically
                if i % 100 == 0 and i > 0:
                    for task_name in ['harmonic_oscillator', 'pendulum']:
                        evolution = self.attention_analyzer.analyze_attention_evolution(task_name)
                        if evolution and 'entropy' in evolution:
                            print(f"\n{task_name} attention entropy trend: ", end="")
                            recent_entropy = evolution['entropy'][-10:]
                            if len(recent_entropy) > 1:
                                trend = "decreasing" if recent_entropy[-1] < recent_entropy[0] else "increasing"
                                print(f"{trend} (from {recent_entropy[0]:.3f} to {recent_entropy[-1]:.3f})")
            
            if i % 100 == 0:
                self.save_checkpoint()


# Example usage
if __name__ == "__main__":
    config = EnhancedMetaLearningConfig(
        meta_iterations=1000,
        tasks_per_batch=5,
        num_attention_heads=4,
        fusion_type="attention",
        use_hierarchical_encoding=True,
        save_attention_maps=True,
        use_wandb=True
    )
    
    trainer = EnhancedMAMLTrainer(config)
    trainer.train()