# src/janus/experiments/runner/distributed_runner.py

"""
Distributed Training Infrastructure for Janus
============================================

This module contains components for scaling training across multiple GPUs/nodes using Ray,
with asynchronous evaluation and parallel hypothesis testing.
"""

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.util.placement_group import placement_group

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from collections import OrderedDict

# --- Internal Janus Imports (adjusted for new structure) ---
from janus.ml.networks.hypothesis_net import HypothesisNet
from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv # Assuming this is the new path
from janus.core.grammar.base_grammar import ProgressiveGrammar # Assuming this is the new path for grammar
from janus.core.expressions.expression import Variable # Assuming this is the new path for Variable


class RLlibHypothesisNet(TorchModelV2, nn.Module):
    """Wrapper to make HypothesisNet compatible with RLlib."""
    
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs, 
                 model_config, 
                 name):
        
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        # Extract custom config
        custom_config = model_config.get("custom_model_config", {})
        
        # Create underlying HypothesisNet
        self.hypothesis_net = HypothesisNet(
            observation_dim=obs_space.shape[0],
            action_dim=action_space.n,
            hidden_dim=custom_config.get("hidden_dim", 256),
            encoder_type=custom_config.get("encoder_type", "transformer"),
            grammar=custom_config.get("grammar", None)
        )
        
        self._last_tree_repr = None
    
    @override(TorchModelV2)
    def forward(self, 
                input_dict: Dict[str, Any], 
                state: List[Any], 
                seq_lens: Any) -> Tuple[torch.Tensor, List[Any]]:
        
        obs = input_dict["obs"]
        
        # Get action mask if available
        action_mask = input_dict.get("action_mask", None)
        
        # Forward through HypothesisNet
        outputs = self.hypothesis_net(obs, action_mask)
        
        # Store tree representation for value function
        self._last_tree_repr = outputs['tree_representation']
        
        # Return logits for RLlib
        return outputs['action_logits'], state
    
    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        """Value function based on last tree representation."""
        if self._last_tree_repr is None:
            return torch.tensor(0.0)
        
        return self.hypothesis_net.value_net(self._last_tree_repr).squeeze()


@ray.remote(num_gpus=0.1)
class AsyncExpressionEvaluator:
    """Asynchronous expression evaluator for parallel hypothesis testing."""
    
    def __init__(self, grammar: ProgressiveGrammar):
        self.grammar = grammar
        self.cache = OrderedDict()
        self.max_cache_size = 4096  # Max number of expressions to cache
    
    def evaluate_batch(self, 
                      expressions: List[str],
                      data: np.ndarray,
                      variables: List[Variable]) -> List[Dict[str, float]]:
        """Evaluate multiple expressions in parallel."""
        
        results = []
        
        for expr_str in expressions:
            if expr_str in self.cache:
                self.cache.move_to_end(expr_str)  # Mark as recently used
                results.append(self.cache[expr_str])
                continue
            
            # If cache is full and expr_str is new, remove the oldest item
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)

            try:
                # Parse expression
                # Assuming create_expression takes type and args, and 'var' is a valid type
                expr = self.grammar.create_expression('var', [expr_str]) 
                
                # Evaluate on data
                predictions = []
                for i in range(len(data)):
                    subs = {var.symbolic: data[i, var.index] for var in variables}
                    pred = float(expr.symbolic.subs(subs)) # sympy expression.subs()
                    predictions.append(pred)
                
                # Calculate metrics
                targets = data[:, -1]
                mse = np.mean((np.array(predictions) - targets) ** 2)
                
                # Physics-specific metrics (placeholder, integrate actual metrics later)
                variance_ratio = np.var(predictions) / np.var(targets)
                
                result = {
                    'mse': mse,
                    'complexity': expr.complexity,
                    'variance_ratio': variance_ratio,
                    'valid': True
                }
                
            except Exception as e:
                # Log the error for debugging
                print(f"Error evaluating expression '{expr_str}': {e}")
                result = {
                    'mse': float('inf'),
                    'complexity': float('inf'),
                    'variance_ratio': 0.0,
                    'valid': False,
                    'error': str(e)
                }
            
            self.cache[expr_str] = result
            results.append(result)
        
        return results


@ray.remote(num_cpus=2, num_gpus=0.5)
class DistributedExperimentWorker:
    """Worker for running distributed experiments."""
    
    def __init__(self, 
                 worker_id: int,
                 grammar: ProgressiveGrammar,
                 base_config: Dict[str, Any]):
        
        self.worker_id = worker_id
        self.grammar = grammar
        self.config = base_config
        
        # Create local environment
        self.env = self._create_env()
        
        # Evaluator pool
        self.evaluator = AsyncExpressionEvaluator.remote(grammar)
        
    def _create_env(self) -> SymbolicDiscoveryEnv:
        """Create environment with worker-specific configuration."""
        
        # Worker-specific modifications
        config = self.config.copy()
        
        # Vary hyperparameters across workers for diversity in search
        config['max_depth'] = config['max_depth'] + (self.worker_id % 3) - 1
        # Ensure reward_config exists before trying to access its keys
        if 'reward_config' in config:
            config['reward_config']['mse_weight'] *= (1 + 0.1 * (self.worker_id % 5))
        
        return SymbolicDiscoveryEnv(**config)
    
    def run_experiment(self, 
                      policy_weights: Dict,
                      n_episodes: int = 10) -> Dict[str, Any]:
        """Run experiment with given policy weights."""
        
        # Create and load policy
        # Ensure the HypothesisNet initialization matches what's expected from its new location
        policy = HypothesisNet(
            observation_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_dim=256 # Default, can be made configurable
        )
        policy.load_state_dict(policy_weights)
        policy.eval() # Set policy to evaluation mode
        
        # Collect episodes
        episodes_data = []
        discovered_expressions = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                # Get action from the policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    # Assuming get_action_mask() exists and returns a boolean mask
                    mask_tensor = torch.BoolTensor(self.env.get_action_mask()).unsqueeze(0) 
                    
                    action_logits, value_estimate, tree_representation = policy(obs_tensor, mask_tensor)
                    # Apply mask and sample action
                    action_probs = torch.softmax(action_logits, dim=-1)
                    # Filter based on mask if necessary, then sample
                    action = torch.multinomial(action_probs, 1).item() # Simple sampling for now
                
                # Step the environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    if 'expression' in info:
                        discovered_expressions.append(info['expression'])
                    
                    episodes_data.append({
                        'reward': episode_reward,
                        'complexity': info.get('complexity', 0),
                        'mse': info.get('mse', float('inf'))
                    })
                    break
        
        # Batch evaluate discovered expressions
        eval_results = []
        if discovered_expressions:
            eval_results = ray.get(
                self.evaluator.evaluate_batch.remote(
                    discovered_expressions,
                    self.config['target_data'], # Requires target_data to be in env_config
                    self.config['variables'] # Requires variables to be in env_config
                )
            )
        
        # Calculate best_mse safely
        best_mse = float('inf')
        if eval_results:
            valid_mses = [r['mse'] for r in eval_results if 'mse' in r and r['valid']]
            if valid_mses:
                best_mse = min(valid_mses)

        return {
            'worker_id': self.worker_id,
            'episodes': episodes_data,
            'discoveries': discovered_expressions,
            'eval_results': eval_results,
            'avg_reward': np.mean([e['reward'] for e in episodes_data]) if episodes_data else 0.0,
            'best_mse': best_mse
        }


class DistributedJanusTrainer:
    """Main distributed training orchestrator."""
    
    def __init__(self,
                 grammar: ProgressiveGrammar,
                 env_config: Dict[str, Any],
                 num_workers: int = 8,
                 num_gpus: int = 4):
        
        self.grammar = grammar
        self.env_config = env_config
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        
        # Initialize Ray
        if not ray.is_initialized():
            # Adjust num_cpus and num_gpus based on available resources and worker needs
            ray.init(num_cpus=num_workers * 2, num_gpus=num_gpus) 
        
        # Register custom model for RLlib
        ModelCatalog.register_custom_model("hypothesis_net", RLlibHypothesisNet)
        
        # Create workers (Ray remote actors)
        self.workers = [
            DistributedExperimentWorker.remote(i, grammar, env_config)
            for i in range(num_workers)
        ]
        
        # Shared discovery repository (Ray remote actor)
        self.discovery_repo = DiscoveryRepository.remote()
        
    def train_with_pbt(self,
                      num_iterations: int = 100,
                      checkpoint_dir: str = "./checkpoints"):
        """Train using Population Based Training (PBT) with RLlib."""
        
        # PBT scheduler configuration
        pbt_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=10, # How often to perturb hyperparameters
            hyperparam_mutations={
                "lr": tune.uniform(1e-5, 1e-3),
                "clip_param": tune.uniform(0.1, 0.3),
                "entropy_coeff": tune.uniform(0.0, 0.01),
                "train_batch_size": [2048, 4096, 8192] # Example batch sizes
            }
        )
        
        # RLlib PPO Algorithm configuration
        config = PPOConfig()
        config.training(
            lr=3e-4, # Initial learning rate
            clip_param=0.2, # PPO clip parameter
            entropy_coeff=0.01, # Entropy regularization coefficient
            train_batch_size=4096, # Samples collected per training batch
            sgd_minibatch_size=256, # Mini-batch size for SGD
            num_sgd_iter=10 # Number of SGD passes per training batch
        )
        config.resources(
            num_gpus=0.5, # GPUs per training worker (can be fractional)
            num_cpus_per_worker=2 # CPUs per training worker
        )
        config.rollouts(
            num_rollout_workers=self.num_workers, # Number of parallel rollout workers
            rollout_fragment_length=512 # Number of steps collected per worker per rollout
        )
        config.environment(
            env=SymbolicDiscoveryEnv, # The RL environment to use
            env_config=self.env_config # Configuration passed to the environment
        )
        config.model(
            custom_model="hypothesis_net", # Name of the custom model registered with ModelCatalog
            custom_model_config={
                "hidden_dim": 256,
                "encoder_type": "transformer",
                "grammar": self.grammar # Pass the grammar to the model
            }
        )
        
        # Run tuning experiment with Ray Tune
        analysis = tune.run(
            "PPO", # Algorithm name
            config=config.to_dict(), # Convert PPOConfig to dictionary
            stop={"training_iteration": num_iterations}, # Stop condition
            num_samples=4,  # Population size for PBT
            scheduler=pbt_scheduler, # Use the PBT scheduler
            checkpoint_at_end=True, # Save checkpoint at the end of training
            local_dir=checkpoint_dir, # Directory for logs and checkpoints
            verbose=1 # Verbosity level
        )
        
        # Get the best performing trial based on episode_reward_mean
        best_trial = analysis.get_best_trial("episode_reward_mean", "max")
        
        return best_trial
    
    def parallel_hypothesis_search(self,
                                 policy_weights: Dict,
                                 num_rounds: int = 10,
                                 episodes_per_round: int = 50):
        """Run parallel hypothesis search across distributed workers."""
        
        all_discoveries = []
        
        for round_idx in range(num_rounds):
            print(f"\nRound {round_idx + 1}/{num_rounds}")
            
            # Distribute experiment work across workers
            futures = []
            episodes_per_worker = episodes_per_round // self.num_workers
            
            for worker in self.workers:
                future = worker.run_experiment.remote(
                    policy_weights,
                    n_episodes=episodes_per_worker
                )
                futures.append(future)
            
            # Collect results from all workers
            results = ray.get(futures)
            
            # Aggregate discoveries from all workers
            round_discoveries = []
            for result in results:
                round_discoveries.extend(result['discoveries'])
                
                # Log individual worker performance for monitoring
                print(f"  Worker {result['worker_id']}: "
                      f"Avg Reward = {result['avg_reward']:.3f}, "
                      f"Best MSE = {result['best_mse']:.3e}")
            
            all_discoveries.extend(round_discoveries)
            
            # Update shared discovery repository with new findings
            ray.get(self.discovery_repo.add_discoveries.remote(round_discoveries))
            
            # Get and print repository statistics
            repo_stats = ray.get(self.discovery_repo.get_statistics.remote())
            print(f"  Repository: {repo_stats['total_unique']} unique discoveries")
            # Safely access 'best_discovery' as it could be None if no discoveries yet
            print(f"  Top discovery: {repo_stats['best_discovery'] or 'N/A'} (Count: {repo_stats.get('best_count', 0)})")
        
        return all_discoveries
    
    def adaptive_curriculum_search(self,
                                 initial_policy_weights: Dict,
                                 num_stages: int = 5):
        """Implement adaptive curriculum learning with distributed evaluation."""
        
        policy_weights = initial_policy_weights
        
        for stage in range(num_stages):
            print(f"\n{'='*60}")
            print(f"Curriculum Stage {stage + 1}/{num_stages}")
            print(f"{'='*60}")
            
            # Adjust environment difficulty based on current stage
            self._update_curriculum_stage(stage)
            
            # Run distributed search for the current curriculum stage
            discoveries = self.parallel_hypothesis_search(
                policy_weights,
                num_rounds=5, # Number of search rounds within this stage
                episodes_per_round=100 # Episodes per round for each worker
            )
            
            # Evaluate success rate for the current stage's discoveries
            success_rate = self._evaluate_stage_success(discoveries)
            print(f"Stage success rate: {success_rate:.2%}")
            
            # Fine-tune policy if performance is below a threshold, adapting to new difficulty
            if success_rate < 0.7:
                print("Fine-tuning policy for current difficulty...")
                policy_weights = self._finetune_policy(policy_weights, stage)
        
        return policy_weights
    
    def _update_curriculum_stage(self, stage: int):
        """Update environment configuration for the given curriculum stage."""
        
        # Define progressive difficulty configurations
        difficulty_configs = [
            {'max_depth': 3, 'max_complexity': 5},
            {'max_depth': 5, 'max_complexity': 10},
            {'max_depth': 7, 'max_complexity': 15},
            {'max_depth': 10, 'max_complexity': 25},
            {'max_depth': 12, 'max_complexity': 40}
        ]
        
        if stage < len(difficulty_configs):
            config_to_apply = difficulty_configs[stage]
            
            # Update the environment config for all distributed workers
            for worker in self.workers:
                # Assuming the SymbolicDiscoveryEnv (or its base) has an update_config remote method
                ray.get(worker.env.update_config.remote(config_to_apply))
        else:
            print(f"Warning: Stage {stage} out of bounds for defined difficulty configs.")
    
    def _evaluate_stage_success(self, discoveries: List[str]) -> float:
        """Evaluate the success rate of discoveries for the current stage."""
        
        if not discoveries:
            return 0.0
        
        # Get evaluation results for all discoveries
        eval_results = ray.get(
            self.evaluator.evaluate_batch.remote(
                discoveries,
                self.env_config['target_data'], # Requires target_data to be in env_config
                self.env_config['variables'] # Requires variables to be in env_config
            )
        )
        
        # Define success criteria: MSE below threshold and reasonable complexity
        successes = sum(
            1 for r in eval_results 
            if r.get('valid', False) and r.get('mse', float('inf')) < 0.1 and r.get('complexity', float('inf')) < 20
        )
        
        return successes / len(eval_results)
    
    def _finetune_policy(self, 
                        policy_weights: Dict,
                        stage: int) -> Dict:
        """Fine-tune the policy for the current stage's specific challenges."""
        
        # Configure a short fine-tuning run for PPO
        config = PPOConfig()
        config.training(
            lr=1e-3, # Higher learning rate for fine-tuning
            train_batch_size=1024
        )
        config.rollouts(num_rollout_workers=2) # Use a smaller number of workers for fine-tuning
        config.environment(
            env=SymbolicDiscoveryEnv,
            env_config=self.env_config # Use the current environment config
        )
        config.model(
            custom_model="hypothesis_net",
            custom_model_config={
                "hidden_dim": 256,
                "encoder_type": "transformer",
                "grammar": self.grammar
            }
        )
        
        # Build the RLlib trainer
        trainer = config.build(env=SymbolicDiscoveryEnv)
        trainer.set_weights({"default_policy": policy_weights}) # Load initial policy weights
        
        # Run a few training iterations for fine-tuning
        print(f"  Running fine-tuning for stage {stage}...")
        for i in range(10): # 10 fine-tuning iterations
            result = trainer.train()
            print(f"    Fine-tune iteration {i+1}: Mean Reward = {result['episode_reward_mean']:.3f}")
        
        # Get the updated weights from the fine-tuned policy
        updated_weights = trainer.get_weights()["default_policy"]
        trainer.stop() # Clean up the trainer
        return updated_weights


@ray.remote
class DiscoveryRepository:
    """Centralized repository for storing and managing discovered expressions."""
    
    def __init__(self):
        self.discoveries = {} # Dictionary to store expressions and their metadata
        # self.metadata = {} # Not used in current implementation, can be removed or expanded
    
    def add_discoveries(self, expressions: List[str]):
        """Add new discoveries to the repository with a timestamp and count."""
        
        timestamp = time.time()
        
        for expr in expressions:
            if expr not in self.discoveries:
                self.discoveries[expr] = {
                    'first_discovered': timestamp,
                    'count': 0, # Initialize count
                    'eval_metrics': {} # Placeholder for storing evaluation metrics if needed
                }
            
            self.discoveries[expr]['count'] += 1
            self.discoveries[expr]['last_seen'] = timestamp
            # Additional logic can be added here to update average metrics or best metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieve overall statistics from the repository."""
        
        if not self.discoveries:
            return {
                'total_unique': 0,
                'total_occurrences': 0,
                'best_discovery': None,
                'best_count': 0
            }
        
        total_occurrences = sum(d['count'] for d in self.discoveries.values())
        
        # Find the most common discovery by occurrence count
        best_discovery_item = max(self.discoveries.items(), key=lambda x: x[1]['count'])
        
        return {
            'total_unique': len(self.discoveries),
            'total_occurrences': total_occurrences,
            'best_discovery': best_discovery_item[0],
            'best_count': best_discovery_item[1]['count']
        }
    
    def get_top_discoveries(self, n: int = 10) -> List[Tuple[str, Dict]]:
        """Get the top N discoveries sorted by occurrence count."""
        
        sorted_discoveries = sorted(
            self.discoveries.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return sorted_discoveries[:n]


# Utility functions for distributed training orchestration
def create_placement_group_for_training(num_workers: int, 
                                      gpus_per_worker: float = 0.5) -> Any:
    """Create a Ray placement group for efficient resource allocation across nodes/GPUs."""
    
    bundles = []
    
    # Bundle for the driver process (if it needs dedicated resources)
    bundles.append({"CPU": 2, "GPU": 0.5}) # Example allocation
    
    # Bundles for each worker
    for _ in range(num_workers):
        bundles.append({
            "CPU": 2, # Example CPU per worker
            "GPU": gpus_per_worker # Fractional or whole GPU per worker
        })
    
    # Create the placement group with a 'SPREAD' strategy to distribute across nodes
    pg = placement_group(bundles, strategy="SPREAD")
    ray.get(pg.ready()) # Wait for the placement group to be ready
    
    return pg


def distributed_hyperparameter_search(grammar: ProgressiveGrammar,
                                    env_config: Dict[str, Any],
                                    search_space: Dict[str, Any],
                                    num_trials: int = 20):
    """Run a distributed hyperparameter search using Ray Tune."""
    
    # Base configuration for the RLlib PPO algorithm
    config = {
        "env": SymbolicDiscoveryEnv, # The environment class
        "env_config": env_config, # Configuration for the environment
        "framework": "torch", # Deep learning framework
        "model": {
            "custom_model": "hypothesis_net", # Name of the custom model
            "custom_model_config": {
                # Tunable model hyperparameters
                "hidden_dim": tune.choice([128, 256, 512]),
                "encoder_type": tune.choice(["transformer", "treelstm"]),
                "grammar": grammar # Pass grammar to the model
            }
        },
        # Tunable RL algorithm hyperparameters
        "lr": tune.loguniform(1e-5, 1e-3), # Learning rate
        "entropy_coeff": tune.uniform(0.0, 0.1), # Entropy regularization
        "train_batch_size": tune.choice([2048, 4096, 8192]), # Batch size for training
        "num_sgd_iter": tune.choice([5, 10, 15]) # Number of SGD iterations
    }
    
    # Overlay any custom search parameters provided by the user
    config.update(search_space)
    
    # Run the hyperparameter search using Ray Tune
    analysis = tune.run(
        "PPO", # Algorithm to tune
        config=config, # Combined configuration and search space
        num_samples=num_trials, # Number of distinct trials to run
        stop={"training_iteration": 50}, # Stop condition for each trial
        metric="episode_reward_mean", # Metric to optimize
        mode="max", # Maximize the metric
        resources_per_trial={"cpu": 4, "gpu": 0.5} # Resources allocated per trial
    )
    
    # Return the best configuration found by the search
    return analysis.best_config

