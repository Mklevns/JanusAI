# File: JanusAI/examples/attention_physics_discovery.py
"""
Demo: Attention-Based Physics Discovery

This example demonstrates how attention mechanisms help discover physical laws
by focusing on relevant parts of the expression tree based on the task.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from janus_ai.ml.networks.attention_meta_policy import EnhancedMetaLearningPolicy
from janus_ai.ml.training.enhanced_meta_trainer import (
    EnhancedMAMLTrainer, 
    EnhancedMetaLearningConfig
)
from janus_ai.physics.data.generators import PhysicsTaskDistribution
from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv


def visualize_attention_for_task(trainer, task_name="harmonic_oscillator"):
    """
    Visualize how attention changes during discovery of a specific physical law.
    """
    # Get a specific task
    task = trainer.task_distribution.get_task_by_name(task_name)
    
    # Generate data
    data = task.generate_data(100, noise=True)
    
    # Create environment
    env = SymbolicDiscoveryEnv(
        data=data,
        target_expr=task.true_law if hasattr(task, 'true_law') else "unknown",
        max_depth=10,
        max_complexity=30
    )
    
    # Collect a few support trajectories
    support_trajectories = trainer._collect_trajectories(
        trainer.policy,
        task,
        n_episodes=3,
        use_attention_stats=True
    )
    
    # Prepare task context
    task_context = trainer._prepare_task_context(support_trajectories)
    
    # Run one episode and collect attention maps
    obs, _ = env.reset()
    attention_maps = []
    actions_taken = []
    rewards = []
    
    for step in range(50):  # Limit steps for visualization
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.config.device)
        
        with torch.no_grad():
            outputs = trainer.policy(obs_tensor, task_context, return_attention=True)
            
            if 'attention_weights' in outputs:
                attention_maps.append(outputs['attention_weights'].cpu().numpy())
            
            # Get action
            action_mask = torch.BoolTensor(env.get_action_mask()).to(trainer.config.device)
            action, _ = trainer.policy.act(
                obs_tensor.squeeze(0),
                task_context,
                action_mask,
                deterministic=True  # Deterministic for visualization
            )
        
        actions_taken.append(action)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        obs = next_obs
        
        if done or truncated:
            if 'expression' in info:
                print(f"Discovered expression: {info['expression']}")
                print(f"True law: {task.true_law if hasattr(task, 'true_law') else 'Unknown'}")
                print(f"MSE: {info.get('mse', 'N/A')}")
            break
    
    # Visualize attention evolution
    if attention_maps:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Attention Evolution for {task_name}', fontsize=16)
        
        # Select 6 time points
        indices = np.linspace(0, len(attention_maps)-1, 6, dtype=int)
        
        for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
            attention = attention_maps[idx].squeeze()
            
            # Create heatmap
            im = ax.imshow(attention, cmap='hot', aspect='auto')
            ax.set_title(f'Step {idx}')
            ax.set_xlabel('Tree Node')
            ax.set_ylabel('Query')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('attention_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Plot reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'Reward Evolution for {task_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig('reward_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return attention_maps, actions_taken, rewards


def compare_attention_across_tasks(trainer, task_names=None):
    """
    Compare attention patterns across different physics tasks.
    """
    if task_names is None:
        task_names = ["harmonic_oscillator", "pendulum", "double_pendulum"]
    
    attention_stats = {}
    
    for task_name in task_names:
        task = trainer.task_distribution.get_task_by_name(task_name)
        
        # Collect trajectories
        trajectories = trainer._collect_trajectories(
            trainer.policy,
            task,
            n_episodes=5,
            use_attention_stats=True
        )
        
        # Extract attention statistics
        all_entropies = []
        all_max_attentions = []
        
        for traj in trajectories:
            if traj['attention_stats']:
                for stats in traj['attention_stats']:
                    if 'entropy' in stats:
                        all_entropies.append(stats['entropy'])
                    if 'weights' in stats:
                        max_att = stats['weights'].max()
                        all_max_attentions.append(max_att)
        
        attention_stats[task_name] = {
            'mean_entropy': np.mean(all_entropies) if all_entropies else 0,
            'std_entropy': np.std(all_entropies) if all_entropies else 0,
            'mean_max_attention': np.mean(all_max_attentions) if all_max_attentions else 0
        }
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Entropy comparison
    task_labels = list(attention_stats.keys())
    entropies = [attention_stats[t]['mean_entropy'] for t in task_labels]
    entropy_stds = [attention_stats[t]['std_entropy'] for t in task_labels]
    
    ax1.bar(task_labels, entropies, yerr=entropy_stds, capsize=10)
    ax1.set_ylabel('Attention Entropy')
    ax1.set_title('Attention Focus by Task Type')
    ax1.set_xticklabels(task_labels, rotation=45)
    
    # Max attention comparison
    max_attentions = [attention_stats[t]['mean_max_attention'] for t in task_labels]
    
    ax2.bar(task_labels, max_attentions)
    ax2.set_ylabel('Mean Max Attention')
    ax2.set_title('Attention Concentration by Task Type')
    ax2.set_xticklabels(task_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('attention_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return attention_stats


def main():
    """
    Main demo function.
    """
    print("=== Attention-Based Physics Discovery Demo ===\n")
    
    # Configure enhanced meta-learning
    config = EnhancedMetaLearningConfig(
        # Use smaller values for demo
        meta_lr=0.001,
        adaptation_lr=0.01,
        adaptation_steps=3,
        tasks_per_batch=3,
        support_episodes=5,
        query_episodes=5,
        
        # Attention configuration
        num_attention_heads=4,
        fusion_type="attention",
        use_hierarchical_encoding=True,
        
        # Enable attention saving
        save_attention_maps=True,
        attention_save_interval=10,
        
        # Short training for demo
        meta_iterations=100,
        
        # Disable wandb for demo
        use_wandb=False,
        
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Device: {config.device}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"Fusion type: {config.fusion_type}\n")
    
    # Initialize components
    task_distribution = PhysicsTaskDistribution(include_noise=True)
    
    # Create trainer
    print("Initializing enhanced MAML trainer...")
    trainer = EnhancedMAMLTrainer(config, task_distribution=task_distribution)
    
    # Quick training
    print("\nTraining for demonstration...")
    for i in range(20):  # Just 20 iterations for demo
        metrics = trainer.meta_train_step()
        if i % 5 == 0:
            print(f"Iteration {i}: Loss = {metrics['meta_loss']:.4f}, "
                  f"Success = {metrics.get('meta/discovery_success', 0):.2%}")
    
    print("\n=== Visualizing Attention Patterns ===\n")
    
    # Visualize attention for a specific task
    print("1. Attention evolution during harmonic oscillator discovery:")
    visualize_attention_for_task(trainer, "harmonic_oscillator")
    
    # Compare attention across tasks
    print("\n2. Comparing attention patterns across different tasks:")
    attention_stats = compare_attention_across_tasks(trainer)
    
    print("\nAttention Statistics Summary:")
    for task, stats in attention_stats.items():
        print(f"\n{task}:")
        print(f"  - Mean entropy: {stats['mean_entropy']:.3f} ± {stats['std_entropy']:.3f}")
        print(f"  - Mean max attention: {stats['mean_max_attention']:.3f}")
    
    # Analyze attention evolution over training
    print("\n3. Attention focus evolution analysis:")
    
    # Get evolution for a specific task
    evolution = trainer.attention_analyzer.analyze_attention_evolution("harmonic_oscillator")
    
    if evolution and 'entropy' in evolution:
        plt.figure(figsize=(10, 5))
        plt.plot(evolution['entropy'], label='Attention Entropy')
        plt.xlabel('Training Step')
        plt.ylabel('Entropy')
        plt.title('Attention Focus Evolution During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('attention_focus_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate trend
        if len(evolution['entropy']) > 1:
            start_entropy = np.mean(evolution['entropy'][:5])
            end_entropy = np.mean(evolution['entropy'][-5:])
            print(f"\nAttention entropy trend: {start_entropy:.3f} → {end_entropy:.3f}")
            print(f"Change: {((end_entropy - start_entropy) / start_entropy * 100):.1f}%")
    
    print("\n=== Demo Complete ===")
    print("\nKey Insights:")
    print("1. Attention helps the model focus on relevant parts of the expression tree")
    print("2. Different physics tasks require different attention patterns")
    print("3. Attention entropy decreases as the model learns (more focused attention)")
    print("4. Task embeddings guide the discovery process effectively")
    
    # Save final checkpoint
    trainer.save_checkpoint("demo_checkpoint.pt")
    print("\nCheckpoint saved to: demo_checkpoint.pt")


if __name__ == "__main__":
    main()