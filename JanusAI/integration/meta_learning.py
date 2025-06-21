# File: src/janus/integration/meta_learning.py

import json
import time
import traceback
import torch

# Import the necessary components from their new locations
from janus_ai.ml.training.meta_trainer import MetaLearningConfig, MetaLearningPolicy, MAMLTrainer, TaskEnvironmentBuilder
from janus_ai.physics.data.generators import PhysicsTaskDistribution


def main():
    """Main entry point for meta-training"""
    
    config = MetaLearningConfig(
        meta_lr=0.0003,
        adaptation_lr=0.01,
        adaptation_steps=5,
        tasks_per_batch=10,
        support_episodes=10,
        query_episodes=10,
        meta_iterations=1000,
        use_intrinsic_rewards=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Meta-Learning for Physics Discovery")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Config: {json.dumps(config.__dict__, indent=2)}")
    print()
    
    print("Initializing physics task distribution...")
    task_distribution = PhysicsTaskDistribution(include_noise=True)
    print(task_distribution.describe_task_distribution())
    print()
    
    print("Determining observation and action space dimensions...")
    env_builder = TaskEnvironmentBuilder(config)

    max_obs_dim = 0
    max_action_dim = 0
    
    sample_tasks = task_distribution.sample_task_batch(5, curriculum=False)
    for task in sample_tasks:
        try:
            sample_env = env_builder.build_env(task)
            obs_dim_task = sample_env.observation_space.shape[0]
            action_dim_task = sample_env.action_space.n

            max_obs_dim = max(max_obs_dim, obs_dim_task)
            max_action_dim = max(max_action_dim, action_dim_task)

            print(f"  Task: {task.name} - Obs: {obs_dim_task}, Actions: {action_dim_task}")
        except Exception as e:
            print(f"  Warning: Could not build environment for {task.name}: {e}")

    obs_dim = max_obs_dim
    action_dim = max_action_dim + 1
    
    print("\nUsing dimensions:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print()
    
    print("Initializing meta-learning policy...")
    policy = MetaLearningPolicy(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        n_layers=3,
        use_task_embedding=True
    )
    
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    print("Initializing trainer...")
    trainer = MAMLTrainer(config, policy, task_distribution)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(f"{config.checkpoint_dir}/interrupted_checkpoint.pt")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(f"{config.checkpoint_dir}/error_checkpoint.pt")
    
    print("\nFinal evaluation on diverse tasks...")
    try:
        final_metrics = trainer.evaluate_on_new_tasks(n_tasks=20)

        print("\nFinal Performance:")
        print("-" * 30)
        for key, value in sorted(final_metrics.items()):
            if "mean" in key:
                print(f"{key}: {value:.3f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
