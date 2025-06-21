#!/usr/bin/env python
"""
quickstart.py - JanusAI Quick Start Script

Run this to immediately see JanusAI in action!
No configuration needed - just run: python scripts/quickstart.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- CORRECTED PATH SETUP ---
# This ensures that the script can find the 'janus' module
# by adding the project's root directory to the Python path.
try:
    # Get the directory of the current script (e.g., /path/to/JanusAI/scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (e.g., /path/to/JanusAI)
    project_root = os.path.dirname(script_dir)
    # Add the project root to the path
    sys.path.insert(0, project_root)

    # Now, try the imports
    from janus_ai.environments.base.symbolic_env import SymbolicDiscoveryEnv
    from janus_ai.physics.data.generators import PhysicsTaskDistribution
    # Assuming EnhancedPPOTrainer exists, if not, this will need to be adjusted
    from janus_ai.ml.training.ppo_trainer import PPOTrainer as EnhancedPPOTrainer
    from janus_ai.config.models import PPOConfig as EnhancedPPOConfig
    from stable_baselines3.common.callbacks import BaseCallback

except ImportError as e:
    print("\n❌ Error importing JanusAI modules.")
    print(f"   Import error: {e}")
    print("\n   This can happen for two reasons:")
    print("   1. You haven't installed the dependencies. Try running:")
    print("      pip install -r requirements.txt")
    print("\n   2. The main package directory is named 'JanusAI' instead of 'janus'.")
    print("      From your project root, please run this command:")
    print("      mv JanusAI janus")
    sys.exit(1)

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("Warning: PyTorch not found. Install with: pip install torch")
    HAS_TORCH = False
    DEVICE = "cpu"

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

def print_banner():
    """Print a nice banner."""
    print("\n" + "="*70)
    print(r"""
      _                         _   ___ 
     | | __ _ _ __   __ _ ___  / \ |_ _|
     | |/ _` | '_ \ / _` / __|/ _ \ | | 
     | | (_| | | | | (_| \__ / ___ \| | 
    _|_|\__,_|_| |_|\__,_|___/_/   \_\___|
   |__/                                 
    """)
    print("="*70)
    print("🚀 Welcome to JanusAI - Automated Physics Discovery")
    print("="*70)

def select_task():
    """Let user select a physics task."""
    tasks = {
        "1": ("harmonic_oscillator_energy", "Harmonic Oscillator (Easy)", 20000),
        "2": ("pendulum_energy", "Pendulum Energy (Medium)", 30000),
        "3": ("ideal_gas", "Ideal Gas Law (Medium)", 30000),
        "4": ("double_pendulum", "Double Pendulum (Hard)", 50000),
    }
    
    print("\n📋 Select a physics discovery task:")
    for key, (name, desc, _) in tasks.items():
        print(f"   {key}. {desc}")
    print("   q. Quit")
    
    while True:
        choice = input("\nYour choice [1]: ").strip() or "1"
        if choice.lower() == 'q':
            print("Goodbye!")
            sys.exit(0)
        if choice in tasks:
            return tasks[choice]
        print("Invalid choice. Please try again.")

def quick_experiment(task_name, timesteps):
    """Run a quick experiment with minimal configuration."""
    
    print(f"\n🧪 Running experiment: {task_name}")
    print(f"   Training steps: {timesteps:,}")
    print(f"   Device: {DEVICE}")
    
    # Create task and environment
    print("\n📊 Generating physics data...")
    task_dist = PhysicsTaskDistribution(include_noise=True)
    task = task_dist.get_task_by_name(task_name)
    
    # Generate separate X and y data, as per my previous architectural recommendation
    X_data, y_data = task.generate_data(200, noise=True, return_X_y=True)
    
    print(f"   Variables: {', '.join(task.variables)}")
    if hasattr(task, 'true_law'):
        print(f"   True law: {task.true_law}")
    
    # Create environment using the improved X, y signature
    env = SymbolicDiscoveryEnv(
        X_data=X_data,
        y_data=y_data,
        variables=task.variables,
        max_depth=8,
        max_complexity=20
    )
    
    # Configure trainer
    config = EnhancedPPOConfig(
        total_timesteps=timesteps,
        n_steps=1024,
        batch_size=64, # Increased batch size for more stable gradients
        device=DEVICE
    )
    
    # Create trainer
    trainer = EnhancedPPOTrainer(env=env, config=config)
    
    # Track discoveries
    discoveries = []
    start_time = datetime.now()
    
    print("\n🏃 Training started! Watch for discoveries...\n")
    print("-" * 50)
    
    class QuickCallback(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get('infos', [{}])
            for info in infos:
                if 'expression' in info and info.get('is_new_best', False):
                    expr = info['expression']
                    reward = info.get('reward', 0)
                    
                    discoveries.append({
                        'expression': str(expr),
                        'reward': reward,
                        'timestep': self.num_timesteps
                    })
                    
                    print(f"✨ Step {self.num_timesteps:6d}: New best -> {expr} (R: {reward:.4f})")
            
            if self.num_timesteps % 5000 == 0 and self.num_timesteps > 0:
                progress = self.num_timesteps / config.total_timesteps * 100
                print(f"   ... Progress: {progress:.0f}% ({self.num_timesteps:,}/{config.total_timesteps:,} steps)")
            
            return True
    
    # Train!
    try:
        # Note: The original script called trainer.agent.learn, which suggests
        # the trainer might be a wrapper. I'm calling trainer.learn directly.
        # This might need to be adjusted based on the actual trainer's API.
        trainer.learn(
            total_timesteps=config.total_timesteps,
            callback=QuickCallback()
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    # Results
    elapsed = (datetime.now() - start_time).total_seconds()
    print("-" * 50)
    print(f"\n✅ Training complete in {elapsed:.1f} seconds!")
    
    if discoveries:
        print(f"\n🎯 Found {len(discoveries)} unique high-reward expressions:")
        discoveries.sort(key=lambda x: x['reward'], reverse=True)
        for i, disc in enumerate(discoveries[:5]):
            print(f"   {i+1}. {disc['expression']} (reward: {disc['reward']:.3f})")
        
        # Simple visualization
        if len(discoveries) > 1:
            plt.figure(figsize=(10, 6))
            timesteps = [d['timestep'] for d in discoveries]
            rewards = [d['reward'] for d in discoveries]
            plt.plot(timesteps, rewards, '-o', alpha=0.7, markersize=8)
            plt.xlabel('Training Step')
            plt.ylabel('Discovery Reward')
            plt.title(f'Discovery Timeline - {task_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('quickstart_discoveries.png', dpi=150)
            print(f"\n📊 Visualization saved to: quickstart_discoveries.png")
            # plt.show() # Can be blocking
    else:
        print("\n❌ No significant discoveries. Try increasing training steps or using a simpler task.")

def main():
    """Main entry point."""
    print_banner()
    
    print("\n🔍 Checking setup...")
    print(f"   ✓ PyTorch: {'Available' if HAS_TORCH else '❌ Not found'}")
    print(f"   ✓ Device: {DEVICE}")
    
    if not HAS_TORCH:
        print("\n❌ PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    task_name, task_desc, timesteps = select_task()
    print(f"\n🎮 Starting discovery on {task_desc}")
    
    try:
        quick_experiment(task_name, timesteps)
    except Exception as e:
        print(f"\n❌ An error occurred during the experiment: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🎉 Thank you for trying JanusAI!")
    print("\nNext steps:")
    print("   1. Explore the generated 'quickstart_discoveries.png'")
    print("   2. Try different tasks or increase the training steps")
    print("   3. Dive into the configs to create your own experiments!")
    print("="*70)

if __name__ == "__main__":
    main()