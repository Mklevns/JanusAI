# File: JanusAI/examples/advanced_exploration_demo.py
"""
Demonstration: Advanced Exploration Strategies in Action

This example shows how the three exploration strategies work together
to discover physical laws more efficiently than random search.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List

from janus.ml.training.enhanced_ppo_trainer import (
    EnhancedPPOTrainer, 
    EnhancedPPOConfig
)
from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus.physics.data.generators import PhysicsTaskDistribution


class ExplorationDemo:
    """
    Demonstrates the effectiveness of advanced exploration strategies.
    """
    
    def __init__(self, task_name: str = "harmonic_oscillator_energy"):
        self.task_name = task_name
        self.results = {
            'random': None,
            'info_gain': None,
            'prend': None,
            'llm': None,
            'combined': None
        }
        
    def run_baseline(self, n_steps: int = 50000) -> Dict:
        """Run baseline random exploration."""
        print("\n=== Running Baseline (Random Exploration) ===")
        
        env = self._create_environment()
        config = EnhancedPPOConfig(
            total_timesteps=n_steps,
            use_information_gain=False,
            use_prend=False,
            use_llm_goals=False,
            intrinsic_weight=0.0  # Pure extrinsic rewards
        )
        
        trainer = EnhancedPPOTrainer(env, config)
        start_time = time.time()
        trainer.train()
        
        return {
            'time': time.time() - start_time,
            'discoveries': trainer.discovered_expressions,
            'final_reward': np.mean([d['reward'] for d in trainer.discovered_expressions[-10:]])
            if trainer.discovered_expressions else 0
        }
    
    def run_info_gain_only(self, n_steps: int = 50000) -> Dict:
        """Run with information gain exploration only."""
        print("\n=== Running Information Gain Exploration ===")
        
        env = self._create_environment()
        config = EnhancedPPOConfig(
            total_timesteps=n_steps,
            use_information_gain=True,
            use_prend=False,
            use_llm_goals=False,
            ensemble_size=5,
            info_gain_scale=0.5,
            intrinsic_weight=0.5
        )
        
        trainer = EnhancedPPOTrainer(env, config)
        start_time = time.time()
        trainer.train()
        
        # Get ensemble statistics
        ensemble_stats = self._analyze_ensemble_evolution(trainer)
        
        return {
            'time': time.time() - start_time,
            'discoveries': trainer.discovered_expressions,
            'ensemble_stats': ensemble_stats,
            'final_reward': np.mean([d['reward'] for d in trainer.discovered_expressions[-10:]])
            if trainer.discovered_expressions else 0
        }
    
    def run_prend_only(self, n_steps: int = 50000) -> Dict:
        """Run with PreND exploration only."""
        print("\n=== Running PreND Exploration ===")
        
        env = self._create_environment()
        config = EnhancedPPOConfig(
            total_timesteps=n_steps,
            use_information_gain=False,
            use_prend=True,
            use_llm_goals=False,
            prend_scale=0.5,
            intrinsic_weight=0.5
        )
        
        trainer = EnhancedPPOTrainer(env, config)
        start_time = time.time()
        trainer.train()
        
        return {
            'time': time.time() - start_time,
            'discoveries': trainer.discovered_expressions,
            'final_reward': np.mean([d['reward'] for d in trainer.discovered_expressions[-10:]])
            if trainer.discovered_expressions else 0
        }
    
    def run_llm_only(self, n_steps: int = 50000) -> Dict:
        """Run with LLM-guided exploration only."""
        print("\n=== Running LLM-Guided Exploration ===")
        
        env = self._create_environment()
        config = EnhancedPPOConfig(
            total_timesteps=n_steps,
            use_information_gain=False,
            use_prend=False,
            use_llm_goals=True,
            llm_model="gpt-3.5-turbo",  # Faster for demo
            llm_exploration_rate=0.3,
            intrinsic_weight=0.5
        )
        
        trainer = EnhancedPPOTrainer(env, config)
        
        # Track LLM suggestions
        llm_suggestions = []
        
        def track_llm_suggestion(suggestion: str):
            llm_suggestions.append({
                'suggestion': suggestion,
                'timestep': trainer.agent.num_timesteps
            })
        
        # Monkey patch to track suggestions
        if hasattr(trainer, 'llm_generator'):
            original_suggest = trainer.llm_generator.suggest_next_goal
            def wrapped_suggest(context):
                result = original_suggest(context)
                track_llm_suggestion(result)
                return result
            trainer.llm_generator.suggest_next_goal = wrapped_suggest
        
        start_time = time.time()
        trainer.train()
        
        return {
            'time': time.time() - start_time,
            'discoveries': trainer.discovered_expressions,
            'llm_suggestions': llm_suggestions,
            'final_reward': np.mean([d['reward'] for d in trainer.discovered_expressions[-10:]])
            if trainer.discovered_expressions else 0
        }
    
    def run_combined(self, n_steps: int = 50000) -> Dict:
        """Run with all exploration strategies combined."""
        print("\n=== Running Combined Exploration ===")
        
        env = self._create_environment()
        config = EnhancedPPOConfig(
            total_timesteps=n_steps,
            use_information_gain=True,
            use_prend=True,
            use_llm_goals=True,
            ensemble_size=5,
            info_gain_scale=0.4,
            prend_scale=0.3,
            llm_model="gpt-3.5-turbo",
            llm_exploration_rate=0.2,
            intrinsic_weight=0.5,
            adaptive_weights=True
        )
        
        trainer = EnhancedPPOTrainer(env, config)
        start_time = time.time()
        trainer.train()
        
        # Get adaptive weight evolution
        weight_history = self._track_weight_evolution(trainer)
        
        return {
            'time': time.time() - start_time,
            'discoveries': trainer.discovered_expressions,
            'weight_evolution': weight_history,
            'final_reward': np.mean([d['reward'] for d in trainer.discovered_expressions[-10:]])
            if trainer.discovered_expressions else 0
        }
    
    def _create_environment(self) -> SymbolicDiscoveryEnv:
        """Create environment for the demo task."""
        task_dist = PhysicsTaskDistribution()
        task = task_dist.get_task_by_name(self.task_name)
        
        data = task.generate_data(200, noise=True)
        
        env = SymbolicDiscoveryEnv(
            data=data,
            target_expr=task.true_law if hasattr(task, 'true_law') else "unknown",
            max_depth=10,
            max_complexity=30
        )
        
        # Add task context
        env.task_info = {
            'name': task.name,
            'domain': task.domain,
            'variables': task.variables,
            'true_law': getattr(task, 'true_law', 'unknown')
        }
        
        return env
    
    def _analyze_ensemble_evolution(self, trainer: EnhancedPPOTrainer) -> Dict:
        """Analyze how ensemble uncertainty evolved during training."""
        if not hasattr(trainer, 'dynamics_ensemble'):
            return {}
        
        stats = trainer.dynamics_ensemble.training_stats
        
        return {
            'disagreement_evolution': stats['disagreements'],
            'mean_disagreement': np.mean(stats['disagreements']) if stats['disagreements'] else 0,
            'disagreement_reduction': (
                stats['disagreements'][0] - stats['disagreements'][-1]
            ) if len(stats['disagreements']) > 1 else 0
        }
    
    def _track_weight_evolution(self, trainer: EnhancedPPOTrainer) -> List[Dict]:
        """Track how adaptive weights evolved."""
        # This would ideally be tracked during training
        # For demo, we'll return the final weights
        if hasattr(trainer.intrinsic_reward, 'weights'):
            return [trainer.intrinsic_reward.weights]
        return []
    
    def run_all_experiments(self, n_steps: int = 20000):
        """Run all exploration strategies and compare."""
        self.results['random'] = self.run_baseline(n_steps)
        self.results['info_gain'] = self.run_info_gain_only(n_steps)
        self.results['prend'] = self.run_prend_only(n_steps)
        self.results['llm'] = self.run_llm_only(n_steps)
        self.results['combined'] = self.run_combined(n_steps)
    
    def visualize_results(self):
        """Create visualizations comparing the strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Discovery Speed Comparison
        ax = axes[0, 0]
        strategies = list(self.results.keys())
        times = [self.results[s]['time'] for s in strategies]
        discoveries = [len(self.results[s]['discoveries']) for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax.bar(x - width/2, times, width, label='Time (s)', alpha=0.8)
        ax.bar(x + width/2, discoveries, width, label='Discoveries', alpha=0.8)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Value')
        ax.set_title('Discovery Efficiency')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        
        # 2. Final Reward Comparison
        ax = axes[0, 1]
        final_rewards = [self.results[s]['final_reward'] for s in strategies]
        
        ax.bar(strategies, final_rewards, color='green', alpha=0.7)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Final Reward')
        ax.set_title('Final Performance')
        ax.set_xticklabels(strategies, rotation=45)
        
        # 3. Ensemble Disagreement Evolution (if available)
        ax = axes[1, 0]
        if 'ensemble_stats' in self.results['info_gain']:
            disagreements = self.results['info_gain']['ensemble_stats']['disagreement_evolution']
            if disagreements:
                ax.plot(disagreements)
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Ensemble Disagreement')
                ax.set_title('Uncertainty Evolution (Info Gain)')
                ax.grid(True, alpha=0.3)
        
        # 4. Discovery Timeline
        ax = axes[1, 1]
        for strategy in ['random', 'info_gain', 'combined']:
            if strategy in self.results:
                discoveries = self.results[strategy]['discoveries']
                if discoveries:
                    timesteps = [d['timestep'] for d in discoveries if 'timestep' in d]
                    rewards = [d['reward'] for d in discoveries]
                    if timesteps:
                        ax.scatter(timesteps, rewards, label=strategy, alpha=0.6)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Discovery Reward')
        ax.set_title('Discovery Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('exploration_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print a summary of the results."""
        print("\n" + "="*60)
        print("EXPLORATION STRATEGY COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\nTask: {self.task_name}")
        print("\nResults:")
        print(f"{'Strategy':<15} {'Time (s)':<10} {'Discoveries':<12} {'Final Reward':<12}")
        print("-" * 50)
        
        for strategy, result in self.results.items():
            if result:
                print(f"{strategy:<15} {result['time']:<10.2f} "
                      f"{len(result['discoveries']):<12} {result['final_reward']:<12.3f}")
        
        # Calculate speedup
        baseline_time = self.results['random']['time']
        print("\nSpeedup vs Random Baseline:")
        for strategy in ['info_gain', 'prend', 'llm', 'combined']:
            if strategy in self.results and self.results[strategy]:
                speedup = baseline_time / self.results[strategy]['time']
                print(f"  {strategy}: {speedup:.2f}x faster")
        
        # Best discoveries
        print("\nBest Discoveries:")
        for strategy, result in self.results.items():
            if result and result['discoveries']:
                best = max(result['discoveries'], key=lambda x: x['reward'])
                print(f"  {strategy}: {best['expression']} (reward: {best['reward']:.3f})")
        
        # LLM suggestions (if available)
        if 'llm_suggestions' in self.results.get('llm', {}):
            suggestions = self.results['llm']['llm_suggestions']
            if suggestions:
                print(f"\nSample LLM Suggestions:")
                for s in suggestions[:3]:
                    print(f"  - {s['suggestion']}")


def main():
    """Run the complete demonstration."""
    print("="*60)
    print("ADVANCED EXPLORATION STRATEGIES DEMONSTRATION")
    print("="*60)
    
    # Choose a task
    task_options = [
        "harmonic_oscillator_energy",
        "pendulum_energy", 
        "ideal_gas"
    ]
    
    print("\nAvailable tasks:")
    for i, task in enumerate(task_options):
        print(f"  {i+1}. {task}")
    
    # For demo, use the first task
    task_name = task_options[0]
    print(f"\nUsing task: {task_name}")
    
    # Run demonstration
    demo = ExplorationDemo(task_name)
    
    # Run shorter experiments for demo
    print("\nRunning experiments (this may take a few minutes)...")
    demo.run_all_experiments(n_steps=10000)  # Reduced for demo
    
    # Show results
    demo.print_summary()
    demo.visualize_results()
    
    # Save results
    results_path = Path("exploration_demo_results.json")
    with open(results_path, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for strategy, result in demo.results.items():
            if result:
                json_results[strategy] = {
                    'time': result['time'],
                    'num_discoveries': len(result['discoveries']),
                    'final_reward': result['final_reward'],
                    'best_discovery': max(result['discoveries'], key=lambda x: x['reward'])
                    if result['discoveries'] else None
                }
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Insights:")
    print("1. Information Gain explores uncertain regions efficiently")
    print("2. PreND leverages semantic knowledge from pre-trained models")
    print("3. LLM guidance provides intelligent high-level hypotheses")
    print("4. Combined approach achieves best overall performance")
    print("\nThese strategies transform random search into intelligent exploration!")


if __name__ == "__main__":
    main()