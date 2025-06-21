# JanusAI/agents/integration.py
"""
Integration module that connects the multi-agent system with existing Janus components.
Provides complete working examples and utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

# Import Janus components
from .multi_agent_system import (
    DynamicAgentPool, PlannerAgent, AgentRole, AgentConfig,
    BaseScientificAgent
)
from .iterative_refinement import (
    IterativeRefinementLoop, JudgeAgent, FeedbackIncorporator
)
from ..memory.dual_memory_system import DualMemorySystem
from ..memory.memory_integration import (
    MemoryIntegratedEnv, MemoryAugmentedAgent
)
from ..grammar.ai_grammar import AIGrammar
from ..rewards.interpretability_reward import InterpretabilityReward
from ..ml.training.advanced_ppo_trainer import AdvancedPPOTrainer, PPOConfig
from ..ml.networks.policy_networks import TransformerPolicy
from ..environments.symbolic_discovery_env import SymbolicDiscoveryEnv


class JanusMultiAgentFramework:
    """
    Complete multi-agent framework for Janus scientific discovery.
    Integrates all components into a unified system.
    """
    
    def __init__(self,
                 memory_db_path: str = "janus_discoveries.db",
                 device: str = "cuda",
                 enable_training: bool = True,
                 enable_visualization: bool = True):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.enable_training = enable_training
        self.enable_visualization = enable_visualization
        
        # Initialize core components
        self._initialize_components(memory_db_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.discovery_sessions = []
        
    def _initialize_components(self, memory_db_path: str):
        """Initialize all framework components"""
        
        # 1. Memory System
        self.memory_system = DualMemorySystem(
            episodic_capacity=10000,
            shared_capacity=10,
            db_path=memory_db_path,
            embedding_dim=768
        )
        
        # 2. Grammar
        self.grammar = AIGrammar(
            primitives=[
                'attention', 'linear', 'softmax', 'scale', 'mask',
                'shift', 'position_encoding', 'layer_norm', 'dropout',
                'residual', 'concat', 'split'
            ],
            max_depth=10,
            use_attention_specific=True
        )
        
        # 3. Reward Function
        self.reward_function = InterpretabilityReward(
            fidelity_weight=0.4,
            simplicity_weight=0.2,
            consistency_weight=0.2,
            insight_weight=0.2
        )
        
        # 4. Base Policy Network
        self.base_policy = TransformerPolicy(
            obs_dim=256,
            action_dim=len(self.grammar.primitives) * 4,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            max_seq_length=200
        ).to(self.device)
        
        # 5. PPO Trainer (if enabled)
        if self.enable_training:
            self.ppo_trainer = AdvancedPPOTrainer(
                policy=self.base_policy,
                env=None,  # Will be set per problem
                config=PPOConfig(
                    learning_rate=1e-4,
                    n_epochs=4,
                    batch_size=64,
                    entropy_coef=0.01
                ),
                device=self.device
            )
        else:
            self.ppo_trainer = None
        
        # 6. Agent Pool
        self.agent_pool = DynamicAgentPool(
            memory_system=self.memory_system,
            grammar=self.grammar,
            base_policy=self.base_policy,
            max_agents=15
        )
        
        # 7. Planner
        self.planner = PlannerAgent(
            agent_pool=self.agent_pool,
            convergence_threshold=0.9,
            min_iterations=3,
            max_iterations=10
        )
        
        # 8. Refinement Loop
        self.refinement_loop = IterativeRefinementLoop(
            memory_system=self.memory_system,
            grammar=self.grammar,
            reward_function=self.reward_function,
            ppo_trainer=self.ppo_trainer,
            convergence_patience=3
        )
        
        # 9. Feedback Incorporator
        self.feedback_incorporator = FeedbackIncorporator(
            learning_rate=0.1
        )
    
    def discover(self,
                problem_type: str,
                problem_config: Optional[Dict[str, Any]] = None,
                visualize: bool = True) -> Dict[str, Any]:
        """
        Run a complete discovery session.
        
        Args:
            problem_type: Type of problem ('attention_pattern', 'physics_law', etc.)
            problem_config: Additional problem configuration
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with discovery results and metrics
        """
        
        session_start = datetime.now()
        
        # Create problem context
        problem_context = self._create_problem_context(problem_type, problem_config)
        
        self.logger.info(f"Starting discovery session: {problem_type}")
        self.logger.info(f"Problem context: {problem_context}")
        
        # Run discovery
        discovery = self.refinement_loop.run_discovery(
            problem_context=problem_context,
            agent_pool=self.agent_pool,
            planner=self.planner,
            max_iterations=problem_context.get('max_iterations', 10)
        )
        
        # Collect metrics
        session_metrics = {
            'problem_type': problem_type,
            'discovery': discovery,
            'duration': (datetime.now() - session_start).total_seconds(),
            'iterations': len(self.refinement_loop.iteration_history),
            'unique_expressions': len(set(
                h.num_unique_expressions 
                for h in self.refinement_loop.iteration_history
            )),
            'final_score': discovery.validation_score,
            'agent_contributions': self._analyze_agent_contributions()
        }
        
        # Visualize if requested
        if visualize and self.enable_visualization:
            self._create_visualizations(session_metrics)
        
        # Store session
        self.discovery_sessions.append(session_metrics)
        
        # Log summary
        self._log_session_summary(session_metrics)
        
        return session_metrics
    
    def _create_problem_context(self, 
                              problem_type: str,
                              config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create problem context based on type"""
        
        base_context = {
            'timestamp': datetime.now(),
            'problem_type': problem_type
        }
        
        if problem_type == 'attention_pattern':
            context = {
                **base_context,
                'domain': 'attention_mechanisms',
                'task': config.get('pattern_type', 'scaled_dot_product'),
                'complexity': config.get('complexity', 'medium'),
                'target_data': self._generate_attention_target(
                    config.get('pattern_type', 'scaled_dot_product')
                )
            }
        
        elif problem_type == 'physics_law':
            context = {
                **base_context,
                'domain': 'physics_laws',
                'task': config.get('law_type', 'harmonic_oscillator'),
                'complexity': config.get('complexity', 'medium'),
                'target_data': self._generate_physics_target(
                    config.get('law_type', 'harmonic_oscillator')
                )
            }
        
        elif problem_type == 'symbolic_regression':
            context = {
                **base_context,
                'domain': 'symbolic_regression',
                'task': 'find_expression',
                'complexity': config.get('complexity', 'medium'),
                'target_data': config.get('target_data')
            }
        
        else:
            context = {
                **base_context,
                'domain': 'general',
                'task': 'discover_pattern',
                'complexity': 'medium'
            }
        
        # Add any custom config
        if config:
            context.update(config)
        
        return context
    
    def _generate_attention_target(self, pattern_type: str) -> np.ndarray:
        """Generate target attention pattern"""
        
        seq_len = 16
        
        if pattern_type == 'scaled_dot_product':
            # Standard attention pattern
            Q = np.random.randn(seq_len, 64)
            K = np.random.randn(seq_len, 64)
            scores = Q @ K.T / np.sqrt(64)
            attention = self._softmax(scores)
            
        elif pattern_type == 'previous_token':
            attention = np.zeros((seq_len, seq_len))
            for i in range(1, seq_len):
                attention[i, i-1] = 1.0
            attention[0, 0] = 1.0
            
        elif pattern_type == 'local_window':
            window = 3
            attention = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(max(0, i-window), min(seq_len, i+window+1)):
                    attention[i, j] = 1.0
            # Normalize rows
            attention = attention / attention.sum(axis=1, keepdims=True)
            
        else:
            # Random pattern
            attention = np.random.rand(seq_len, seq_len)
            attention = self._softmax(attention)
        
        return attention
    
    def _generate_physics_target(self, law_type: str) -> Dict[str, np.ndarray]:
        """Generate target physics data"""
        
        t = np.linspace(0, 10, 100)
        
        if law_type == 'harmonic_oscillator':
            # x = A * cos(ωt + φ)
            A, omega, phi = 2.0, 2 * np.pi, 0
            x = A * np.cos(omega * t + phi)
            data = {'t': t, 'x': x}
            
        elif law_type == 'projectile_motion':
            # y = v0*t - 0.5*g*t^2
            v0, g = 20.0, 9.81
            y = v0 * t - 0.5 * g * t**2
            data = {'t': t, 'y': y}
            
        elif law_type == 'exponential_decay':
            # N = N0 * exp(-λt)
            N0, lambda_decay = 100.0, 0.5
            N = N0 * np.exp(-lambda_decay * t)
            data = {'t': t, 'N': N}
            
        else:
            # Generic quadratic
            y = 2 * t**2 + 3 * t + 1
            data = {'t': t, 'y': y}
        
        return data
    
    def _softmax(self, x: np.ndarray, axis=-1) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _analyze_agent_contributions(self) -> Dict[str, Any]:
        """Analyze how different agents contributed"""
        
        contributions = {}
        
        for agent_id, agent in self.agent_pool.active_agents.items():
            perf = self.agent_pool.agent_performance[agent_id]
            
            contributions[agent.role.value] = {
                'total_attempts': perf['num_actions'],
                'average_score': self.agent_pool._get_average_score(agent_id),
                'best_score': perf['best_score'],
                'discoveries': agent.discovery_count
            }
        
        return contributions
    
    def _create_visualizations(self, session_metrics: Dict[str, Any]):
        """Create visualizations for the discovery session"""
        
        try:
            from ..memory.advanced_features import MemoryVisualizer
            import matplotlib.pyplot as plt
            
            visualizer = MemoryVisualizer(self.memory_system)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Iteration progress
            ax = axes[0, 0]
            iterations = [m.iteration for m in self.refinement_loop.iteration_history]
            best_scores = [m.best_score for m in self.refinement_loop.iteration_history]
            avg_scores = [m.average_score for m in self.refinement_loop.iteration_history]
            
            ax.plot(iterations, best_scores, 'b-', label='Best Score', linewidth=2)
            ax.plot(iterations, avg_scores, 'g--', label='Average Score', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.set_title('Discovery Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Agent contributions
            ax = axes[0, 1]
            contributions = session_metrics['agent_contributions']
            
            roles = list(contributions.keys())
            avg_scores = [c['average_score'] for c in contributions.values()]
            
            ax.bar(roles, avg_scores)
            ax.set_xlabel('Agent Role')
            ax.set_ylabel('Average Score')
            ax.set_title('Agent Performance')
            ax.tick_params(axis='x', rotation=45)
            
            # 3. Expression diversity
            ax = axes[1, 0]
            unique_counts = [m.num_unique_expressions for m in self.refinement_loop.iteration_history]
            
            ax.plot(iterations, unique_counts, 'r-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Unique Expressions')
            ax.set_title('Expression Diversity')
            ax.grid(True, alpha=0.3)
            
            # 4. Time per iteration
            ax = axes[1, 1]
            times = [m.time_elapsed for m in self.refinement_loop.iteration_history]
            
            ax.bar(iterations, times)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Computation Time')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'discovery_session_{timestamp}.png')
            plt.close()
            
            self.logger.info("Created visualization")
            
        except Exception as e:
            self.logger.warning(f"Could not create visualization: {e}")
    
    def _log_session_summary(self, metrics: Dict[str, Any]):
        """Log summary of discovery session"""
        
        self.logger.info("\n=== Discovery Session Summary ===")
        self.logger.info(f"Problem: {metrics['problem_type']}")
        self.logger.info(f"Duration: {metrics['duration']:.2f} seconds")
        self.logger.info(f"Iterations: {metrics['iterations']}")
        self.logger.info(f"Final Score: {metrics['final_score']:.3f}")
        self.logger.info(f"Expression: {metrics['discovery'].expression}")
        
        self.logger.info("\nAgent Contributions:")
        for role, contrib in metrics['agent_contributions'].items():
            self.logger.info(f"  {role}:")
            self.logger.info(f"    Attempts: {contrib['total_attempts']}")
            self.logger.info(f"    Avg Score: {contrib['average_score']:.3f}")
            self.logger.info(f"    Best Score: {contrib['best_score']:.3f}")


class MultiAgentEnvironmentWrapper:
    """Wrapper that makes any Janus environment multi-agent compatible"""
    
    def __init__(self,
                 base_env: SymbolicDiscoveryEnv,
                 memory_system: DualMemorySystem):
        
        self.base_env = base_env
        self.memory_system = memory_system
        self.agent_observations = {}
        
    def reset(self, **kwargs):
        """Reset environment with multi-agent support"""
        obs, info = self.base_env.reset(**kwargs)
        
        # Clear agent observations
        self.agent_observations.clear()
        
        # Add memory context
        info['episodic_memories'] = self._get_relevant_memories()
        info['shared_memory'] = []
        
        return obs, info
    
    def step(self, action: int, agent_id: str):
        """Step with agent tracking"""
        
        # Execute action
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Track agent-specific information
        self.agent_observations[agent_id] = {
            'observation': obs,
            'reward': reward,
            'action': action,
            'expression': info.get('expression_str', '')
        }
        
        # Add multi-agent context
        info['agent_observations'] = self.agent_observations
        info['agent_id'] = agent_id
        
        return obs, reward, terminated, truncated, info
    
    def get_agent_mask(self, agent_role: AgentRole) -> np.ndarray:
        """Get action mask specific to agent role"""
        
        base_mask = self.base_env.get_action_mask()
        
        # Modify mask based on role specialization
        if agent_role == AgentRole.HYPOTHESIS_GENERATOR:
            # Prefer exploratory actions
            pass
        elif agent_role == AgentRole.VALIDATOR:
            # Prefer validation actions
            pass
        
        return base_mask
    
    def _get_relevant_memories(self) -> List[Dict[str, Any]]:
        """Get relevant memories for current problem"""
        
        # Simplified - would use actual retrieval
        memories = self.memory_system.episodic.get_top_validated(3)
        
        return [
            {
                'expression': m.expression,
                'score': m.validation_score,
                'domain': m.domain
            }
            for m in memories
        ]


# Example usage functions
def run_attention_discovery_example():
    """Example: Discover attention mechanism patterns"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create framework
    framework = JanusMultiAgentFramework(
        memory_db_path="attention_discoveries.db",
        device="cuda",
        enable_training=True
    )
    
    # Define problem configurations
    attention_problems = [
        {
            'pattern_type': 'scaled_dot_product',
            'complexity': 'medium',
            'max_iterations': 8
        },
        {
            'pattern_type': 'previous_token',
            'complexity': 'simple',
            'max_iterations': 5
        },
        {
            'pattern_type': 'local_window',
            'complexity': 'complex',
            'max_iterations': 10
        }
    ]
    
    # Run discoveries
    results = []
    
    for config in attention_problems:
        print(f"\n{'='*60}")
        print(f"Discovering: {config['pattern_type']} attention pattern")
        print(f"{'='*60}")
        
        result = framework.discover(
            problem_type='attention_pattern',
            problem_config=config,
            visualize=True
        )
        
        results.append(result)
        
        print(f"\nDiscovered: {result['discovery'].expression}")
        print(f"Score: {result['final_score']:.3f}")
        print(f"Iterations: {result['iterations']}")
    
    # Analyze results
    print(f"\n{'='*60}")
    print("Session Analysis")
    print(f"{'='*60}")
    
    total_time = sum(r['duration'] for r in results)
    avg_score = np.mean([r['final_score'] for r in results])
    
    print(f"Total problems solved: {len(results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average score: {avg_score:.3f}")
    
    # Show memory statistics
    stats = framework.memory_system.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"Episodic discoveries: {stats['episodic']['total_discoveries']}")
    print(f"Average validation: {stats['episodic
