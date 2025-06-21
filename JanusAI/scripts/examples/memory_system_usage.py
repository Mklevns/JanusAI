# JanusAI/scripts/examples/memory_system_usage.py
"""
Practical examples showing how to integrate the Dual Memory System
with existing Janus components for enhanced scientific discovery.
"""

import torch
import numpy as np
from datetime import datetime
import logging

# Import Janus components
from janus_ai.memory.dual_memory_system import (

    DualMemorySystem, EmbeddingGenerator
)
from janus_ai.memory.memory_integration import (

    MemoryIntegratedEnv, MemoryAugmentedAgent,
    MemoryReplayBuffer, MemoryMetrics
)
from janus_ai.ml.training.advanced_ppo_trainer import AdvancedPPOTrainer, PPOConfig

from janus_ai.ml.networks.policy_networks import TransformerPolicy


from janus_ai.grammar.ai_grammar import AIGrammar

from janus_ai.rewards.interpretability_reward import InterpretabilityReward



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1. Basic Setup and Initialization
def setup_memory_system():
    """Initialize the dual memory system with default settings"""
    
    # Create memory system
    memory_system = DualMemorySystem(
        episodic_capacity=10000,  # Store up to 10k discoveries
        shared_capacity=10,       # Keep top 10 results per problem
        db_path="janus_discoveries.db",  # Persistent storage
        embedding_dim=768         # Standard embedding size
    )
    
    # Create embedding generator (optional but recommended)
    embedding_gen = EmbeddingGenerator()
    
    logger.info("Memory system initialized")
    return memory_system, embedding_gen


# 2. Memory-Augmented Discovery Agent
class DiscoveryAgentWithMemory(MemoryAugmentedAgent):
    """Example agent that uses memory for better discovery"""
    
    def __init__(self, 
                 role: str,
                 policy: torch.nn.Module,
                 grammar: AIGrammar,
                 memory_system: DualMemorySystem):
        
        super().__init__(role, memory_system)
        self.policy = policy
        self.grammar = grammar
        
    def discover(self, problem_context: Dict[str, Any], problem_id: str):
        """Discover solution using memory-augmented reasoning"""
        
        # 1. Retrieve relevant past discoveries
        memories = self.get_relevant_memories(problem_context, k=5)
        
        logger.info(f"{self.role}: Found {len(memories)} relevant memories")
        
        # 2. Build context from memories
        memory_context = []
        for mem in memories:
            memory_context.append({
                'expression': mem.expression,
                'confidence': mem.confidence,
                'domain': mem.domain
            })
        
        # 3. Get current shared memory state
        shared_context = self.get_shared_memory_context(problem_id)
        
        # 4. Generate new hypothesis
        thought = f"Building on {len(memories)} past discoveries and {len(shared_context)} current attempts..."
        
        # Use policy to generate expression (simplified)
        with torch.no_grad():
            # In practice, would encode contexts properly
            expression = self._generate_expression(memory_context, shared_context)
        
        response = f"Proposed expression: {expression}"
        
        # 5. Evaluate expression
        score = self._evaluate_expression(expression, problem_context)
        
        # 6. Add to shared memory
        self.add_to_shared_memory(
            expression=expression,
            thought=thought,
            response=response,
            score=score,
            problem_id=problem_id,
            detailed_scores={
                'novelty': self._calculate_novelty(expression, memories),
                'complexity': self._calculate_complexity(expression),
                'validity': score
            }
        )
        
        return expression, score
    
    def _generate_expression(self, memory_context, shared_context):
        """Generate expression based on contexts"""
        # Simplified - in practice would use policy network
        if memory_context:
            # Adapt from best memory
            base_expr = memory_context[0]['expression']
            return base_expr.replace('softmax', 'log_softmax')  # Example modification
        else:
            return "softmax(attention(Q, K) / sqrt(d))"
    
    def _evaluate_expression(self, expression, context):
        """Evaluate expression quality"""
        # Simplified - in practice would compile and test
        return np.random.uniform(0.6, 0.95)
    
    def _calculate_novelty(self, expression, memories):
        """Calculate novelty compared to memories"""
        if not memories:
            return 1.0
        
        # Check if expression exists in memories
        for mem in memories:
            if expression == mem.expression:
                return 0.1
        
        return 0.8  # Simplified
    
    def _calculate_complexity(self, expression):
        """Calculate expression complexity"""
        # Simple metric based on length and nesting
        return 1.0 / (1.0 + len(expression) / 100)


# 3. Multi-Agent Discovery Session
def run_multi_agent_discovery(memory_system: DualMemorySystem):
    """Run a discovery session with multiple agents using shared memory"""
    
    # Initialize components
    grammar = AIGrammar()
    policy = TransformerPolicy(
        obs_dim=128,
        action_dim=len(grammar.primitives),
        hidden_dim=256
    )
    
    # Create multiple agents with different roles
    agents = [
        DiscoveryAgentWithMemory("HypothesisGenerator", policy, grammar, memory_system),
        DiscoveryAgentWithMemory("SymbolicReasoner", policy, grammar, memory_system),
        DiscoveryAgentWithMemory("Validator", policy, grammar, memory_system)
    ]
    
    # Define problem
    problem_context = {
        'domain': 'attention_mechanisms',
        'task': 'discover_multi_head_attention',
        'description': 'Find expression for multi-head attention mechanism'
    }
    
    # Start problem in memory system
    problem_id = f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shared_memory = memory_system.start_problem(problem_id)
    
    # Run multiple iterations
    for iteration in range(3):
        logger.info(f"\n--- Iteration {iteration + 1} ---")
        
        # Each agent contributes
        for agent in agents:
            expression, score = agent.discover(problem_context, problem_id)
            logger.info(f"{agent.role}: {expression} (score: {score:.3f})")
        
        # Check shared memory state
        top_results = shared_memory.get_top(3)
        logger.info(f"\nTop expressions in shared memory:")
        for i, result in enumerate(top_results):
            logger.info(f"{i+1}. {result.expression} (score: {result.score:.3f}) by {result.agent_role}")
    
    # End problem and save best to episodic
    discovery = memory_system.end_problem(problem_id, problem_context['domain'])
    
    if discovery:
        logger.info(f"\nSaved discovery to episodic memory:")
        logger.info(f"Expression: {discovery.expression}")
        logger.info(f"Confidence: {discovery.confidence:.3f}")


# 4. Memory-Enhanced Training Loop
def train_with_memory_replay(memory_system: DualMemorySystem):
    """Training loop that uses memory replay"""
    
    # Setup
    grammar = AIGrammar()
    reward_fn = InterpretabilityReward()
    
    # Create memory-integrated environment
    env = MemoryIntegratedEnv(
        memory_system=memory_system,
        grammar=grammar,
        reward_fn=reward_fn,
        task_type="symbolic_regression"
    )
    
    # Create policy
    policy = TransformerPolicy(
        obs_dim=128,
        action_dim=len(grammar.primitives),
        hidden_dim=256
    )
    
    # Create trainer with PPO
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        n_epochs=4,
        batch_size=64
    )
    
    trainer = AdvancedPPOTrainer(
        policy=policy,
        env=env,
        config=ppo_config
    )
    
    # Create replay buffer
    replay_buffer = MemoryReplayBuffer(memory_system)
    
    # Training loop
    for episode in range(100):
        # Regular environment episode
        obs, info = env.reset()
        
        # Get episodic memories for context
        episodic_memories = info.get('episodic_memories', [])
        
        done = False
        episode_reward = 0
        
        while not done:
            # Policy uses episodic context
            action = policy.get_action(obs, context=episodic_memories)
            
            obs, reward, terminated, truncated, info = env.step(action, agent_role="Learner")
            done = terminated or truncated
            episode_reward += reward
        
        logger.info(f"Episode {episode}: Reward = {episode_reward:.3f}")
        
        # Periodically train on replay
        if episode % 10 == 0 and episode > 0:
            # Sample high-quality discoveries
            replay_batch = replay_buffer.sample(
                batch_size=16,
                min_score=0.7  # Only good discoveries
            )
            
            if replay_batch:
                logger.info(f"Training on {len(replay_batch)} replay samples")
                
                # Convert to training batch
                training_batch = replay_buffer.create_training_batch(replay_batch)
                
                # Train policy on replay
                # ... (training logic here)


# 5. Analyze Memory System Performance
def analyze_memory_performance(memory_system: DualMemorySystem):
    """Analyze what the system has learned"""
    
    metrics = MemoryMetrics(memory_system)
    
    # Get overall statistics
    stats = memory_system.get_memory_stats()
    
    logger.info("\n=== Memory System Analysis ===")
    logger.info(f"Total discoveries: {stats['episodic']['total_discoveries']}")
    logger.info(f"Average confidence: {stats['episodic']['average_confidence']:.3f}")
    logger.info(f"Average validation: {stats['episodic']['average_validation']:.3f}")
    
    # Analyze by domain
    domain_progress = metrics.analyze_domain_progress()
    
    logger.info("\nProgress by domain:")
    for domain, progress in domain_progress.items():
        logger.info(f"\n{domain}:")
        logger.info(f"  Discoveries: {progress['total_discoveries']}")
        logger.info(f"  Best validation: {progress['best_validation']:.3f}")
        logger.info(f"  Avg validation: {progress['average_validation']:.3f}")
    
    # Show top discoveries
    logger.info("\nTop 5 discoveries overall:")
    top_discoveries = memory_system.episodic.get_top_validated(5)
    
    for i, discovery in enumerate(top_discoveries):
        logger.info(f"\n{i+1}. {discovery.expression}")
        logger.info(f"   Domain: {discovery.domain}")
        logger.info(f"   Validation: {discovery.validation_score:.3f}")
        logger.info(f"   Confidence: {discovery.confidence:.3f}")
        logger.info(f"   Agents: {', '.join(discovery.agent_roles)}")


# 6. Advanced Usage: Cross-Problem Learning
def demonstrate_cross_problem_learning(memory_system: DualMemorySystem):
    """Show how system learns across problems"""
    
    problems = [
        {
            'domain': 'attention_mechanisms',
            'task': 'scaled_dot_product',
            'target': 'softmax(Q @ K.T / sqrt(d))'
        },
        {
            'domain': 'attention_mechanisms', 
            'task': 'multi_head_attention',
            'target': 'concat([head_i(Q, K, V) for i in range(h)]) @ W_O'
        },
        {
            'domain': 'physics_laws',
            'task': 'harmonic_oscillator',
            'target': 'x = A * cos(omega * t + phi)'
        }
    ]
    
    # Create agent
    agent = DiscoveryAgentWithMemory(
        "CrossLearner",
        TransformerPolicy(obs_dim=128, action_dim=20, hidden_dim=256),
        AIGrammar(),
        memory_system
    )
    
    # Solve problems sequentially
    for i, problem in enumerate(problems):
        logger.info(f"\n=== Problem {i+1}: {problem['task']} ===")
        
        # Get relevant memories (will increase over time)
        memories = agent.get_relevant_memories(problem, k=3)
        logger.info(f"Using {len(memories)} relevant past discoveries")
        
        # Start problem
        problem_id = f"{problem['task']}_{i}"
        shared_mem = memory_system.start_problem(problem_id)
        
        # Discover (simplified)
        expression, score = agent.discover(problem, problem_id)
        
        # End problem and save
        discovery = memory_system.end_problem(problem_id, problem['domain'])
        
        if discovery:
            logger.info(f"Discovered: {discovery.expression}")
            logger.info(f"Target was: {problem['target']}")
    
    # Analyze cross-problem learning
    logger.info("\n=== Cross-Problem Learning Analysis ===")
    
    # Check if later problems benefited from earlier ones
    all_discoveries = memory_system.episodic.get_top_validated(10)
    
    # Group by timestamp to see improvement
    discoveries_by_order = sorted(all_discoveries, key=lambda d: d.timestamp)
    
    scores_over_time = [d.validation_score for d in discoveries_by_order]
    if len(scores_over_time) > 1:
        improvement = scores_over_time[-1] - scores_over_time[0]
        logger.info(f"Score improvement: {improvement:.3f}")
        logger.info(f"Average score increased from {scores_over_time[0]:.3f} to {scores_over_time[-1]:.3f}")


# 7. Main execution
if __name__ == "__main__":
    # Initialize memory system
    memory_system, embedding_gen = setup_memory_system()
    
    # Run different examples
    logger.info("\n=== Example 1: Multi-Agent Discovery ===")
    run_multi_agent_discovery(memory_system)
    
    logger.info("\n=== Example 2: Cross-Problem Learning ===")
    demonstrate_cross_problem_learning(memory_system)
    
    logger.info("\n=== Example 3: Memory Analysis ===")
    analyze_memory_performance(memory_system)
    
    # Save memory state
    memory_system.save_state("janus_memory_checkpoint")
    logger.info("\n=== Memory state saved ===")
    
    # Example loading from checkpoint
    # new_memory_system = DualMemorySystem()
    # new_memory_system.load_state("janus_memory_checkpoint")
    
    logger.info("\n=== Memory System Demo Complete ===")
