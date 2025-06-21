# JanusAI/memory/memory_integration.py
"""
Integration utilities for connecting the Dual Memory System with Janus components.
Provides adapters, hooks, and helper functions for seamless integration.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging

from janus_ai.dual_memory_system import (

    DualMemorySystem, Discovery, IntermediateResult, 
    EmbeddingGenerator, SharedMemory
)
from janus_ai.environments.symbolic_discovery_env import SymbolicDiscoveryEnv
from janus_ai.grammar.expression_tree import ExpressionTree


class MemoryIntegratedEnv(SymbolicDiscoveryEnv):
    """
    Extended SymbolicDiscoveryEnv with integrated memory system.
    Automatically tracks discoveries and provides memory-augmented observations.
    """
    
    def __init__(self, 
                 memory_system: DualMemorySystem,
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.memory_system = memory_system
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.current_problem_id = None
        self.shared_memory = None
        
        # Track agent interactions
        self.agent_tracker = {}
        
    def reset(self, **kwargs) -> tuple:
        """Reset environment with memory initialization"""
        obs, info = super().reset(**kwargs)
        
        # Start new problem in memory system
        self.current_problem_id = f"{self.task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.shared_memory = self.memory_system.start_problem(self.current_problem_id)
        
        # Augment observation with relevant episodic memories
        if self.embedding_generator.available:
            # Generate embedding for current problem
            problem_text = f"{self.task_type} {kwargs.get('target_expression', '')}"
            query_embedding = self.embedding_generator.generate(problem_text)
            
            if query_embedding is not None:
                relevant_discoveries = self.memory_system.get_relevant_discoveries(
                    query_embedding=query_embedding,
                    domain=self.task_type,
                    k=3
                )
                
                # Add to info
                info['episodic_memories'] = [
                    {
                        'expression': d.expression,
                        'confidence': d.confidence,
                        'validation': d.validation_score
                    }
                    for d in relevant_discoveries
                ]
        
        return obs, info
    
    def step(self, action: int, agent_role: str = "Unknown") -> tuple:
        """Step with automatic memory tracking"""
        # Execute action
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track agent
        if agent_role not in self.agent_tracker:
            self.agent_tracker[agent_role] = {
                'steps': 0,
                'total_reward': 0.0,
                'expressions': []
            }
        
        self.agent_tracker[agent_role]['steps'] += 1
        self.agent_tracker[agent_role]['total_reward'] += reward
        
        # Get current expression
        current_expr = info.get('expression_str', '')
        if current_expr:
            self.agent_tracker[agent_role]['expressions'].append(current_expr)
        
        # Add to shared memory if significant
        if reward > 0.5 or terminated:  # Threshold for significance
            result = IntermediateResult(
                id=f"{self.current_problem_id}_{agent_role}_{self.agent_tracker[agent_role]['steps']}",
                timestamp=datetime.now(),
                domain=self.task_type,
                agent_role=agent_role,
                expression=current_expr,
                thought=f"Step {self.agent_tracker[agent_role]['steps']}: Exploring {current_expr}",
                response=f"Achieved reward {reward:.3f}",
                score=reward,
                detailed_scores={
                    'step_reward': reward,
                    'cumulative_reward': self.agent_tracker[agent_role]['total_reward']
                },
                iteration=self.agent_tracker[agent_role]['steps'] // 10  # Rough iteration estimate
            )
            
            self.shared_memory.add(result)
        
        # Augment info with shared memory state
        info['shared_memory_top'] = [
            {
                'expression': r.expression,
                'score': r.score,
                'agent': r.agent_role
            }
            for r in self.shared_memory.get_top(3)
        ]
        
        # If episode ends, potentially save to episodic
        if terminated and reward > 0.8:  # High-quality discovery
            discovery = self.memory_system.end_problem(
                self.current_problem_id,
                domain=self.task_type,
                save_to_episodic=True
            )
            
            if discovery:
                info['saved_to_episodic'] = True
                info['discovery_id'] = discovery.id
        
        return obs, reward, terminated, truncated, info


class MemoryAugmentedAgent:
    """
    Base class for agents that use the memory system.
    Provides utilities for memory access and update.
    """
    
    def __init__(self,
                 role: str,
                 memory_system: DualMemorySystem,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        
        self.role = role
        self.memory_system = memory_system
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.logger = logging.getLogger(f"{__name__}.{role}")
        
    def get_relevant_memories(self, 
                            context: Dict[str, Any], 
                            k: int = 5) -> List[Discovery]:
        """Retrieve relevant memories for current context"""
        
        # Try embedding-based retrieval first
        if self.embedding_generator.available:
            context_text = self._context_to_text(context)
            embedding = self.embedding_generator.generate(context_text)
            
            if embedding is not None:
                return self.memory_system.get_relevant_discoveries(
                    query_embedding=embedding,
                    domain=context.get('domain'),
                    k=k
                )
        
        # Fallback to domain-based retrieval
        domain = context.get('domain')
        if domain:
            return self.memory_system.episodic.search_by_domain(domain, k)
        
        # Last resort: top validated
        return self.memory_system.episodic.get_top_validated(k)
    
    def add_to_shared_memory(self,
                           expression: str,
                           thought: str,
                           response: str,
                           score: float,
                           problem_id: str,
                           detailed_scores: Optional[Dict[str, float]] = None):
        """Add result to shared memory"""
        
        if problem_id not in self.memory_system.active_problems:
            self.logger.warning(f"Problem {problem_id} not active, cannot add to shared memory")
            return
        
        shared_mem = self.memory_system.active_problems[problem_id]
        
        result = IntermediateResult(
            id=f"{problem_id}_{self.role}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            domain="unknown",  # Would be set by context
            agent_role=self.role,
            expression=expression,
            thought=thought,
            response=response,
            score=score,
            detailed_scores=detailed_scores or {},
            iteration=0  # Would be tracked externally
        )
        
        shared_mem.add(result)
        self.logger.debug(f"Added result to shared memory (score: {score:.3f})")
    
    def get_shared_memory_context(self, problem_id: str) -> List[Dict[str, Any]]:
        """Get current shared memory state for context building"""
        
        if problem_id not in self.memory_system.active_problems:
            return []
        
        shared_mem = self.memory_system.active_problems[problem_id]
        
        return [
            {
                'agent': r.agent_role,
                'expression': r.expression,
                'score': r.score,
                'thought': r.thought[:200] + '...' if len(r.thought) > 200 else r.thought
            }
            for r in shared_mem.get_top(5)
        ]
    
    def _context_to_text(self, context: Dict[str, Any]) -> str:
        """Convert context dictionary to text for embedding"""
        parts = []
        
        if 'domain' in context:
            parts.append(f"Domain: {context['domain']}")
        
        if 'task' in context:
            parts.append(f"Task: {context['task']}")
            
        if 'expression' in context:
            parts.append(f"Expression: {context['expression']}")
            
        if 'description' in context:
            parts.append(context['description'])
        
        return " ".join(parts)


class MemoryReplayBuffer:
    """
    Replay buffer that samples from episodic memory for training.
    Useful for meta-learning and continual learning scenarios.
    """
    
    def __init__(self, 
                 memory_system: DualMemorySystem,
                 prioritize_by: str = "validation_score"):
        
        self.memory_system = memory_system
        self.prioritize_by = prioritize_by
        
    def sample(self, 
              batch_size: int, 
              domain: Optional[str] = None,
              min_score: float = 0.0) -> List[Discovery]:
        """Sample discoveries for replay"""
        
        # Get all eligible discoveries
        if domain:
            candidates = self.memory_system.episodic.search_by_domain(domain, limit=1000)
        else:
            candidates = list(self.memory_system.episodic.memories.values())
        
        # Filter by minimum score
        candidates = [d for d in candidates if getattr(d, self.prioritize_by) >= min_score]
        
        if not candidates:
            return []
        
        # Prioritized sampling
        scores = np.array([getattr(d, self.prioritize_by) for d in candidates])
        
        # Convert to probabilities
        probs = scores / scores.sum()
        
        # Sample indices
        n_samples = min(batch_size, len(candidates))
        indices = np.random.choice(len(candidates), size=n_samples, p=probs, replace=False)
        
        return [candidates[i] for i in indices]
    
    def create_training_batch(self, 
                            discoveries: List[Discovery],
                            include_negatives: bool = True) -> Dict[str, torch.Tensor]:
        """Create training batch from discoveries"""
        
        batch = {
            'expressions': [],
            'domains': [],
            'scores': [],
            'embeddings': []
        }
        
        for discovery in discoveries:
            batch['expressions'].append(discovery.expression)
            batch['domains'].append(discovery.domain)
            batch['scores'].append(discovery.validation_score)
            
            if discovery.embedding is not None:
                batch['embeddings'].append(torch.from_numpy(discovery.embedding))
        
        # Convert to tensors
        batch['scores'] = torch.tensor(batch['scores'], dtype=torch.float32)
        
        if batch['embeddings']:
            batch['embeddings'] = torch.stack(batch['embeddings'])
        
        # Add negative examples if requested
        if include_negatives:
            # Sample low-scoring discoveries as negatives
            negatives = self.sample(
                len(discoveries), 
                min_score=0.0
            )
            
            # Filter to get truly negative examples
            negatives = [d for d in negatives if d.validation_score < 0.5][:len(discoveries)]
            
            for neg in negatives:
                batch['expressions'].append(neg.expression)
                batch['domains'].append(neg.domain)
                batch['scores'].append(0.0)  # Label as negative
        
        return batch


class MemoryMetrics:
    """Track and analyze memory system performance"""
    
    def __init__(self, memory_system: DualMemorySystem):
        self.memory_system = memory_system
        self.metrics_history = []
        
    def record_snapshot(self):
        """Record current memory state metrics"""
        
        metrics = {
            'timestamp': datetime.now(),
            'episodic_stats': self.memory_system.episodic.get_statistics(),
            'active_problems': len(self.memory_system.active_problems),
            'shared_memory_scores': {}
        }
        
        # Get average scores from active shared memories
        for problem_id, shared_mem in self.memory_system.active_problems.items():
            stats = shared_mem.get_statistics()
            metrics['shared_memory_scores'][problem_id] = {
                'average_score': stats.get('average_score', 0.0),
                'best_score': stats.get('best_score', 0.0)
            }
        
        self.metrics_history.append(metrics)
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curves over time"""
        
        curves = {
            'episodic_size': [],
            'average_confidence': [],
            'average_validation': [],
            'discovery_rate': []
        }
        
        for i, metrics in enumerate(self.metrics_history):
            ep_stats = metrics['episodic_stats']
            
            curves['episodic_size'].append(ep_stats['total_discoveries'])
            curves['average_confidence'].append(ep_stats['average_confidence'])
            curves['average_validation'].append(ep_stats['average_validation'])
            
            # Calculate discovery rate
            if i > 0:
                prev_size = self.metrics_history[i-1]['episodic_stats']['total_discoveries']
                curr_size = ep_stats['total_discoveries']
                rate = curr_size - prev_size
            else:
                rate = ep_stats['total_discoveries']
            
            curves['discovery_rate'].append(rate)
        
        return curves
    
    def analyze_domain_progress(self) -> Dict[str, Dict[str, float]]:
        """Analyze progress by domain"""
        
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]['episodic_stats']
        domains = latest.get('domains', {})
        
        # Get domain-specific metrics
        domain_progress = {}
        
        for domain, count in domains.items():
            # Get top discoveries in domain
            top_in_domain = self.memory_system.episodic.search_by_domain(domain, limit=100)
            
            if top_in_domain:
                domain_progress[domain] = {
                    'total_discoveries': count,
                    'average_validation': np.mean([d.validation_score for d in top_in_domain]),
                    'best_validation': max(d.validation_score for d in top_in_domain),
                    'average_confidence': np.mean([d.confidence for d in top_in_domain])
                }
        
        return domain_progress


# Example integration with training loop
class MemoryAugmentedTrainer:
    """Example trainer that uses memory system"""
    
    def __init__(self,
                 policy: torch.nn.Module,
                 memory_system: DualMemorySystem,
                 env: MemoryIntegratedEnv):
        
        self.policy = policy
        self.memory_system = memory_system
        self.env = env
        self.replay_buffer = MemoryReplayBuffer(memory_system)
        
    def train_step(self, use_replay: bool = True):
        """Single training step with optional memory replay"""
        
        # Regular environment interaction
        obs, info = self.env.reset()
        
        # Get episodic context
        episodic_context = info.get('episodic_memories', [])
        
        # Run episode
        done = False
        while not done:
            # Get action from policy (would include episodic context)
            action = self.policy.get_action(obs, episodic_context)
            
            obs, reward, terminated, truncated, info = self.env.step(action, agent_role="Trainer")
            done = terminated or truncated
            
            # Update policy (simplified)
            # ... policy update logic ...
        
        # Memory replay
        if use_replay and np.random.random() < 0.3:  # 30% chance
            replay_batch = self.replay_buffer.sample(batch_size=8)
            
            if replay_batch:
                # Train on replay batch
                # ... replay training logic ...
                pass


# Usage example
if __name__ == "__main__":
    # Initialize memory system
    memory_system = DualMemorySystem(
        episodic_capacity=5000,
        shared_capacity=10,
        db_path="janus_memory.db"
    )
    
    # Create memory-integrated environment
    from ..grammar.ai_grammar import AIGrammar
    from ..rewards.interpretability_reward import InterpretabilityReward
    
    grammar = AIGrammar()
    reward_fn = InterpretabilityReward()
    
    env = MemoryIntegratedEnv(
        memory_system=memory_system,
        grammar=grammar,
        reward_fn=reward_fn,
        task_type="attention_pattern"
    )
    
    # Create memory-augmented agent
    agent = MemoryAugmentedAgent(
        role="Explorer",
        memory_system=memory_system
    )
    
    # Run discovery session
    obs, info = env.reset()
    
    # Agent gets relevant memories
    memories = agent.get_relevant_memories({
        'domain': 'attention_pattern',
        'task': 'discover_softmax_attention'
    })
    
    print(f"Found {len(memories)} relevant memories")
    for mem in memories[:3]:
        print(f"- {mem.expression} (score: {mem.validation_score:.3f})")
    
    # Simulate discovery
    agent.add_to_shared_memory(
        expression="softmax(Q @ K.T / sqrt(d))",
        thought="This captures scaled dot-product attention",
        response="Standard attention mechanism discovered",
        score=0.85,
        problem_id=env.current_problem_id
    )
    
    # Check metrics
    metrics = MemoryMetrics(memory_system)
    metrics.record_snapshot()
    
    print("\nMemory System Status:")
    print(f"Episodic discoveries: {memory_system.episodic.get_statistics()['total_discoveries']}")
    print(f"Active problems: {len(memory_system.active_problems)}")
