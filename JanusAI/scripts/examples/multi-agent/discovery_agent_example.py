# JanusAI/scripts/examples/multi_agent/discovery_agent_example.py
"""
Complete Multi-Agent Discovery System Example
============================================

This example demonstrates how to integrate all components of the JanusAI
multi-agent system in a working simulation with phased communication.

Author: JanusAI Team
Date: 2024
"""

import torch.nn as nn
import numpy as np
import logging
from datetime import datetime
import time

# Import all JanusAI components
from schemas import AgentRole, MessageType, Message
from knowledge import MessageBus, SharedKnowledgeBase
from discovery_agent import (
    DiscoveryAgent, 
    DualMemorySystem
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplePolicy(nn.Module):
    """Simple policy network for action selection."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SymbolicEnvironment:
    """
    Simple symbolic regression environment for testing.
    Agents try to discover mathematical expressions.
    """
    
    def __init__(self, target_expr: str = "x^2 + 2*x + 1", obs_dim: int = 10):
        self.target_expr = target_expr
        self.obs_dim = obs_dim
        self.observation_space = type('obj', (object,), {'shape': (obs_dim,)})
        self.action_space = type('obj', (object,), {
            'n': 10,  # 10 possible expression components
            'sample': lambda: np.random.randint(10)
        })
        self.current_expr = []
        self.max_steps = 20
        self.steps = 0
    
    def reset(self):
        self.current_expr = []
        self.steps = 0
        obs = np.random.randn(self.obs_dim)
        return obs, {}
    
    def step(self, action):
        self.steps += 1
        
        # Build expression based on actions
        expr_components = ['x', '+', '-', '*', '^', '1', '2', '3', '(', ')']
        if action < len(expr_components):
            self.current_expr.append(expr_components[action])
        
        # Generate observation
        obs = np.random.randn(self.obs_dim)
        
        # Calculate reward based on expression similarity (simplified)
        expr_str = ''.join(self.current_expr)
        if expr_str == self.target_expr:
            reward = 1.0
        elif self.target_expr.startswith(expr_str):
            reward = 0.1 * len(expr_str) / len(self.target_expr)
        else:
            reward = np.random.uniform(0, 0.3)  # Small random reward
        
        # Check if done
        done = self.steps >= self.max_steps or reward >= 0.99
        
        info = {'expression': expr_str if expr_str else "empty"}
        
        return obs, reward, done, False, info


class MultiAgentDiscoverySystem:
    """
    Complete multi-agent discovery system with phased training.
    """
    
    def __init__(self, num_explorers: int = 2, num_validators: int = 2):
        # Initialize infrastructure
        self.message_bus = MessageBus(max_queue_size=1000)
        self.shared_knowledge = SharedKnowledgeBase(
            validation_threshold=2,
            message_bus=self.message_bus
        )
        
        # Create agents
        self.agents = []
        self.create_agents(num_explorers, num_validators)
        
        # Training phase
        self.current_phase = 1
        
        # Statistics
        self.phase_stats = {1: {}, 2: {}, 3: {}}
    
    def create_agents(self, num_explorers: int, num_validators: int):
        """Create explorer and validator agents."""
        
        # Create explorers
        for i in range(num_explorers):
            agent = self._create_agent(
                agent_id=f"explorer_{i:03d}",
                role=AgentRole.EXPLORER
            )
            self.agents.append(agent)
        
        # Create validators
        for i in range(num_validators):
            agent = self._create_agent(
                agent_id=f"validator_{i:03d}",
                role=AgentRole.VALIDATOR
            )
            self.agents.append(agent)
        
        # Add one refiner and one specialist
        self.agents.append(self._create_agent("refiner_001", AgentRole.REFINER))
        self.agents.append(self._create_agent("specialist_001", AgentRole.SPECIALIST))
        
        logger.info(f"Created {len(self.agents)} agents")
    
    def _create_agent(self, agent_id: str, role: AgentRole) -> DiscoveryAgent:
        """Create a single agent."""
        env = SymbolicEnvironment()
        policy = SimplePolicy(env.obs_dim, env.action_space.n)
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        agent = DiscoveryAgent(
            agent_id=agent_id,
            role=role,
            policy_network=policy,
            environment=env,
            memory_system=memory,
            shared_knowledge=self.shared_knowledge
        )
        
        return agent
    
    def set_phase(self, phase: int):
        """Set training phase for all agents."""
        self.current_phase = phase
        for agent in self.agents:
            agent.set_training_phase(phase)
        logger.info(f"\n{'='*60}")
        logger.info(f"ENTERING PHASE {phase}")
        logger.info(f"{'='*60}")
    
    def run_episode(self, episode_num: int):
        """Run one episode with all agents."""
        episode_discoveries = []
        communication_vectors = {}
        
        # Each agent explores
        for agent in self.agents:
            # Get messages for this agent
            messages = self.message_bus.get_messages(agent.agent_id, max_messages=20)
            
            # Add tactical communications from previous agents this episode
            for sender_id, comm_vec in communication_vectors.items():
                if sender_id != agent.agent_id:
                    tactical_msg = Message(
                        msg_type=MessageType.TACTICAL_VECTOR,
                        sender_id=sender_id,
                        timestamp=datetime.now(),
                        content=comm_vec.tolist(),
                        ttl=30.0
                    )
                    messages.append(tactical_msg)
            
            # Agent explores
            discovery, comm_vector = agent.explore(messages)
            episode_discoveries.append(discovery)
            
            # Store communication vector
            if comm_vector is not None:
                communication_vectors[agent.agent_id] = comm_vector
                
                # Broadcast tactical vector
                if self.current_phase >= 2:
                    tactical_msg = Message(
                        msg_type=MessageType.TACTICAL_VECTOR,
                        sender_id=agent.agent_id,
                        timestamp=datetime.now(),
                        content=comm_vector.tolist(),
                        ttl=60.0
                    )
                    self.message_bus.publish(tactical_msg)
        
        # Log episode summary
        best_discovery = max(episode_discoveries, key=lambda d: d.reward)
        logger.info(f"Episode {episode_num}: Best discovery: {best_discovery.expression} "
                   f"(reward: {best_discovery.reward:.3f}) by {best_discovery.discovered_by}")
        
        return episode_discoveries
    
    def run_phase(self, num_episodes: int = 10):
        """Run a complete training phase."""
        self.phase_stats[self.current_phase] = {
            'discoveries': [],
            'communications': {'tier1': 0, 'tier2': 0},
            'validations': {'proposed': 0, 'approved': 0}
        }
        
        for episode in range(num_episodes):
            discoveries = self.run_episode(episode)
            self.phase_stats[self.current_phase]['discoveries'].extend(discoveries)
            
            # Collect statistics
            for agent in self.agents:
                stats = agent.get_stats()
                self.phase_stats[self.current_phase]['communications']['tier1'] += stats.get('tier1_sent', 0)
                self.phase_stats[self.current_phase]['communications']['tier2'] += stats.get('tier2_sent', 0)
        
        # Phase summary
        self._print_phase_summary()
    
    def _print_phase_summary(self):
        """Print summary statistics for the current phase."""
        stats = self.phase_stats[self.current_phase]
        discoveries = stats['discoveries']
        
        print(f"\n--- Phase {self.current_phase} Summary ---")
        print(f"Total discoveries: {len(discoveries)}")
        
        if discoveries:
            avg_reward = np.mean([d.reward for d in discoveries])
            best_reward = max(d.reward for d in discoveries)
            print(f"Average reward: {avg_reward:.3f}")
            print(f"Best reward: {best_reward:.3f}")
        
        print(f"Tier 1 messages: {stats['communications']['tier1']}")
        print(f"Tier 2 messages: {stats['communications']['tier2']}")
        
        # Knowledge base statistics
        kb_summary = self.shared_knowledge.get_knowledge_summary()
        print(f"Confirmed discoveries: {kb_summary['total_discoveries']}")
        print(f"Pending validations: {kb_summary['pending_validations']}")
        
        # Message bus statistics
        bus_stats = self.message_bus.get_stats()
        print(f"Message bus - Tier 1 queue: {bus_stats['tier1_queue_size']}, "
              f"Tier 2 queue: {bus_stats['tier2_queue_size']}")
    
    def run_full_curriculum(self, episodes_per_phase: int = 10):
        """Run the full three-phase curriculum."""
        
        # Phase 1: No communication
        self.set_phase(1)
        print("\nPhase 1: Agents develop individual competence")
        print("- No communication enabled")
        print("- Agents explore independently")
        self.run_phase(episodes_per_phase)
        
        time.sleep(1)  # Brief pause between phases
        
        # Phase 2: Tactical communication
        self.set_phase(2)
        print("\nPhase 2: Tactical coordination via latent vectors")
        print("- Tier 2 communication enabled")
        print("- Agents share 32-dim tactical vectors")
        print("- Real-time coordination begins")
        self.run_phase(episodes_per_phase)
        
        time.sleep(1)
        
        # Phase 3: Full communication
        self.set_phase(3)
        print("\nPhase 3: Strategic knowledge sharing")
        print("- Tier 1 communication enabled (with cost)")
        print("- Discovery validation protocol active")
        print("- Full multi-agent collaboration")
        self.run_phase(episodes_per_phase)
        
        # Final analysis
        self._final_analysis()
    
    def _final_analysis(self):
        """Perform final analysis across all phases."""
        print("\n" + "="*60)
        print("FINAL ANALYSIS")
        print("="*60)
        
        # Compare phases
        for phase in [1, 2, 3]:
            discoveries = self.phase_stats[phase]['discoveries']
            if discoveries:
                avg_reward = np.mean([d.reward for d in discoveries])
                print(f"\nPhase {phase} average reward: {avg_reward:.3f}")
        
        # Agent performance
        print("\nAgent Performance:")
        agent_stats = self.shared_knowledge.get_agent_statistics()
        for agent_id, stats in sorted(agent_stats.items())[:5]:  # Top 5
            print(f"  {agent_id}: {stats['confirmed_discoveries']} confirmed, "
                  f"avg reward: {stats['avg_reward']:.3f}")
        
        # Best discoveries
        best_discoveries = self.shared_knowledge.get_best_discoveries(n=5)
        if best_discoveries:
            print("\nTop Discoveries:")
            for i, disc in enumerate(best_discoveries, 1):
                print(f"  {i}. {disc.expression} (reward: {disc.reward:.3f}, "
                      f"version: {disc.version}, by: {disc.discovered_by})")
        
        # Communication efficiency
        total_tier1 = sum(p['communications']['tier1'] for p in self.phase_stats.values())
        total_tier2 = sum(p['communications']['tier2'] for p in self.phase_stats.values())
        
        print(f"\nCommunication Summary:")
        print(f"  Total Tier 1 messages: {total_tier1}")
        print(f"  Total Tier 2 messages: {total_tier2}")
        
        if total_tier1 > 0:
            efficiency = len(best_discoveries) / total_tier1
            print(f"  Discovery efficiency: {efficiency:.2f} discoveries per Tier 1 message")


def main():
    """Run the complete multi-agent discovery demonstration."""
    
    print("="*60)
    print("JanusAI Multi-Agent Discovery System")
    print("Phased Communication Curriculum Demo")
    print("="*60)
    
    # Create system
    system = MultiAgentDiscoverySystem(
        num_explorers=2,
        num_validators=2
    )
    
    # Run full curriculum
    system.run_full_curriculum(episodes_per_phase=5)
    
    print("\n✅ Demonstration complete!")


if __name__ == "__main__":
    main()