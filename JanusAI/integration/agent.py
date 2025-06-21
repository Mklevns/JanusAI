# JanusAI/integration/agent.py
"""
Discovery Agent with Neural Communication Components
===================================================

This module implements the core DiscoveryAgent class and its neural communication
components for the JanusAI multi-agent system.

Components:
- CommunicationEncoder: Encodes agent state to latent communication vector
- CommunicationAggregator: Attention-based aggregation of peer communications
- DiscoveryAgent: Main agent class with exploration and validation logic

Author: JanusAI Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


# Import schemas and infrastructure
from janus_ai.integration.schemas import AgentRole, MessageType, Discovery, Message
from janus_ai.integration.knowledge import SharedKnowledgeBase


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for DiscoveryAgent behavior and neural components."""
    latent_dim: int = 32
    num_heads: int = 4
    exploration_rate: Dict[AgentRole, float] = field(default_factory=lambda: {
        AgentRole.EXPLORER: 0.3,
        AgentRole.REFINER: 0.1,
        AgentRole.VALIDATOR: 0.05,
        AgentRole.SPECIALIST: 0.2
    })
    tier1_cost: float = 0.1
    empirical_ambiguity_threshold: float = 0.15
    empirical_weight: float = 0.7
    llm_weight: float = 0.3


class CommunicationEncoder(nn.Module):
    """
    Neural encoder for Tier 2 tactical communication.
    Encodes high-dimensional agent state into low-dimensional latent vectors.
    """
    def __init__(self, input_dim: int, config: AgentConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, config.latent_dim),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)


class CommunicationAggregator(nn.Module):
    """
    Attention-based aggregator for incoming communication vectors.
    Uses multi-head attention to weigh messages based on relevance.
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.latent_dim,
            num_heads=config.num_heads,
            batch_first=False # PyTorch MHA expects (seq, batch, embed) by default
        )
        self.layer_norm = nn.LayerNorm(config.latent_dim)

    def forward(self, own_state: torch.Tensor, comm_vectors: List[torch.Tensor]) -> torch.Tensor:
        if not comm_vectors:
            return torch.zeros_like(own_state)

        # own_state: [latent_dim] -> query: [1, 1, latent_dim] (seq_len=1, batch_size=1)
        query = own_state.unsqueeze(0).unsqueeze(0)

        # comm_vectors: List of [latent_dim] -> key/value: [N, 1, latent_dim] (seq_len=N, batch_size=1)
        comm_stack = torch.stack(comm_vectors, dim=0).unsqueeze(1)

        # attended_output shape: [1, 1, latent_dim]
        attended_output, _ = self.multihead_attn(query=query, key=comm_stack, value=comm_stack)
        
        # Squeeze to get back to [latent_dim]
        squeezed_output = attended_output.squeeze(0).squeeze(0)

        # Apply residual connection and layer norm
        output = self.layer_norm(squeezed_output + own_state)

        return output


class DiscoveryAgent:
    """
    The fundamental "citizen" of the AI society, capable of exploration,
    communication, and validation based on its assigned role.
    """
    def __init__(self, agent_id: str, role: AgentRole, policy_network: nn.Module,
                 environment: Any, memory_system: DualMemorySystem,
                 shared_knowledge: SharedKnowledgeBase, config: AgentConfig):

        self.agent_id = agent_id
        self.role = role
        self.policy_network = policy_network
        self.environment = environment
        self.memory_system = memory_system
        self.shared_knowledge = shared_knowledge
        self.config = config

        obs_dim = environment.observation_space.shape[0]
        self.comm_encoder = CommunicationEncoder(obs_dim, config)
        self.comm_aggregator = CommunicationAggregator(config)
        self.exploration_rate = config.exploration_rate.get(role, 0.15)
        self.training_phase = 1
        self.stats = {'discoveries': 0, 'tier1_sent': 0, 'tier2_sent': 0, 'validations': 0}

        logger.info(f"Initialized {role.value} agent {agent_id}")

    def _send_tactical_communication(self, comm_vector: torch.Tensor):
        """Helper to create and publish a Tier 2 message."""
        tactical_msg = Message(
            msg_type=MessageType.TACTICAL_VECTOR,
            sender_id=self.agent_id,
            timestamp=datetime.now(),
            content=comm_vector.detach().numpy().tolist() # Convert to list for serialization
        )
        self.shared_knowledge.message_bus.publish(tactical_msg)
        self.stats['tier2_sent'] += 1

    def explore(self, incoming_messages: List[Message]) -> Tuple[Optional[Discovery], Optional[torch.Tensor]]:
        """Main exploration and action loop for the agent."""
        # Validators have a different primary loop: validation.
        if self.role == AgentRole.VALIDATOR:
            for msg in incoming_messages:
                if msg.msg_type == MessageType.VALIDATION_REQUEST:
                    self._handle_validation_request(msg)
            return None, None # Validators don't generate new discoveries in this loop

        tactical_vectors = [
            torch.FloatTensor(msg.content)
            for msg in incoming_messages
            if msg.msg_type == MessageType.TACTICAL_VECTOR and self.training_phase >= 2
        ]

        obs, _ = self.environment.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_tensor = torch.FloatTensor(obs)
            final_obs_tensor = obs_tensor.clone() # Start with the base observation

            if self.training_phase >= 2:
                # Generate this step's communication vector
                comm_vector = self.comm_encoder(obs_tensor)
                self._send_tactical_communication(comm_vector)
                
                # Incorporate peer communication
                if tactical_vectors:
                    aggregated_comm = self.comm_aggregator(comm_vector, tactical_vectors)
                    # Additive influence on the observation
                    final_obs_tensor += 0.1 * aggregated_comm
            
            # Action selection
            if np.random.random() < self.exploration_rate:
                action = self.environment.action_space.sample()
            else:
                with torch.no_grad():
                    action_probs = self.policy_network(final_obs_tensor.unsqueeze(0))
                    action = torch.argmax(F.softmax(action_probs, dim=-1)).item()
            
            next_obs, reward, terminated, truncated, info = self.environment.step(action)
            total_reward += reward
            obs = next_obs
            done = terminated or truncated

        # Create discovery object with keyword arguments for correctness
        discovery = Discovery(
            expression=info.get('expression', f'action_{action}'),
            reward=float(total_reward),
            timestamp=datetime.now(),
            discovered_by=self.agent_id,
            metadata={'role': self.role.value, 'phase': self.training_phase}
        )
        self.memory_system.add_discovery(discovery)
        self.stats['discoveries'] += 1

        if self.training_phase >= 3 and self._should_publish_discovery(discovery):
            self._publish_discovery(discovery)

        # The final comm_vector is not used here, but could be returned for analysis
        return discovery, self.comm_encoder(torch.FloatTensor(obs)) if self.training_phase >= 2 else None

    def _publish_discovery(self, discovery: Discovery):
        """Applies cost and proposes a discovery."""
        discovery.reward -= self.config.tier1_cost
        discovery_id = self.shared_knowledge.propose_discovery(self.agent_id, discovery)
        self.stats['tier1_sent'] += 1
        logger.info(f"Agent {self.agent_id} proposed discovery {discovery_id} (cost: {self.config.tier1_cost})")

    def _should_publish_discovery(self, discovery: Discovery) -> bool:
        """Decide whether to publish a discovery to Tier 1."""
        if discovery.reward < self.config.tier1_cost * 2:
            return False
        
        best_known = self.shared_knowledge.get_best_discoveries(n=1)
        if best_known and discovery.reward <= best_known[0].reward * 1.1:
            return False
            
        return True
    
    def _handle_validation_request(self, msg: Message):
        """Handles a validation request using the Augmented Consensus pattern."""
        discovery = msg.content['discovery']
        discovery_id = msg.content['discovery_id']
        logger.info(f"Validator {self.agent_id} evaluating {discovery_id}")

        empirical_score = self._empirical_validation(discovery)
        llm_similarity = 0.5  # Neutral default
        evidence = {'empirical_score': empirical_score, 'method': 'empirical_only'}

        if abs(empirical_score - 0.5) < self.config.empirical_ambiguity_threshold:
            logger.info(f"Score {empirical_score:.3f} is ambiguous, consulting LLM.")
            llm_similarity = self._consult_llm(discovery)
            evidence.update({'llm_consulted': True, 'llm_similarity': llm_similarity, 'method': 'empirical_with_llm'})

        final_score = (self.config.empirical_weight * empirical_score + self.config.llm_weight * llm_similarity) if evidence.get('llm_consulted') else empirical_score
        approve = final_score > 0.6
        evidence.update({'final_score': final_score, 'decision_threshold': 0.6})

        self.shared_knowledge.vote_on_discovery(self.agent_id, discovery_id, approve, evidence)
        self.stats['validations'] += 1
        logger.info(f"Validator {self.agent_id} voted {'APPROVE' if approve else 'REJECT'} on {discovery_id} (score: {final_score:.3f})")

    def _empirical_validation(self, discovery: Discovery) -> float:
        """Placeholder for empirical validation."""
        # In practice: re-evaluate on a held-out validation dataset
        base_score = np.clip(discovery.reward, 0, 1)
        complexity_penalty = min(len(str(discovery.expression)) / 100.0, 0.2)
        return max(0, base_score - complexity_penalty)

    def _consult_llm(self, discovery: Discovery) -> float:
        """Placeholder for LLM consultation."""
        # In practice: Format prompt, call local LLM API, parse response
        logger.debug(f"Simulating LLM consultation for: {discovery.expression}")
        return np.clip(np.random.normal(loc=discovery.reward, scale=0.15), 0, 1)

    def set_training_phase(self, phase: int):
        self.training_phase = phase
        logger.info(f"Agent {self.agent_id} set to training phase {phase}")

    def get_stats(self) -> Dict[str, Any]:
        return {'agent_id': self.agent_id, 'role': self.role.value, **self.stats}


# The mock environment and __main__ block are great for testing and can be kept.
class MockEnvironment:
    def __init__(self, obs_dim: int = 10):
        self.observation_space = type('obj', (object,), {'shape': (obs_dim,)})
        self.action_space = type('obj', (object,), {'n': 4, 'sample': lambda: np.random.randint(4)})
    def reset(self): return np.random.randn(self.observation_space.shape[0]), {}
    def step(self, action):
        obs = np.random.randn(self.observation_space.shape[0])
        reward = np.random.uniform(0, 1)
        done = np.random.random() < 0.1
        return obs, reward, done, False, {'expression': f"x**{action}"}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Setup
    config = AgentConfig()
    bus = MessageBus()
    kb = SharedKnowledgeBase(validation_threshold=1, message_bus=bus)
    env = MockEnvironment(obs_dim=10)
    
    class MockPolicy(nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Linear(obs_dim, action_dim)
        def forward(self, x): return self.net(x)

    policy = MockPolicy(10, 4)

    # Create agents
    explorer = DiscoveryAgent("explorer_001", AgentRole.EXPLORER, policy, env, DualMemorySystem(), kb, config)
    validator = DiscoveryAgent("validator_001", AgentRole.VALIDATOR, policy, env, DualMemorySystem(), kb, config)
    
    # --- Simulate Phase 3 ---
    print("\n--- Phase 3: Full Communication Simulation ---")
    explorer.set_training_phase(3)
    validator.set_training_phase(3)

    # Explorer makes a discovery and automatically proposes it if good enough
    discovery, _ = explorer.explore([])
    print(f"Explorer stats: {explorer.get_stats()}")

    # Validator checks for and handles validation requests
    messages = bus.get_messages("validator_001")
    if messages:
        print(f"\nValidator received {len(messages)} validation requests.")
        validator.explore(messages)
        print(f"Validator stats: {validator.get_stats()}")
    else:
        print("\nValidator received no validation requests.")
        
    print(f"\nKnowledge Base State: {kb.get_knowledge_summary()}")
