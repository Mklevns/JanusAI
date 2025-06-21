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
from dataclasses import dataclass

# Import schemas and infrastructure
from janus_ai.integration.schemas import AgentRole, MessageType, Discovery, Message
from janus_ai.integration.knowledge import SharedKnowledgeBase

logger = logging.getLogger(__name__)


class CommunicationEncoder(nn.Module):
    """
    Neural encoder for Tier 2 tactical communication.
    
    Encodes high-dimensional agent state into low-dimensional latent vectors
    for efficient inter-agent communication.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 64):
        """
        Initialize the communication encoder.
        
        Args:
            input_dim: Dimension of input state vector
            latent_dim: Dimension of output latent communication vector
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state to communication vector.
        
        Args:
            state: Agent state tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Latent communication vector of shape (batch_size, latent_dim) or (latent_dim,)
        """
        # Handle both batched and single inputs
        single_input = state.dim() == 1
        if single_input:
            state = state.unsqueeze(0)
            
        encoded = self.encoder(state)
        
        if single_input:
            encoded = encoded.squeeze(0)
            
        return encoded


class CommunicationAggregator(nn.Module):
    """
    Attention-based aggregator for incoming communication vectors.
    
    Uses scaled dot-product attention to compute weighted averages of
    incoming messages based on relevance to the agent's own state.
    """
    
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, num_heads: int = 4):
        """
        Initialize the communication aggregator.
        
        Args:
            latent_dim: Dimension of communication vectors
            hidden_dim: Dimension of attention projections
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention projections
        self.query_proj = nn.Linear(latent_dim, hidden_dim)
        self.key_proj = nn.Linear(latent_dim, hidden_dim)
        self.value_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, own_state: torch.Tensor, comm_vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate incoming communication vectors using attention.
        
        Args:
            own_state: Agent's own state/communication vector (latent_dim,)
            comm_vectors: List of incoming communication vectors
            
        Returns:
            Aggregated communication vector (latent_dim,)
        """
        if not comm_vectors:
            # No incoming communications, return zeros
            return torch.zeros(self.latent_dim)
        
        # Stack communication vectors
        comm_stack = torch.stack(comm_vectors)  # [n_agents, latent_dim]
        n_agents = comm_stack.shape[0]
        
        # Compute query from own state
        query = self.query_proj(own_state.unsqueeze(0))  # [1, hidden_dim]
        
        # Reshape for multi-head attention
        query = query.view(1, self.num_heads, self.head_dim)  # [1, heads, head_dim]
        
        # Compute keys and values from incoming communications
        keys = self.key_proj(comm_stack)  # [n_agents, hidden_dim]
        values = self.value_proj(comm_stack)  # [n_agents, hidden_dim]
        
        # Reshape for multi-head attention
        keys = keys.view(n_agents, self.num_heads, self.head_dim)  # [n_agents, heads, head_dim]
        values = values.view(n_agents, self.num_heads, self.head_dim)  # [n_agents, heads, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1))  # [1, heads, n_agents]
        scores = scores / np.sqrt(self.head_dim)  # Scale
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)  # [1, heads, n_agents]
        
        # Apply attention to values
        attended = torch.matmul(weights, values)  # [1, heads, head_dim]
        
        # Concatenate heads
        attended = attended.view(1, self.hidden_dim)  # [1, hidden_dim]
        
        # Final output projection
        output = self.output_proj(attended).squeeze(0)  # [latent_dim]
        
        # Residual connection and layer norm
        output = self.layer_norm(output + own_state)
        
        return output


@dataclass
class DualMemorySystem:
    """Simple dual memory system for the agent."""
    short_term: List[Discovery]
    long_term: List[Discovery]
    
    def add_discovery(self, discovery: Discovery):
        """Add discovery to memory."""
        self.short_term.append(discovery)
        if discovery.reward > 0.8:  # Simple threshold for long-term
            self.long_term.append(discovery)
    
    def get_relevant_memories(self, context: Dict[str, Any], n: int = 5) -> List[Discovery]:
        """Get relevant memories (simplified)."""
        all_memories = self.short_term + self.long_term
        # Sort by reward and return top n
        sorted_memories = sorted(all_memories, key=lambda d: d.reward, reverse=True)
        return sorted_memories[:n]


class DiscoveryAgent:
    """
    Individual discovery agent with tiered communication capabilities.
    
    This is the fundamental "citizen" of the AI society, capable of:
    - Exploration based on role
    - Neural communication with peers
    - Validation of discoveries (if VALIDATOR role)
    - Strategic decision making about when to publish discoveries
    """
    
    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 policy_network: nn.Module,
                 environment: Any,
                 memory_system: DualMemorySystem,
                 shared_knowledge: SharedKnowledgeBase,
                 obs_dim: int = None):
        """
        Initialize a discovery agent.
        
        Args:
            agent_id: Unique identifier for this agent
            role: Agent's role (EXPLORER, REFINER, VALIDATOR, SPECIALIST)
            policy_network: Neural network policy for action selection
            environment: Environment instance for exploration
            memory_system: Local dual memory system
            shared_knowledge: Reference to shared knowledge base
            obs_dim: Observation dimension (if None, inferred from environment)
        """
        self.agent_id = agent_id
        self.role = role
        self.policy = policy_network
        self.env = environment
        self.memory = memory_system
        self.shared_knowledge = shared_knowledge
        
        # Infer observation dimension if not provided
        if obs_dim is None:
            obs_dim = self.env.observation_space.shape[0]
        
        # Communication components
        self.comm_encoder = CommunicationEncoder(obs_dim, latent_dim=32)
        self.comm_aggregator = CommunicationAggregator(latent_dim=32)
        
        # Communication costs and parameters
        self.tier1_cost = 0.1  # Cost for publishing major discoveries
        self.training_phase = 1  # Start with no communication
        
        # Role-specific parameters
        self.exploration_rate = self._get_exploration_rate()
        
        # Validation parameters (for VALIDATOR role)
        self.empirical_ambiguity_threshold = 0.15  # When to consult LLM
        self.empirical_weight = 0.7
        self.llm_weight = 0.3
        
        # Statistics
        self.stats = {
            'discoveries': 0,
            'tier1_sent': 0,
            'tier2_sent': 0,
            'validations': 0
        }
        
        logger.info(f"Initialized {agent_id} with role {role.value}, "
                   f"exploration rate: {self.exploration_rate}")
    
    def _get_exploration_rate(self) -> float:
        """Get exploration rate based on role."""
        rates = {
            AgentRole.EXPLORER: 0.3,
            AgentRole.REFINER: 0.1,
            AgentRole.VALIDATOR: 0.05,
            AgentRole.SPECIALIST: 0.2
        }
        return rates.get(self.role, 0.15)
    
    def set_training_phase(self, phase: int):
        """Update training phase for curriculum learning."""
        self.training_phase = phase
        logger.info(f"Agent {self.agent_id} entering training phase {phase}")
    
    def explore(self, incoming_messages: List[Message]) -> Tuple[Discovery, Optional[torch.Tensor]]:
        """
        Main exploration loop for the agent.
        
        This method:
        1. Processes incoming messages
        2. Handles validation requests if VALIDATOR
        3. Generates communication vectors (phase >= 2)
        4. Aggregates peer communications
        5. Takes environment steps
        6. Formulates discoveries
        7. Decides whether to publish (phase >= 3)
        
        Args:
            incoming_messages: Messages from the message bus
            
        Returns:
            Tuple of (discovery, communication_vector)
        """
        # Process incoming messages
        tactical_vectors = []
        discovery_pointers = []
        validation_requests = []
        
        for msg in incoming_messages:
            if msg.msg_type == MessageType.TACTICAL_VECTOR and self.training_phase >= 2:
                tactical_vectors.append(torch.FloatTensor(msg.content))
            elif msg.msg_type == MessageType.DISCOVERY_POINTER and self.training_phase >= 3:
                discovery_pointers.append(msg.content)
            elif msg.msg_type == MessageType.VALIDATION_REQUEST and self.role == AgentRole.VALIDATOR:
                validation_requests.append(msg)
        
        # Handle validation requests if validator
        for val_request in validation_requests:
            self._handle_validation_request(val_request)
        
        # Get relevant memories
        context = {
            'role': self.role.value,
            'phase': self.training_phase,
            'recent_discoveries': len(discovery_pointers)
        }
        relevant_memories = self.memory.get_relevant_memories(context)
        
        # Initialize environment episode
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        trajectory = []
        
        # Initialize communication vector
        comm_vector = None
        
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs)
            
            # Generate communication vector if in phase 2+
            if self.training_phase >= 2:
                comm_vector = self.comm_encoder(obs_tensor)
                
                # Aggregate incoming tactical communications
                if tactical_vectors:
                    aggregated_comm = self.comm_aggregator(comm_vector, tactical_vectors)
                    # In practice, this would modify the observation or policy input
                    # For now, we'll add it as a bias to exploration
                    exploration_bias = aggregated_comm.mean().item() * 0.1
                else:
                    exploration_bias = 0.0
            else:
                exploration_bias = 0.0
            
            # Action selection with exploration
            if np.random.random() < self.exploration_rate + exploration_bias:
                action = self.env.action_space.sample()
            else:
                # Use policy network (placeholder - adapt to your policy interface)
                with torch.no_grad():
                    action_probs = self.policy(obs_tensor.unsqueeze(0))
                    action = torch.multinomial(F.softmax(action_probs, dim=-1), 1).item()
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            trajectory.append((obs, action, reward))
            
            total_reward += reward
            obs = next_obs
            done = terminated or truncated
        
        # Create discovery
        discovery = Discovery(
            expression=info.get('expression', f"expr_{np.random.randint(1000)}"),
            reward=total_reward,
            timestamp=datetime.now(),
            discovered_by=self.agent_id,
            metadata={
                'role': self.role.value,
                'phase': self.training_phase,
                'trajectory_length': len(trajectory),
                'exploration_rate': self.exploration_rate
            }
        )
        
        # Add to personal memory
        self.memory.add_discovery(discovery)
        self.stats['discoveries'] += 1
        
        # Decide whether to publish (Tier 1 communication)
        if self.training_phase >= 3 and self._should_publish_discovery(discovery):
            # Apply cost for Tier 1 communication
            discovery.reward -= self.tier1_cost
            
            # Propose to shared knowledge base
            discovery_id = self.shared_knowledge.propose_discovery(self.agent_id, discovery)
            self.stats['tier1_sent'] += 1
            
            logger.info(f"Agent {self.agent_id} proposed discovery {discovery_id} "
                       f"(reward: {discovery.reward:.3f}, cost: {self.tier1_cost})")
        
        # Update Tier 2 statistics if we generated a communication vector
        if comm_vector is not None:
            self.stats['tier2_sent'] += 1
        
        return discovery, comm_vector
    
    def _should_publish_discovery(self, discovery: Discovery) -> bool:
        """
        Decide whether to publish a discovery to Tier 1.
        
        This is a strategic decision considering:
        - The cost of Tier 1 communication
        - The quality of the discovery
        - Whether it's significantly better than known discoveries
        - The agent's role
        
        Args:
            discovery: The discovery to evaluate
            
        Returns:
            Boolean decision to publish or not
        """
        # Don't publish if reward doesn't justify the cost
        if discovery.reward < self.tier1_cost * 2:
            return False
        
        # Validators typically don't propose discoveries
        if self.role == AgentRole.VALIDATOR:
            return False
        
        # Check if significantly better than best known
        best_known = self.shared_knowledge.get_best_discoveries(n=1)
        if best_known:
            best_reward = best_known[0].reward
            # Only publish if at least 10% better
            if discovery.reward <= best_reward * 1.1:
                return False
        
        # Role-specific logic
        if self.role == AgentRole.EXPLORER:
            # Explorers are more likely to publish novel findings
            return discovery.reward > 0.7
        elif self.role == AgentRole.REFINER:
            # Refiners only publish significant improvements
            return discovery.reward > 0.85
        elif self.role == AgentRole.SPECIALIST:
            # Specialists publish domain-specific breakthroughs
            return discovery.reward > 0.8
        
        return True
    
    def _handle_validation_request(self, msg: Message):
        """
        Handle validation request with Augmented Consensus using LLM feedback.
        
        This implements the recommended validation pattern:
        1. Empirical validation (primary evidence)
        2. LLM consultation for ambiguous cases
        3. Weighted decision making
        
        Args:
            msg: Validation request message
        """
        request_content = msg.content
        discovery_id = request_content['discovery_id']
        discovery = request_content['discovery']
        
        logger.info(f"Validator {self.agent_id} evaluating discovery {discovery_id}")
        
        # Step 1: Empirical validation (most important)
        empirical_score = self._empirical_validation(discovery)
        
        # Step 2: Check if empirical score is ambiguous
        llm_similarity = 0.5  # Default neutral
        evidence = {
            'empirical_score': empirical_score,
            'method': 'empirical_only'
        }
        
        if abs(empirical_score - 0.5) < self.empirical_ambiguity_threshold:
            # Empirical score is ambiguous, consult LLM
            logger.info(f"Empirical score {empirical_score:.3f} is ambiguous, consulting LLM")
            
            llm_similarity = self._consult_llm(discovery)
            evidence['llm_consulted'] = True
            evidence['llm_similarity'] = llm_similarity
            evidence['method'] = 'empirical_with_llm'
        
        # Step 3: Make weighted decision
        if evidence.get('llm_consulted', False):
            final_score = (self.empirical_weight * empirical_score + 
                          self.llm_weight * llm_similarity)
        else:
            final_score = empirical_score
        
        # Vote based on final score
        approve = final_score > 0.6  # Approval threshold
        evidence['final_score'] = final_score
        evidence['decision_threshold'] = 0.6
        
        # Submit vote
        self.shared_knowledge.vote_on_discovery(
            self.agent_id,
            discovery_id,
            approve,
            evidence
        )
        
        self.stats['validations'] += 1
        logger.info(f"Validator {self.agent_id} voted {'APPROVE' if approve else 'REJECT'} "
                   f"on {discovery_id} (score: {final_score:.3f})")
    
    def _empirical_validation(self, discovery: Discovery) -> float:
        """
        Perform empirical validation of a discovery.
        
        In practice, this would re-evaluate the expression on held-out data.
        For now, we simulate with the discovery's reward plus noise.
        
        Args:
            discovery: Discovery to validate
            
        Returns:
            Empirical score between 0 and 1
        """
        # Simulate empirical validation
        # In practice: re-evaluate expression on validation dataset
        base_score = discovery.reward
        
        # Add noise to simulate evaluation uncertainty
        noise = np.random.normal(0, 0.1)
        empirical_score = np.clip(base_score + noise, 0, 1)
        
        # Penalize overly complex expressions (Occam's Razor)
        expr_length = len(str(discovery.expression))
        complexity_penalty = min(expr_length / 100, 0.2)
        
        return max(0, empirical_score - complexity_penalty)
    
    def _consult_llm(self, discovery: Discovery) -> float:
        """
        Consult LLM for peer review of ambiguous discovery.
        
        This is a placeholder for actual LLM integration.
        In practice, this would:
        1. Formulate a prompt with the discovery details
        2. Query a local LLM for feedback
        3. Compare LLM suggestion with the discovery
        
        Args:
            discovery: Discovery to review
            
        Returns:
            Similarity score between discovery and LLM suggestion (0-1)
        """
        # Placeholder LLM consultation
        prompt = f"""
        As an expert in symbolic regression and scientific discovery, please evaluate:
        
        Expression: {discovery.expression}
        Reported Performance: {discovery.reward:.3f}
        Discovered by: {discovery.discovered_by}
        
        Questions:
        1. Is this expression mathematically sound?
        2. Does the performance seem reasonable?
        3. Can you suggest a similar or improved expression?
        
        Please provide your assessment.
        """
        
        # Simulate LLM response
        # In practice: llm_response = llm_client.query(prompt)
        
        # For now, return a similarity score based on discovery quality
        if discovery.reward > 0.8:
            similarity = 0.8 + np.random.uniform(-0.1, 0.1)
        elif discovery.reward > 0.6:
            similarity = 0.6 + np.random.uniform(-0.2, 0.2)
        else:
            similarity = 0.4 + np.random.uniform(-0.2, 0.2)
        
        return np.clip(similarity, 0, 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'phase': self.training_phase,
            **self.stats
        }


# Example environment wrapper for testing
class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, obs_dim: int = 10):
        self.observation_space = type('obj', (object,), {'shape': (obs_dim,)})
        self.action_space = type('obj', (object,), {'n': 4, 'sample': lambda: np.random.randint(4)})
    
    def reset(self):
        return np.random.randn(self.observation_space.shape[0]), {}
    
    def step(self, action):
        obs = np.random.randn(self.observation_space.shape[0])
        reward = np.random.uniform(0, 1)
        done = np.random.random() < 0.1
        return obs, reward, done, False, {'expression': f"x^{action}"}


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create mock components
    from .knowledge import SharedKnowledgeBase, MessageBus
    
    message_bus = MessageBus()
    shared_knowledge = SharedKnowledgeBase(validation_threshold=2, message_bus=message_bus)
    
    # Create mock policy network
    class MockPolicy(nn.Module):
        def __init__(self, obs_dim, action_dim):
            super().__init__()
            self.net = nn.Linear(obs_dim, action_dim)
        
        def forward(self, x):
            return self.net(x)
    
    # Create environment and memory
    env = MockEnvironment(obs_dim=10)
    memory = DualMemorySystem(short_term=[], long_term=[])
    policy = MockPolicy(10, 4)
    
    # Create agents with different roles
    explorer = DiscoveryAgent(
        agent_id="explorer_001",
        role=AgentRole.EXPLORER,
        policy_network=policy,
        environment=env,
        memory_system=memory,
        shared_knowledge=shared_knowledge
    )
    
    validator = DiscoveryAgent(
        agent_id="validator_001",
        role=AgentRole.VALIDATOR,
        policy_network=policy,
        environment=env,
        memory_system=DualMemorySystem(short_term=[], long_term=[]),
        shared_knowledge=shared_knowledge
    )
    
    # Simulate exploration
    print("\n--- Phase 1: No Communication ---")
    explorer.set_training_phase(1)
    discovery1, comm1 = explorer.explore([])
    print(f"Discovery: {discovery1.expression}, Reward: {discovery1.reward:.3f}")
    print(f"Communication vector: {comm1}")
    
    print("\n--- Phase 2: Tactical Communication ---")
    explorer.set_training_phase(2)
    discovery2, comm2 = explorer.explore([])
    print(f"Discovery: {discovery2.expression}, Reward: {discovery2.reward:.3f}")
    print(f"Communication vector shape: {comm2.shape if comm2 is not None else None}")
    
    print("\n--- Phase 3: Full Communication ---")
    explorer.set_training_phase(3)
    validator.set_training_phase(3)
    
    # Explorer makes discovery
    discovery3, comm3 = explorer.explore([])
    
    # Get validation requests for validator
    val_messages = message_bus.get_messages("validator_001")
    print(f"\nValidator received {len(val_messages)} messages")
    
    # Validator processes messages
    if val_messages:
        validator.explore(val_messages)
    
    # Check stats
    print("\n--- Agent Statistics ---")
    print(f"Explorer: {explorer.get_stats()}")
    print(f"Validator: {validator.get_stats()}")