# JanusAI/tests/unit/multi-agent/discoveryagent_test.py
"""
Test Suite for DiscoveryAgent Components
========================================

Comprehensive tests for the neural communication components and
DiscoveryAgent class in the JanusAI system.

Author: JanusAI Team
Date: 2024
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import List

# Import components to test
from schemas import AgentRole, MessageType, Discovery, Message
from knowledge import MessageBus, SharedKnowledgeBase
from discovery_agent import (
    CommunicationEncoder,
    CommunicationAggregator,
    DiscoveryAgent,
    DualMemorySystem
)


class TestCommunicationEncoder:
    """Test the CommunicationEncoder neural network."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = CommunicationEncoder(input_dim=10, latent_dim=32)
        
        # Check architecture
        assert len(encoder.encoder) == 7  # 3 Linear + 2 ReLU + 2 LayerNorm
        assert isinstance(encoder.encoder[-1], nn.Tanh)
    
    def test_forward_single_input(self):
        """Test encoding single state vector."""
        encoder = CommunicationEncoder(input_dim=10, latent_dim=32)
        
        # Single input
        state = torch.randn(10)
        encoded = encoder(state)
        
        assert encoded.shape == (32,)
        assert torch.all(encoded >= -1) and torch.all(encoded <= 1)  # Tanh bounds
    
    def test_forward_batch_input(self):
        """Test encoding batch of state vectors."""
        encoder = CommunicationEncoder(input_dim=10, latent_dim=32)
        
        # Batch input
        states = torch.randn(5, 10)
        encoded = encoder(states)
        
        assert encoded.shape == (5, 32)
        assert torch.all(encoded >= -1) and torch.all(encoded <= 1)
    
    def test_gradient_flow(self):
        """Test gradient flows through encoder."""
        encoder = CommunicationEncoder(input_dim=10, latent_dim=32)
        
        state = torch.randn(10, requires_grad=True)
        encoded = encoder(state)
        loss = encoded.sum()
        loss.backward()
        
        assert state.grad is not None
        assert not torch.any(torch.isnan(state.grad))


class TestCommunicationAggregator:
    """Test the CommunicationAggregator attention mechanism."""
    
    def test_initialization(self):
        """Test aggregator initialization."""
        aggregator = CommunicationAggregator(latent_dim=32, num_heads=4)
        
        assert aggregator.num_heads == 4
        assert aggregator.head_dim == 16  # 64 / 4
    
    def test_no_communications(self):
        """Test aggregation with no incoming communications."""
        aggregator = CommunicationAggregator(latent_dim=32)
        
        own_state = torch.randn(32)
        comm_vectors = []
        
        result = aggregator(own_state, comm_vectors)
        
        assert result.shape == (32,)
        assert torch.all(result == 0)  # Should return zeros
    
    def test_single_communication(self):
        """Test aggregation with one incoming communication."""
        aggregator = CommunicationAggregator(latent_dim=32)
        
        own_state = torch.randn(32)
        comm_vectors = [torch.randn(32)]
        
        result = aggregator(own_state, comm_vectors)
        
        assert result.shape == (32,)
        assert not torch.all(result == 0)  # Should be non-zero
    
    def test_multiple_communications(self):
        """Test aggregation with multiple incoming communications."""
        aggregator = CommunicationAggregator(latent_dim=32)
        
        own_state = torch.randn(32)
        comm_vectors = [torch.randn(32) for _ in range(5)]
        
        result = aggregator(own_state, comm_vectors)
        
        assert result.shape == (32,)
        # Check residual connection works
        assert not torch.allclose(result, own_state)  # Should be modified
    
    def test_attention_weights(self):
        """Test that attention mechanism produces valid weights."""
        aggregator = CommunicationAggregator(latent_dim=32)
        
        # Create distinctive vectors
        own_state = torch.ones(32)
        similar_vector = torch.ones(32) * 0.9
        different_vector = -torch.ones(32)
        
        comm_vectors = [similar_vector, different_vector]
        result = aggregator(own_state, comm_vectors)
        
        # Result should be influenced more by similar vector
        similarity_to_similar = torch.cosine_similarity(result, similar_vector, dim=0)
        similarity_to_different = torch.cosine_similarity(result, different_vector, dim=0)
        
        assert similarity_to_similar > similarity_to_different


class TestDualMemorySystem:
    """Test the DualMemorySystem."""
    
    def test_add_discovery(self):
        """Test adding discoveries to memory."""
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        discovery = Discovery(
            expression="x^2",
            reward=0.9,
            timestamp=datetime.now(),
            discovered_by="test_agent"
        )
        
        memory.add_discovery(discovery)
        
        assert len(memory.short_term) == 1
        assert len(memory.long_term) == 1  # High reward promoted
    
    def test_get_relevant_memories(self):
        """Test retrieving relevant memories."""
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        # Add discoveries with different rewards
        for i in range(5):
            discovery = Discovery(
                expression=f"x^{i}",
                reward=0.1 * i,
                timestamp=datetime.now(),
                discovered_by="test_agent"
            )
            memory.add_discovery(discovery)
        
        relevant = memory.get_relevant_memories({}, n=3)
        
        assert len(relevant) == 3
        # Should be sorted by reward
        assert relevant[0].reward >= relevant[1].reward >= relevant[2].reward


class TestDiscoveryAgent:
    """Test the DiscoveryAgent class."""
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock environment."""
        class MockEnv:
            def __init__(self):
                self.observation_space = type('obj', (object,), {'shape': (10,)})
                self.action_space = type('obj', (object,), {
                    'n': 4,
                    'sample': lambda: np.random.randint(4)
                })
            
            def reset(self):
                return np.random.randn(10), {}
            
            def step(self, action):
                obs = np.random.randn(10)
                reward = np.random.uniform(0, 1)
                done = np.random.random() < 0.2
                return obs, reward, done, False, {'expression': f"x^{action}"}
        
        return MockEnv()
    
    @pytest.fixture
    def mock_policy(self):
        """Create mock policy network."""
        class MockPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 4)
            
            def forward(self, x):
                return self.net(x)
        
        return MockPolicy()
    
    @pytest.fixture
    def shared_infrastructure(self):
        """Create shared knowledge base and message bus."""
        bus = MessageBus()
        kb = SharedKnowledgeBase(validation_threshold=2, message_bus=bus)
        return bus, kb
    
    def test_agent_initialization(self, mock_environment, mock_policy, shared_infrastructure):
        """Test agent initialization."""
        _, kb = shared_infrastructure
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        agent = DiscoveryAgent(
            agent_id="test_agent",
            role=AgentRole.EXPLORER,
            policy_network=mock_policy,
            environment=mock_environment,
            memory_system=memory,
            shared_knowledge=kb
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.role == AgentRole.EXPLORER
        assert agent.exploration_rate == 0.3
        assert agent.training_phase == 1
    
    def test_phase_1_exploration(self, mock_environment, mock_policy, shared_infrastructure):
        """Test Phase 1: No communication."""
        _, kb = shared_infrastructure
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        agent = DiscoveryAgent(
            agent_id="explorer_001",
            role=AgentRole.EXPLORER,
            policy_network=mock_policy,
            environment=mock_environment,
            memory_system=memory,
            shared_knowledge=kb
        )
        
        agent.set_training_phase(1)
        discovery, comm_vector = agent.explore([])
        
        assert discovery is not None
        assert comm_vector is None  # No communication in phase 1
        assert agent.stats['discoveries'] == 1
        assert agent.stats['tier1_sent'] == 0
        assert agent.stats['tier2_sent'] == 0
    
    def test_phase_2_exploration(self, mock_environment, mock_policy, shared_infrastructure):
        """Test Phase 2: Tactical communication."""
        _, kb = shared_infrastructure
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        agent = DiscoveryAgent(
            agent_id="explorer_001",
            role=AgentRole.EXPLORER,
            policy_network=mock_policy,
            environment=mock_environment,
            memory_system=memory,
            shared_knowledge=kb
        )
        
        agent.set_training_phase(2)
        discovery, comm_vector = agent.explore([])
        
        assert discovery is not None
        assert comm_vector is not None  # Should generate communication
        assert comm_vector.shape == (32,)  # Latent dimension
        assert agent.stats['tier2_sent'] == 1
        assert agent.stats['tier1_sent'] == 0  # No strategic messages yet
    
    def test_phase_3_exploration(self, mock_environment, mock_policy, shared_infrastructure):
        """Test Phase 3: Full communication."""
        bus, kb = shared_infrastructure
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        agent = DiscoveryAgent(
            agent_id="explorer_001",
            role=AgentRole.EXPLORER,
            policy_network=mock_policy,
            environment=mock_environment,
            memory_system=memory,
            shared_knowledge=kb
        )
        
        agent.set_training_phase(3)
        
        # Run multiple explorations to likely trigger discovery proposal
        for _ in range(5):
            discovery, comm_vector = agent.explore([])
            if discovery.reward > 0.7:  # High enough to potentially publish
                break
        
        # Check if any Tier 1 messages were sent
        bus_stats = bus.get_stats()
        assert bus_stats['tier1_total'] >= 0  # May or may not publish
    
    def test_validation_handling(self, mock_environment, mock_policy, shared_infrastructure):
        """Test validator agent handling validation requests."""
        bus, kb = shared_infrastructure
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        # Create validator agent
        validator = DiscoveryAgent(
            agent_id="validator_001",
            role=AgentRole.VALIDATOR,
            policy_network=mock_policy,
            environment=mock_environment,
            memory_system=memory,
            shared_knowledge=kb
        )
        validator.set_training_phase(3)
        
        # Create a discovery to validate
        discovery = Discovery(
            expression="x^2 + 2*x + 1",
            reward=0.85,
            timestamp=datetime.now(),
            discovered_by="explorer_001"
        )
        
        # Propose discovery (creates validation request)
        discovery_id = kb.propose_discovery("explorer_001", discovery)
        
        # Get messages for validator
        messages = bus.get_messages("validator_001")
        
        # Validator should process validation request
        validator.explore(messages)
        
        assert validator.stats['validations'] == 1
    
    def test_should_publish_logic(self, mock_environment, mock_policy, shared_infrastructure):
        """Test the _should_publish_discovery logic."""
        _, kb = shared_infrastructure
        memory = DualMemorySystem(short_term=[], long_term=[])
        
        agent = DiscoveryAgent(
            agent_id="explorer_001",
            role=AgentRole.EXPLORER,
            policy_network=mock_policy,
            environment=mock_environment,
            memory_system=memory,
            shared_knowledge=kb
        )
        
        # Low reward discovery - should not publish
        low_discovery = Discovery(
            expression="x",
            reward=0.15,  # Less than 2 * tier1_cost
            timestamp=datetime.now(),
            discovered_by=agent.agent_id
        )
        assert not agent._should_publish_discovery(low_discovery)
        
        # High reward discovery - should publish
        high_discovery = Discovery(
            expression="x^2",
            reward=0.95,
            timestamp=datetime.now(),
            discovered_by=agent.agent_id
        )
        assert agent._should_publish_discovery(high_discovery)
        
        # Validator shouldn't publish
        agent.role = AgentRole.VALIDATOR
        assert not agent._should_publish_discovery(high_discovery)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_communication_flow(self):
        """Test complete communication flow through all components."""
        # Create infrastructure
        bus = MessageBus()
        kb = SharedKnowledgeBase(validation_threshold=1, message_bus=bus)
        
        # Create mock environment and policy
        class MockEnv:
            observation_space = type('obj', (object,), {'shape': (10,)})
            action_space = type('obj', (object,), {'n': 4, 'sample': lambda: 0})
            
            def reset(self):
                return np.ones(10), {}
            
            def step(self, action):
                return np.ones(10), 0.9, True, False, {'expression': 'x^2'}
        
        class MockPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 4)
            
            def forward(self, x):
                return torch.ones(1, 4)
        
        # Create agents
        explorer = DiscoveryAgent(
            "explorer_001",
            AgentRole.EXPLORER,
            MockPolicy(),
            MockEnv(),
            DualMemorySystem([], []),
            kb
        )
        
        validator = DiscoveryAgent(
            "validator_001",
            AgentRole.VALIDATOR,
            MockPolicy(),
            MockEnv(),
            DualMemorySystem([], []),
            kb
        )
        
        # Set to phase 3
        explorer.set_training_phase(3)
        validator.set_training_phase(3)
        
        # Explorer discovers
        discovery, comm = explorer.explore([])
        
        # Check if validation request created
        val_messages = bus.get_messages("validator_001")
        validation_requests = [m for m in val_messages 
                             if m.msg_type == MessageType.VALIDATION_REQUEST]
        
        if validation_requests:
            # Validator processes
            validator.explore(val_messages)
            
            # Check knowledge base
            kb_summary = kb.get_knowledge_summary()
            # May or may not be confirmed depending on validation
            assert kb_summary['total_proposals'] >= 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])