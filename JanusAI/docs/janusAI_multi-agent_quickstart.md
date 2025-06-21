# JanusAI Multi-Agent System - Quick Start Guide

## 🚀 5-Minute Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/janusai-multiagent.git
cd janusai-multiagent

# Install dependencies
pip install -r requirements.txt
```

### Basic Example

```python
# quick_start.py
from datetime import datetime
from janus_multiagent import (
    MessageBus, SharedKnowledgeBase, Discovery,
    AgentRole, MessageType, Message
)

# 1. Initialize the system
message_bus = MessageBus()
knowledge_base = SharedKnowledgeBase(
    validation_threshold=2,  # Need 2 validators
    message_bus=message_bus
)

# 2. Create an Explorer agent discovery
discovery = Discovery(
    expression="x^2 - 4*x + 4",
    reward=0.95,
    timestamp=datetime.now(),
    discovered_by="explorer_001"
)

# 3. Propose the discovery
discovery_id = knowledge_base.propose_discovery("explorer_001", discovery)
print(f"✅ Discovery proposed: {discovery_id}")

# 4. Validators vote on it
knowledge_base.vote_on_discovery(
    "validator_001", 
    discovery_id, 
    approve=True,
    evidence={"error": 0.001}
)

knowledge_base.vote_on_discovery(
    "validator_002",
    discovery_id,
    approve=True,
    evidence={"verified": True}
)

# 5. Check the results
best = knowledge_base.get_best_discoveries(n=1)
if best:
    print(f"🏆 Best discovery: {best[0].expression} (reward: {best[0].reward})")
```

## 🏗️ Building Your First Agent

### Simple Explorer Agent

```python
import numpy as np
from typing import List

class SimpleExplorerAgent:
    def __init__(self, agent_id: str, message_bus: MessageBus, knowledge_base: SharedKnowledgeBase):
        self.agent_id = agent_id
        self.role = AgentRole.EXPLORER
        self.message_bus = message_bus
        self.knowledge_base = knowledge_base
        
    def explore(self):
        """Perform exploration and propose discoveries."""
        # Simulate finding a mathematical expression
        expression = f"x^{np.random.randint(1, 4)} + {np.random.randint(-5, 5)}"
        reward = np.random.uniform(0.5, 0.99)
        
        discovery = Discovery(
            expression=expression,
            reward=reward,
            timestamp=datetime.now(),
            discovered_by=self.agent_id
        )
        
        # Propose the discovery
        discovery_id = self.knowledge_base.propose_discovery(self.agent_id, discovery)
        print(f"🔍 {self.agent_id} discovered: {expression} (reward: {reward:.3f})")
        
        return discovery_id
    
    def communicate_tactical(self, state_vector: List[float]):
        """Share tactical information with other agents."""
        msg = Message(
            msg_type=MessageType.TACTICAL_VECTOR,
            sender_id=self.agent_id,
            timestamp=datetime.now(),
            content=state_vector,
            ttl=30.0  # 30 seconds
        )
        self.message_bus.publish(msg)

# Usage
explorer = SimpleExplorerAgent("explorer_001", message_bus, knowledge_base)
discovery_id = explorer.explore()
explorer.communicate_tactical([0.1, 0.5, -0.3, 0.8])
```

### Simple Validator Agent

```python
class SimpleValidatorAgent:
    def __init__(self, agent_id: str, message_bus: MessageBus, knowledge_base: SharedKnowledgeBase):
        self.agent_id = agent_id
        self.role = AgentRole.VALIDATOR
        self.message_bus = message_bus
        self.knowledge_base = knowledge_base
    
    def process_messages(self):
        """Check for validation requests and vote."""
        messages = self.message_bus.get_messages(self.agent_id, max_messages=10)
        
        for msg in messages:
            if msg.msg_type == MessageType.VALIDATION_REQUEST:
                self._validate_discovery(msg.content)
    
    def _validate_discovery(self, request_content):
        """Validate a proposed discovery."""
        discovery_id = request_content['discovery_id']
        discovery = request_content['discovery']
        
        # Simple validation logic
        is_valid = discovery.reward > 0.7
        evidence = {
            "checked": True,
            "method": "threshold_validation",
            "threshold": 0.7
        }
        
        self.knowledge_base.vote_on_discovery(
            self.agent_id,
            discovery_id,
            approve=is_valid,
            evidence=evidence
        )
        
        print(f"✓ {self.agent_id} voted {'YES' if is_valid else 'NO'} on {discovery_id}")

# Usage
validator = SimpleValidatorAgent("validator_001", message_bus, knowledge_base)
validator.process_messages()
```

## 📊 Monitoring Your System

### Real-time Dashboard

```python
def print_dashboard(message_bus, knowledge_base):
    """Display system status dashboard."""
    print("\n" + "="*50)
    print("🎯 JANUSAI SYSTEM DASHBOARD")
    print("="*50)
    
    # Message Bus Status
    bus_stats = message_bus.get_stats()
    print(f"\n📬 Message Bus:")
    print(f"  Strategic Messages: {bus_stats['tier1_total']}")
    print(f"  Tactical Messages: {bus_stats['tier2_total']}")
    print(f"  Queue Status: {bus_stats['tier2_queue_size']}/{message_bus.max_queue_size}")
    
    # Knowledge Base Status
    kb_summary = knowledge_base.get_knowledge_summary()
    print(f"\n🧠 Knowledge Base:")
    print(f"  Discoveries: {kb_summary['total_discoveries']}")
    print(f"  Pending: {kb_summary['pending_validations']}")
    print(f"  Success Rate: {kb_summary['successful_validations']}/{kb_summary['total_proposals']}")
    
    # Top Discoveries
    top_discoveries = knowledge_base.get_best_discoveries(n=3)
    if top_discoveries:
        print(f"\n🏆 Top Discoveries:")
        for i, disc in enumerate(top_discoveries, 1):
            print(f"  {i}. {disc.expression} (reward: {disc.reward:.3f})")
    
    # Agent Performance
    agent_stats = knowledge_base.get_agent_statistics()
    if agent_stats:
        print(f"\n👥 Agent Performance:")
        for agent_id, stats in list(agent_stats.items())[:3]:
            print(f"  {agent_id}: {stats['confirmed_discoveries']} discoveries, "
                  f"avg reward: {stats['avg_reward']:.3f}")

# Run dashboard
print_dashboard(message_bus, knowledge_base)
```

## 🎮 Complete Working Example

```python
# complete_example.py
import time
import random
from datetime import datetime

def run_multi_agent_simulation(duration_seconds=30):
    """Run a complete multi-agent simulation."""
    
    # Initialize system
    message_bus = MessageBus(max_queue_size=1000)
    knowledge_base = SharedKnowledgeBase(validation_threshold=2, message_bus=message_bus)
    
    # Create agents
    explorers = [
        SimpleExplorerAgent(f"explorer_{i:03d}", message_bus, knowledge_base)
        for i in range(3)
    ]
    
    validators = [
        SimpleValidatorAgent(f"validator_{i:03d}", message_bus, knowledge_base)
        for i in range(3)
    ]
    
    print(f"🚀 Starting {duration_seconds}s simulation with {len(explorers)} explorers and {len(validators)} validators")
    
    start_time = time.time()
    cycle = 0
    
    while time.time() - start_time < duration_seconds:
        cycle += 1
        print(f"\n--- Cycle {cycle} ---")
        
        # Explorers make discoveries
        for explorer in explorers:
            if random.random() < 0.3:  # 30% chance to discover
                explorer.explore()
        
        # Small delay to simulate processing
        time.sleep(0.5)
        
        # Validators process messages
        for validator in validators:
            validator.process_messages()
        
        # Show dashboard every 5 cycles
        if cycle % 5 == 0:
            print_dashboard(message_bus, knowledge_base)
        
        time.sleep(1)
    
    print("\n🏁 Simulation Complete!")
    print_dashboard(message_bus, knowledge_base)

# Run the simulation
if __name__ == "__main__":
    run_multi_agent_simulation(duration_seconds=20)
```

## 🔍 Common Patterns

### Pattern 1: Coordinated Search

```python
# Multiple explorers coordinate to avoid redundant search
def coordinated_exploration(explorers: List[SimpleExplorerAgent]):
    """Explorers share search regions via tactical messages."""
    
    for i, explorer in enumerate(explorers):
        # Each explorer broadcasts its search region
        search_region = [i * 0.1, i * 0.2, random.random(), random.random()]
        explorer.communicate_tactical(search_region)
    
    # Each explorer checks messages to avoid overlap
    for explorer in explorers:
        messages = explorer.message_bus.get_messages(explorer.agent_id)
        tactical_messages = [m for m in messages if m.msg_type == MessageType.TACTICAL_VECTOR]
        
        if tactical_messages:
            print(f"{explorer.agent_id} received {len(tactical_messages)} coordination messages")
```

### Pattern 2: Discovery Refinement Chain

```python
# Agents build upon each other's discoveries
def refinement_chain(knowledge_base, initial_discovery_id):
    """Create a chain of refinements."""
    
    # Get the original discovery
    discoveries = knowledge_base.get_best_discoveries(n=10)
    original = next((d for d in discoveries if str(d.timestamp.timestamp()) in initial_discovery_id), None)
    
    if original:
        # Create refined version
        refined = Discovery(
            expression=f"simplified({original.expression})",
            reward=original.reward * 1.1,  # 10% improvement
            timestamp=datetime.now(),
            discovered_by="refiner_001",
            metadata={"refined_from": initial_discovery_id}
        )
        
        # Propose refinement
        refined_id = knowledge_base.propose_discovery("refiner_001", refined)
        print(f"🔧 Refined {original.expression} -> {refined.expression}")
        
        return refined_id
```

## 📚 Next Steps

1. **Extend Agent Behavior**: Create specialized agents with domain knowledge
2. **Add Neural Networks**: Integrate with PyTorch/TensorFlow for learned behaviors
3. **Scale Up**: Use Ray or multiprocessing for parallel agent execution
4. **Persistence**: Add database backend for long-term knowledge storage
5. **Visualization**: Create real-time visualization of agent interactions

## 🆘 Getting Help

- 📖 [Full Documentation](./DOCUMENTATION.md)
- 💬 [Community Discord](https://discord.gg/janusai)
- 🐛 [Issue Tracker](https://github.com/your-org/janusai/issues)
- 📧 [Email Support](mailto:support@janusai.org)

Happy Discovering! 🚀