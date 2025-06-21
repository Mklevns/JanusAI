# JanusAI Multi-Agent System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Communication Protocol](#communication-protocol)
5. [Use Case Examples](#use-case-examples)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## System Overview

The JanusAI Multi-Agent System is a sophisticated framework for collaborative scientific discovery through autonomous agents. It implements a **tiered communication architecture** that enables both strategic knowledge sharing and tactical real-time coordination among specialized agents.

### Key Features
- **Dual-tier communication**: Strategic (Tier 1) and Tactical (Tier 2) messaging
- **Augmented Consensus**: Multi-agent validation for quality assurance
- **Role specialization**: Explorer, Refiner, Validator, and Specialist agents
- **Version control**: Semantic versioning for discovered knowledge
- **Rate limiting**: Prevents system abuse and ensures fair resource usage
- **Thread-safe**: Designed for concurrent multi-agent operations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Explorer   │  │  Refiner    │  │ Validator   │ ...     │
│  │   Agent     │  │   Agent     │  │   Agent     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                 │                 │                │
│         └─────────────────┴─────────────────┘                │
│                           │                                   │
│                    ┌──────▼──────┐                          │
│                    │ MessageBus  │                          │
│                    │             │                          │
│                    │ ┌─────────┐ │                          │
│                    │ │ Tier 1  │ │ Priority Queue          │
│                    │ └─────────┘ │                          │
│                    │ ┌─────────┐ │                          │
│                    │ │ Tier 2  │ │ Bounded Queue           │
│                    │ └─────────┘ │                          │
│                    └──────┬──────┘                          │
│                           │                                  │
│                    ┌──────▼──────────┐                      │
│                    │ SharedKnowledge │                      │
│                    │      Base       │                      │
│                    │                 │                      │
│                    │ • Discoveries   │                      │
│                    │ • Consensus     │                      │
│                    │ • Versioning    │                      │
│                    └─────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Structures

#### AgentRole
Defines the specialized roles agents can assume:
- **EXPLORER**: High exploration rate, broad search strategies
- **REFINER**: Improves existing discoveries
- **VALIDATOR**: Verifies and validates proposed discoveries
- **SPECIALIST**: Domain-specific or targeted search

#### MessageType
Categorizes messages in the communication protocol:
- **DISCOVERY_POINTER** (Tier 1): Reference to validated discovery
- **VALIDATION_REQUEST** (Tier 1): Request for discovery validation
- **VALIDATION_VOTE** (Tier 1): Validation vote submission
- **TACTICAL_VECTOR** (Tier 2): Real-time coordination data

#### Discovery
Represents a scientific discovery with:
- Expression and reward value
- Version tracking
- Validation scores and votes
- Contributing agents
- Metadata and provenance

#### Message
Communication packet containing:
- Message type and sender ID
- Content payload
- Priority level
- Time-to-live (TTL)
- Timestamp and metadata

### 2. MessageBus

The high-throughput communication backbone with:
- **Dual-queue system**: Priority queue for Tier 1, bounded queue for Tier 2
- **Rate limiting**: Token bucket algorithm per agent
- **Automatic expiration**: TTL-based message filtering
- **Statistics tracking**: Message counts and queue states

### 3. SharedKnowledgeBase

The persistent knowledge store featuring:
- **Augmented Consensus**: Multi-agent validation protocol
- **Version control**: Semantic versioning for discoveries
- **Agent tracking**: Contribution statistics per agent
- **Thread safety**: Concurrent access protection

## Communication Protocol

### Tier 1: Strategic Communication
Used for high-value, persistent information:
- Discovery announcements
- Validation requests and votes
- System-wide notifications

**Characteristics:**
- High priority
- No size limits
- Longer TTL (1 hour default)
- Auditable and traceable

### Tier 2: Tactical Communication
Used for real-time coordination:
- Latent state vectors
- Coordination signals
- Ephemeral information

**Characteristics:**
- Lower priority
- Size-limited queue
- Short TTL (60 seconds default)
- High throughput

## Use Case Examples

### Example 1: Basic Discovery Workflow

```python
import logging
from datetime import datetime
from your_module import (MessageBus, SharedKnowledgeBase, Discovery, 
                        AgentRole, MessageType)

# Setup
logging.basicConfig(level=logging.INFO)
message_bus = MessageBus(max_queue_size=1000)
knowledge_base = SharedKnowledgeBase(validation_threshold=2, message_bus=message_bus)

# Explorer discovers something
discovery = Discovery(
    expression="E = mc^2",
    reward=0.98,
    timestamp=datetime.now(),
    discovered_by="explorer_1",
    metadata={"domain": "physics", "complexity": 3}
)

# Propose for validation
discovery_id = knowledge_base.propose_discovery("explorer_1", discovery)
print(f"Discovery proposed: {discovery_id}")

# Validators receive and process validation request
validator_messages = message_bus.get_messages("validator_1", max_messages=10)
for msg in validator_messages:
    if msg.msg_type == MessageType.VALIDATION_REQUEST:
        # Validator evaluates the discovery
        discovery_data = msg.content['discovery']
        # ... validation logic ...
        
        # Submit vote with evidence
        knowledge_base.vote_on_discovery(
            "validator_1", 
            msg.content['discovery_id'],
            approve=True,
            evidence={"verification": "reproduced", "error": 0.001}
        )

# Second validator approves
knowledge_base.vote_on_discovery(
    "validator_2",
    discovery_id, 
    approve=True,
    evidence={"method": "analytical", "confidence": 0.99}
)

# Discovery is now confirmed and available
best_discoveries = knowledge_base.get_best_discoveries(n=5)
print(f"Top discovery: {best_discoveries[0].expression}")
```

### Example 2: Multi-Agent Coordination with Tactical Communication

```python
# Agents sharing tactical information
import numpy as np

# Explorer broadcasts its current search region
tactical_vector = np.array([0.5, -0.3, 0.8, 0.1])  # Latent representation
tactical_msg = Message(
    msg_type=MessageType.TACTICAL_VECTOR,
    sender_id="explorer_1",
    timestamp=datetime.now(),
    content=tactical_vector.tolist(),
    ttl=30.0  # 30 second relevance
)
message_bus.publish(tactical_msg)

# Refiner receives and adjusts its strategy
refiner_messages = message_bus.get_messages("refiner_1", max_messages=20)
tactical_vectors = []
for msg in refiner_messages:
    if msg.msg_type == MessageType.TACTICAL_VECTOR:
        tactical_vectors.append(np.array(msg.content))

if tactical_vectors:
    # Aggregate tactical information
    avg_vector = np.mean(tactical_vectors, axis=0)
    print(f"Refiner adjusting strategy based on {len(tactical_vectors)} tactical inputs")
```

### Example 3: Version Evolution

```python
# Initial discovery
discovery_v1 = Discovery(
    expression="x^2 + 2*x + 1",
    reward=0.85,
    timestamp=datetime.now(),
    discovered_by="explorer_1"
)
discovery_id_v1 = knowledge_base.propose_discovery("explorer_1", discovery_v1)

# Get validators to approve
knowledge_base.vote_on_discovery("validator_1", discovery_id_v1, True)
knowledge_base.vote_on_discovery("validator_2", discovery_id_v1, True)

# Refiner finds improved version
discovery_v2 = Discovery(
    expression="(x + 1)^2",  # Simplified form
    reward=0.92,  # Better reward
    timestamp=datetime.now(),
    discovered_by="refiner_1",
    metadata={"refined_from": discovery_id_v1}
)
discovery_id_v2 = knowledge_base.propose_discovery("refiner_1", discovery_v2)

# Validate improved version
knowledge_base.vote_on_discovery("validator_1", discovery_id_v2, True)
knowledge_base.vote_on_discovery("validator_3", discovery_id_v2, True)

# Check version history
best = knowledge_base.get_best_discoveries(n=1)
print(f"Current best version: {best[0].version} with reward {best[0].reward}")
```

### Example 4: Rate Limiting and System Protection

```python
# Simulate agent spam attempt
spam_agent = "malicious_agent"
spam_count = 0
blocked_count = 0

for i in range(150):  # Try to send 150 messages rapidly
    msg = Message(
        msg_type=MessageType.TACTICAL_VECTOR,
        sender_id=spam_agent,
        timestamp=datetime.now(),
        content=[i],
        ttl=10.0
    )
    
    if message_bus.publish(msg):
        spam_count += 1
    else:
        blocked_count += 1

print(f"Spam agent sent {spam_count} messages, {blocked_count} blocked by rate limit")
```

### Example 5: Agent Performance Analytics

```python
# After running for a while, analyze agent performance
agent_stats = knowledge_base.get_agent_statistics()

print("Agent Performance Report:")
print("-" * 50)
for agent_id, stats in agent_stats.items():
    print(f"\nAgent: {agent_id}")
    print(f"  Total Discoveries: {stats['total_discoveries']}")
    print(f"  Confirmed: {stats['confirmed_discoveries']}")
    print(f"  Average Reward: {stats['avg_reward']:.3f}")
    print(f"  Best Reward: {stats['best_reward']:.3f}")
    print(f"  Validation Score: {stats['avg_validation_score']:.3f}")

# System-wide statistics
kb_summary = knowledge_base.get_knowledge_summary()
bus_stats = message_bus.get_stats()

print(f"\nSystem Summary:")
print(f"  Total Discoveries: {kb_summary['total_discoveries']}")
print(f"  Pending Validations: {kb_summary['pending_validations']}")
print(f"  Success Rate: {kb_summary['successful_validations'] / max(kb_summary['total_proposals'], 1):.2%}")
print(f"  Message Traffic: {bus_stats['tier1_total']} strategic, {bus_stats['tier2_total']} tactical")
```

## API Reference

### MessageBus

#### `__init__(max_queue_size: int = 10000, tokens_per_second: float = 10.0)`
Initialize the message bus with queue size and rate limits.

#### `publish(message: Message) -> bool`
Publish a message to the appropriate queue.
- Returns: `True` if published, `False` if rate limited or dropped

#### `get_messages(agent_id: str, max_messages: int = 10) -> List[Message]`
Retrieve messages for an agent with priority handling.

#### `get_stats() -> Dict[str, Any]`
Get current message bus statistics.

### SharedKnowledgeBase

#### `__init__(validation_threshold: int = 2, message_bus: Optional[MessageBus] = None)`
Initialize knowledge base with validation requirements.

#### `propose_discovery(agent_id: str, discovery: Discovery) -> str`
Propose a new discovery for validation.
- Returns: Unique discovery ID

#### `vote_on_discovery(agent_id: str, discovery_id: str, approve: bool, evidence: Optional[Dict] = None)`
Submit a validation vote for a pending discovery.

#### `get_best_discoveries(n: int = 10, min_version: Optional[str] = None) -> List[Discovery]`
Get top discoveries by reward, optionally filtered by version.

#### `get_agent_statistics() -> Dict[str, Dict[str, Any]]`
Get comprehensive statistics for each agent.

#### `get_knowledge_summary() -> Dict[str, Any]`
Get summary of knowledge base state.

## Best Practices

### 1. Agent Design
- **Specialize roles**: Don't make jack-of-all-trades agents
- **Respect rate limits**: Implement backoff strategies
- **Use appropriate tiers**: Strategic for important, tactical for coordination

### 2. Discovery Validation
- **Provide evidence**: Always include evidence with votes
- **Avoid self-validation**: System prevents it, but don't attempt
- **Set appropriate thresholds**: Balance quality vs. speed

### 3. Communication
- **Set reasonable TTLs**: Don't keep tactical messages too long
- **Monitor queue sizes**: Watch for congestion
- **Handle message expiration**: Check `is_expired` before processing

### 4. Performance
- **Batch message retrieval**: Get multiple messages per call
- **Use version filtering**: Query only relevant discovery versions
- **Monitor statistics**: Track system health metrics

### 5. Error Handling
```python
try:
    discovery_id = knowledge_base.propose_discovery(agent_id, discovery)
except Exception as e:
    logger.error(f"Failed to propose discovery: {e}")
    # Implement retry logic or alternative strategy
```

## Troubleshooting

### Common Issues

#### 1. Messages Not Delivered
- Check rate limits: Agent may be throttled
- Verify queue sizes: Tier 2 may be dropping old messages
- Check TTL: Messages may be expiring

#### 2. Discoveries Not Confirmed
- Verify validation threshold is met
- Check validator agents are active
- Ensure validators aren't trying to self-validate

#### 3. Performance Issues
- Monitor queue sizes with `get_stats()`
- Check for agents flooding the system
- Verify thread contention isn't excessive

#### 4. Version Conflicts
- Ensure proper version comparison
- Check reward values for updates
- Verify version history tracking

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Check Script

```python
def system_health_check(message_bus, knowledge_base):
    """Perform system health check."""
    print("System Health Check")
    print("=" * 50)
    
    # Check message bus
    bus_stats = message_bus.get_stats()
    print(f"Message Bus:")
    print(f"  Tier 1 Queue: {bus_stats['tier1_queue_size']} messages")
    print(f"  Tier 2 Queue: {bus_stats['tier2_queue_size']}/{message_bus.max_queue_size}")
    print(f"  Active Agents: {bus_stats['active_agents']}")
    print(f"  Dropped Messages: {bus_stats['dropped_total']}")
    
    # Check knowledge base
    kb_summary = knowledge_base.get_knowledge_summary()
    print(f"\nKnowledge Base:")
    print(f"  Confirmed Discoveries: {kb_summary['total_discoveries']}")
    print(f"  Pending Validations: {kb_summary['pending_validations']}")
    print(f"  Validation Success Rate: {kb_summary['successful_validations'] / max(kb_summary['total_proposals'], 1):.1%}")
    
    # Performance metrics
    if bus_stats['tier2_total'] > 0:
        drop_rate = bus_stats['dropped_total'] / bus_stats['tier2_total']
        if drop_rate > 0.1:
            print(f"\n⚠️  WARNING: High message drop rate: {drop_rate:.1%}")
    
    if kb_summary['pending_validations'] > 50:
        print(f"\n⚠️  WARNING: High pending validation count: {kb_summary['pending_validations']}")
    
    print("\n✅ Health check complete")

# Run health check
system_health_check(message_bus, knowledge_base)
```

## Conclusion

The JanusAI Multi-Agent System provides a robust framework for collaborative discovery with quality assurance through consensus validation. By following the tiered communication architecture and best practices outlined in this documentation, you can build sophisticated multi-agent systems capable of complex collaborative tasks while maintaining system integrity and performance.