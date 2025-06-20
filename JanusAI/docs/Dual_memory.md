# Janus Dual Memory System Documentation

## Overview

The Janus Dual Memory System implements a sophisticated memory architecture inspired by the Xolver paper, providing both long-term episodic memory and short-term shared memory for multi-agent scientific discovery.

## Architecture

### 1. **Episodic Memory** (Long-term)
- Stores successful discoveries across all problems
- Persists to SQLite database
- Supports similarity-based retrieval using FAISS
- Automatic capacity management with eviction

### 2. **Shared Memory** (Per-problem)
- Maintains top-k intermediate results during discovery
- Enables agent collaboration within a problem
- Thread-safe for concurrent access
- Automatic ranking by score

## Installation

```bash
# Required dependencies
pip install numpy pandas matplotlib seaborn networkx scikit-learn
pip install faiss-cpu  # or faiss-gpu for CUDA
pip install sentence-transformers  # Optional, for embeddings

# Install Janus memory system
cd JanusAI
pip install -e .
```

## Quick Start

### Basic Usage

```python
from memory.dual_memory_system import DualMemorySystem, Discovery, IntermediateResult

# Initialize memory system
memory_system = DualMemorySystem(
    episodic_capacity=10000,  # Max discoveries to store
    shared_capacity=10,       # Max results per problem
    db_path="janus_memory.db"
)

# Start a new discovery problem
problem_id = "attention_discovery_001"
shared_memory = memory_system.start_problem(problem_id)

# Agents add intermediate results
result = IntermediateResult(
    id="result_1",
    timestamp=datetime.now(),
    domain="attention_mechanisms",
    agent_role="HypothesisGenerator",
    expression="softmax(Q @ K.T / sqrt(d))",
    thought="This captures scaled dot-product attention",
    response="Standard attention mechanism discovered",
    score=0.85,
    detailed_scores={
        'fidelity': 0.9,
        'simplicity': 0.8,
        'novelty': 0.85
    }
)
shared_memory.add(result)

# End problem and save best to episodic memory
discovery = memory_system.end_problem(
    problem_id, 
    domain="attention_mechanisms",
    save_to_episodic=True
)

# Search episodic memory
top_discoveries = memory_system.episodic.get_top_validated(5)
```

### Memory-Augmented Agents

```python
from memory.memory_integration import MemoryAugmentedAgent

class DiscoveryAgent(MemoryAugmentedAgent):
    def discover(self, problem_context, problem_id):
        # Get relevant past discoveries
        memories = self.get_relevant_memories(problem_context, k=5)
        
        # Get current shared memory state
        shared_context = self.get_shared_memory_context(problem_id)
        
        # Generate new hypothesis based on memories
        expression = self.generate_hypothesis(memories, shared_context)
        
        # Add to shared memory
        self.add_to_shared_memory(
            expression=expression,
            thought="Building on past discoveries...",
            response=f"Proposed: {expression}",
            score=0.75,
            problem_id=problem_id
        )
```

### Memory-Integrated Environment

```python
from memory.memory_integration import MemoryIntegratedEnv

# Create environment with memory
env = MemoryIntegratedEnv(
    memory_system=memory_system,
    grammar=grammar,
    reward_fn=reward_fn,
    task_type="attention_pattern"
)

# Reset provides episodic context
obs, info = env.reset()
episodic_memories = info.get('episodic_memories', [])

# Steps automatically track to shared memory
obs, reward, done, truncated, info = env.step(action, agent_role="Explorer")
```

## Advanced Features

### 1. Memory Consolidation

```python
from memory.advanced_features import MemoryConsolidator

consolidator = MemoryConsolidator(memory_system)

# Consolidate old memories to save space
result = consolidator.consolidate_old_memories(
    age_threshold_days=30,
    min_importance=0.5
)
print(f"Consolidated {result['removed_count']} memories into {result['summary_count']} summaries")
```

### 2. Visualization

```python
from memory.advanced_features import MemoryVisualizer

visualizer = MemoryVisualizer(memory_system)

# Plot discovery timeline
visualizer.plot_discovery_timeline(domain="attention_mechanisms")

# Visualize embedding space
visualizer.plot_embedding_space()

# Create knowledge graph
visualizer.plot_knowledge_graph(max_nodes=50)
```

### 3. Export/Import

```python
from memory.advanced_features import MemoryExporter

exporter = MemoryExporter(memory_system)

# Export discoveries
exporter.export_discoveries(
    "janus_discoveries.zip",
    domain="attention_mechanisms",
    min_score=0.7
)

# Import to another system
new_memory_system = DualMemorySystem()
new_exporter = MemoryExporter(new_memory_system)
new_exporter.import_discoveries(
    "janus_discoveries.zip",
    merge_strategy="keep_better"
)
```

### 4. Importance Sampling

```python
from memory.advanced_features import ImportanceSampler

sampler = ImportanceSampler(memory_system)

# Sample important discoveries for training
important_discoveries = sampler.sample_by_importance(
    n_samples=10,
    temperature=0.5,  # Lower = more greedy
    domain="physics_laws"
)
```

### 5. Memory Replay for Training

```python
from memory.memory_integration import MemoryReplayBuffer

replay_buffer = MemoryReplayBuffer(memory_system)

# Sample high-quality discoveries
replay_batch = replay_buffer.sample(
    batch_size=16,
    domain="attention_mechanisms",
    min_score=0.7
)

# Create training batch
training_data = replay_buffer.create_training_batch(
    replay_batch,
    include_negatives=True
)
```

## Memory System API

### DualMemorySystem

| Method | Description |
|--------|-------------|
| `start_problem(problem_id)` | Start new problem with fresh shared memory |
| `end_problem(problem_id, domain, save_to_episodic)` | End problem and optionally save to episodic |
| `get_relevant_discoveries(query_embedding, domain, k)` | Get relevant past discoveries |
| `get_memory_stats()` | Get statistics for both memory types |
| `save_state(path)` | Save complete memory state |
| `load_state(path)` | Load memory state |

### EpisodicMemory

| Method | Description |
|--------|-------------|
| `add(discovery)` | Add discovery to memory |
| `get(discovery_id)` | Retrieve specific discovery |
| `search_by_similarity(embedding, k, domain)` | Find similar discoveries |
| `search_by_domain(domain, limit)` | Get discoveries in domain |
| `search_by_expression(pattern, limit)` | Search by expression pattern |
| `get_top_validated(n, domain)` | Get highest scoring discoveries |

### SharedMemory

| Method | Description |
|--------|-------------|
| `add(result)` | Add intermediate result |
| `get_top(k)` | Get top k results by score |
| `get_by_agent(agent_role)` | Get results from specific agent |
| `get_unique_expressions()` | Get all unique expressions |
| `to_episodic_entry(domain, problem_id)` | Convert to discovery |

## Data Structures

### Discovery

```python
@dataclass
class Discovery:
    id: str                          # Unique identifier
    timestamp: datetime              # When discovered
    domain: str                      # Problem domain
    expression: str                  # Symbolic expression
    hypothesis: str                  # Natural language hypothesis
    evidence: List[Dict[str, Any]]   # Supporting evidence
    confidence: float                # Agent confidence (0-1)
    validation_score: float          # Validation score (0-1)
    reasoning_trace: List[str]       # Reasoning steps
    agent_roles: List[str]          # Contributing agents
    embedding: Optional[np.ndarray]  # Semantic embedding
    metadata: Dict[str, Any]        # Additional metadata
```

### IntermediateResult

```python
@dataclass
class IntermediateResult:
    id: str                         # Unique identifier
    timestamp: datetime             # When created
    domain: str                     # Problem domain
    agent_role: str                 # Agent that created it
    expression: str                 # Proposed expression
    thought: str                    # Agent's reasoning
    response: str                   # Agent's response
    score: float                    # Overall score
    detailed_scores: Dict[str, float]  # Breakdown of scores
    feedback: str                   # Judge feedback
    iteration: int                  # Iteration number
```

## Best Practices

### 1. Memory Capacity
- Set episodic capacity based on available disk space
- Use consolidation for long-running systems
- Monitor memory statistics regularly

### 2. Embeddings
- Use embeddings for better similarity search
- Keep embedding dimension consistent
- Consider GPU acceleration for FAISS with large memories

### 3. Thread Safety
- The system is thread-safe for multi-agent use
- Each agent should have its own MemoryAugmentedAgent instance
- Use problem IDs to isolate shared memories

### 4. Persistence
- Regularly save memory state for backup
- Export important discoveries before major changes
- Use consolidation to manage database size

### 5. Performance
- Index episodic memory by domain for faster queries
- Limit shared memory capacity to maintain speed
- Use importance sampling for large memories

## Example: Complete Discovery Session

```python
import logging
from datetime import datetime

# Setup
logging.basicConfig(level=logging.INFO)
memory_system = DualMemorySystem()

# Multi-agent discovery
agents = [
    DiscoveryAgent("HypothesisGenerator", memory_system),
    DiscoveryAgent("Experimenter", memory_system),
    DiscoveryAgent("Theorist", memory_system)
]

# Run discovery session
problem_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shared_memory = memory_system.start_problem(problem_id)

for iteration in range(3):
    logging.info(f"Iteration {iteration + 1}")
    
    for agent in agents:
        # Each agent contributes
        discovery = agent.discover(
            {"domain": "attention_mechanisms", "task": "multi_head_attention"},
            problem_id
        )
    
    # Check progress
    top_results = shared_memory.get_top(3)
    for i, result in enumerate(top_results):
        logging.info(f"{i+1}. {result.expression} (score: {result.score:.3f})")

# Finalize
final_discovery = memory_system.end_problem(problem_id, "attention_mechanisms")

# Analyze results
stats = memory_system.get_memory_stats()
logging.info(f"Total discoveries: {stats['episodic']['total_discoveries']}")
```

## Troubleshooting

### Common Issues

1. **FAISS Import Error**
   ```bash
   # Install CPU version
   pip install faiss-cpu
   # OR GPU version (requires CUDA)
   pip install faiss-gpu
   ```

2. **Database Lock Error**
   - Ensure only one process writes to the database at a time
   - Use proper connection closing in multi-process setups

3. **Memory Growth**
   - Monitor with `get_memory_stats()`
   - Use consolidation for old memories
   - Adjust capacity limits as needed

4. **Embedding Dimension Mismatch**
   - Keep embedding_dim consistent across sessions
   - Re-generate embeddings if model changes

## Future Enhancements

1. **Distributed Memory**
   - Support for distributed storage across machines
   - Redis integration for shared memory

2. **Advanced Retrieval**
   - Learned retrieval models
   - Hybrid sparse-dense retrieval

3. **Memory Compression**
   - Automatic compression of old memories
   - Learned summarization models

4. **Visualization Dashboard**
   - Real-time memory monitoring
   - Interactive exploration tools

## Contributing

See the main Janus contributing guidelines. Key areas for contribution:
- Additional memory consolidation strategies
- New visualization types
- Performance optimizations
- Integration with other Janus components
