# Integration Guide: Enhanced Attention-Based Meta-Learning

## Overview

This guide shows how to integrate the enhanced attention-based meta-learning components into the Janus project. The key improvement is using task embeddings as queries for attention over tree representations, allowing dynamic focus based on the current physics task.

## Key Components

### 1. Enhanced Meta-Learning Policy (`attention_meta_policy.py`)

The enhanced policy includes:
- **Multi-head attention mechanism** for task-aware tree focusing
- **Multiple fusion strategies** (attention, gating, bilinear)
- **Hierarchical task encoding** with local and global context
- **Auxiliary predictions** for complexity and convergence time

### 2. Enhanced MAML Trainer (`enhanced_meta_trainer.py`)

The trainer adds:
- **Attention analysis and visualization**
- **Multi-task learning** with auxiliary losses
- **Attention entropy regularization**
- **Performance tracking** per task type

## Quick Start

### Basic Usage

```python
from janus.ml.networks.attention_meta_policy import EnhancedMetaLearningPolicy
from janus.ml.training.enhanced_meta_trainer import (
    EnhancedMAMLTrainer, 
    EnhancedMetaLearningConfig
)
from janus.physics.data.generators import PhysicsTaskDistribution

# Configure enhanced meta-learning
config = EnhancedMetaLearningConfig(
    # Base MAML parameters
    meta_lr=0.0003,
    adaptation_lr=0.01,
    adaptation_steps=5,
    tasks_per_batch=10,
    
    # Attention-specific parameters
    num_attention_heads=4,
    fusion_type="attention",  # Options: "attention", "gating", "bilinear"
    use_hierarchical_encoding=True,
    
    # Multi-task learning weights
    policy_loss_weight=1.0,
    value_loss_weight=0.5,
    complexity_loss_weight=0.1,
    convergence_loss_weight=0.1,
    attention_entropy_weight=0.01,
    
    # Visualization
    save_attention_maps=True,
    attention_save_interval=100,
    
    # Device
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize task distribution
task_distribution = PhysicsTaskDistribution(include_noise=True)

# Create and train the enhanced model
trainer = EnhancedMAMLTrainer(config, task_distribution=task_distribution)
trainer.train(n_iterations=1000)
```

### Advanced Usage with Custom Policy

```python
# Create a custom enhanced policy
policy = EnhancedMetaLearningPolicy(
    observation_dim=128,
    action_dim=50,
    hidden_dim=256,
    num_attention_heads=8,
    fusion_type="attention",
    use_hierarchical_encoding=True
)

# Use with trainer
trainer = EnhancedMAMLTrainer(
    config=config,
    policy=policy,
    task_distribution=task_distribution
)
```

## How Attention-Based Adaptation Works

### 1. Task Encoding

The policy encodes task trajectories using a bidirectional LSTM:

```python
def encode_task(self, task_trajectories: torch.Tensor) -> Dict[str, torch.Tensor]:
    # task_trajectories shape: (batch, num_traj, traj_len, feat_dim)
    
    # LSTM encoding
    lstm_out, (hidden, cell) = self.task_encoder(trajectories_flat)
    
    # Hierarchical encoding (optional)
    if self.use_hierarchical_encoding:
        local_features = self.local_task_encoder(trajectories_flat)
        global_features = self.global_task_aggregator(task_embeddings)
        
    return {
        'task_embedding': task_embedding,
        'local_features': local_features,
        'global_features': global_features
    }
```

### 2. Attention-Based Tree Focus

The task embedding serves as a query for attending over the tree representation:

```python
# In forward pass
query = task_embedding.unsqueeze(1)      # Task as query
key_value = tree_repr.unsqueeze(1)       # Tree as key/value

# Apply attention
adapted_repr, attention_weights = self.task_attention(query, key_value)

# The model now focuses on task-relevant parts of the expression tree
```

### 3. Multi-Modal Fusion

The attended representation is fused with the task embedding:

```python
# Fusion strategies
if fusion_type == "attention":
    fused = attention_mechanism(task_embedding, tree_repr)
elif fusion_type == "gating":
    gate = sigmoid(linear([task_embedding, tree_repr]))
    fused = gate * task_embedding + (1 - gate) * tree_repr
elif fusion_type == "bilinear":
    fused = bilinear(task_embedding, tree_repr)
```

## Attention Analysis and Visualization

### Accessing Attention Statistics

```python
# Get attention statistics for analysis
obs_tensor = torch.FloatTensor(observations).to(device)
task_context = trainer._prepare_task_context(trajectories)

attention_stats = policy.get_attention_stats(obs_tensor, task_context)
# Returns:
# - attention_weights: Raw attention maps
# - attention_entropy: Measure of focus (lower = more focused)
# - attention_max: Maximum attention value per sample
# - attention_argmax: Most attended position
```

### Visualizing Attention Evolution

The trainer automatically saves attention heatmaps during training:

```python
# Attention maps saved to: enhanced_meta_logs/attention_maps/
# Format: attention_{task_name}_{iteration}.png
```

### Analyzing Attention Patterns

```python
# Get attention evolution for a specific task
evolution = trainer.attention_analyzer.analyze_attention_evolution("harmonic_oscillator")

# Plot attention entropy over training
import matplotlib.pyplot as plt
plt.plot(evolution['entropy'])
plt.xlabel('Training Step')
plt.ylabel('Attention Entropy')
plt.title('Attention Focus Evolution')
```

## Benefits of Attention-Based Adaptation

1. **Dynamic Focus**: The model learns to focus on relevant parts of the expression tree based on the task
2. **Better Generalization**: Task-specific attention helps transfer learning across physics domains
3. **Interpretability**: Attention maps show what the model considers important for each task
4. **Faster Adaptation**: Focused attention accelerates inner-loop learning

## Performance Tips

1. **Start Simple**: Begin with `fusion_type="attention"` and 4 attention heads
2. **Monitor Entropy**: Decreasing attention entropy indicates the model is learning to focus
3. **Regularization**: Use `attention_entropy_weight` to balance focus vs exploration
4. **Hierarchical Encoding**: Enable for complex tasks with multiple scales of behavior

## Troubleshooting

### High Attention Entropy
- Increase `attention_entropy_weight` to encourage focus
- Ensure sufficient support trajectories for task encoding
- Check if tasks are too diverse for single attention pattern

### Slow Convergence
- Reduce `num_attention_heads` if computation is bottleneck
- Try `fusion_type="gating"` for simpler but effective adaptation
- Increase `adaptation_steps` for more inner-loop updates

### Memory Issues
- Reduce `tasks_per_batch` or trajectory lengths
- Disable `save_attention_maps` during long training runs
- Use gradient checkpointing for very deep policies

## Next Steps

1. **Experiment with Fusion Types**: Try different fusion strategies for your tasks
2. **Custom Attention Mechanisms**: Implement domain-specific attention patterns
3. **Multi-Task Learning**: Leverage auxiliary predictions for better representations
4. **Attention Transfer**: Pre-train attention patterns on simple tasks

## References

- Original MAML paper: Finn et al., "Model-Agnostic Meta-Learning"
- FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
- Cross-Modal Attention: Various works on vision-language models

---

For more details, see the implementation files:
- `JanusAI/ml/networks/attention_meta_policy.py`
- `JanusAI/ml/training/enhanced_meta_trainer.py`
