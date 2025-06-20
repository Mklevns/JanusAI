# Integration Guide: Advanced Exploration Strategies for JanusAI

## Overview

This guide shows how to integrate three cutting-edge exploration strategies into JanusAI:

1. **MaxInfoRL (Information Gain)**: Rewards exploring uncertain regions
2. **PreND (Pre-trained Network Distillation)**: Uses powerful models to guide exploration
3. **LLM-Driven Exploration**: Leverages language models for high-level hypothesis generation

## Quick Start

### Basic Setup

```python
from janus.ml.training.enhanced_ppo_trainer import EnhancedPPOTrainer, EnhancedPPOConfig
from janus.environments.base.symbolic_env import SymbolicDiscoveryEnv
from janus.physics.data.generators import PhysicsTaskDistribution

# Create environment
task_dist = PhysicsTaskDistribution()
task = task_dist.sample_task()
env = SymbolicDiscoveryEnv(
    data=task.generate_data(200),
    target_expr="unknown",
    max_depth=10
)

# Configure enhanced training
config = EnhancedPPOConfig(
    # Enable all exploration strategies
    use_information_gain=True,
    use_prend=True,
    use_llm_goals=True,
    
    # Adjust weights
    info_gain_scale=0.5,
    prend_scale=0.3,
    intrinsic_weight=0.5,
    
    # LLM settings
    llm_model="gpt-4",
    llm_exploration_rate=0.2
)

# Train with enhanced exploration
trainer = EnhancedPPOTrainer(env, config)
trainer.train()
```

## Strategy Details

### 1. MaxInfoRL (Information Gain)

The information gain reward encourages exploring expressions where the dynamics model ensemble is most uncertain.

#### How it Works:
- Maintains an ensemble of neural networks that predict rewards from expression embeddings
- Variance in predictions indicates epistemic uncertainty
- High uncertainty = high intrinsic reward

#### Configuration:
```python
config = EnhancedPPOConfig(
    use_information_gain=True,
    ensemble_size=5,  # Number of models in ensemble
    ensemble_hidden_dim=128,
    info_gain_scale=0.5,  # Scale factor for reward
)
```

#### When to Use:
- Early in training when the model knows little
- For open-ended discovery tasks
- When the reward landscape is complex

### 2. PreND (Pre-trained Network Distillation)

Uses a powerful pre-trained model as a fixed target to identify "interesting" states.

#### How it Works:
- A predictor network tries to match features from a frozen pre-trained model
- States that are hard to predict get higher intrinsic rewards
- Focuses exploration on semantically meaningful regions

#### Configuration:
```python
config = EnhancedPPOConfig(
    use_prend=True,
    prend_model_name="clip-vit-base-patch32",  # Or any HuggingFace model
    prend_scale=0.3,
)
```

#### When to Use:
- For AI interpretability tasks
- When you want to leverage existing knowledge
- To avoid getting stuck on trivial patterns

### 3. LLM-Driven Exploration

Uses language models to suggest high-level hypotheses and exploration goals.

#### How it Works:
- LLM analyzes context (variables, recent discoveries, performance)
- Suggests mathematical expressions to try
- Agent gets rewards for matching these goals

#### Configuration:
```python
config = EnhancedPPOConfig(
    use_llm_goals=True,
    llm_model="gpt-4",  # Or "claude-2", "gpt-3.5-turbo"
    llm_exploration_rate=0.2,  # Probability of using LLM goal
    llm_goal_duration=100,  # Steps per goal
)
```

#### Setting up API Keys:
```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

#### When to Use:
- For complex physics domains
- When random exploration is too slow
- To incorporate domain knowledge

## Advanced Usage

### Combining Strategies

The strategies work synergistically:

```python
# Adaptive weight combination
config = EnhancedPPOConfig(
    use_information_gain=True,
    use_prend=True,
    use_llm_goals=True,
    
    # Weights will adapt based on performance
    adaptive_weights=True,
    
    # Initial weights
    extrinsic_weight=1.0,
    intrinsic_weight=0.5,
)
```

### Custom Expression Embedder

For better information gain calculations:

```python
from janus.ml.networks.dynamics_ensemble import DynamicsEnsemble

class CustomExpressionEmbedder(nn.Module):
    def __init__(self, grammar):
        super().__init__()
        self.grammar = grammar
        # Your custom architecture
        
    def forward(self, expression):
        # Convert expression to embedding
        features = self.extract_features(expression)
        return self.encode(features)

# Use in trainer
trainer.expression_embedder = CustomExpressionEmbedder(env.grammar)
```

### Domain-Specific LLM Prompts

Customize prompts for your domain:

```python
from janus.utils.ai.llm_exploration import LLMGoalGenerator

# Create custom generator
generator = LLMGoalGenerator(model_name="gpt-4")

# Add custom prompt template
generator.prompt_templates['quantum'] = """
You are a quantum physicist discovering wave functions.
Variables: {variables}
Recent discoveries: {discoveries}

Suggest a wavefunction expression considering:
1. Normalization requirements
2. Boundary conditions
3. Symmetry properties

Expression:"""

trainer.llm_generator = generator
```

## Performance Monitoring

### Key Metrics to Track

```python
# Access exploration statistics
stats = trainer.intrinsic_reward.get_statistics()

print("Information Gain Stats:")
print(f"  Mean disagreement: {stats['components']['information_gain']['mean_disagreement']}")
print(f"  Disagreement trend: {stats['components']['information_gain']['disagreement_trend']}")

print("\nPreND Stats:")
print(f"  Mean prediction error: {stats['components']['prend']['mean_prediction_error']}")

print("\nLLM Goal Stats:")
print(f"  Achievement rate: {stats['components']['goal_matching']['achievement_rate']}")
print(f"  Diversity score: {stats['components']['goal_matching']['diversity_score']}")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot intrinsic reward contributions
components = stats['components']
names = list(components.keys())
contributions = [components[n]['mean_contribution'] for n in names]

plt.bar(names, contributions)
plt.xlabel('Exploration Strategy')
plt.ylabel('Mean Contribution')
plt.title('Intrinsic Reward Contributions')
plt.show()
```

## Best Practices

### 1. Start Simple
Begin with one strategy and add others gradually:
```python
# Phase 1: Information gain only
config.use_information_gain = True
config.use_prend = False
config.use_llm_goals = False

# Phase 2: Add PreND
config.use_prend = True

# Phase 3: Add LLM goals
config.use_llm_goals = True
```

### 2. Balance Exploration vs Exploitation
```python
# High exploration early
config.intrinsic_weight = 0.7  # Early training
config.llm_exploration_rate = 0.3

# Reduce over time
config.intrinsic_weight = 0.3  # Late training
config.llm_exploration_rate = 0.1
```

### 3. Domain-Specific Tuning

#### For Physics Discovery:
```python
config.use_information_gain = True  # Primary
config.use_llm_goals = True  # Secondary
config.use_prend = False  # Less useful here
```

#### For AI Interpretability:
```python
config.use_prend = True  # Primary
config.use_information_gain = True  # Secondary
config.use_llm_goals = False  # Optional
```

### 4. Handle API Limits

For LLM usage:
```python
config.llm_exploration_rate = 0.1  # Reduce API calls
config.llm_goal_duration = 200  # Longer goal persistence
```

## Troubleshooting

### High Variance in Rewards
- Reduce `info_gain_scale` and `prend_scale`
- Enable reward normalization
- Increase ensemble size

### LLM Suggestions Not Helpful
- Improve context in prompts
- Add more discovered expressions to context
- Try different models or temperatures

### Slow Training
- Reduce ensemble size
- Use smaller pre-trained models
- Cache LLM responses

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Disable some exploration strategies

## Example: Complete Physics Discovery Pipeline

```python
# Full example for discovering conservation laws
from janus.ml.training.enhanced_ppo_trainer import (
    EnhancedPPOTrainer, 
    EnhancedPPOConfig
)

# 1. Setup environment with double pendulum
task = task_dist.get_task_by_name("double_pendulum")
env = SymbolicDiscoveryEnv(
    data=task.generate_data(500),
    target_expr="unknown",
    max_depth=15,
    max_complexity=50
)

# 2. Configure for physics discovery
config = EnhancedPPOConfig(
    # Core settings
    total_timesteps=500000,
    n_steps=2048,
    
    # Enable information gain and LLM
    use_information_gain=True,
    use_prend=False,  # Not needed for physics
    use_llm_goals=True,
    
    # Information gain settings
    ensemble_size=7,
    info_gain_scale=0.6,
    
    # LLM settings for physics
    llm_model="gpt-4",
    llm_exploration_rate=0.25,
    
    # Reward balance
    extrinsic_weight=1.0,
    intrinsic_weight=0.4,
    adaptive_weights=True
)

# 3. Add physics context to environment
env.task_info = {
    'name': 'Double Pendulum',
    'domain': 'mechanics',
    'variable_descriptions': {
        'theta1': 'Angle of first pendulum',
        'theta2': 'Angle of second pendulum',
        'omega1': 'Angular velocity of first pendulum',
        'omega2': 'Angular velocity of second pendulum',
        'm1': 'Mass of first pendulum',
        'm2': 'Mass of second pendulum',
        'L1': 'Length of first pendulum',
        'L2': 'Length of second pendulum',
        'g': 'Gravitational acceleration',
        'E': 'Total energy'
    }
}

# 4. Train with monitoring
trainer = EnhancedPPOTrainer(env, config)

# Custom callback for physics-specific logging
class PhysicsDiscoveryCallback(BaseCallback):
    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if 'expression' in info and info.get('is_conservation_law'):
                print(f"Discovered conservation law: {info['expression']}")
        return True

trainer.agent.learn(
    total_timesteps=config.total_timesteps,
    callback=[trainer.intrinsic_callback, PhysicsDiscoveryCallback()]
)

# 5. Analyze results
print("\nDiscovered Expressions:")
for disc in trainer.discovered_expressions:
    print(f"  {disc['expression']} (reward: {disc['reward']:.3f})")
```

## Conclusion

These advanced exploration strategies transform JanusAI from a random searcher into an intelligent discovery system that:

1. **Knows what it doesn't know** (Information Gain)
2. **Leverages existing knowledge** (PreND)
3. **Forms intelligent hypotheses** (LLM Goals)

By combining these approaches, JanusAI can tackle complex discovery tasks that would be intractable with traditional methods.
