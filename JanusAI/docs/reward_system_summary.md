# Formalized Reward System: Executive Summary

## 🎯 Overview

We've created a centralized, extensible reward system for the Janus project that replaces scattered reward calculations with a unified `RewardHandler`. This system makes it trivial to experiment with different reward formulations by simply changing configurations.

## 📁 Key Components

### 1. **RewardHandler** (`ml/rewards/reward_handler.py`)
The core class that composes multiple reward components with configurable weights.

```python
# Simple usage
handler = RewardHandler(
    reward_components={
        FidelityReward(): 0.5,
        NoveltyReward(): 0.3,
        ComplexityReward(): 0.2
    }
)

# Calculate reward
total_reward = handler.calculate_total_reward(**step_data)
```

### 2. **Reward Registry** (`ml/rewards/reward_registry.py`)
A registry system for managing and creating reward components.

```python
# Use presets
handler = create_handler_from_preset('physics_discovery')

# Or create by name
handler = RewardHandler({
    "novelty": 0.4,
    "conservation": 0.6
})
```

### 3. **Existing Reward Components**
All existing rewards now integrate seamlessly:
- `NoveltyReward` - Rewards novel expressions
- `ComplexityReward` - Controls expression complexity
- `ConservationLawReward` - Rewards physical law adherence
- `InterpretabilityReward` - Rewards interpretable AI explanations

## 🚀 Key Features

### 1. **Composable Design**
Mix and match reward components with different weights:
```python
reward_components = {
    NoveltyReward(threshold=0.1): 0.3,
    ComplexityReward(target=10): 0.2,
    ConservationLawReward(law='energy'): 0.5
}
```

### 2. **Configuration-Based**
Define rewards in config files:
```yaml
reward_handler:
  components:
    novelty:
      threshold: 0.1
    complexity:
      target_complexity: 10
    conservation:
      law_type: energy
  weights:
    novelty: 0.3
    complexity: 0.2
    conservation: 0.5
```

### 3. **Preset Configurations**
Ready-to-use configurations for common scenarios:
- `physics_discovery` - For discovering physical laws
- `ai_interpretability` - For explaining AI models
- `balanced_exploration` - General-purpose exploration
- `curriculum_learning` - Progressive difficulty

### 4. **Detailed Logging**
Track individual component contributions:
```python
reward_info = handler.calculate_detailed_reward(**data)
print(f"Novelty contribution: {reward_info.component_rewards['novelty']}")
print(f"Total reward: {reward_info.total_reward}")
```

### 5. **Dynamic Adaptation**
Adjust rewards during training:
```python
# Manual adjustment
handler.update_weights({'novelty': 0.5, 'complexity': 0.5})

# Or use adaptive handler
handler = AdaptiveRewardHandler(
    components,
    target_balance={'novelty': 0.3, 'complexity': 0.7}
)
```

## 💡 Usage Examples

### Basic Physics Discovery
```python
from janus.ml.rewards.reward_registry import create_handler_from_preset

# Use physics preset
reward_handler = create_handler_from_preset('physics_discovery')

# Create environment with handler
env = SymbolicDiscoveryEnv(
    grammar=grammar,
    X_data=X,
    y_data=y,
    variables=variables,
    reward_handler=reward_handler
)
```

### Custom AI Interpretability
```python
# Create custom configuration
handler = RewardHandler(
    reward_components={
        InterpretabilityReward(
            fidelity_weight=0.6,
            simplicity_weight=0.4
        ): 0.8,
        NoveltyReward(): 0.2
    },
    normalize=True,
    clip_range=(-5.0, 5.0)
)
```

### Experiment with Different Rewards
```python
# Try different reward schemes easily
for preset in ['physics_discovery', 'balanced_exploration']:
    handler = create_handler_from_preset(preset)
    env.reward_handler = handler
    
    # Run experiment
    results = run_experiment(env)
    print(f"{preset}: {results['mean_reward']}")
```

## 🔄 Migration Path

### Before (Scattered Rewards)
```python
def _calculate_reward(self):
    mse_reward = -mse * self.mse_weight
    complexity_penalty = -complexity * self.penalty_factor
    novelty_bonus = self.calculate_novelty()
    
    return mse_reward + complexity_penalty + novelty_bonus
```

### After (Centralized)
```python
def _calculate_reward(self):
    return self.reward_handler.calculate_total_reward(
        current_observation=obs,
        action=action,
        next_observation=next_obs,
        reward_from_env=base_reward,
        done=done,
        info=info
    )
```

## 📊 Benefits

1. **Easier Experimentation**: Change reward schemes without code changes
2. **Better Organization**: All reward logic in one place
3. **Improved Debugging**: Track individual reward components
4. **Flexible Configuration**: Use code, config files, or presets
5. **Extensible Design**: Easy to add new reward components

## 🎮 Advanced Features

### Adaptive Rewards
```python
handler = AdaptiveRewardHandler(
    components,
    adaptation_rate=0.01,
    target_balance={'fidelity': 0.7, 'novelty': 0.3}
)
```

### Reward Scheduling
```python
scheduler = RewardScheduler(handler, {
    0: {'novelty': 0.6, 'complexity': 0.4},      # Early
    5000: {'novelty': 0.3, 'complexity': 0.7}    # Late
})
```

### Custom Components
```python
class MyCustomReward(BaseReward):
    def calculate_reward(self, **kwargs):
        # Custom logic
        return reward_value

handler.add_component('custom', MyCustomReward(), weight=0.1)
```

## 📈 Impact on Janus Project

This formalized reward system:

1. **Accelerates Research**: Quickly test different reward hypotheses
2. **Improves Reproducibility**: Reward configurations are explicit
3. **Enables Comparison**: Easy to compare reward schemes
4. **Supports Both Goals**: Physics discovery and AI interpretability
5. **Facilitates Collaboration**: Clear, modular reward design

## 🔮 Future Extensions

The system is designed to support:
- Multi-objective optimization
- Learned reward functions
- User-defined reward components
- Automatic reward tuning
- Reward visualization tools

## 📝 Summary

The formalized `RewardHandler` system transforms reward engineering in Janus from an ad-hoc process to a systematic, configurable approach. By centralizing reward calculation and providing a flexible composition system, researchers can now focus on designing effective reward schemes rather than implementing reward infrastructure.

**Key Takeaway**: Changing reward formulations is now as simple as changing a configuration dictionary, enabling rapid experimentation and discovery.