# Integration Guide: Migrating to RewardHandler

## Overview

This guide shows how to integrate the new formalized `RewardHandler` system into existing Janus code. The RewardHandler centralizes reward calculation and makes it easy to experiment with different reward formulations.

## Key Benefits

1. **Centralized Configuration**: All reward weights and components in one place
2. **Easy Experimentation**: Switch reward schemes with configuration changes
3. **Detailed Logging**: Track individual reward component contributions
4. **Dynamic Adjustment**: Modify rewards during training
5. **Preset Configurations**: Ready-to-use reward schemes for common scenarios

## Migration Examples

### 1. Updating SymbolicDiscoveryEnv

**Before:**
```python
class SymbolicDiscoveryEnv(gym.Env):
    def _calculate_reward(self, action_valid: bool) -> float:
        # Scattered reward logic
        if not action_valid:
            return self.reward_config['invalid_penalty']
        
        # MSE calculation
        mse = self._calculate_mse()
        mse_reward = self.reward_config['mse_weight'] * mse
        
        # Complexity penalty
        complexity = calculate_expression_complexity(expr)
        complexity_penalty = self.reward_config['complexity_penalty'] * complexity
        
        # Manual combination
        return mse_reward + complexity_penalty + completion_bonus
```

**After:**
```python
class SymbolicDiscoveryEnv(gym.Env):
    def __init__(self, *args, reward_handler: RewardHandler = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Use provided handler or create from preset
        self.reward_handler = reward_handler or create_handler_from_preset(
            'physics_discovery'
        )
    
    def _calculate_reward(self, action_valid: bool) -> float:
        # Prepare info for reward handler
        info = self._get_info()
        info['action_valid'] = action_valid
        info['expression'] = self.current_expression
        info['complexity'] = self.current_complexity
        
        # Delegate to reward handler
        return self.reward_handler.calculate_total_reward(
            current_observation=self._get_observation(),
            action=self._last_action,
            next_observation=self._get_observation(),
            reward_from_env=0.0,  # No base reward
            done=self._is_terminated(),
            info=info
        )
```

### 2. Updating EnhancedSymbolicDiscoveryEnv

**Before:**
```python
class EnhancedSymbolicDiscoveryEnv(SymbolicDiscoveryEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Multiple calculators
        self.intrinsic_calculator = IntrinsicRewardCalculator(
            novelty_weight=kwargs.get('novelty_weight', 0.3),
            diversity_weight=kwargs.get('diversity_weight', 0.2),
            complexity_growth_weight=kwargs.get('complexity_growth_weight', 0.1),
            conservation_weight=kwargs.get('conservation_weight', 0.4),
        )
    
    def step(self, action):
        # Complex reward combination
        base_reward = super()._calculate_reward(action_valid)
        intrinsic_reward = self.intrinsic_calculator.calculate_total_intrinsic_reward(...)
        total_reward = base_reward + intrinsic_reward
```

**After:**
```python
class EnhancedSymbolicDiscoveryEnv(SymbolicDiscoveryEnv):
    def __init__(self, *args, reward_config: Dict = None, **kwargs):
        # Create comprehensive reward handler
        if reward_config:
            reward_handler = RewardHandler.from_config(reward_config)
        else:
            # Use all available components
            reward_handler = RewardHandler(
                reward_components={
                    "novelty": kwargs.get('novelty_weight', 0.3),
                    "complexity": kwargs.get('complexity_weight', 0.2),
                    "conservation": kwargs.get('conservation_weight', 0.4),
                    "fidelity": kwargs.get('fidelity_weight', 0.1)
                },
                normalize=True,
                clip_range=(-10.0, 10.0)
            )
        
        super().__init__(*args, reward_handler=reward_handler, **kwargs)
```

### 3. Updating PPOTrainer for Dynamic Rewards

**Before:**
```python
class PPOTrainer:
    def __init__(self, policy, env, **kwargs):
        self.policy = policy
        self.env = env
        # Fixed reward configuration
```

**After:**
```python
class PPOTrainer:
    def __init__(self, policy, env, reward_handler: RewardHandler = None, **kwargs):
        self.policy = policy
        self.env = env
        self.reward_handler = reward_handler
        
        # Inject reward handler into environment if supported
        if hasattr(env, 'set_reward_handler'):
            env.set_reward_handler(self.reward_handler)
    
    def adapt_rewards(self, metrics: Dict[str, float]):
        """Adapt reward weights based on training metrics."""
        if isinstance(self.reward_handler, AdaptiveRewardHandler):
            # Automatic adaptation
            pass
        else:
            # Manual adaptation based on metrics
            if metrics['success_rate'] < 0.3:
                # Increase exploration
                self.reward_handler.update_weights({
                    'novelty': 0.5,
                    'complexity': 0.2,
                    'conservation': 0.3
                })
```

### 4. Configuration-Based Training Scripts

**Before:**
```python
# In train_physics.py
env = SymbolicDiscoveryEnv(
    grammar=grammar,
    X_data=X,
    y_data=y,
    variables=variables,
    reward_config={
        'mse_weight': -1.0,
        'complexity_penalty': -0.01,
        'completion_bonus': 1.0
    }
)
```

**After:**
```python
# In train_physics.py
from janus.ml.rewards.reward_handler import RewardHandler
from janus.ml.rewards.reward_registry import get_preset_config

# Load configuration
reward_config = get_preset_config('physics_discovery')

# Override for specific experiment
reward_config['weights']['conservation'] = 0.6  # Emphasize conservation

# Create handler
reward_handler = RewardHandler.from_config(reward_config)

# Create environment
env = SymbolicDiscoveryEnv(
    grammar=grammar,
    X_data=X,
    y_data=y,
    variables=variables,
    reward_handler=reward_handler
)
```

### 5. Experiment Configuration Files

**Before (YAML):**
```yaml
experiment:
  name: harmonic_oscillator
  reward_config:
    mse_weight: -1.0
    complexity_penalty: -0.01
    novelty_weight: 0.2
    conservation_weight: 0.5
```

**After (YAML):**
```yaml
experiment:
  name: harmonic_oscillator
  reward_handler:
    components:
      conservation:
        law_type: energy
        weight: 1.0
      complexity:
        target_complexity: 8
        tolerance: 3
      novelty:
        novelty_threshold: 0.1
      fidelity: {}  # Use defaults
    weights:
      conservation: 0.5
      complexity: 0.2
      novelty: 0.2
      fidelity: 0.1
    normalize: true
    clip_range: [-10.0, 10.0]
```

## Integration Patterns

### Pattern 1: Environment with Configurable Rewards

```python
class ConfigurableRewardEnv(SymbolicDiscoveryEnv):
    """Environment that accepts reward configuration."""
    
    def __init__(self, *args, reward_preset: str = None, **kwargs):
        # Extract reward configuration
        reward_config = kwargs.pop('reward_config', None)
        reward_handler = kwargs.pop('reward_handler', None)
        
        # Create handler based on inputs
        if reward_handler is None:
            if reward_config:
                reward_handler = RewardHandler.from_config(reward_config)
            elif reward_preset:
                reward_handler = create_handler_from_preset(reward_preset)
            else:
                # Default handler
                reward_handler = create_handler_from_preset('balanced_exploration')
        
        super().__init__(*args, reward_handler=reward_handler, **kwargs)
```

### Pattern 2: Reward Scheduling

```python
class RewardScheduler:
    """Schedule reward weights during training."""
    
    def __init__(self, reward_handler: RewardHandler, schedule: Dict[int, Dict[str, float]]):
        self.reward_handler = reward_handler
        self.schedule = schedule
    
    def update(self, timestep: int):
        """Update weights based on schedule."""
        for milestone, weights in sorted(self.schedule.items()):
            if timestep >= milestone:
                self.reward_handler.update_weights(weights)

# Usage
scheduler = RewardScheduler(
    reward_handler=env.reward_handler,
    schedule={
        0: {'novelty': 0.6, 'complexity': 0.4},      # Early: explore
        1000: {'novelty': 0.4, 'complexity': 0.6},   # Mid: balance
        5000: {'novelty': 0.2, 'complexity': 0.8}    # Late: refine
    }
)
```

### Pattern 3: Multi-Objective Optimization

```python
class MultiObjectiveRewardHandler(RewardHandler):
    """Handler that tracks Pareto frontier of rewards."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pareto_frontier = []
    
    def calculate_detailed_reward(self, **kwargs) -> RewardInfo:
        info = super().calculate_detailed_reward(**kwargs)
        
        # Track multi-objective performance
        objectives = list(info.component_rewards.values())
        self._update_pareto_frontier(objectives)
        
        return info
```

## Best Practices

### 1. Use Presets for Standard Tasks

```python
# Physics discovery
handler = create_handler_from_preset('physics_discovery')

# AI interpretability
handler = create_handler_from_preset('ai_interpretability')

# Balanced exploration
handler = create_handler_from_preset('balanced_exploration')
```

### 2. Log Reward Details During Development

```python
handler = RewardHandler(
    reward_components={...},
    log_rewards=True  # Enable detailed logging
)

# Get detailed breakdown
reward_info = handler.calculate_detailed_reward(...)
print(f"Components: {reward_info.component_rewards}")
print(f"Weighted: {reward_info.weighted_rewards}")
```

### 3. Use Configuration Files

```python
# config.yaml
reward_handler:
  preset: physics_discovery
  overrides:
    weights:
      conservation: 0.7
      complexity: 0.2
      novelty: 0.1

# Loading
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)

handler = create_handler_from_preset(
    config['reward_handler']['preset'],
    **config['reward_handler'].get('overrides', {})
)
```

### 4. Monitor Reward Components

```python
class RewardMonitor:
    """Track reward components over time."""
    
    def __init__(self, handler: RewardHandler):
        self.handler = handler
        self.history = {name: [] for name in handler.get_component_names()}
    
    def record(self, reward_info: RewardInfo):
        for name, value in reward_info.component_rewards.items():
            self.history[name].append(value)
    
    def plot_history(self):
        # Plot component contributions over time
        pass
```

## Testing Reward Configurations

```python
def test_reward_configuration(handler: RewardHandler, test_cases: List[Dict]):
    """Test reward handler on specific cases."""
    for i, case in enumerate(test_cases):
        reward = handler.calculate_total_reward(**case)
        print(f"Test case {i}: {case['info']['expression']} -> {reward:.4f}")

# Test physics reward
test_cases = [
    {'info': {'expression': 'F = m*a', 'complexity': 3}},
    {'info': {'expression': 'E = m*c^2', 'complexity': 4}},
    {'info': {'expression': 'x^17 + sin(cos(tan(x)))', 'complexity': 20}}
]

handler = create_handler_from_preset('physics_discovery')
test_reward_configuration(handler, test_cases)
```

## Summary

The RewardHandler system provides:

1. **Centralized Control**: All reward logic in one place
2. **Flexibility**: Easy to experiment with different reward schemes
3. **Modularity**: Add/remove components without changing core code
4. **Transparency**: Clear visibility into reward calculations
5. **Reusability**: Share reward configurations across experiments

By migrating to this system, the Janus project gains a powerful tool for reward engineering that will accelerate research into both physics discovery and AI interpretability.