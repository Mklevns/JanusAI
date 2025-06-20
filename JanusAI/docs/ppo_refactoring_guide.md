# PPO Trainer Refactoring Guide

## Overview

The PPO trainer has been refactored to separate data collection and learning phases, following modern RL library patterns (e.g., Stable Baselines3, RLlib). This decoupling provides better modularity, flexibility, and code reusability.

## Key Changes

### 1. Separated Concerns

**Before:** Monolithic `train` method handled everything
```python
def train(self, total_timesteps, ...):
    # Everything mixed together
    for update in range(n_updates):
        # Collect data
        # Train on data
        # Log and checkpoint
```

**After:** Clean separation of responsibilities
```python
def collect_rollouts(self, n_steps) -> Dict[str, Tensor]:
    # Only handles environment interaction
    
def learn(self, rollout_data) -> Dict[str, float]:
    # Only handles training on collected data
    
def train(self, total_timesteps):
    # Orchestrates collection and learning
```

### 2. Benefits of Decoupling

#### **Flexibility**
- Can collect data without training
- Can train on pre-collected data
- Easy to implement different collection strategies

#### **Testing**
- Test collection and learning independently
- Mock one phase while testing the other

#### **Advanced Use Cases**
- Distributed collection with centralized learning
- Asynchronous data collection
- Experience replay from disk

## Usage Examples

### Basic Training Loop

```python
# Standard training - works the same as before
trainer = PPOTrainer(
    policy=policy,
    env=env,
    learning_rate=3e-4,
    n_epochs=10,
    batch_size=64
)

# Simple training call
history = trainer.train(
    total_timesteps=100000,
    rollout_length=2048,
    log_interval=10
)
```

### Custom Training Loop

```python
# Now you can create custom training loops!
trainer = PPOTrainer(policy, env)

# Collect more data before training
all_rollouts = []
for i in range(5):
    rollout_data = trainer.collect_rollouts(n_steps=1000)
    all_rollouts.append(rollout_data)

# Train on combined data
combined_data = combine_rollouts(all_rollouts)
metrics = trainer.learn(combined_data)
```

### Distributed Collection

```python
# Collect data on multiple environments in parallel
envs = [create_env() for _ in range(4)]
trainers = [PPOTrainer(policy, env) for env in envs]

# Parallel collection
rollouts = []
for trainer in trainers:
    rollout = trainer.collect_rollouts(n_steps=512)
    rollouts.append(rollout)

# Centralized learning
combined_data = combine_rollouts(rollouts)
metrics = trainers[0].learn(combined_data)
```

### Curriculum Learning Integration

```python
class CurriculumPPOTrainer(PPOTrainer):
    def collect_rollouts(self, n_steps, difficulty_level=1):
        # Adjust environment difficulty
        self.env.set_difficulty(difficulty_level)
        
        # Collect with current difficulty
        return super().collect_rollouts(n_steps)
    
    def train(self, total_timesteps, **kwargs):
        difficulty = 1
        
        while self.total_timesteps < total_timesteps:
            # Gradually increase difficulty
            if self.total_timesteps > total_timesteps * 0.3:
                difficulty = 2
            if self.total_timesteps > total_timesteps * 0.6:
                difficulty = 3
            
            # Collect at current difficulty
            rollout_data = self.collect_rollouts(
                n_steps=2048,
                difficulty_level=difficulty
            )
            
            # Learn from collected data
            metrics = self.learn(rollout_data)
            
            # Log progress
            print(f"Timesteps: {self.total_timesteps}, "
                  f"Difficulty: {difficulty}, "
                  f"Reward: {np.mean(self.episode_rewards):.2f}")
```

### Experience Replay from Disk

```python
# Save rollouts for later use
def save_rollouts_to_disk(trainer, n_rollouts=10):
    for i in range(n_rollouts):
        rollout_data = trainer.collect_rollouts(n_steps=2048)
        torch.save(rollout_data, f"rollout_{i}.pt")

# Train from saved rollouts
def train_from_saved_rollouts(trainer, rollout_files):
    for epoch in range(5):
        for file in rollout_files:
            rollout_data = torch.load(file)
            metrics = trainer.learn(rollout_data)
            print(f"Epoch {epoch}, File {file}: Loss={metrics['loss']:.4f}")
```

### Meta-Learning with Task-Specific Collection

```python
class MetaLearningPPOTrainer(PPOTrainer):
    def collect_task_specific_rollouts(self, task, n_steps):
        # Set task in environment
        self.env.set_task(task)
        
        # Get task representation
        task_trajectories = self.get_task_trajectories(task)
        
        # Collect rollouts for this task
        return self.collect_rollouts(
            n_steps=n_steps,
            task_trajectories=task_trajectories
        )
    
    def meta_train(self, tasks, steps_per_task=1000):
        for epoch in range(10):
            for task in tasks:
                # Collect data for task
                rollout_data = self.collect_task_specific_rollouts(
                    task, steps_per_task
                )
                
                # Learn from task data
                metrics = self.learn(rollout_data)
                
                print(f"Task {task.name}: {metrics}")
```

## Implementation Details

### RolloutBuffer Improvements

The `RolloutBuffer` class now:
- Has a clean interface with `add()`, `reset()`, and `get()` methods
- Handles advantage computation internally
- Supports variable-length episodes
- Returns data in a format ready for training

### Device Management

The refactored trainer properly handles device placement:
```python
# Automatic device selection
self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move data to device in learn()
for key, value in rollout_data.items():
    if isinstance(value, torch.Tensor):
        rollout_data[key] = value.to(self.device)
```

### Checkpointing

Improved checkpointing with the decoupled design:
```python
# Save after learning phase
if self.n_updates % save_interval == 0:
    self.save_checkpoint()

# Checkpoint includes all necessary state
checkpoint_data = {
    'policy_state_dict': self.policy.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'total_timesteps': self.total_timesteps,
    'n_updates': self.n_updates,
    # ... other state
}
```

## Migration Guide

### For Basic Users

If you're using the standard training loop, no changes needed:
```python
# This still works exactly the same
trainer.train(total_timesteps=100000)
```

### For Advanced Users

Take advantage of the new flexibility:
```python
# Old way - had to modify train() method
class CustomPPOTrainer(PPOTrainer):
    def train(self, ...):
        # Override entire training loop
        
# New way - just override what you need
class CustomPPOTrainer(PPOTrainer):
    def collect_rollouts(self, n_steps):
        # Custom collection logic
        return super().collect_rollouts(n_steps)
```

### API Compatibility

The refactored trainer maintains backward compatibility:
- All existing parameters work the same
- The `train()` method signature is unchanged
- Default behavior is identical

## Testing

The decoupled design makes testing much easier:

```python
def test_collection():
    """Test rollout collection independently"""
    trainer = PPOTrainer(mock_policy, mock_env)
    rollout_data = trainer.collect_rollouts(n_steps=100)
    
    assert len(rollout_data['observations']) == 100
    assert 'actions' in rollout_data
    assert 'rewards' in rollout_data

def test_learning():
    """Test learning independently"""
    trainer = PPOTrainer(policy, mock_env)
    
    # Create fake rollout data
    fake_data = create_fake_rollout_data(n_samples=1000)
    
    # Test learning
    metrics = trainer.learn(fake_data)
    
    assert metrics['loss'] > 0
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
```

## Performance Considerations

The refactored design maintains performance while adding flexibility:

1. **No overhead**: The standard training loop performs identically
2. **Memory efficient**: RolloutBuffer only stores what's needed
3. **Batch processing**: Efficient mini-batch creation
4. **Device handling**: Automatic GPU utilization when available

## Conclusion

The refactored PPO trainer provides:
- **Cleaner code**: Separated concerns are easier to understand
- **More flexibility**: Custom training loops are now possible
- **Better testing**: Components can be tested independently
- **Extensibility**: Easy to add new features without modifying core logic

This design follows best practices from modern RL libraries while maintaining the simplicity of the original interface.