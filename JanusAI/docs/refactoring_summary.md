# SymbolicDiscoveryEnv Refactoring Summary

## Overview

We've refactored `SymbolicDiscoveryEnv` to separate input features (`X_data`) and target values (`y_data`), eliminating the confusing pattern of combining them in a single `target_data` array.

## Key Changes

### 1. Constructor Signature

**Before:**
```python
def __init__(self, grammar, target_data, variables, target_variable_index=None, ...):
    # target_data contains both X and y mixed together
    # Need to track which column is the target
```

**After:**
```python  
def __init__(self, grammar, X_data, y_data, variables, ...):
    # Clean separation - X_data for inputs, y_data for outputs
    # No index tracking needed
```

### 2. Data Handling

**Before:**
```python
# Split combined data internally
X_data = np.delete(self.target_data, self.target_variable_index, axis=1)
y_true = self.target_data[:, self.target_variable_index]
```

**After:**
```python
# Direct access to separated data
X = self.X_data
y = self.y_data[:, 0]  # Explicit handling of multi-output
```

### 3. Input Validation

**Before:**
- Check `target_data` is 2D
- Validate `target_variable_index` is within bounds
- Complex logic for multi-output cases

**After:**
- Validate `X_data` and `y_data` shapes separately
- Check sample counts match
- Verify variables match X features
- Clear handling of 1D/2D y_data

## Benefits

### 1. **Clarity and Readability**
- No confusion about data layout
- Self-documenting parameter names
- Easier to understand for new contributors

### 2. **Simplified Child Classes**
```python
# AI Interpretability Environment - much cleaner!
class AIInterpretabilityEnv(SymbolicDiscoveryEnv):
    def __init__(self, ai_model, behavior_data, ...):
        super().__init__(
            grammar=grammar,
            X_data=behavior_data.inputs,      # Clear!
            y_data=behavior_data.outputs,     # Simple!
            variables=self._extract_variables()
        )
```

### 3. **Better Error Messages**
- "X_data and y_data must have same number of samples" vs
- "Invalid target_variable_index: 5. Must be within bounds [0, 4]"

### 4. **Easier Testing**
```python
# Test data setup is more intuitive
env = SymbolicDiscoveryEnv(
    grammar=grammar,
    X_data=np.array([[1, 2], [3, 4]]),
    y_data=np.array([5, 6]),
    variables=[Variable('x1', 0), Variable('x2', 1)]
)
```

### 5. **Multi-Output Support**
- Explicit handling of multi-output scenarios
- No ambiguity about which outputs to use
- Easy to select specific outputs

## Migration Path

### For Environment Developers

1. Update constructor calls to use `X_data` and `y_data`
2. Remove `target_variable_index` logic
3. Update any `_evaluate_expression` overrides
4. Simplify data preparation code

### For Users

1. Split your combined data before creating environments:
   ```python
   # Old
   env = SymbolicDiscoveryEnv(grammar, combined_data, vars, target_index)
   
   # New  
   X = combined_data[:, :target_index]
   y = combined_data[:, target_index]
   env = SymbolicDiscoveryEnv(grammar, X, y, vars)
   ```

2. Or use the compatibility wrapper during transition

## Example Use Cases

### Physics Discovery
```python
# Newton's Second Law: F = ma
X_data = np.column_stack([mass_data, acceleration_data])
y_data = force_data
variables = [Variable('m', 0), Variable('a', 1)]

env = SymbolicDiscoveryEnv(grammar, X_data, y_data, variables)
```

### Neural Network Interpretation
```python
# Explain neural network: output = f(inputs)
X_data = neural_inputs
y_data = neural_outputs
variables = [Variable(f'x{i}', i) for i in range(input_dim)]

env = SymbolicDiscoveryEnv(grammar, X_data, y_data, variables)
```

### Time Series Prediction
```python
# Predict future from past: y(t) = f(y(t-1), y(t-2), ...)
X_data = np.column_stack([y_lag1, y_lag2, y_lag3])
y_data = y_current
variables = [Variable(f'y_t-{i}', i-1) for i in range(1, 4)]

env = SymbolicDiscoveryEnv(grammar, X_data, y_data, variables)
```

## Conclusion

This refactoring makes `SymbolicDiscoveryEnv` cleaner, more intuitive, and easier to extend. The separation of concerns between input features and target values eliminates a common source of bugs and confusion, leading to more maintainable code throughout the Janus project.