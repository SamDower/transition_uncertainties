# Policy Training with Reward Ensembles

This guide explains how to train policies using learned reward models from ensembles.

## Overview

The codebase now supports training tabular Q-learning policies with flexible reward functions. This allows you to:

1. Train policies using reward ensembles (mean, optimistic, pessimistic)
2. Train policies with any custom reward function
3. Evaluate and compare policies using comprehensive metrics

## Quick Start

### 1. Train a Reward Ensemble (if not already done)

```bash
python example.py
```

This creates `reward_ensemble.pt` with a trained ensemble.

### 2. Train a Policy Using the Ensemble

```bash
python train_policy.py
```

This script:
- Loads the trained reward ensemble
- Creates a reward function from ensemble mean predictions
- Trains a Q-learning agent using this reward function
- Evaluates the trained policy
- Compares with a baseline policy trained on true rewards

## Key Components

### Tabular Q-Learning (`src/algorithms/q_learning.py`)

The `TabularQLearning` class implements Q-learning with:
- **Flexible reward functions**: Pass any `reward_fn(state, action, next_state)`
- **Epsilon-greedy exploration**: Configurable exploration rate with decay
- **Built-in evaluation**: Comprehensive metrics for policy evaluation
- **Save/Load**: Persist Q-tables and training statistics

#### Example Usage

```python
from src.algorithms import TabularQLearning
from src.utils import create_ensemble_mean_reward_fn

# Create reward function from ensemble
reward_fn = create_ensemble_mean_reward_fn(ensemble, state_dim, device)

# Create Q-learning agent
q_learner = TabularQLearning(
    mdp=env,
    reward_fn=reward_fn,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Train
q_learner.train(num_episodes=2000, max_steps_per_episode=100)

# Evaluate
metrics = q_learner.evaluate(num_episodes=100, use_greedy=True)
```

### Reward Function Utilities (`src/utils/reward_functions.py`)

Multiple reward function creators are available:

#### 1. Ensemble Mean (Default)
```python
reward_fn = create_ensemble_mean_reward_fn(ensemble, state_dim, device)
# Returns: mean(ensemble predictions)
```

#### 2. Optimistic (Exploration Bonus)
```python
reward_fn = create_ensemble_optimistic_reward_fn(
    ensemble, state_dim, optimism_factor=1.0, device=device
)
# Returns: mean + optimism_factor * std
# Encourages exploration in uncertain regions
```

#### 3. Pessimistic (Conservative)
```python
reward_fn = create_ensemble_pessimistic_reward_fn(
    ensemble, state_dim, pessimism_factor=1.0, device=device
)
# Returns: mean - pessimism_factor * std
# Conservative approach, avoids uncertain regions
```

#### 4. True Environment Rewards
```python
reward_fn = create_true_reward_fn(env)
# For baseline comparisons
```

## Evaluation Metrics

The `evaluate()` method returns comprehensive metrics:

```python
{
    'mean_return': float,      # Average discounted return
    'std_return': float,       # Standard deviation of returns
    'min_return': float,       # Minimum return observed
    'max_return': float,       # Maximum return observed
    'mean_length': float,      # Average episode length
    'std_length': float,       # Std dev of episode lengths
    'success_rate': float,     # Fraction reaching goal (treasure)
    'num_episodes': int        # Number of evaluation episodes
}
```

## Customizing Training

### Training Parameters

```python
q_learner = TabularQLearning(
    mdp=env,
    reward_fn=reward_fn,
    learning_rate=0.1,        # Alpha: Q-value update rate
    discount_factor=0.99,     # Gamma: future reward weight
    epsilon=1.0,              # Initial exploration rate
    epsilon_decay=0.995,      # Multiplicative decay per episode
    epsilon_min=0.01,         # Minimum exploration rate
    seed=42                   # For reproducibility
)

q_learner.train(
    num_episodes=2000,        # Number of training episodes
    max_steps_per_episode=100, # Max steps before episode terminates
    eval_interval=200,        # Print stats every N episodes
    verbose=True              # Print training progress
)
```

### Custom Reward Functions

You can create any custom reward function:

```python
def my_custom_reward_fn(state, action, next_state):
    """Custom reward logic."""
    # Your logic here
    return reward

q_learner = TabularQLearning(mdp=env, reward_fn=my_custom_reward_fn, ...)
```

## Experiments with Modified Rewards

The flexible architecture makes it easy to experiment:

### Example 1: Uncertainty-Aware Exploration

```python
# Try different optimism factors
for optimism in [0.5, 1.0, 2.0]:
    reward_fn = create_ensemble_optimistic_reward_fn(
        ensemble, state_dim, optimism_factor=optimism
    )
    q_learner = TabularQLearning(mdp=env, reward_fn=reward_fn, ...)
    q_learner.train(...)
    metrics = q_learner.evaluate(...)
    print(f"Optimism {optimism}: success rate = {metrics['success_rate']}")
```

### Example 2: Conservative vs Optimistic

```python
# Compare pessimistic and optimistic approaches
reward_fns = {
    'pessimistic': create_ensemble_pessimistic_reward_fn(ensemble, state_dim),
    'mean': create_ensemble_mean_reward_fn(ensemble, state_dim),
    'optimistic': create_ensemble_optimistic_reward_fn(ensemble, state_dim)
}

results = {}
for name, reward_fn in reward_fns.items():
    q_learner = TabularQLearning(mdp=env, reward_fn=reward_fn, ...)
    q_learner.train(...)
    results[name] = q_learner.evaluate(...)
```

### Example 3: Hybrid Reward Functions

```python
def hybrid_reward_fn(state, action, next_state):
    """Combine ensemble prediction with domain knowledge."""
    ensemble_reward = ensemble_mean_fn(state, action, next_state)

    # Add domain-specific bonus/penalty
    if next_state == env.treasure:
        ensemble_reward += 5.0  # Extra treasure bonus

    return ensemble_reward
```

## Files Created

- **`src/algorithms/q_learning.py`**: Tabular Q-learning implementation
- **`src/algorithms/__init__.py`**: Algorithm module exports
- **`src/utils/reward_functions.py`**: Reward function utilities
- **`train_policy.py`**: Example training script

## Saved Outputs

Running `train_policy.py` creates:
- `q_learning_ensemble_reward.npy`: Policy trained with ensemble rewards
- `q_learning_true_reward.npy`: Baseline policy trained with true rewards

Load saved policies:
```python
q_learner = TabularQLearning(mdp=env, reward_fn=reward_fn, ...)
q_learner.load('q_learning_ensemble_reward.npy')
```

## Tips for Experimentation

1. **Start with baseline**: Always compare against true reward policy
2. **Tune exploration**: Adjust epsilon decay for your environment
3. **Monitor convergence**: Check if Q-table size stabilizes
4. **Multiple seeds**: Run with different seeds for robust comparisons
5. **Reward scaling**: Normalize rewards if they have very different scales

## Next Steps

- Experiment with different reward function modifications
- Try different ensemble sizes or architectures
- Implement value iteration for comparison
- Add policy visualization tools
- Experiment with different exploration strategies
