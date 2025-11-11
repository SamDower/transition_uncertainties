# Transition Uncertainties

A machine learning framework for training ensembles of reward models from trajectory preferences, with support for transition-level epistemic uncertainty estimation.

## Overview

This codebase enables:
- Creation of Markov Decision Processes (MDPs)
- Sampling trajectory pairs from MDPs using customizable policies
- Labeling trajectory pairs with ground truth preferences
- Training ensembles of reward models R(s,a,s') using PyTorch
- Computing epistemic uncertainty over transitions

## Project Structure

```
transition_uncertainties/
├── src/
│   ├── environments/       # MDP implementations
│   │   ├── mdp.py         # Base MDP class
│   │   └── gridworld.py   # Simple gridworld environment
│   ├── policies/          # Policy implementations
│   │   └── policy.py      # Various policy classes
│   ├── models/            # Reward model implementations
│   │   ├── reward_model.py  # Neural network R(s,a,s')
│   │   └── ensemble.py      # Ensemble training logic
│   └── utils/             # Utility functions
│       ├── trajectory.py   # Trajectory sampling
│       └── preferences.py  # Preference labeling
├── example.py             # Full pipeline demonstration
└── requirements.txt       # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the example script to see the full pipeline in action:

```bash
python example.py
```

This will:
1. Create a 5x5 GridWorld environment
2. Sample 100 trajectory pairs using a uniform random policy
3. Label pairs based on ground truth returns
4. Train an ensemble of 5 reward models
5. Evaluate the ensemble on test trajectories
6. Save the trained ensemble

## Usage

### Creating an MDP

```python
from src.environments import GridWorld

env = GridWorld(grid_size=5, discount_factor=0.99)
```

### Defining a Policy

```python
from src.policies import UniformPolicy, EpsilonGreedyPolicy

# Uniform random policy
policy = UniformPolicy(num_actions=env.get_num_actions())

# Or epsilon-greedy policy
policy = EpsilonGreedyPolicy(num_actions=env.get_num_actions(), epsilon=0.1)
```

### Sampling Trajectories

```python
from src.utils import sample_trajectory_pairs

pairs = sample_trajectory_pairs(
    mdp=env,
    policy=policy,
    num_pairs=100,
    max_steps=20
)
```

### Labeling Preferences

```python
from src.utils import label_trajectory_pairs

preferences = label_trajectory_pairs(pairs, discount_factor=0.99)
```

### Training an Ensemble

```python
from src.models import RewardModelEnsemble

ensemble = RewardModelEnsemble(
    ensemble_size=5,
    state_dim=25,  # 5x5 grid with one-hot encoding
    num_actions=4,
    hidden_dims=[64, 64]
)

ensemble.train(
    trajectory_pairs=pairs,
    preferences=preferences,
    num_epochs=50,
    bootstrap=True
)
```

### Making Predictions

```python
# Predict return with uncertainty
mean_return, std_return = ensemble.predict_returns(trajectory, return_std=True)
```

## Components

### Environments

- **MDP**: Abstract base class for Markov Decision Processes
- **GridWorld**: Simple gridworld where the agent navigates to a goal

### Policies

- **UniformPolicy**: Selects actions uniformly at random
- **EpsilonGreedyPolicy**: Epsilon-greedy with respect to Q-values
- **SoftmaxPolicy**: Boltzmann exploration with temperature
- **CustomPolicy**: Define your own policy function

### Models

- **TransitionRewardModel**: Neural network that maps (s,a,s') → R
- **RewardModelEnsemble**: Manages training and inference for multiple reward models

### Utilities

- **Trajectory**: Data structure for sequences of transitions
- **sample_trajectory**: Sample a single trajectory from an MDP
- **sample_trajectory_pairs**: Sample multiple trajectory pairs
- **label_trajectory_pairs**: Assign preferences based on returns

## Next Steps

This is the foundation for implementing novel transition epistemic uncertainty measures. The ensemble structure enables computing uncertainty estimates like:

- Variance across ensemble predictions
- Disagreement on transition rewards
- Epistemic uncertainty U(s,a,s')

You can extend this codebase by:
1. Implementing your novel uncertainty measure
2. Adding more complex MDP environments
3. Implementing different policy types
4. Comparing uncertainty methods