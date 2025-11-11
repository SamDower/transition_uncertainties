"""PyTorch reward model R(s, a, s')."""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple


class TransitionRewardModel(nn.Module):
    """
    Neural network reward model R(s, a, s').

    Takes a transition (state, action, next_state) and outputs a scalar reward.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: list = [64, 64],
        activation: str = 'relu'
    ):
        """
        Initialize the reward model.

        Args:
            state_dim: Dimension of the state representation.
            num_actions: Number of possible actions.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function ('relu', 'tanh', 'elu').
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions

        # Build network layers
        layers = []
        input_dim = state_dim + num_actions + state_dim  # s, a (one-hot), s'

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            input_dim = hidden_dim

        # Output layer (single scalar reward)
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards for transitions.

        Args:
            states: Batch of states [batch_size, state_dim].
            actions: Batch of actions [batch_size] (integer indices).
            next_states: Batch of next states [batch_size, state_dim].

        Returns:
            Predicted rewards [batch_size, 1].
        """
        # One-hot encode actions
        actions_onehot = torch.nn.functional.one_hot(
            actions.long(),
            num_classes=self.num_actions
        ).float()

        # Concatenate (s, a, s')
        transition_input = torch.cat([states, actions_onehot, next_states], dim=-1)

        # Pass through network
        rewards = self.network(transition_input)

        return rewards

    def predict_trajectory_return(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        discount_factor: float = 0.99
    ) -> torch.Tensor:
        """
        Predict the discounted return of a trajectory.

        Args:
            states: Trajectory states [seq_len, state_dim].
            actions: Trajectory actions [seq_len].
            next_states: Trajectory next states [seq_len, state_dim].
            discount_factor: Discount factor.

        Returns:
            Predicted discounted return (scalar).
        """
        with torch.no_grad():
            # Get rewards for all transitions
            rewards = self.forward(states, actions, next_states).squeeze(-1)

            # Compute discounted return
            discounts = torch.tensor(
                [discount_factor ** i for i in range(len(rewards))],
                dtype=torch.float32,
                device=rewards.device
            )

            return torch.sum(rewards * discounts)


def state_to_tensor(state: Any, state_dim: int, device: str = 'cpu') -> torch.Tensor:
    """
    Convert state to tensor representation.

    For gridworld, this converts (x, y) tuples to one-hot or normalized vectors.

    Args:
        state: State representation.
        state_dim: Dimension of state tensor.
        device: Device to place tensor on ('cpu' or 'cuda').

    Returns:
        State tensor.
    """
    if isinstance(state, tuple) and len(state) == 2:
        # Gridworld state (x, y) -> normalized coordinates
        # Assume square grid, infer grid size from state_dim
        # For one-hot: state_dim = grid_size^2
        # For normalized: state_dim = 2
        if state_dim == 2:
            # Normalized representation
            # Need to know grid size - for now assume it's in range [0, sqrt(state_dim))
            return torch.tensor(state, dtype=torch.float32, device=device)
        else:
            # One-hot representation
            grid_size = int(np.sqrt(state_dim))
            x, y = state
            index = y * grid_size + x
            one_hot = torch.zeros(state_dim, dtype=torch.float32, device=device)
            one_hot[index] = 1.0
            return one_hot
    elif isinstance(state, np.ndarray):
        return torch.tensor(state, dtype=torch.float32, device=device)
    elif isinstance(state, torch.Tensor):
        return state.float().to(device)
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")


def trajectory_to_tensors(
    trajectory,
    state_dim: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a trajectory to tensor format.

    Args:
        trajectory: Trajectory object.
        state_dim: Dimension of state representation.
        device: Device to place tensors on ('cpu' or 'cuda').

    Returns:
        Tuple of (states, actions, next_states) tensors.
    """
    states = []
    actions = []
    next_states = []

    for transition in trajectory.transitions:
        states.append(state_to_tensor(transition.state, state_dim, device))
        actions.append(transition.action)
        next_states.append(state_to_tensor(transition.next_state, state_dim, device))

    states_tensor = torch.stack(states)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    next_states_tensor = torch.stack(next_states)

    return states_tensor, actions_tensor, next_states_tensor
