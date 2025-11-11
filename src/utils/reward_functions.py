"""Utility functions for creating reward functions from ensembles and other sources."""

import torch
import numpy as np
from typing import Any, Callable
from ..models.ensemble import RewardModelEnsemble


def create_ensemble_mean_reward_fn(
    ensemble: RewardModelEnsemble,
    state_dim: int,
    device: str = 'cpu'
) -> Callable[[Any, int, Any], float]:
    """
    Create a reward function that returns the mean prediction from an ensemble.

    Args:
        ensemble: Trained reward model ensemble.
        state_dim: Dimension of state representation (for one-hot encoding).
        device: Device to run inference on.

    Returns:
        Reward function that takes (state, action, next_state) and returns mean ensemble reward.
    """
    def reward_fn(state: Any, action: int, next_state: Any) -> float:
        """Compute mean ensemble reward for a transition."""
        # Convert state to one-hot encoding
        state_row, state_col = state
        state_onehot = np.zeros(state_dim)
        state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

        next_state_row, next_state_col = next_state
        next_state_onehot = np.zeros(state_dim)
        next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

        # Convert to tensors
        state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
        action_tensor = torch.LongTensor([action]).to(device)

        # Get predictions from each model in ensemble
        predictions = []
        for model in ensemble.models:
            model.eval()
            with torch.no_grad():
                reward_pred = model(state_tensor, action_tensor, next_state_tensor)
                predictions.append(reward_pred.item())

        # Return mean prediction
        return np.mean(predictions)

    return reward_fn


def create_ensemble_optimistic_reward_fn(
    ensemble: RewardModelEnsemble,
    state_dim: int,
    optimism_factor: float = 1.0,
    device: str = 'cpu'
) -> Callable[[Any, int, Any], float]:
    """
    Create a reward function that adds optimism (UCB-style exploration bonus).

    Reward = mean(ensemble) + optimism_factor * std(ensemble)

    Args:
        ensemble: Trained reward model ensemble.
        state_dim: Dimension of state representation.
        optimism_factor: Weight for the exploration bonus (typically 0.5-2.0).
        device: Device to run inference on.

    Returns:
        Reward function with optimistic exploration bonus.
    """
    def reward_fn(state: Any, action: int, next_state: Any) -> float:
        """Compute optimistic ensemble reward for a transition."""
        # Convert state to one-hot encoding
        state_row, state_col = state
        state_onehot = np.zeros(state_dim)
        state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

        next_state_row, next_state_col = next_state
        next_state_onehot = np.zeros(state_dim)
        next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

        # Convert to tensors
        state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
        action_tensor = torch.LongTensor([action]).to(device)

        # Get predictions from each model in ensemble
        predictions = []
        for model in ensemble.models:
            model.eval()
            with torch.no_grad():
                reward_pred = model(state_tensor, action_tensor, next_state_tensor)
                predictions.append(reward_pred.item())

        # Return mean + exploration bonus
        mean_reward = np.mean(predictions)
        std_reward = np.std(predictions)
        return mean_reward + optimism_factor * std_reward

    return reward_fn


def create_ensemble_pessimistic_reward_fn(
    ensemble: RewardModelEnsemble,
    state_dim: int,
    pessimism_factor: float = 1.0,
    device: str = 'cpu'
) -> Callable[[Any, int, Any], float]:
    """
    Create a reward function that is pessimistic (conservative).

    Reward = mean(ensemble) - pessimism_factor * std(ensemble)

    Args:
        ensemble: Trained reward model ensemble.
        state_dim: Dimension of state representation.
        pessimism_factor: Weight for the pessimism penalty.
        device: Device to run inference on.

    Returns:
        Reward function with pessimistic penalty.
    """
    def reward_fn(state: Any, action: int, next_state: Any) -> float:
        """Compute pessimistic ensemble reward for a transition."""
        # Convert state to one-hot encoding
        state_row, state_col = state
        state_onehot = np.zeros(state_dim)
        state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

        next_state_row, next_state_col = next_state
        next_state_onehot = np.zeros(state_dim)
        next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

        # Convert to tensors
        state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
        action_tensor = torch.LongTensor([action]).to(device)

        # Get predictions from each model in ensemble
        predictions = []
        for model in ensemble.models:
            model.eval()
            with torch.no_grad():
                reward_pred = model(state_tensor, action_tensor, next_state_tensor)
                predictions.append(reward_pred.item())

        # Return mean - pessimism penalty
        mean_reward = np.mean(predictions)
        std_reward = np.std(predictions)
        return mean_reward - pessimism_factor * std_reward

    return reward_fn


def create_true_reward_fn(mdp: Any) -> Callable[[Any, int, Any], float]:
    """
    Create a reward function that uses the true environment rewards.

    This is useful for comparison with learned reward functions.

    Args:
        mdp: The MDP environment.

    Returns:
        Reward function that returns true environment rewards.
    """
    def reward_fn(state: Any, action: int, next_state: Any) -> float:
        """Get true reward from environment."""
        # Temporarily set state and take action to get reward
        original_state = mdp.state
        mdp.state = state
        _, reward, _ = mdp.step(state, action)
        mdp.state = original_state
        return reward

    return reward_fn
