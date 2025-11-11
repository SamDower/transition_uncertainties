"""Policy classes for action selection."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """
        Get the probability distribution over actions for a given state.

        Args:
            state: The current state.

        Returns:
            Array of action probabilities.
        """
        pass

    def sample_action(self, state: Any) -> int:
        """
        Sample an action from the policy distribution.

        Args:
            state: The current state.

        Returns:
            Sampled action.
        """
        probs = self.get_action_probabilities(state)
        return np.random.choice(len(probs), p=probs)


class UniformPolicy(Policy):
    """Policy that selects actions uniformly at random."""

    def __init__(self, num_actions: int):
        """
        Initialize uniform policy.

        Args:
            num_actions: Number of available actions.
        """
        self.num_actions = num_actions

    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """Get uniform distribution over actions."""
        return np.ones(self.num_actions) / self.num_actions


class EpsilonGreedyPolicy(Policy):
    """Epsilon-greedy policy with respect to a Q-function."""

    def __init__(self, num_actions: int, epsilon: float = 0.1):
        """
        Initialize epsilon-greedy policy.

        Args:
            num_actions: Number of available actions.
            epsilon: Probability of taking a random action.
        """
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.q_values = {}  # state -> action values

    def set_q_values(self, state: Any, q_values: np.ndarray):
        """
        Set Q-values for a state.

        Args:
            state: The state.
            q_values: Array of Q-values for each action.
        """
        self.q_values[state] = q_values

    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """Get epsilon-greedy action probabilities."""
        probs = np.ones(self.num_actions) * (self.epsilon / self.num_actions)

        if state in self.q_values:
            best_action = np.argmax(self.q_values[state])
            probs[best_action] += (1.0 - self.epsilon)
        else:
            # If no Q-values, use uniform distribution
            probs = np.ones(self.num_actions) / self.num_actions

        return probs


class SoftmaxPolicy(Policy):
    """Softmax (Boltzmann) policy with temperature parameter."""

    def __init__(self, num_actions: int, temperature: float = 1.0):
        """
        Initialize softmax policy.

        Args:
            num_actions: Number of available actions.
            temperature: Temperature parameter (higher = more random).
        """
        self.num_actions = num_actions
        self.temperature = temperature
        self.q_values = {}  # state -> action values

    def set_q_values(self, state: Any, q_values: np.ndarray):
        """
        Set Q-values for a state.

        Args:
            state: The state.
            q_values: Array of Q-values for each action.
        """
        self.q_values[state] = q_values

    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """Get softmax action probabilities."""
        if state in self.q_values:
            scaled_values = self.q_values[state] / self.temperature
            # Subtract max for numerical stability
            exp_values = np.exp(scaled_values - np.max(scaled_values))
            probs = exp_values / np.sum(exp_values)
        else:
            # If no Q-values, use uniform distribution
            probs = np.ones(self.num_actions) / self.num_actions

        return probs


class CustomPolicy(Policy):
    """Custom policy defined by a user-provided function."""

    def __init__(self, policy_fn):
        """
        Initialize custom policy.

        Args:
            policy_fn: Function that takes a state and returns action probabilities.
        """
        self.policy_fn = policy_fn

    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """Get action probabilities from custom function."""
        return self.policy_fn(state)
