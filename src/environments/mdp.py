"""Base class for Markov Decision Processes."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np


class MDP(ABC):
    """Abstract base class for Markov Decision Processes."""

    def __init__(self, discount_factor: float = 0.99):
        """
        Initialize MDP.

        Args:
            discount_factor: Discount factor for future rewards (gamma).
        """
        self.discount_factor = discount_factor

    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment to an initial state.

        Returns:
            Initial state.
        """
        pass

    @abstractmethod
    def step(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        """
        Take a step in the environment.

        Args:
            state: Current state.
            action: Action to take.

        Returns:
            Tuple of (next_state, reward, done).
        """
        pass

    @abstractmethod
    def get_num_states(self) -> int:
        """Get the number of states in the MDP."""
        pass

    @abstractmethod
    def get_num_actions(self) -> int:
        """Get the number of actions in the MDP."""
        pass

    def get_discount_factor(self) -> float:
        """Get the discount factor."""
        return self.discount_factor
