"""Simple gridworld environment."""

import numpy as np
from typing import Tuple
from .mdp import MDP


class GridWorldMDP(MDP):
    """
    A simple gridworld environment.

    The agent starts at a random position and must navigate to a goal.
    States are represented as (x, y) coordinates.
    Actions are: 0=up, 1=down, 2=left, 3=right.
    """

    def __init__(
        self,
        grid_size: int = 5,
        discount_factor: float = 0.99,
        step_reward: float = -0.1,
        goal_reward: float = 1.0,
        wall_penalty: float = -0.5,
        seed: int = None
    ):
        """
        Initialize GridWorld.

        Args:
            grid_size: Size of the square grid.
            discount_factor: Discount factor for future rewards.
            step_reward: Reward for each step taken.
            goal_reward: Reward for reaching the goal.
            wall_penalty: Penalty for hitting a wall.
            seed: Random seed for reproducibility.
        """
        super().__init__(discount_factor)
        self.grid_size = grid_size
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty

        if seed is not None:
            np.random.seed(seed)

        # Goal is at top-right corner
        self.goal_state = (grid_size - 1, grid_size - 1)

        # Current state
        self.current_state = None

    def reset(self) -> Tuple[int, int]:
        """
        Reset to a random starting position (not the goal).

        Returns:
            Initial state as (x, y) tuple.
        """
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) != self.goal_state:
                self.current_state = (x, y)
                return self.current_state

    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment.

        Args:
            state: Current state as (x, y) tuple.
            action: Action to take (0=up, 1=down, 2=left, 3=right).

        Returns:
            Tuple of (next_state, reward, done).
        """
        x, y = state

        # Determine next position based on action
        if action == 0:  # up
            next_x, next_y = x, max(0, y - 1)
        elif action == 1:  # down
            next_x, next_y = x, min(self.grid_size - 1, y + 1)
        elif action == 2:  # left
            next_x, next_y = max(0, x - 1), y
        elif action == 3:  # right
            next_x, next_y = min(self.grid_size - 1, x + 1), y
        else:
            raise ValueError(f"Invalid action: {action}")

        next_state = (next_x, next_y)

        # Determine reward
        if next_state == state:  # Hit a wall
            reward = self.wall_penalty
        elif next_state == self.goal_state:  # Reached goal
            reward = self.goal_reward
        else:  # Normal step
            reward = self.step_reward

        # Check if episode is done
        done = (next_state == self.goal_state)

        return next_state, reward, done

    def get_num_states(self) -> int:
        """Get the number of states (grid positions)."""
        return self.grid_size * self.grid_size

    def get_num_actions(self) -> int:
        """Get the number of actions."""
        return 4

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert state tuple to flat index."""
        x, y = state
        return y * self.grid_size + x

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index to state tuple."""
        y = index // self.grid_size
        x = index % self.grid_size
        return (x, y)
