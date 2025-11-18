"""Randomly generated deterministic gridworld environment."""

import numpy as np
from typing import Tuple
from .mdp import MDP


class RandomDeterministicGridworldMDP(MDP):
    """
    A randomly generated deterministic gridworld environment.

    The agent navigates an NxN grid with 4 directional actions (up, down, left, right).
    The ground truth reward function is randomly generated using a multi-step procedure
    that includes sparsification, scaling, translation, and potential shaping.

    States are represented as (row, col) coordinates.
    Actions are: 0=up, 1=down, 2=left, 3=right.
    """

    def __init__(
        self,
        grid_size: int = 8,
        discount_factor: float = 0.99,
        seed: int = None
    ):
        """
        Initialize RandomDeterministicGridworldMDP.

        Args:
            grid_size: Size of the square grid (NxN).
            discount_factor: Discount factor for future rewards (gamma).
            seed: Random seed for reproducibility.
        """
        super().__init__(discount_factor)
        self.grid_size = grid_size
        self.current_state = None

        if seed is not None:
            np.random.seed(seed)

        # Generate the ground truth reward function
        self.reward_fn = self._generate_reward_function()

    def _generate_reward_function(self) -> np.ndarray:
        """
        Generate a randomly constructed reward function R(s, a).

        Returns:
            Reward matrix of shape (NxN, 4) where entry [s, a] = R(s, a).
        """
        N = self.grid_size

        # Step 1: Sample i.i.d. Gaussians (mean=0, std=1)
        R = np.random.standard_normal((N * N, 4))

        # Step 2: With 20% probability, apply sparsification
        if np.random.rand() < 0.2:
            R[R < 3] = 0

        # Step 3: With 70% probability, apply scaling
        if np.random.rand() < 0.7:
            scale_factor = np.random.uniform(0, 10)
            R = R * scale_factor

        # Step 4: With 30% probability, apply translation
        if np.random.rand() < 0.3:
            translation = np.random.uniform(0, 10)
            R = R + translation

        # Step 5: With 50% probability, apply potential shaping
        if np.random.rand() < 0.5:
            # Sample potential vector Φ
            phi = np.random.standard_normal(N * N)

            # Scale the potential vector
            phi_scale = np.random.uniform(0, 10)
            phi = phi * phi_scale

            # Translate the potential vector
            phi_translation = np.random.uniform(0, 1)
            phi = phi + phi_translation

            # Apply potential shaping: R_new(s, a, s') = R(s, a, s') + gamma*Φ(s') - Φ(s)
            # We need to compute this for all (s, a) pairs
            R_shaped = np.zeros_like(R)
            for state_idx in range(N * N):
                for action in range(4):
                    # Get the next state from the deterministic transition
                    row, col = self.index_to_state(state_idx)
                    next_row, next_col = self._get_next_state(row, col, action)
                    next_state_idx = self.state_to_index((next_row, next_col))

                    # Apply potential shaping
                    R_shaped[state_idx, action] = (
                        R[state_idx, action]
                        + self.discount_factor * phi[next_state_idx]
                        - phi[state_idx]
                    )
            R = R_shaped

        return R

    def _get_next_state(self, row: int, col: int, action: int) -> Tuple[int, int]:
        """
        Get the deterministic next state given current state and action.

        Args:
            row: Current row.
            col: Current column.
            action: Action to take (0=up, 1=down, 2=left, 3=right).

        Returns:
            Tuple of (next_row, next_col).
        """
        if action == 0:  # up
            next_row = max(0, row - 1)
            next_col = col
        elif action == 1:  # down
            next_row = min(self.grid_size - 1, row + 1)
            next_col = col
        elif action == 2:  # left
            next_row = row
            next_col = max(0, col - 1)
        elif action == 3:  # right
            next_row = row
            next_col = min(self.grid_size - 1, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        return next_row, next_col

    def reset(self) -> Tuple[int, int]:
        """
        Reset to the top-left starting position (0, 0).

        Returns:
            Initial state as (row, col) tuple.
        """
        self.current_state = (0, 0)
        return self.current_state

    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a deterministic step in the environment.

        Args:
            state: Current state as (row, col) tuple.
            action: Action to take (0=up, 1=down, 2=left, 3=right).

        Returns:
            Tuple of (next_state, reward, done).
        """
        row, col = state

        # Get the deterministic next state
        next_row, next_col = self._get_next_state(row, col, action)
        next_state = (next_row, next_col)

        # Get the reward from the reward function
        state_idx = self.state_to_index(state)
        reward = float(self.reward_fn[state_idx, action])

        # Episode never terminates in this environment
        done = False

        return next_state, reward, done

    def get_num_states(self) -> int:
        """Get the number of states (grid positions)."""
        return self.grid_size * self.grid_size

    def get_num_actions(self) -> int:
        """Get the number of actions."""
        return 4

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert state tuple to flat index."""
        row, col = state
        return row * self.grid_size + col

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index to state tuple."""
        row = index // self.grid_size
        col = index % self.grid_size
        return (row, col)
