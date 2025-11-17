from typing import Any, Tuple, List
from abc import ABC
import numpy as np
from .mdp import MDP


class SparseSummitMDP(MDP):
    """
    8x8 deterministic gridworld MDP with:
      - start (0,0)
      - treasure at (7,7) [+10], absorbing state
      - local lure at (1,0) [+2]
      - plateau in region rows 2–4, cols 2–4 [+0.5]
      - optional trap at (6,7) [-8]
    """

    def __init__(
        self,
        discount_factor: float = 0.99,
        grid_size: int = 8,
        use_trap: bool = True,
    ):
        super().__init__(discount_factor)
        self.grid_size = grid_size
        self.use_trap = use_trap

        # Define special rewards
        self.local_lure = (1, 0)
        self.plateau_coords = [(r, c) for r in range(2, 5) for c in range(2, 5)]
        self.treasure = (7, 7)
        self.trap = (6, 7) if use_trap else None

        # Define actions: up, down, left, right
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }

        # Define start state
        self.start_state = (0, 0)
        self.state = self.start_state

    # -------------------------------
    # MDP interface implementations
    # -------------------------------

    def reset(self) -> Tuple[int, int]:
        """Reset to the start position."""
        self.state = self.start_state
        return self.state

    def step(self, state: Any, action: int) -> Tuple[Any, float, bool]:
        """Take a deterministic step in the grid."""
        row, col = state

        # If already at the absorbing treasure state, stay there forever
        if state == self.treasure:
            return self.treasure, 10.0, False  # absorbing, not terminal

        dr, dc = self.actions[action]
        next_row, next_col = row + dr, col + dc

        # Check bounds
        if not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
            next_row, next_col = row, col  # stay in place if hitting boundary

        next_state = (next_row, next_col)

        # Determine reward
        reward = 0.0
        if next_state == self.local_lure:
            reward = 2.0
        elif next_state in self.plateau_coords:
            reward = 0.5
        elif next_state == self.treasure:
            reward = 10.0
        elif self.trap and next_state == self.trap:
            reward = -8.0

        # Termination condition: only trap ends episode now
        done = self.trap and next_state == self.trap

        self.state = next_state
        return next_state, reward, done

    def get_num_states(self) -> int:
        """Total number of states"""
        return self.grid_size * self.grid_size

    def get_num_actions(self) -> int:
        """Four deterministic moves."""
        return len(self.actions)

    # -------------------------------
    # Utilities
    # -------------------------------

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to a unique index (row-major)."""
        return state[0] * self.grid_size + state[1]

    def index_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert index to (row, col)."""
        return divmod(idx, self.grid_size)

    def visualize_grid(self) -> None:
        """Print a simple ASCII grid with rewards"""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        for (r, c) in self.plateau_coords:
            grid[r, c] = "p"
        grid[self.local_lure] = "L"
        grid[self.treasure] = "T"
        if self.trap:
            grid[self.trap] = "X"
        grid[self.start_state] = "S"
        print("\n".join(" ".join(row) for row in grid))
