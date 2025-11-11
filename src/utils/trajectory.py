"""Trajectory sampling and utilities."""

from typing import List, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Transition:
    """A single transition (s, a, s', r)."""
    state: Any
    action: int
    next_state: Any
    reward: float


@dataclass
class Trajectory:
    """A trajectory consisting of a sequence of transitions."""
    transitions: List[Transition]

    def get_return(self, discount_factor: float = 0.99) -> float:
        """
        Calculate the discounted return of the trajectory.

        Args:
            discount_factor: Discount factor for future rewards.

        Returns:
            Discounted return.
        """
        total_return = 0.0
        for i, transition in enumerate(self.transitions):
            total_return += (discount_factor ** i) * transition.reward
        return total_return

    def get_undiscounted_return(self) -> float:
        """
        Calculate the undiscounted return (sum of rewards).

        Returns:
            Sum of all rewards in the trajectory.
        """
        return sum(t.reward for t in self.transitions)

    def __len__(self) -> int:
        """Get the length of the trajectory."""
        return len(self.transitions)


def sample_trajectory(
    mdp,
    policy,
    max_steps: int = 100,
    start_state: Any = None
) -> Trajectory:
    """
    Sample a single trajectory from an MDP using a policy.

    Args:
        mdp: The MDP to sample from.
        policy: Policy to use for action selection.
        max_steps: Maximum number of steps in the trajectory.
        start_state: Optional starting state (if None, uses mdp.reset()).

    Returns:
        Sampled trajectory.
    """
    if start_state is None:
        state = mdp.reset()
    else:
        state = start_state

    transitions = []

    for _ in range(max_steps):
        # Sample action from policy
        action = policy.sample_action(state)

        # Take step in environment
        next_state, reward, done = mdp.step(state, action)

        # Record transition
        transitions.append(Transition(state, action, next_state, reward))

        if done:
            break

        state = next_state

    return Trajectory(transitions)


def sample_trajectory_pairs(
    mdp,
    policy,
    num_pairs: int,
    max_steps: int = 100,
    seed: int = None
) -> List[Tuple[Trajectory, Trajectory]]:
    """
    Sample pairs of trajectories from an MDP.

    Args:
        mdp: The MDP to sample from.
        policy: Policy to use for action selection.
        num_pairs: Number of trajectory pairs to sample.
        max_steps: Maximum steps per trajectory.
        seed: Random seed for reproducibility.

    Returns:
        List of trajectory pairs.
    """
    if seed is not None:
        np.random.seed(seed)

    pairs = []
    for _ in range(num_pairs):
        traj1 = sample_trajectory(mdp, policy, max_steps)
        traj2 = sample_trajectory(mdp, policy, max_steps)
        pairs.append((traj1, traj2))

    return pairs
