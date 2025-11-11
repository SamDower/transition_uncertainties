"""Preference labeling utilities."""

from typing import List, Tuple
import numpy as np
from .trajectory import Trajectory


def label_trajectory_pair(
    traj1: Trajectory,
    traj2: Trajectory,
    discount_factor: float = 0.99,
    use_undiscounted: bool = False
) -> int:
    """
    Label a trajectory pair based on ground truth returns.

    Args:
        traj1: First trajectory.
        traj2: Second trajectory.
        discount_factor: Discount factor for computing returns.
        use_undiscounted: If True, use undiscounted returns instead.

    Returns:
        Preference label: 0 if traj1 is preferred, 1 if traj2 is preferred.
    """
    if use_undiscounted:
        return1 = traj1.get_undiscounted_return()
        return2 = traj2.get_undiscounted_return()
    else:
        return1 = traj1.get_return(discount_factor)
        return2 = traj2.get_return(discount_factor)

    # Return 0 if traj1 is better, 1 if traj2 is better
    return 0 if return1 >= return2 else 1


def label_trajectory_pairs(
    pairs: List[Tuple[Trajectory, Trajectory]],
    discount_factor: float = 0.99,
    use_undiscounted: bool = False
) -> np.ndarray:
    """
    Label multiple trajectory pairs.

    Args:
        pairs: List of trajectory pairs.
        discount_factor: Discount factor for computing returns.
        use_undiscounted: If True, use undiscounted returns instead.

    Returns:
        Array of preference labels (0 or 1 for each pair).
    """
    labels = []
    for traj1, traj2 in pairs:
        label = label_trajectory_pair(traj1, traj2, discount_factor, use_undiscounted)
        labels.append(label)

    return np.array(labels)


def get_trajectory_statistics(trajectories: List[Trajectory]) -> dict:
    """
    Get statistics about a set of trajectories.

    Args:
        trajectories: List of trajectories.

    Returns:
        Dictionary with statistics.
    """
    lengths = [len(traj) for traj in trajectories]
    returns = [traj.get_undiscounted_return() for traj in trajectories]

    return {
        'num_trajectories': len(trajectories),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
    }
