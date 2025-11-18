import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for headless environments
import matplotlib.pyplot as plt
from src.environments import MDP
from src.models import RewardModelEnsemble, CanonicalizedRewardEnsemble
from src.utils import Trajectory, Transition

def plot_ensemble_gridworld_deterministic(
    mdp: MDP,
    ensemble: RewardModelEnsemble,
    canonicalized_ensemble: CanonicalizedRewardEnsemble = None,
    cmap='viridis',
    save_path: str = "figures/heatmaps.png"
):
    """
    Plot reward heatmaps for each action.

    Shows:
        Row 0: Ground truth rewards
        Row 1: Ensemble mean rewards
        Row 2: Ensemble std rewards
        Row 3 (optional): Canonicalized ensemble mean rewards
        Row 4 (optional): Canonicalized ensemble std rewards

    Args:
        mdp: The MDP environment.
        ensemble: The trained RewardModelEnsemble.
        canonicalized_ensemble: Optional CanonicalizedRewardEnsemble for comparison.
        cmap: matplotlib colormap name.
        save_path: Path to save the figure (default: "figures/heatmaps.png").
    """
    grid_size = mdp.grid_size
    num_rows = 5 if canonicalized_ensemble is not None else 3
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 3 * num_rows + 1))

    # Get a copy of the chosen colormap and set NaN color to black
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='black')

    gt_reward_table = np.zeros((grid_size*grid_size, mdp.get_num_actions()))
    for s in range(grid_size*grid_size):
        state = mdp.index_to_state(s)
        for a in range(mdp.get_num_actions()):
            # For stochastic MDPs, compute expected reward; for deterministic, just sample
            if hasattr(mdp, 'get_all_transitions'):
                # Stochastic: compute expected reward over all transitions
                all_transitions = mdp.get_all_transitions()
                expected_reward = 0.0
                for s_curr, a_curr, next_st, prob in all_transitions:
                    if s_curr == state and a_curr == a:
                        reward = mdp.R.get(state, {}).get(a, {}).get(next_st, 0.0)
                        expected_reward += prob * reward
                gt_reward_table[s][a] = expected_reward
            else:
                # Deterministic: single trajectory sample
                next_state, reward, done = mdp.step(state, a)
                gt_reward_table[s][a] = reward

    # Compute common vmin/vmax for consistent color scaling
    vmin = min(np.nanmin(a) for a in gt_reward_table)
    vmax = max(np.nanmax(a) for a in gt_reward_table)

    for i in range(4):
        ax = axes[0, i]
        im = ax.imshow(gt_reward_table[:,i].reshape((grid_size, grid_size)), cmap=cmap, origin='upper',
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Action {i}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ensemble_mean_reward_table = np.zeros((grid_size*grid_size, mdp.get_num_actions()))
    ensemble_std_reward_table = np.zeros((grid_size*grid_size, mdp.get_num_actions()))
    for s in range(grid_size*grid_size):
        state = mdp.index_to_state(s)
        for a in range(mdp.get_num_actions()):
            # For stochastic MDPs, use expected next state (or first transition); for deterministic, step
            if hasattr(mdp, 'get_all_transitions'):
                # Stochastic: use first transition as representative
                all_transitions = mdp.get_all_transitions()
                for s_curr, a_curr, next_st, prob in all_transitions:
                    if s_curr == state and a_curr == a:
                        reward = mdp.R.get(state, {}).get(a, {}).get(next_st, 0.0)
                        trajectory = Trajectory([Transition(state, a, next_st, reward)])
                        mean_pred, std_pred = ensemble.predict_returns(trajectory, return_std = True)
                        ensemble_mean_reward_table[s][a] = mean_pred
                        ensemble_std_reward_table[s][a] = std_pred
                        break  # Just use the first transition
            else:
                # Deterministic: single sample
                next_state, reward, done = mdp.step(state, a)
                trajectory = Trajectory([Transition(state, a, next_state, reward)])
                mean_pred, std_pred = ensemble.predict_returns(trajectory, return_std = True)
                ensemble_mean_reward_table[s][a] = mean_pred
                ensemble_std_reward_table[s][a] = std_pred
    
    # Compute common vmin/vmax for consistent color scaling
    vmin = min(np.nanmin(a) for a in ensemble_mean_reward_table)
    vmax = max(np.nanmax(a) for a in ensemble_mean_reward_table)

    for i in range(4):
        ax = axes[1, i]
        im = ax.imshow(ensemble_mean_reward_table[:,i].reshape((grid_size, grid_size)), cmap=cmap, origin='upper',
                       vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Compute common vmin/vmax for consistent color scaling
    vmin = min(np.nanmin(a) for a in ensemble_std_reward_table)
    vmax = max(np.nanmax(a) for a in ensemble_std_reward_table)

    for i in range(4):
        ax = axes[2, i]
        im = ax.imshow(ensemble_std_reward_table[:,i].reshape((grid_size, grid_size)), cmap=cmap, origin='upper',
                       vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add canonicalized ensemble rows if provided
    if canonicalized_ensemble is not None:
        canonicalized_mean_reward_table = np.zeros((grid_size*grid_size, mdp.get_num_actions()))
        canonicalized_std_reward_table = np.zeros((grid_size*grid_size, mdp.get_num_actions()))

        for s in range(grid_size*grid_size):
            state = mdp.index_to_state(s)
            for a in range(mdp.get_num_actions()):
                # For stochastic MDPs, use first transition; for deterministic, step
                if hasattr(mdp, 'get_all_transitions'):
                    # Stochastic: use first transition as representative
                    all_transitions = mdp.get_all_transitions()
                    for s_curr, a_curr, next_st, prob in all_transitions:
                        if s_curr == state and a_curr == a:
                            reward = mdp.R.get(state, {}).get(a, {}).get(next_st, 0.0)
                            trajectory = Trajectory([Transition(state, a, next_st, reward)])
                            mean_pred, std_pred = canonicalized_ensemble.predict_returns(trajectory, return_std=True)
                            canonicalized_mean_reward_table[s][a] = mean_pred
                            canonicalized_std_reward_table[s][a] = std_pred
                            break  # Just use the first transition
                else:
                    # Deterministic: single sample
                    next_state, reward, done = mdp.step(state, a)
                    trajectory = Trajectory([Transition(state, a, next_state, reward)])
                    mean_pred, std_pred = canonicalized_ensemble.predict_returns(trajectory, return_std=True)
                    canonicalized_mean_reward_table[s][a] = mean_pred
                    canonicalized_std_reward_table[s][a] = std_pred

        # Compute common vmin/vmax for canonicalized mean rewards
        vmin = min(np.nanmin(a) for a in canonicalized_mean_reward_table)
        vmax = max(np.nanmax(a) for a in canonicalized_mean_reward_table)

        for i in range(4):
            ax = axes[3, i]
            im = ax.imshow(canonicalized_mean_reward_table[:,i].reshape((grid_size, grid_size)), cmap=cmap, origin='upper',
                           vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Compute common vmin/vmax for canonicalized std rewards
        vmin = min(np.nanmin(a) for a in canonicalized_std_reward_table)
        vmax = max(np.nanmax(a) for a in canonicalized_std_reward_table)

        for i in range(4):
            ax = axes[4, i]
            im = ax.imshow(canonicalized_std_reward_table[:,i].reshape((grid_size, grid_size)), cmap=cmap, origin='upper',
                           vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_frequency_vs_uncertainty(
    mdp: MDP,
    ensemble: RewardModelEnsemble,
    canonicalized_ensemble: CanonicalizedRewardEnsemble,
    pairs: list,
    all_trajectories: list = None,
    figsize=(14, 6),
    save_path: str = "figures/frequency_vs_uncertainty.png"
):
    """
    Plot frequency vs uncertainty for transitions in side-by-side scatter plots.

    Shows the relationship between transition frequency in the training data and
    the uncertainty (standard deviation) of the reward model ensemble predictions.

    Args:
        mdp: The MDP environment.
        ensemble: The trained RewardModelEnsemble.
        canonicalized_ensemble: The canonicalized version of the ensemble.
        pairs: List of trajectory pairs (used to compute transition frequencies).
        all_trajectories: Optional list of all trajectories (alternative to pairs).
        figsize: Figure size as (width, height).
        save_path: Path to save the figure (default: "figures/frequency_vs_uncertainty.png").
    """
    grid_size = mdp.grid_size
    num_states = grid_size * grid_size
    num_actions = mdp.get_num_actions()

    # Count transition frequencies from training data
    transition_counts = {}

    # Use all_trajectories if provided, otherwise extract from pairs
    trajectories = all_trajectories
    if trajectories is None and pairs is not None:
        trajectories = []
        for traj1, traj2 in pairs:
            trajectories.extend([traj1, traj2])

    if trajectories is not None:
        for traj in trajectories:
            for transition in traj.transitions:
                # Find state index
                state_idx = mdp.state_to_index(transition.state)
                action = transition.action
                next_state_idx = mdp.state_to_index(transition.next_state)
                key = (state_idx, action, next_state_idx)
                transition_counts[key] = transition_counts.get(key, 0) + 1

    # Collect data for all possible transitions
    frequencies = []
    uncertainties_ensemble = []
    uncertainties_canonical = []

    # For stochastic MDPs, iterate over all possible transitions; for deterministic, enumerate states
    if hasattr(mdp, 'get_all_transitions'):
        # Stochastic: use all transitions
        all_transitions = mdp.get_all_transitions()
        for state, action, next_state, prob in all_transitions:
            state_idx = mdp.state_to_index(state)
            next_state_idx = mdp.state_to_index(next_state)
            key = (state_idx, action, next_state_idx)

            # Get frequency
            freq = transition_counts.get(key, 0)
            frequencies.append(freq)

            # Get reward for trajectory
            reward = mdp.R.get(state, {}).get(action, {}).get(next_state, 0.0)

            # Get uncertainty from ensemble
            trajectory = Trajectory([Transition(state, action, next_state, reward)])
            _, std_ensemble = ensemble.predict_returns(trajectory, return_std=True)
            uncertainties_ensemble.append(std_ensemble)

            # Get uncertainty from canonicalized ensemble
            _, std_canonical = canonicalized_ensemble.predict_returns(trajectory, return_std=True)
            uncertainties_canonical.append(std_canonical)
    else:
        # Deterministic: original logic
        for s in range(num_states):
            for a in range(num_actions):
                next_state, reward, done = mdp.step(mdp.index_to_state(s), a)
                next_state_idx = mdp.state_to_index(next_state)
                key = (s, a, next_state_idx)

                # Get frequency
                freq = transition_counts.get(key, 0)
                frequencies.append(freq)

                # Get uncertainty from ensemble
                trajectory = Trajectory([Transition(mdp.index_to_state(s), a, next_state, reward)])
                _, std_ensemble = ensemble.predict_returns(trajectory, return_std=True)
                uncertainties_ensemble.append(std_ensemble)

                # Get uncertainty from canonicalized ensemble
                _, std_canonical = canonicalized_ensemble.predict_returns(trajectory, return_std=True)
                uncertainties_canonical.append(std_canonical)

    # Create side-by-side scatter plots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Trained ensemble
    ax_left = axes[0]
    scatter_left = ax_left.scatter(frequencies, uncertainties_ensemble, alpha=0.6, s=50, c='steelblue')
    ax_left.set_xlabel('Transition Frequency', fontsize=12)
    ax_left.set_ylabel('Uncertainty (Std Dev)', fontsize=12)
    ax_left.set_title('Trained Reward Ensemble', fontsize=13, fontweight='bold')
    ax_left.grid(True, alpha=0.3)

    # Right plot: Canonicalized ensemble
    ax_right = axes[1]
    scatter_right = ax_right.scatter(frequencies, uncertainties_canonical, alpha=0.6, s=50, c='coral')
    ax_right.set_xlabel('Transition Frequency', fontsize=12)
    ax_right.set_ylabel('Uncertainty (Std Dev)', fontsize=12)
    ax_right.set_title('Canonicalized Reward Ensemble', fontsize=13, fontweight='bold')
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()