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
    cmap='viridis'
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
        cmap: matplotlib colormap name
    """
    grid_size = mdp.grid_size
    num_rows = 5 if canonicalized_ensemble is not None else 3
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 3 * num_rows + 1))

    # Get a copy of the chosen colormap and set NaN color to black
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color='black')

    gt_reward_table = np.zeros((grid_size*grid_size, mdp.get_num_actions()))
    for s in range(grid_size*grid_size):
        for a in range(mdp.get_num_actions()):
            next_state, reward, done = mdp.step(mdp.index_to_state(s), a)
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
        for a in range(mdp.get_num_actions()):
            next_state, reward, done = mdp.step(mdp.index_to_state(s), a)
            trajectory = Trajectory([Transition(mdp.index_to_state(s), a, next_state, reward)])
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
            for a in range(mdp.get_num_actions()):
                next_state, reward, done = mdp.step(mdp.index_to_state(s), a)
                trajectory = Trajectory([Transition(mdp.index_to_state(s), a, next_state, reward)])
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
    plt.savefig("figures/heatmaps.png", dpi=300, bbox_inches='tight')