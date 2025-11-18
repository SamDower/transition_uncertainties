"""Example script demonstrating the full pipeline."""

import numpy as np
import torch
from src.environments import GridWorldMDP, SparseSummitMDP, RandomGridworldMDP
from src.policies import UniformPolicy, CustomPolicy
from src.utils import sample_trajectory_pairs, label_trajectory_pairs, get_trajectory_statistics
from src.models import RewardModelEnsemble, CanonicalizedRewardEnsemble, ValueAdjustedLevelling, L1Norm
from src.utils.plotting import plot_ensemble_gridworld_deterministic, plot_frequency_vs_uncertainty
import random


def main():

    """Run the full pipeline example."""
    print("=" * 60)
    print("Transition Uncertainty Experiment Pipeline")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Step 1: Create GridWorld MDP
    print("\n1. Creating SparseSummit MDP...")
    #env = GridWorld(grid_size=grid_size, seed=42)
    grid_size = 8
    env = RandomGridworldMDP(
        N=8,
        p_slip=0.10,          # mild global stochasticity
        trap_rate=0.08,       # a few traps but not too many
        wall_rate=0.12,       # roughly ~8 walls expected
        risky_region_bias=1.5,# bottom-right 3× more stochastic
        teleporter_rate=0.03, # occasional teleport surprises
        small_goal_rate=0.16,
        reward_params={
            "trap_range": (-8.0, -3.0),
            "normal_range": (-0.2, 0.2),
            "goal": 1.0,
            "small_goal": 0.3
        },
        discount_factor=0.99,
        seed=10,            # or set a seed for reproducibility
    )


    print("\n2. Creating policy...")
    policy = UniformPolicy(4)
    #policy = CustomPolicy(lambda s: [0.2, 0.3, 0.2, 0.3])
    print("   - Using uniform random policy")

    # Step 3: Sample trajectory pairs
    print("\n3. Sampling trajectory pairs...")
    num_pairs = 1000
    max_steps = 20
    pairs = sample_trajectory_pairs(
        mdp=env,
        policy=policy,
        num_pairs=num_pairs,
        max_steps=max_steps,
        seed=42
    )
    print(f"   - Sampled {num_pairs} trajectory pairs")
    print(f"   - Max steps per trajectory: {max_steps}")

    # Get statistics
    all_trajectories = []
    for traj1, traj2 in pairs:
        all_trajectories.extend([traj1, traj2])

    # Count state visitation frequencies
    state_frequencies = np.zeros((grid_size, grid_size))
    for traj in all_trajectories:
        for transition in traj.transitions:
            # State is a tuple (row, col)
            row, col = transition.state
            state_frequencies[row, col] += 1

    # Visualize the frequencies of each state as a grid heatmap with the numbers shown
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create heatmap
    im = ax.imshow(state_frequencies, cmap='YlOrRd', origin='upper')

    # Add text annotations with frequencies
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax.text(j, i, int(state_frequencies[i, j]),
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax.set_title('State Visitation Frequencies', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    fig.colorbar(im, ax=ax, label='Frequency')

    # Save the figure
    import os
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/state_frequencies.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n4. State Visitation Frequencies:")
    print(f"   - Total visits: {int(state_frequencies.sum())}")
    print(f"   - Average visits per state: {state_frequencies.mean():.2f}")
    print(f"   - Max visits: {int(state_frequencies.max())}")
    print(f"   - Min visits: {int(state_frequencies.min())}")
    print(f"\n   State frequency grid:")
    for i in range(grid_size):
        print(f"   {state_frequencies[i].astype(int)}")
    
    env.visualize_grid()


if __name__ == '__main__':
    main()
