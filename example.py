"""Example script demonstrating the full pipeline."""

import numpy as np
import torch
from src.environments import GridWorld
from src.policies import UniformPolicy
from src.utils import sample_trajectory_pairs, label_trajectory_pairs, get_trajectory_statistics
from src.models import RewardModelEnsemble


def main():
    """Run the full pipeline example."""
    print("=" * 60)
    print("Transition Uncertainty Experiment Pipeline")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Step 1: Create GridWorld MDP
    print("\n1. Creating GridWorld MDP...")
    grid_size = 5
    env = GridWorld(grid_size=grid_size, seed=42)
    print(f"   - Grid size: {grid_size}x{grid_size}")
    print(f"   - Number of states: {env.get_num_states()}")
    print(f"   - Number of actions: {env.get_num_actions()}")
    print(f"   - Goal state: {env.goal_state}")

    # Step 2: Create policy
    print("\n2. Creating policy...")
    policy = UniformPolicy(num_actions=env.get_num_actions())
    print("   - Using uniform random policy")

    # Step 3: Sample trajectory pairs
    print("\n3. Sampling trajectory pairs...")
    num_pairs = 100
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

    stats = get_trajectory_statistics(all_trajectories)
    print(f"\n   Trajectory Statistics:")
    print(f"   - Mean length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
    print(f"   - Mean return: {stats['mean_return']:.3f} ± {stats['std_return']:.3f}")
    print(f"   - Return range: [{stats['min_return']:.3f}, {stats['max_return']:.3f}]")

    # Step 4: Label trajectory pairs
    print("\n4. Labeling trajectory pairs with ground truth preferences...")
    preferences = label_trajectory_pairs(pairs, discount_factor=env.get_discount_factor())
    print(f"   - Labeled {len(preferences)} pairs")
    print(f"   - Preference distribution: {np.sum(preferences == 0)} prefer first, {np.sum(preferences == 1)} prefer second")

    # Step 5: Create and train reward model ensemble
    print("\n5. Training reward model ensemble...")
    ensemble_size = 5
    state_dim = grid_size * grid_size  # One-hot encoding
    num_actions = env.get_num_actions()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ensemble = RewardModelEnsemble(
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=[64, 64],
        lr=1e-3,
        device=device
    )
    print(f"   - Ensemble size: {ensemble_size}")
    print(f"   - State dimension: {state_dim}")
    print(f"   - Hidden layers: [64, 64]")

    # Train the ensemble
    print("\n   Training...")
    ensemble.train(
        trajectory_pairs=pairs,
        preferences=preferences,
        num_epochs=50,
        batch_size=10,
        bootstrap=True,
        verbose=True
    )

    # Step 6: Evaluate the ensemble
    print("\n6. Evaluating trained ensemble...")
    test_pairs = sample_trajectory_pairs(
        mdp=env,
        policy=policy,
        num_pairs=10,
        max_steps=max_steps,
        seed=123
    )

    correct_predictions = 0
    for traj1, traj2 in test_pairs:
        # Get ground truth preference
        return1 = traj1.get_return(env.get_discount_factor())
        return2 = traj2.get_return(env.get_discount_factor())
        true_preference = 0 if return1 >= return2 else 1

        # Get ensemble predictions
        pred_return1, std1 = ensemble.predict_returns(traj1, return_std=True)
        pred_return2, std2 = ensemble.predict_returns(traj2, return_std=True)
        pred_preference = 0 if pred_return1 >= pred_return2 else 1

        if pred_preference == true_preference:
            correct_predictions += 1

        print(f"\n   Pair:")
        print(f"   - True returns: {return1:.3f} vs {return2:.3f} -> Preference: {true_preference}")
        print(f"   - Pred returns: {pred_return1:.3f} (±{std1:.3f}) vs {pred_return2:.3f} (±{std2:.3f}) -> Preference: {pred_preference}")
        print(f"   - Correct: {pred_preference == true_preference}")

    accuracy = correct_predictions / len(test_pairs)
    print(f"\n   Overall accuracy: {accuracy:.1%}")

    # Step 7: Save the ensemble
    print("\n7. Saving ensemble...")
    ensemble.save('reward_ensemble.pt')
    print("   - Saved to: reward_ensemble.pt")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
