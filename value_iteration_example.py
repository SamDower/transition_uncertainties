"""Example script demonstrating value iteration with precomputed rewards from ensemble."""

import numpy as np
import torch
from src.environments import SparseSummitMDP
from src.policies import CustomPolicy, UniformPolicy
from src.models import RewardModelEnsemble
from src.algorithms import ValueIteration


def main():
    """Run value iteration example with precomputed rewards."""
    print("=" * 70)
    print("Value Iteration with Precomputed Ensemble Rewards")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    grid_size = 8
    ensemble_size = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nConfiguration:")
    print(f"  - Grid size: {grid_size}x{grid_size}")
    print(f"  - Ensemble size: {ensemble_size}")
    print(f"  - Device: {device}")

    # =========================================================================
    # Step 1: Create environment and load pretrained ensemble
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Create environment and load pretrained ensemble")
    print("=" * 70)

    env = SparseSummitMDP(discount_factor=0.99, grid_size=grid_size, use_trap=True)
    print(f"\nCreated SparseSummitMDP:")
    print(f"  - Treasure location: {env.treasure}")
    print(f"  - Trap location: {env.trap}")
    print(f"  - Start state: (0, 0)")

    # Load pretrained ensemble
    print(f"\nLoading pretrained ensemble...")
    state_dim = grid_size * grid_size
    num_actions = env.get_num_actions()

    ensemble = RewardModelEnsemble(
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=[64, 64],
        lr=1e-3,
        max_steps=20,
        device=device.type,
    )

    ensemble.load('reward_ensemble.pt')
    print(f"  - Ensemble loaded successfully!")

    # =========================================================================
    # Step 2: Precompute rewards
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Precompute rewards for all transitions")
    print("=" * 70)

    precomputed_rewards = ensemble.precompute_rewards(env, verbose=True)

    # Create a fast lookup reward function
    def precomputed_reward_fn(state, action, next_state):
        """Fast lookup of precomputed rewards."""
        return precomputed_rewards.get((state, action, next_state), 0.0)

    print(f"\nReward statistics:")
    rewards_list = list(precomputed_rewards.values())
    print(f"  - Min reward: {np.min(rewards_list):.4f}")
    print(f"  - Max reward: {np.max(rewards_list):.4f}")
    print(f"  - Mean reward: {np.mean(rewards_list):.4f}")
    print(f"  - Std dev: {np.std(rewards_list):.4f}")

    # =========================================================================
    # Step 3: Create evaluation policies
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Create evaluation policies")
    print("=" * 70)

    # Uniform policy (baseline)
    uniform_policy = UniformPolicy(num_actions=env.get_num_actions())
    print(f"  - Created UniformPolicy")

    # Custom deterministic policy (prefer right and down movements)
    def biased_policy_fn(state):
        """Policy that prefers moving towards treasure at (7,7)."""
        x, y = state
        # Actions: 0=up, 1=down, 2=left, 3=right
        probs = [0.1, 0.4, 0.1, 0.4]  # Prefer down and right
        return probs

    biased_policy = CustomPolicy(biased_policy_fn)
    print(f"  - Created CustomPolicy (biased towards right/down)")

    # =========================================================================
    # Step 4: Run value iteration with different policies
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Run value iteration with precomputed rewards")
    print("=" * 70)

    scenarios = [
        ("Uniform policy", uniform_policy),
        ("Biased policy (towards treasure)", biased_policy),
    ]

    results = {}

    for scenario_name, policy in scenarios:
        print(f"\n{'─' * 70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'─' * 70}")

        vi = ValueIteration(
            mdp=env,
            reward_fn=precomputed_reward_fn,
            discount_factor=0.99,
            convergence_threshold=1e-6,
            max_iterations=500,
            seed=42
        )

        # Run value iteration
        value_fn = vi.solve_for_all_states(policy, verbose=True)

        # Store results
        results[scenario_name] = {
            'vi': vi,
            'value_fn': value_fn,
            'num_iterations': vi.num_iterations,
            'converged': vi.converged,
        }

        # Get value array for visualization
        value_array = vi.get_state_values_as_array(size=grid_size)

        # Print summary statistics
        print(f"\nValue Function Statistics:")
        print(f"  - Min value: {np.min(value_array):.4f}")
        print(f"  - Max value: {np.max(value_array):.4f}")
        print(f"  - Mean value: {np.mean(value_array):.4f}")
        print(f"  - Std dev: {np.std(value_array):.4f}")

        # Print values at key locations
        print(f"\nValues at key states:")
        print(f"  - Start (0,0): {vi.get_value((0, 0)):.4f}")
        if hasattr(env, 'treasure'):
            print(f"  - Treasure {env.treasure}: {vi.get_value(env.treasure):.4f}")
        if hasattr(env, 'trap'):
            print(f"  - Trap {env.trap}: {vi.get_value(env.trap):.4f}")

        # Print value grid visualization (8x8)
        print(f"\nValue Grid Heatmap:")
        for i in range(grid_size):
            row_vals = [f"{value_array[i, j]:6.2f}" for j in range(grid_size)]
            print(f"  {' '.join(row_vals)}")

    # =========================================================================
    # Step 5: Compare value estimates across scenarios
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Summary of Results")
    print("=" * 70)

    print(f"\n{'Scenario':<45} {'Converged':<12} {'Iterations':<12}")
    print("─" * 69)
    for scenario_name, data in results.items():
        converged = "Yes" if data['converged'] else "No"
        iterations = str(data['num_iterations'])
        print(f"{scenario_name:<45} {converged:<12} {iterations:<12}")

    # Compare start state values across scenarios
    print(f"\nValue at start state (0,0):")
    print("─" * 69)
    start_state = (0, 0)
    for scenario_name, data in results.items():
        value = data['vi'].get_value(start_state)
        print(f"  {scenario_name:<43}: {value:>10.4f}")

    # Compare treasure state values
    if hasattr(env, 'treasure'):
        print(f"\nValue at treasure state {env.treasure}:")
        print("─" * 69)
        for scenario_name, data in results.items():
            value = data['vi'].get_value(env.treasure)
            print(f"  {scenario_name:<43}: {value:>10.4f}")

    # =========================================================================
    # Step 6: Demonstrate usage pattern
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Usage Pattern Summary")
    print("=" * 70)

    print(f"""
Value Iteration with Precomputed Rewards:

1. Create an environment (MDP):
   env = SparseSummitMDP(...)

2. Load or train a reward ensemble:
   ensemble = RewardModelEnsemble(...)
   ensemble.load('reward_ensemble.pt')  # or train it

3. Precompute all rewards at once (one-time cost):
   precomputed_rewards = ensemble.precompute_rewards(env)

4. Create a fast lookup reward function:
   def reward_fn(state, action, next_state):
       return precomputed_rewards.get((state, action, next_state), 0.0)

5. Define a policy to evaluate:
   policy = UniformPolicy(num_actions)

6. Run value iteration with fast reward lookups:
   vi = ValueIteration(mdp=env, reward_fn=reward_fn, ...)
   values = vi.solve_for_all_states(policy)

7. Access computed values:
   value = vi.get_value(state)
   values_array = vi.get_state_values_as_array(size=8)

PERFORMANCE BENEFIT:
  - Without precomputation: ~0.1-0.5ms per reward lookup (5 forward passes)
  - With precomputation: ~0.001ms per reward lookup (dictionary lookup)
  - Speedup: 100-500x faster for value iteration!

This is especially valuable for small, discrete state spaces where you need
to evaluate policies multiple times with different reward functions.
""")

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
