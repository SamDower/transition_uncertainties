"""Example script demonstrating STARC canonicalization of reward models."""

import numpy as np
import torch
from src.environments import SparseSummitMDP
from src.policies import UniformPolicy, CustomPolicy
from src.models import (
    RewardModelEnsemble,
    CanonicalizedRewardEnsemble,
    ValueAdjustedLevelling,
    L1Norm,
    L2Norm
)
from src.utils import sample_trajectory_pairs, label_trajectory_pairs


def evaluate_ensemble(ensemble, pairs, preferences, verbose=False):
    """Evaluate ensemble accuracy on preference prediction."""
    correct = 0

    for idx, (traj1, traj2) in enumerate(pairs):
        pred_return1 = ensemble.predict_returns(traj1)
        pred_return2 = ensemble.predict_returns(traj2)

        # Predict preference based on returns
        if pred_return1 > pred_return2:
            pred_pref = 0
        else:
            pred_pref = 1

        if pred_pref == preferences[idx]:
            correct += 1

    accuracy = correct / len(pairs)
    if verbose:
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(pairs)} correct)")

    return accuracy


def main():
    """Run STARC canonicalization example."""
    print("=" * 70)
    print("STARC Canonicalization of Reward Models")
    print("=" * 70)

    # Set random seed
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
    print(f"\nCreated SparseSummitMDP")

    # Load pretrained ensemble
    print(f"Loading pretrained ensemble...")
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
    # Step 2: Generate test trajectories for evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Generate test trajectories")
    print("=" * 70)

    policy = UniformPolicy(num_actions=env.get_num_actions())
    num_test_pairs = 100
    max_steps = 20

    print(f"Sampling {num_test_pairs} test trajectory pairs...")
    test_pairs = sample_trajectory_pairs(
        mdp=env,
        policy=policy,
        num_pairs=num_test_pairs,
        max_steps=max_steps,
        seed=42
    )

    test_preferences = label_trajectory_pairs(
        test_pairs,
        discount_factor=env.get_discount_factor()
    )

    print(f"  - Sampled {num_test_pairs} test pairs")

    # =========================================================================
    # Step 3: Evaluate original ensemble on test set
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Evaluate original ensemble")
    print("=" * 70)

    print(f"\nOriginal ensemble performance on test set:")
    original_accuracy = evaluate_ensemble(ensemble, test_pairs, test_preferences, verbose=True)

    # =========================================================================
    # Step 4: Canonicalize ensemble with different configurations
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Canonicalize ensemble with different normalizers")
    print("=" * 70)

    canonicalization_configs = [
        ("ValueAdjustedLevelling + L1Norm", ValueAdjustedLevelling(0.99), L1Norm()),
        ("ValueAdjustedLevelling + L2Norm", ValueAdjustedLevelling(0.99), L2Norm()),
    ]

    results = {}

    for config_name, canonicalizer, normalizer in canonicalization_configs:
        print(f"\n{'─' * 70}")
        print(f"Configuration: {config_name}")
        print(f"{'─' * 70}")

        # Create canonicalized ensemble
        print(f"Creating canonicalized ensemble...")
        canon_ensemble = CanonicalizedRewardEnsemble(
            ensemble=ensemble,
            canonicalizer=canonicalizer,
            normalizer=normalizer,
            mdp=env,
            state_dim=state_dim,
            device=device.type
        )

        # Canonicalize all models
        print(f"Canonicalizing models...")
        canon_ensemble.canonicalize(verbose=True)

        # Evaluate canonicalized ensemble
        print(f"\nCanonicalizedEnsemble performance on test set:")
        canon_accuracy = evaluate_ensemble(
            canon_ensemble, test_pairs, test_preferences, verbose=True
        )

        # Store results
        results[config_name] = {
            'ensemble': canon_ensemble,
            'accuracy': canon_accuracy,
        }

        # Print comparison
        print(f"\nComparison:")
        print(f"  - Original accuracy: {original_accuracy:.4f}")
        print(f"  - Canonicalized accuracy: {canon_accuracy:.4f}")
        print(f"  - Difference: {canon_accuracy - original_accuracy:+.4f}")

    # =========================================================================
    # Step 5: Demonstrate accessing canonicalized models
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Access canonicalized models individually")
    print("=" * 70)

    config_name = "ValueAdjustedLevelling + L1Norm"
    canon_ensemble = results[config_name]['ensemble']
    canonicalized_models = canon_ensemble.get_canonicalized_models()

    print(f"\nEnsemble has {len(canonicalized_models)} canonicalized models")
    print(f"Each model is a CanonicalizedRewardModel wrapper")

    # Show that we can use individual models
    print(f"\nExample: Evaluating first canonicalized model on a trajectory:")
    test_traj = test_pairs[0][0]
    from src.models.reward_model import trajectory_to_tensors

    states, actions, next_states = trajectory_to_tensors(
        test_traj, state_dim, device.type
    )

    first_model = canonicalized_models[0]
    first_model.eval()
    with torch.no_grad():
        first_rewards = first_model(states, actions, next_states)
        print(f"  - Rewards shape: {first_rewards.shape}")
        print(f"  - First 5 canonicalized rewards: {first_rewards[:5].squeeze().cpu().numpy()}")

    # =========================================================================
    # Step 6: Summary and usage guide
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Summary and Usage Guide")
    print("=" * 70)

    print(f"""
STARC Canonicalization Overview:

STARC (Scaling and Translating with a Reward Canonicalization) consists of:
1. A canonicalization function c(R) that transforms rewards
2. A normalization function ||c(R)|| that normalizes the result

The final canonicalized reward is: c_R(s,a,s') = c(R)(s,a,s') / ||c(R)||

Available Canonicalizers:
  - ValueAdjustedLevelling: c(R)(s,a,s') = R(s,a,s') - V(s) + γ*V(s')
    (Removes value function component for comparability)

Available Normalizers:
  - L1Norm: ||c(R)||_1 = Σ|c(R)(i)|
  - L2Norm: ||c(R)||_2 = √(Σ(c(R)(i))²)
  - MaxNorm: ||c(R)||_∞ = max|c(R)(i)|

Usage Pattern:

1. Create or load a RewardModelEnsemble:
   ensemble = RewardModelEnsemble(...)
   ensemble.load('reward_ensemble.pt')

2. Create a CanonicalizedRewardEnsemble:
   canon_ensemble = CanonicalizedRewardEnsemble(
       ensemble=ensemble,
       canonicalizer=ValueAdjustedLevelling(discount_factor=0.99),
       normalizer=L1Norm(),
       mdp=env,
       state_dim=state_dim,
       device='cuda'
   )

3. Canonicalize all models:
   canon_ensemble.canonicalize(verbose=True)

4. Use the canonicalized ensemble:
   # Get trajectory returns
   return = canon_ensemble.predict_returns(trajectory)

   # Access individual canonicalized models
   models = canon_ensemble.get_canonicalized_models()
   for model in models:
       model.eval()
       rewards = model(states, actions, next_states)

Key Benefits:
  - Makes reward models more comparable across different training runs
  - Removes spurious differences due to scale/translation
  - Helps identify true differences in learned reward structure
  - Can use different canonicalizers and normalizers for different analyses

Test Set Results:
  - Original ensemble accuracy: {original_accuracy:.4f}
  - Best canonicalized accuracy: {max(r['accuracy'] for r in results.values()):.4f}
""")

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
