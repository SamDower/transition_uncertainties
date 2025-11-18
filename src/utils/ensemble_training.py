"""Utilities for training and managing reward model ensembles."""

import os
import numpy as np
import torch
from typing import Optional, Tuple
from src.environments import RandomGridworldMDP
from src.policies import UniformPolicy
from src.utils import sample_trajectory_pairs, label_trajectory_pairs
from src.models import RewardModelEnsemble


def train_and_save_ensemble(
    grid_size: int = 8,
    ensemble_size: int = 5,
    num_pairs: int = 1000,
    max_steps: int = 20,
    num_epochs: int = 50,
    batch_size: int = 10,
    seed: int = 42,
    mdp_seed: int = None,
    bootstrap: bool = False,
    device: str = None,
    output_dir: str = "saved_ensembles",
    verbose: bool = True
) -> Tuple[RewardModelEnsemble, RandomGridworldMDP, str]:
    """
    Train a reward model ensemble on a random stochastic gridworld and save it.

    Args:
        grid_size: Size of the NxN gridworld.
        ensemble_size: Number of models in the ensemble.
        num_pairs: Number of trajectory pairs to sample for training.
        max_steps: Maximum steps per trajectory.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        seed: Random seed for reproducibility (trajectory sampling and training).
        mdp_seed: Random seed for the MDP reward generation (defaults to same as seed).
        bootstrap: Whether to use bootstrap sampling during training.
        device: Device to use ('cpu', 'cuda', or None for auto-detection).
        output_dir: Directory to save the ensemble.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (trained_ensemble, mdp, save_path).
    """
    if mdp_seed is None:
        mdp_seed = seed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if verbose:
        print("=" * 70)
        print("Training Reward Model Ensemble on Random Stochastic Gridworld")
        print("=" * 70)

    # Create environment
    if verbose:
        print(f"\n1. Creating RandomGridworldMDP...")
        print(f"   - Grid size: {grid_size}x{grid_size}")
        print(f"   - MDP seed: {mdp_seed}")

    mdp = RandomGridworldMDP(
        N=grid_size,
        p_slip=0.10,
        trap_rate=0.08,
        wall_rate=0.12,
        risky_region_bias=1.5,
        teleporter_rate=0.03,
        small_goal_rate=0.16,
        reward_params={
            "trap_range": (-8.0, -3.0),
            "normal_range": (-0.2, 0.2),
            "goal": 1.0,
            "small_goal": 0.3
        },
        discount_factor=0.99,
        seed=mdp_seed
    )

    # Create policy
    if verbose:
        print(f"\n2. Creating uniform random policy...")
    policy = UniformPolicy(num_actions=mdp.get_num_actions())

    # Sample trajectory pairs
    if verbose:
        print(f"\n3. Sampling trajectory pairs...")
        print(f"   - Number of pairs: {num_pairs}")
        print(f"   - Max steps per trajectory: {max_steps}")
        print(f"   - Seed: {seed}")

    np.random.seed(seed)
    pairs = sample_trajectory_pairs(
        mdp=mdp,
        policy=policy,
        num_pairs=num_pairs,
        max_steps=max_steps,
        seed=seed
    )

    # Label trajectory pairs
    if verbose:
        print(f"\n4. Labeling trajectory pairs with ground truth preferences...")

    preferences = label_trajectory_pairs(pairs, discount_factor=mdp.get_discount_factor())

    if verbose:
        print(f"   - Labeled {len(preferences)} pairs")
        print(f"   - Preference distribution: {np.sum(preferences == 0)} prefer first, {np.sum(preferences == 1)} prefer second")

    # Create and train ensemble
    if verbose:
        print(f"\n5. Creating reward model ensemble...")
        print(f"   - Ensemble size: {ensemble_size}")
        print(f"   - State dimension: {mdp.get_num_states()}")
        print(f"   - Number of actions: {mdp.get_num_actions()}")

    state_dim = mdp.get_num_states()
    num_actions = mdp.get_num_actions()

    ensemble = RewardModelEnsemble(
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=[64, 64],
        lr=1e-3,
        max_steps=max_steps,
        device=str(device),
    )

    if verbose:
        print(f"\n6. Training ensemble...")
        print(f"   - Number of epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Bootstrap: {bootstrap}")
        print(f"   - Device: {device}")

    ensemble.train(
        trajectory_pairs=pairs,
        preferences=preferences,
        num_epochs=num_epochs,
        batch_size=batch_size,
        bootstrap=bootstrap,
        verbose=verbose
    )

    # Generate save path
    os.makedirs(output_dir, exist_ok=True)
    filename = f"ensemble_g{grid_size}_e{ensemble_size}_s{mdp_seed}_p{num_pairs}_ep{num_epochs}_b{int(bootstrap)}.pt"
    save_path = os.path.join(output_dir, filename)

    if verbose:
        print(f"\n7. Saving ensemble...")
        print(f"   - Save path: {save_path}")

    ensemble.save(save_path)

    if verbose:
        print("\n" + "=" * 70)
        print("Training and saving complete!")
        print("=" * 70)

    return ensemble, mdp, save_path


def load_ensemble(
    ensemble_size: int,
    grid_size: int,
    state_dim: int,
    num_actions: int,
    save_path: str,
    device: str = None
) -> RewardModelEnsemble:
    """
    Load a saved reward model ensemble.

    Args:
        ensemble_size: Number of models in the ensemble.
        grid_size: Size of the gridworld (used for reference).
        state_dim: Dimension of state representation.
        num_actions: Number of actions.
        save_path: Path to the saved ensemble.
        device: Device to load to ('cpu', 'cuda', or None for auto-detection).

    Returns:
        Loaded RewardModelEnsemble.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    ensemble = RewardModelEnsemble(
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=[64, 64],
        lr=1e-3,
        max_steps=20,
        device=str(device),
    )

    ensemble.load(save_path)
    return ensemble


def plot_ensemble(
    grid_size: int = 8,
    ensemble_size: int = 5,
    num_pairs: int = 1000,
    max_steps: int = 20,
    num_epochs: int = 50,
    seed: int = 42,
    mdp_seed: int = None,
    bootstrap: bool = False,
    device: str = None,
    ensemble_dir: str = "saved_ensembles",
    output_dir: str = "figures",
    verbose: bool = True
) -> Tuple[str, str]:
    """
    Load a trained ensemble, canonicalize it, and generate visualization plots.

    Generates two plots:
    1. Heatmaps of ground truth, ensemble mean, and standard deviations
    2. Frequency vs uncertainty scatter plots for both ensembles

    Args:
        grid_size: Size of the NxN gridworld.
        ensemble_size: Number of models in the ensemble.
        num_pairs: Number of trajectory pairs used during training.
        max_steps: Maximum steps per trajectory during training.
        num_epochs: Number of training epochs.
        seed: Random seed used for trajectory sampling/training.
        mdp_seed: Random seed used for MDP reward generation (defaults to seed).
        bootstrap: Whether bootstrap sampling was used during training.
        device: Device to use ('cpu', 'cuda', or None for auto-detection).
        ensemble_dir: Directory where the ensemble is saved.
        output_dir: Directory to save the plots.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (heatmap_plot_path, frequency_plot_path).
    """
    if mdp_seed is None:
        mdp_seed = seed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if verbose:
        print("=" * 70)
        print("Plotting Ensemble Visualizations")
        print("=" * 70)

    # Create environment
    if verbose:
        print(f"\n1. Creating RandomGridworldMDP...")
        print(f"   - Grid size: {grid_size}x{grid_size}")
        print(f"   - MDP seed: {mdp_seed}")

    mdp = RandomGridworldMDP(
        N=grid_size,
        p_slip=0.10,
        trap_rate=0.08,
        wall_rate=0.12,
        risky_region_bias=1.5,
        teleporter_rate=0.03,
        small_goal_rate=0.16,
        reward_params={
            "trap_range": (-8.0, -3.0),
            "normal_range": (-0.2, 0.2),
            "goal": 1.0,
            "small_goal": 0.3
        },
        discount_factor=0.99,
        seed=mdp_seed
    )
    state_dim = mdp.get_num_states()
    num_actions = mdp.get_num_actions()

    # Load ensemble
    if verbose:
        print(f"\n2. Loading trained ensemble...")
        print(f"   - Ensemble size: {ensemble_size}")

    ensemble_filename = f"ensemble_g{grid_size}_e{ensemble_size}_s{mdp_seed}_p{num_pairs}_ep{num_epochs}_b{int(bootstrap)}.pt"
    ensemble_path = os.path.join(ensemble_dir, ensemble_filename)

    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble not found at {ensemble_path}")

    if verbose:
        print(f"   - Load path: {ensemble_path}")

    ensemble = load_ensemble(
        ensemble_size=ensemble_size,
        grid_size=grid_size,
        state_dim=state_dim,
        num_actions=num_actions,
        save_path=ensemble_path,
        device=str(device)
    )

    if verbose:
        print(f"   - Ensemble loaded successfully!")

    # Create canonicalized ensemble
    if verbose:
        print(f"\n3. Creating and canonicalizing ensemble...")

    from src.models import CanonicalizedRewardEnsemble, ValueAdjustedLevelling, L1Norm

    canon_ensemble = CanonicalizedRewardEnsemble(
        ensemble=ensemble,
        canonicalizer=ValueAdjustedLevelling(discount_factor=0.99),
        normalizer=L1Norm(),
        mdp=mdp,
        state_dim=state_dim,
        device=str(device)
    )

    canon_ensemble.canonicalize(verbose=verbose)

    # Sample trajectory pairs for frequency calculation
    if verbose:
        print(f"\n4. Sampling trajectory pairs for frequency analysis...")
        print(f"   - Number of pairs: {num_pairs}")
        print(f"   - Max steps: {max_steps}")

    policy = UniformPolicy(num_actions=mdp.get_num_actions())
    np.random.seed(seed)
    pairs = sample_trajectory_pairs(
        mdp=mdp,
        policy=policy,
        num_pairs=num_pairs,
        max_steps=max_steps,
        seed=seed
    )

    # Extract all trajectories
    all_trajectories = []
    for traj1, traj2 in pairs:
        all_trajectories.extend([traj1, traj2])

    # Generate plots
    if verbose:
        print(f"\n5. Generating heatmap plots...")

    from src.utils.plotting import plot_ensemble_gridworld_deterministic, plot_frequency_vs_uncertainty

    os.makedirs(output_dir, exist_ok=True)

    heatmap_filename = f"heatmap_g{grid_size}_e{ensemble_size}_s{mdp_seed}_p{num_pairs}_ep{num_epochs}_b{int(bootstrap)}.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)

    plot_ensemble_gridworld_deterministic(
        mdp, ensemble, canonicalized_ensemble=canon_ensemble, save_path=heatmap_path
    )

    if verbose:
        print(f"   - Heatmap saved to: {heatmap_path}")

    if verbose:
        print(f"\n6. Generating frequency vs uncertainty plots...")

    freq_filename = f"frequency_g{grid_size}_e{ensemble_size}_s{mdp_seed}_p{num_pairs}_ep{num_epochs}_b{int(bootstrap)}.png"
    freq_path = os.path.join(output_dir, freq_filename)

    plot_frequency_vs_uncertainty(
        mdp, ensemble, canon_ensemble, pairs, all_trajectories, save_path=freq_path
    )

    if verbose:
        print(f"   - Frequency plot saved to: {freq_path}")

    if verbose:
        print("\n" + "=" * 70)
        print("Plotting complete!")
        print("=" * 70)

    return heatmap_path, freq_path


def evaluate_ensemble(
    grid_size: int = 8,
    ensemble_size: int = 5,
    num_pairs: int = 1000,
    num_epochs: int = 50,
    seed: int = 42,
    mdp_seed: int = None,
    bootstrap: bool = False,
    test_num_pairs: int = 100,
    test_seed: int = 123,
    device: str = None,
    ensemble_dir: str = "saved_ensembles",
    verbose: bool = True
) -> dict:
    """
    Evaluate a trained ensemble on test trajectory pairs.

    Measures the ensemble's ability to correctly rank pairs of trajectories
    based on their true discounted returns.

    Args:
        grid_size: Size of the NxN gridworld.
        ensemble_size: Number of models in the ensemble.
        num_pairs: Number of trajectory pairs used during training (for loading).
        num_epochs: Number of training epochs (for loading).
        seed: Random seed used during training (for loading).
        mdp_seed: Random seed used for MDP reward generation (defaults to seed).
        bootstrap: Whether bootstrap sampling was used during training.
        test_num_pairs: Number of test trajectory pairs to evaluate.
        test_seed: Random seed for generating test pairs.
        device: Device to use ('cpu', 'cuda', or None for auto-detection).
        ensemble_dir: Directory where the ensemble is saved.
        verbose: Whether to print progress messages.

    Returns:
        Dictionary containing evaluation metrics:
        - 'accuracy': Fraction of correctly ranked pairs
        - 'num_correct': Number of correctly ranked pairs
        - 'num_total': Total number of pairs evaluated
        - 'mean_return_diff': Mean absolute difference in true returns (for sorting)
        - 'mean_confidence': Mean absolute difference in ensemble predictions
        - 'correct_predictions': List of (true_return1, true_return2, pred_return1, pred_return2, correct)
    """
    if mdp_seed is None:
        mdp_seed = seed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if verbose:
        print("=" * 70)
        print("Evaluating Ensemble on Test Set")
        print("=" * 70)

    # Create environment
    if verbose:
        print(f"\n1. Creating RandomGridworldMDP...")
        print(f"   - Grid size: {grid_size}x{grid_size}")
        print(f"   - MDP seed: {mdp_seed}")

    mdp = RandomGridworldMDP(
        N=grid_size,
        p_slip=0.10,
        trap_rate=0.08,
        wall_rate=0.12,
        risky_region_bias=1.5,
        teleporter_rate=0.03,
        small_goal_rate=0.16,
        reward_params={
            "trap_range": (-8.0, -3.0),
            "normal_range": (-0.2, 0.2),
            "goal": 1.0,
            "small_goal": 0.3
        },
        discount_factor=0.99,
        seed=mdp_seed
    )
    state_dim = mdp.get_num_states()
    num_actions = mdp.get_num_actions()
    discount_factor = mdp.get_discount_factor()

    # Load ensemble
    if verbose:
        print(f"\n2. Loading trained ensemble...")

    ensemble_filename = f"ensemble_g{grid_size}_e{ensemble_size}_s{mdp_seed}_p{num_pairs}_ep{num_epochs}_b{int(bootstrap)}.pt"
    ensemble_path = os.path.join(ensemble_dir, ensemble_filename)

    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble not found at {ensemble_path}")

    if verbose:
        print(f"   - Load path: {ensemble_path}")

    ensemble = load_ensemble(
        ensemble_size=ensemble_size,
        grid_size=grid_size,
        state_dim=state_dim,
        num_actions=num_actions,
        save_path=ensemble_path,
        device=str(device)
    )

    if verbose:
        print(f"   - Ensemble loaded successfully!")

    # Generate test pairs
    if verbose:
        print(f"\n3. Generating test trajectory pairs...")
        print(f"   - Number of pairs: {test_num_pairs}")
        print(f"   - Test seed: {test_seed}")

    policy = UniformPolicy(num_actions=mdp.get_num_actions())
    np.random.seed(test_seed)
    test_pairs = sample_trajectory_pairs(
        mdp=mdp,
        policy=policy,
        num_pairs=test_num_pairs,
        max_steps=20,
        seed=test_seed
    )

    # Evaluate ensemble
    if verbose:
        print(f"\n4. Evaluating ensemble on test pairs...")

    correct_predictions = 0
    evaluation_results = []
    return_diffs = []
    confidence_diffs = []

    for idx, (traj1, traj2) in enumerate(test_pairs):
        # Get ground truth returns
        return1 = traj1.get_return(discount_factor)
        return2 = traj2.get_return(discount_factor)
        true_preference = 0 if return1 >= return2 else 1

        # Get ensemble predictions
        pred_return1 = ensemble.predict_returns(traj1, return_std=False)
        pred_return2 = ensemble.predict_returns(traj2, return_std=False)
        pred_preference = 0 if pred_return1 >= pred_return2 else 1

        # Check if prediction is correct
        is_correct = (pred_preference == true_preference)
        if is_correct:
            correct_predictions += 1

        # Record metrics
        return_diff = abs(return1 - return2)
        confidence_diff = abs(pred_return1 - pred_return2)

        return_diffs.append(return_diff)
        confidence_diffs.append(confidence_diff)

        evaluation_results.append({
            'true_return1': return1,
            'true_return2': return2,
            'pred_return1': pred_return1,
            'pred_return2': pred_return2,
            'correct': is_correct
        })

        if verbose and (idx + 1) % max(1, test_num_pairs // 10) == 0:
            print(f"   - Evaluated {idx + 1}/{test_num_pairs} pairs...")

    # Compute statistics
    accuracy = correct_predictions / len(test_pairs)
    mean_return_diff = np.mean(return_diffs)
    mean_confidence = np.mean(confidence_diffs)

    if verbose:
        print(f"\n5. Evaluation Results:")
        print(f"   - Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_pairs)})")
        print(f"   - Mean true return difference: {mean_return_diff:.4f}")
        print(f"   - Mean predicted return difference: {mean_confidence:.4f}")

    if verbose:
        print("\n" + "=" * 70)
        print("Evaluation complete!")
        print("=" * 70)

    return {
        'accuracy': accuracy,
        'num_correct': correct_predictions,
        'num_total': len(test_pairs),
        'mean_return_diff': mean_return_diff,
        'mean_confidence': mean_confidence,
        'predictions': evaluation_results
    }
