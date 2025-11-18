"""Script to train and save a reward model ensemble on a random deterministic gridworld."""

import argparse
from src.utils import train_and_save_ensemble


def main():
    """Train and save an ensemble based on command-line arguments."""

    # Train and save the ensemble
    ensemble, mdp, save_path = train_and_save_ensemble(
        grid_size=8,
        ensemble_size=5,
        num_pairs=1000,
        max_steps=20,
        num_epochs=30,
        batch_size=10,
        seed=1,
        mdp_seed=1,
        bootstrap=True,
        output_dir="saved_ensembles",
        verbose=True
    )

    print(f"\nEnsemble saved to: {save_path}")


if __name__ == "__main__":
    main()
