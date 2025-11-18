"""Script to load an ensemble and generate visualization plots."""

import argparse
from src.utils import plot_ensemble


def main():

    # Generate plots
    heatmap_path, freq_path = plot_ensemble(
        grid_size=8,
        ensemble_size=5,
        num_pairs=1000,
        max_steps=20,
        num_epochs=30,
        seed=1,
        mdp_seed=1,
        bootstrap=True,
        ensemble_dir="saved_ensembles",
        output_dir="figures",
        verbose=True
    )

    print(f"\nPlots generated:")
    print(f"  Heatmap: {heatmap_path}")
    print(f"  Frequency: {freq_path}")


if __name__ == "__main__":
    main()
