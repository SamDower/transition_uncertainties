"""Script to evaluate a trained ensemble on test trajectory pairs."""

import argparse
import json
from src.utils import evaluate_ensemble


def main():

    # Evaluate ensemble
    results = evaluate_ensemble(
        grid_size=8,
        ensemble_size=5,
        num_pairs=1000,
        num_epochs=30,
        seed=1,
        mdp_seed=1,
        bootstrap=True,
        test_num_pairs=1000,
        test_seed=2,
        ensemble_dir="saved_ensembles",
        verbose=True
    )

    # Print summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {results['accuracy']:.1%} ({results['num_correct']}/{results['num_total']})")
    print(f"Mean true return difference: {results['mean_return_diff']:.4f}")
    print(f"Mean predicted return difference: {results['mean_confidence']:.4f}")

    # # Save to JSON if requested
    # if args.output_json:
    #     # Convert numpy types for JSON serialization
    #     json_results = {
    #         'accuracy': float(results['accuracy']),
    #         'num_correct': int(results['num_correct']),
    #         'num_total': int(results['num_total']),
    #         'mean_return_diff': float(results['mean_return_diff']),
    #         'mean_confidence': float(results['mean_confidence']),
    #         'predictions': [
    #             {
    #                 'true_return1': float(p['true_return1']),
    #                 'true_return2': float(p['true_return2']),
    #                 'pred_return1': float(p['pred_return1']),
    #                 'pred_return2': float(p['pred_return2']),
    #                 'correct': bool(p['correct'])
    #             }
    #             for p in results['predictions']
    #         ]
    #     }

    #     with open(args.output_json, 'w') as f:
    #         json.dump(json_results, f, indent=2)

    #     print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
