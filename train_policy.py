"""Example script for training a policy using a learned reward ensemble."""

import numpy as np
import torch
from src.environments import SparseSummitMDP
from src.models import RewardModelEnsemble
from src.algorithms import TabularQLearning
from src.utils import create_ensemble_mean_reward_fn, create_true_reward_fn


def main():
    """Train a policy using reward ensemble and evaluate it."""
    print("=" * 70)
    print("Training Policy with Reward Ensemble")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Step 1: Create environment
    print("\n1. Creating SparseSummitMDP environment...")
    grid_size = 8
    env = SparseSummitMDP(discount_factor=0.99, grid_size=grid_size, use_trap=True)
    print(f"   - Grid size: {grid_size}x{grid_size}")
    print(f"   - Number of states: {env.get_num_states()}")
    print(f"   - Number of actions: {env.get_num_actions()}")
    print(f"   - Discount factor: {env.get_discount_factor()}")

    # Visualize the environment
    print("\n   Environment layout:")
    env.visualize_grid()
    print("   Legend: S=Start, T=Treasure(+10), X=Trap(-8), L=Lure(+2), p=Plateau(+0.5), #=Wall")

    # Step 2: Load the reward ensemble
    print("\n2. Loading trained reward ensemble...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - Device: {device}")

    # Initialize ensemble with same architecture as training
    ensemble_size = 5
    state_dim = grid_size * grid_size
    num_actions = env.get_num_actions()

    ensemble = RewardModelEnsemble(
        ensemble_size=ensemble_size,
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dims=[64, 64],
        lr=1e-3,
        max_steps=20,
        device=device,
    )

    # Load trained weights
    ensemble.load('reward_ensemble.pt')
    print(f"   - Ensemble loaded successfully")
    print(f"   - Ensemble size: {ensemble_size}")
    print(f"   - State dimension: {state_dim}")

    # Step 3: Create reward function from ensemble
    print("\n3. Creating reward function from ensemble...")
    reward_fn = create_ensemble_mean_reward_fn(
        ensemble=ensemble,
        state_dim=state_dim,
        device=device
    )
    print("   - Using mean of ensemble predictions as reward")

    # Test the reward function on a few transitions
    print("\n   Testing reward function on sample transitions:")
    test_transitions = [
        ((0, 0), 3, (0, 1)),  # Start -> right
        ((1, 0), 0, (0, 0)),  # Lure -> up to start
        ((7, 6), 3, (7, 7)),  # Near treasure -> to treasure
        ((6, 6), 1, (7, 6)),  # Moving toward trap
    ]
    for state, action, next_state in test_transitions:
        pred_reward = reward_fn(state, action, next_state)
        print(f"     {state} --[action {action}]--> {next_state}: reward = {pred_reward:.3f}")

    # Step 4: Create Q-learning agent
    print("\n4. Creating Q-learning agent...")
    q_learner = TabularQLearning(
        mdp=env,
        reward_fn=reward_fn,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,  # Start with high exploration
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42
    )
    print(f"   - Learning rate: {q_learner.learning_rate}")
    print(f"   - Discount factor: {q_learner.discount_factor}")
    print(f"   - Initial epsilon: {q_learner.epsilon}")
    print(f"   - Epsilon decay: {q_learner.epsilon_decay}")

    # Step 5: Train the agent
    print("\n5. Training Q-learning agent...")
    num_episodes = 2000
    max_steps_per_episode = 20
    eval_interval = 200

    q_learner.train(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        eval_interval=eval_interval,
        verbose=True
    )

    # Step 6: Evaluate the trained policy
    print("\n6. Evaluating trained policy...")

    # Evaluate with learned rewards
    print("\n   A. Evaluation using learned ensemble rewards:")
    metrics_learned = q_learner.evaluate(
        num_episodes=100,
        max_steps=20,
        use_greedy=True,
        use_true_rewards=False,
        seed=123
    )
    print(f"      Mean return: {metrics_learned['mean_return']:.3f} ± {metrics_learned['std_return']:.3f}")
    print(f"      Return range: [{metrics_learned['min_return']:.3f}, {metrics_learned['max_return']:.3f}]")
    print(f"      Mean episode length: {metrics_learned['mean_length']:.1f} ± {metrics_learned['std_length']:.1f}")
    print(f"      Success rate (reached treasure): {metrics_learned['success_rate']:.1%}")

    # Evaluate with true rewards for comparison
    print("\n   B. Evaluation using TRUE environment rewards:")
    metrics_true = q_learner.evaluate(
        num_episodes=100,
        max_steps=20,
        use_greedy=True,
        use_true_rewards=True,
        seed=123
    )
    print(f"      Mean return: {metrics_true['mean_return']:.3f} ± {metrics_true['std_return']:.3f}")
    print(f"      Return range: [{metrics_true['min_return']:.3f}, {metrics_true['max_return']:.3f}]")
    print(f"      Mean episode length: {metrics_true['mean_length']:.1f} ± {metrics_true['std_length']:.1f}")
    print(f"      Success rate (reached treasure): {metrics_true['success_rate']:.1%}")

    # Step 7: Compare with policy trained on true rewards
    print("\n7. Training baseline policy with TRUE rewards (for comparison)...")
    true_reward_fn = create_true_reward_fn(env)

    baseline_q_learner = TabularQLearning(
        mdp=env,
        reward_fn=true_reward_fn,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42
    )

    baseline_q_learner.train(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        eval_interval=eval_interval,
        verbose=True
    )

    print("\n   Evaluating baseline policy...")
    baseline_metrics = baseline_q_learner.evaluate(
        num_episodes=100,
        max_steps=100,
        use_greedy=True,
        use_true_rewards=True,
        seed=123
    )
    print(f"      Mean return: {baseline_metrics['mean_return']:.3f} ± {baseline_metrics['std_return']:.3f}")
    print(f"      Return range: [{baseline_metrics['min_return']:.3f}, {baseline_metrics['max_return']:.3f}]")
    print(f"      Mean episode length: {baseline_metrics['mean_length']:.1f} ± {baseline_metrics['std_length']:.1f}")
    print(f"      Success rate: {baseline_metrics['success_rate']:.1%}")

    # Step 8: Save the trained policies
    print("\n8. Saving trained policies...")
    q_learner.save('q_learning_ensemble_reward.npy')
    baseline_q_learner.save('q_learning_true_reward.npy')
    print("   - Saved ensemble-trained policy to: q_learning_ensemble_reward.npy")
    print("   - Saved baseline policy to: q_learning_true_reward.npy")

    # Step 9: Summary comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    print("\nPolicy trained with ENSEMBLE REWARDS (evaluated on true rewards):")
    print(f"  Mean return: {metrics_true['mean_return']:.3f} ± {metrics_true['std_return']:.3f}")
    print(f"  Success rate: {metrics_true['success_rate']:.1%}")

    print("\nPolicy trained with TRUE REWARDS:")
    print(f"  Mean return: {baseline_metrics['mean_return']:.3f} ± {baseline_metrics['std_return']:.3f}")
    print(f"  Success rate: {baseline_metrics['success_rate']:.1%}")

    print("\nPerformance gap:")
    gap = baseline_metrics['mean_return'] - metrics_true['mean_return']
    print(f"  True reward baseline - Ensemble reward: {gap:.3f}")
    if abs(gap) < 0.5:
        print("  -> Ensemble-trained policy performs comparably to ground truth!")
    elif gap > 0:
        print("  -> Ensemble-trained policy slightly underperforms")
    else:
        print("  -> Ensemble-trained policy outperforms (possibly due to noise in small sample)")

    print("\n" + "=" * 70)
    print("Training complete! Policies saved and ready for further experiments.")
    print("=" * 70)


if __name__ == '__main__':
    main()
