"""Tabular Q-learning implementation with flexible reward functions."""

import numpy as np
from typing import Callable, Any, Dict, Tuple, Optional, List
from ..policies import Policy, EpsilonGreedyPolicy


class TabularQLearning:
    """
    Tabular Q-learning algorithm with support for custom reward functions.

    This implementation allows you to train policies using any reward function,
    including learned reward models from ensembles.
    """

    def __init__(
        self,
        mdp: Any,
        reward_fn: Callable[[Any, int, Any], float],
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize tabular Q-learning agent.

        Args:
            mdp: The MDP environment to train on.
            reward_fn: Reward function that takes (state, action, next_state) and returns a reward.
                      This allows using custom rewards like ensemble means.
            learning_rate: Learning rate (alpha) for Q-value updates.
            discount_factor: Discount factor (gamma) for future rewards.
            epsilon: Initial exploration rate for epsilon-greedy policy.
            epsilon_decay: Multiplicative decay factor for epsilon after each episode.
            epsilon_min: Minimum epsilon value.
            seed: Random seed for reproducibility.
        """
        self.mdp = mdp
        self.reward_fn = reward_fn
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        if seed is not None:
            np.random.seed(seed)

        # Initialize Q-table as dictionary: (state, action) -> value
        self.q_table: Dict[Tuple[Any, int], float] = {}

        # Create epsilon-greedy policy
        self.policy = EpsilonGreedyPolicy(
            num_actions=mdp.get_num_actions(),
            epsilon=epsilon
        )

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def get_q_value(self, state: Any, action: int) -> float:
        """Get Q-value for a state-action pair, defaulting to 0 if not seen."""
        return self.q_table.get((state, action), 0.0)

    def get_max_q_value(self, state: Any) -> float:
        """Get maximum Q-value for a state across all actions."""
        q_values = [self.get_q_value(state, a) for a in range(self.mdp.get_num_actions())]
        return max(q_values) if q_values else 0.0

    def get_state_q_values(self, state: Any) -> np.ndarray:
        """Get Q-values for all actions in a state."""
        return np.array([
            self.get_q_value(state, a)
            for a in range(self.mdp.get_num_actions())
        ])

    def update_policy_for_state(self, state: Any):
        """Update the epsilon-greedy policy's Q-values for a state."""
        q_values = self.get_state_q_values(state)
        self.policy.set_q_values(state, q_values)

    def train_episode(
        self,
        max_steps: int = 100,
        verbose: bool = False
    ) -> Tuple[float, int]:
        """
        Train for one episode using Q-learning.

        Args:
            max_steps: Maximum number of steps per episode.
            verbose: If True, print episode progress.

        Returns:
            Tuple of (total_reward, episode_length)
        """
        state = self.mdp.reset()
        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # Update policy for current state
            self.update_policy_for_state(state)

            # Select action using epsilon-greedy policy
            action = self.policy.sample_action(state)

            # Take action in environment (using true dynamics)
            next_state, true_reward, done = self.mdp.step(state, action)

            # Get reward from custom reward function
            reward = self.reward_fn(state, action, next_state)

            # Q-learning update
            current_q = self.get_q_value(state, action)
            max_next_q = self.get_max_q_value(next_state)

            # TD update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )

            self.q_table[(state, action)] = new_q

            total_reward += reward
            steps += 1

            if verbose and step % 10 == 0:
                print(f"  Step {step}: state={state}, action={action}, reward={reward:.3f}, q_value={new_q:.3f}")

            if done:
                break

            state = next_state

        return total_reward, steps

    def train(
        self,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100,
        eval_interval: int = 100,
        verbose: bool = True
    ):
        """
        Train the Q-learning agent for multiple episodes.

        Args:
            num_episodes: Number of episodes to train.
            max_steps_per_episode: Maximum steps per episode.
            eval_interval: Evaluate and print stats every N episodes.
            verbose: If True, print training progress.
        """
        if verbose:
            print(f"Starting Q-learning training for {num_episodes} episodes...")
            print(f"Learning rate: {self.learning_rate}, Discount: {self.discount_factor}")
            print(f"Epsilon: {self.epsilon} (decay: {self.epsilon_decay}, min: {self.epsilon_min})")

        for episode in range(num_episodes):
            # Train one episode
            episode_reward, episode_length = self.train_episode(
                max_steps=max_steps_per_episode,
                verbose=False
            )

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.policy.epsilon = self.epsilon

            # Print progress
            if verbose and (episode + 1) % eval_interval == 0:
                recent_rewards = self.episode_rewards[-eval_interval:]
                recent_lengths = self.episode_lengths[-eval_interval:]
                print(f"Episode {episode + 1}/{num_episodes}:")
                print(f"  Mean reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}")
                print(f"  Mean length: {np.mean(recent_lengths):.1f} ± {np.std(recent_lengths):.1f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Q-table size: {len(self.q_table)}")

        if verbose:
            print("\nTraining complete!")
            print(f"Final Q-table size: {len(self.q_table)}")

    def get_greedy_policy(self) -> Policy:
        """
        Get a greedy policy based on learned Q-values (epsilon=0).

        Returns:
            Greedy policy for evaluation.
        """
        greedy_policy = EpsilonGreedyPolicy(
            num_actions=self.mdp.get_num_actions(),
            epsilon=0.0
        )

        # Update Q-values for all states we've seen
        for (state, action), _ in self.q_table.items():
            if state not in [s for s, _ in greedy_policy.q_values.keys() if hasattr(greedy_policy.q_values, 'keys')]:
                q_values = self.get_state_q_values(state)
                greedy_policy.set_q_values(state, q_values)

        return greedy_policy

    def evaluate(
        self,
        num_episodes: int = 100,
        max_steps: int = 100,
        use_greedy: bool = True,
        use_true_rewards: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the learned policy.

        Args:
            num_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.
            use_greedy: If True, use greedy policy (epsilon=0). Otherwise use current epsilon.
            use_true_rewards: If True, compute returns using true environment rewards.
                            If False, use the reward function the agent was trained with.
            seed: Random seed for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        if seed is not None:
            np.random.seed(seed)

        # Save current epsilon and set to 0 for greedy evaluation
        original_epsilon = self.policy.epsilon
        if use_greedy:
            self.policy.epsilon = 0.0

        returns = []
        lengths = []
        success_count = 0

        for episode in range(num_episodes):
            state = self.mdp.reset()
            episode_return = 0.0
            steps = 0

            for step in range(max_steps):
                # Update policy for current state
                self.update_policy_for_state(state)

                # Select action
                action = self.policy.sample_action(state)

                # Take action
                next_state, true_reward, done = self.mdp.step(state, action)

                # Use appropriate reward for evaluation
                if use_true_rewards:
                    reward = true_reward
                else:
                    reward = self.reward_fn(state, action, next_state)

                episode_return += reward * (self.discount_factor ** steps)
                steps += 1

                # Check if reached goal (for SparseSummitMDP, this is (7,7))
                if hasattr(self.mdp, 'treasure') and next_state == self.mdp.treasure:
                    success_count += 1

                if done:
                    break

                state = next_state

            returns.append(episode_return)
            lengths.append(steps)

        # Restore original epsilon
        self.policy.epsilon = original_epsilon

        # Compute metrics
        metrics = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': success_count / num_episodes,
            'num_episodes': num_episodes
        }

        return metrics

    def save(self, path: str):
        """Save Q-table and training statistics to disk."""
        save_dict = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        np.save(path, save_dict)
        print(f"Q-learning agent saved to {path}")

    def load(self, path: str):
        """Load Q-table and training statistics from disk."""
        save_dict = np.load(path, allow_pickle=True).item()
        self.q_table = save_dict['q_table']
        self.learning_rate = save_dict['learning_rate']
        self.discount_factor = save_dict['discount_factor']
        self.epsilon = save_dict['epsilon']
        self.initial_epsilon = save_dict['initial_epsilon']
        self.epsilon_decay = save_dict['epsilon_decay']
        self.epsilon_min = save_dict['epsilon_min']
        self.episode_rewards = save_dict['episode_rewards']
        self.episode_lengths = save_dict['episode_lengths']
        print(f"Q-learning agent loaded from {path}")
