"""Value iteration algorithm for computing optimal state values."""

import numpy as np
from typing import Callable, Any, Dict, Optional, Tuple
from ..policies import Policy, EpsilonGreedyPolicy


class ValueIteration:
    """
    Value iteration algorithm for computing state values.

    This implementation computes the value function V(s) for all states
    given a policy and reward function. It can be used to evaluate the
    expected return from each state under the given policy.
    """

    def __init__(
        self,
        mdp: Any,
        reward_fn: Callable[[Any, int, Any], float],
        discount_factor: float = 0.99,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize value iteration solver.

        Args:
            mdp: The MDP environment to compute values for.
            reward_fn: Reward function that takes (state, action, next_state) and returns a reward.
                      This allows using custom rewards like ensemble means.
            discount_factor: Discount factor (gamma) for future rewards.
            convergence_threshold: Convergence threshold for value updates.
            max_iterations: Maximum number of iterations to run.
            seed: Random seed for reproducibility.
        """
        self.mdp = mdp
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

        if seed is not None:
            np.random.seed(seed)

        # Initialize value function: state -> value
        self.value_function: Dict[Any, float] = {}

        # Track convergence
        self.iteration_history: list = []
        self.converged = False
        self.num_iterations = 0

    def get_value(self, state: Any) -> float:
        """Get value for a state, defaulting to 0 if not computed."""
        return self.value_function.get(state, 0.0)

    def set_value(self, state: Any, value: float):
        """Set value for a state."""
        self.value_function[state] = value

    def compute_state_value(self, state: Any, policy: Policy) -> float:
        """
        Compute the value of a state under a given policy.

        Uses the Bellman expectation equation:
        V(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]

        Works with both deterministic and stochastic MDPs.

        Args:
            state: The state to compute value for.
            policy: The policy to evaluate.

        Returns:
            The expected value of the state.
        """
        action_probs = policy.get_action_probabilities(state)
        state_value = 0.0

        for action in range(self.mdp.get_num_actions()):
            action_value = 0.0

            # Check if MDP has get_all_transitions (stochastic)
            if hasattr(self.mdp, 'get_all_transitions'):
                # Stochastic MDP: need to handle multiple next states
                # Get all transitions for this state-action pair
                all_transitions = self.mdp.get_all_transitions()
                for s, a, next_state, prob in all_transitions:
                    if s == state and a == action:
                        # Get reward for this transition
                        reward = self.reward_fn(state, action, next_state)

                        # Bellman update: V(s) = π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
                        action_value += prob * (reward + self.discount_factor * self.get_value(next_state))
            else:
                # Deterministic MDP: original logic
                next_state, _, _ = self.mdp.step(state, action)

                # Get reward for this transition
                reward = self.reward_fn(state, action, next_state)

                # Bellman update: V(s) = π(a|s) * [R(s,a,s') + γ * V(s')]
                action_value = reward + self.discount_factor * self.get_value(next_state)

            # Accumulate weighted by policy probability
            state_value += action_probs[action] * action_value

        return state_value

    def solve(self, policy: Policy, verbose: bool = True) -> Dict[Any, float]:
        """
        Compute state values for all reachable states under a policy.

        Uses iterative policy evaluation with value iteration.

        Args:
            policy: The policy to evaluate.
            verbose: If True, print convergence progress.

        Returns:
            Dictionary mapping states to their values.
        """
        if verbose:
            print(f"Starting value iteration with policy evaluation...")
            print(f"Discount factor: {self.discount_factor}")
            print(f"Convergence threshold: {self.convergence_threshold}")
            print(f"Max iterations: {self.max_iterations}")

        self.value_function = {}
        self.iteration_history = []

        for iteration in range(self.max_iterations):
            # Store old values to check convergence
            old_value = dict(self.value_function)
            max_delta = 0.0

            # First, we need to enumerate reachable states
            # We'll iterate through states as we discover them during rollouts
            # For now, collect all states we might need
            states_to_update = set(old_value.keys())

            # Also add new states discovered from environment
            # Sample a few trajectories to discover states
            for _ in range(min(10, self.mdp.get_num_states())):
                state = self.mdp.reset()
                for _ in range(100):
                    if state not in states_to_update:
                        states_to_update.add(state)
                    action = np.random.randint(0, self.mdp.get_num_actions())
                    next_state, _, done = self.mdp.step(state, action)
                    state = next_state
                    if done:
                        break

            # Update values for all states
            for state in states_to_update:
                new_value = self.compute_state_value(state, policy)
                self.set_value(state, new_value)

                # Track maximum change
                old_val = old_value.get(state, 0.0)
                delta = abs(new_value - old_val)
                max_delta = max(max_delta, delta)

            self.iteration_history.append(max_delta)
            self.num_iterations = iteration + 1

            if verbose and (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                print(f"Iteration {iteration + 1}: max_delta={max_delta:.6f}, states={len(states_to_update)}")

            # Check convergence
            if max_delta < self.convergence_threshold:
                self.converged = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations!")
                break

        if verbose and not self.converged:
            print(f"Reached max iterations ({self.max_iterations}) without convergence.")
            print(f"Final max_delta: {max_delta:.6f}")

        return self.value_function

    def solve_for_all_states(self, policy: Policy, verbose: bool = True) -> Dict[Any, float]:
        """
        Compute state values for ALL states in the MDP (if enumerable).

        This is more thorough than solve() and enumerates all states explicitly.

        Args:
            policy: The policy to evaluate.
            verbose: If True, print convergence progress.

        Returns:
            Dictionary mapping states to their values.
        """
        # Try to enumerate all states
        num_states = self.mdp.get_num_states()

        if verbose:
            print(f"Starting value iteration for all {num_states} states...")
            print(f"Discount factor: {self.discount_factor}")
            print(f"Convergence threshold: {self.convergence_threshold}")

        self.value_function = {}
        self.iteration_history = []

        # Get all states - this assumes the environment has a way to enumerate states
        # For gridworlds, we can reconstruct states from coordinates
        all_states = []
        if hasattr(self.mdp, 'size'):  # GridWorld
            size = self.mdp.size
            for x in range(size):
                for y in range(size):
                    all_states.append((x, y))
        else:
            # Fall back to solve() which discovers states incrementally
            return self.solve(policy, verbose=verbose)

        for iteration in range(self.max_iterations):
            max_delta = 0.0

            for state in all_states:
                old_value = self.get_value(state)
                new_value = self.compute_state_value(state, policy)
                self.set_value(state, new_value)

                delta = abs(new_value - old_value)
                max_delta = max(max_delta, delta)

            self.iteration_history.append(max_delta)
            self.num_iterations = iteration + 1

            if verbose and (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                print(f"Iteration {iteration + 1}: max_delta={max_delta:.6f}")

            # Check convergence
            if max_delta < self.convergence_threshold:
                self.converged = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations!")
                break

        if verbose and not self.converged:
            print(f"Reached max iterations ({self.max_iterations}) without convergence.")
            print(f"Final max_delta: {max_delta:.6f}")

        return self.value_function

    def get_state_values_as_array(self, size: Optional[int] = None) -> np.ndarray:
        """
        Get state values as a 2D array (for gridworld visualization).

        Args:
            size: Grid size. If None, infers from MDP.

        Returns:
            2D numpy array of state values.
        """
        if size is None:
            if hasattr(self.mdp, 'size'):
                size = self.mdp.size
            elif hasattr(self.mdp, 'grid_size'):
                size = self.mdp.grid_size
            else:
                raise ValueError("Cannot infer grid size. Please provide size parameter.")

        values = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                state = (x, y)
                values[x, y] = self.get_value(state)

        return values
