"""STARC canonicalization for reward models.

STARC (Scaling and Translating with a Reward Canonicalization) is a method
for canonicalizing reward functions to make them more comparable.

The canonicalization consists of two steps:
1. Apply a canonicalization function c(R) that transforms the reward
2. Apply a normalization function n(c(R)) to normalize the canonicalized reward

The final canonicalized reward is: c_R(s,a,s') = c(R)(s,a,s') / ||c(R)||
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Dict
from .reward_model import TransitionRewardModel


class MeanRewardModel(nn.Module):
    """Wrapper for the mean reward model from an ensemble."""

    def __init__(self, ensemble_models, device: str = 'cpu'):
        """
        Initialize mean reward model.

        Args:
            ensemble_models: List of models in the ensemble.
            device: Device to run on.
        """
        super().__init__()
        self.ensemble_models = ensemble_models
        self.device = device

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean reward predictions across ensemble.

        Args:
            states: Batch of states [batch_size, state_dim].
            actions: Batch of actions [batch_size].
            next_states: Batch of next states [batch_size, state_dim].

        Returns:
            Mean rewards [batch_size, 1].
        """
        rewards = []
        for model in self.ensemble_models:
            model.eval()
            with torch.no_grad():
                reward = model(states, actions, next_states)
                rewards.append(reward)

        # Stack and compute mean
        rewards_stacked = torch.stack(rewards, dim=0)
        mean_reward = torch.mean(rewards_stacked, dim=0)
        return mean_reward


class Canonicalizer(ABC):
    """Base class for canonicalization functions."""

    @abstractmethod
    def canonicalize(
        self,
        reward_model: TransitionRewardModel,
        mdp: Any,
        state_dim: int,
        device: str = 'cpu'
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Return a function that canonicalizes rewards.

        Args:
            reward_model: The reward model to canonicalize.
            mdp: The MDP environment.
            state_dim: Dimension of state representation.
            device: Device to run on ('cpu' or 'cuda').

        Returns:
            A function that takes (states, actions, next_states) tensors
            and returns canonicalized rewards.
        """
        pass


class ValueAdjustedLevelling(Canonicalizer):
    """
    Value-adjusted levelling canonicalization.

    c(R)(s,a,s') = R(s,a,s') - V(s) + gamma * V(s')

    This removes the value function component, making rewards more comparable
    across different state spaces and policies.
    """

    def __init__(self, discount_factor: float = 0.99):
        """
        Initialize value-adjusted levelling canonicalizer.

        Args:
            discount_factor: Discount factor (gamma).
        """
        self.discount_factor = discount_factor

    def _precompute_model_rewards(
        self,
        reward_model: TransitionRewardModel,
        mdp: Any,
        state_dim: int,
        device: str = 'cpu'
    ) -> Dict[tuple, float]:
        """
        Precompute rewards for all state-action-next_state transitions.

        This creates a lookup table for fast reward lookups during value iteration.
        Works with both deterministic and stochastic MDPs.

        Args:
            reward_model: The reward model to precompute.
            mdp: The MDP environment.
            state_dim: Dimension of state representation.
            device: Device to run on.

        Returns:
            Dictionary mapping (state, action, next_state) to reward.
        """
        precomputed_rewards = {}

        # Check if MDP has get_all_transitions (stochastic) or enumerate states manually (deterministic)
        if hasattr(mdp, 'get_all_transitions'):
            # Stochastic MDP: use all possible transitions
            all_transitions = mdp.get_all_transitions()

            for state, action, next_state, prob in all_transitions:
                # Convert states to tensors
                state_row, state_col = state
                state_onehot = np.zeros(state_dim)
                state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                action_tensor = torch.LongTensor([action]).to(device)

                # Get reward from model
                reward_model.eval()
                with torch.no_grad():
                    reward_pred = reward_model(state_tensor, action_tensor, next_state_tensor)
                    precomputed_rewards[(state, action, next_state)] = reward_pred.item()
        else:
            # Deterministic MDP: original logic
            if hasattr(mdp, 'grid_size'):
                size = mdp.grid_size
                all_states = [(x, y) for x in range(size) for y in range(size)]
            else:
                raise ValueError("MDP must have a 'grid_size' attribute")

            num_states = len(all_states)
            num_actions = mdp.get_num_actions()

            for state in all_states:
                for action in range(num_actions):
                    # Take action in environment
                    next_state, _, _ = mdp.step(state, action)

                    # Convert states to tensors
                    state_row, state_col = state
                    state_onehot = np.zeros(state_dim)
                    state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                    next_state_row, next_state_col = next_state
                    next_state_onehot = np.zeros(state_dim)
                    next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                    state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                    next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                    action_tensor = torch.LongTensor([action]).to(device)

                    # Get reward from model
                    reward_model.eval()
                    with torch.no_grad():
                        reward_pred = reward_model(state_tensor, action_tensor, next_state_tensor)
                        precomputed_rewards[(state, action, next_state)] = reward_pred.item()

        return precomputed_rewards

    def canonicalize(
        self,
        reward_model: TransitionRewardModel,
        mdp: Any,
        state_dim: int,
        device: str = 'cpu'
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Create a canonicalized reward function using value-adjusted levelling.

        Args:
            reward_model: The reward model to canonicalize.
            mdp: The MDP environment.
            state_dim: Dimension of state representation.
            device: Device to run on.

        Returns:
            A function that computes canonicalized rewards.
        """
        from ..algorithms import ValueIteration
        from ..policies import UniformPolicy

        # Compute value function under uniform random policy
        uniform_policy = UniformPolicy(num_actions=mdp.get_num_actions())

        # Precompute rewards for fast lookup during value iteration
        print("  - Precomputing rewards for value function calculation...")
        precomputed_rewards = self._precompute_model_rewards(
            reward_model, mdp, state_dim, device
        )

        # Create a fast lookup reward function from precomputed rewards
        def model_reward_fn(state, action, next_state):
            return precomputed_rewards.get((state, action, next_state), 0.0)

        # Compute value function
        vi = ValueIteration(
            mdp=mdp,
            reward_fn=model_reward_fn,
            discount_factor=self.discount_factor,
            convergence_threshold=1e-6,
            max_iterations=5000,
            seed=None
        )
        value_fn = vi.solve_for_all_states(uniform_policy, verbose=True)

        # Return canonicalization function
        def canonicalized_reward_fn(
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute canonicalized rewards.

            c(R)(s,a,s') = R(s,a,s') - V(s) + gamma * V(s')
            """
            # Get original rewards
            reward_model.eval()
            with torch.no_grad():
                original_rewards = reward_model(states, actions, next_states)

            # Convert states to state tuples for value lookup
            # Assumes states are one-hot encoded [batch_size, state_dim]
            batch_size = states.shape[0]
            state_values = []
            next_state_values = []

            for i in range(batch_size):
                # Find the state index from one-hot encoding
                state_idx = torch.argmax(states[i]).item()
                next_state_idx = torch.argmax(next_states[i]).item()

                grid_size = int(np.sqrt(state_dim))
                state = (state_idx // grid_size, state_idx % grid_size)
                next_state = (next_state_idx // grid_size, next_state_idx % grid_size)

                state_val = value_fn.get(state, 0.0)
                next_state_val = value_fn.get(next_state, 0.0)

                state_values.append(state_val)
                next_state_values.append(next_state_val)

            state_values = torch.FloatTensor(state_values).unsqueeze(1).to(device)
            next_state_values = torch.FloatTensor(next_state_values).unsqueeze(1).to(device)

            # Apply canonicalization: R(s,a,s') - V(s) + gamma * V(s')
            canonicalized = (
                original_rewards - state_values + self.discount_factor * next_state_values
            )

            return canonicalized

        return canonicalized_reward_fn


class Normalizer(ABC):
    """Base class for normalization functions."""

    def __init__(self):
        """Initialize normalizer."""
        self.normalization_constant = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        canonicalize_fn: Callable[[Any, int, Any], float],
        mdp: Any,
        state_dim: int,
        device: str = 'cpu'
    ):
        """
        Fit the normalizer on the entire function space.

        Args:
            canonicalize_fn: Function that takes (state, action, next_state) and returns
                           canonicalized reward.
            mdp: The MDP environment.
            state_dim: Dimension of state representation.
            device: Device to use ('cpu' or 'cuda').
        """
        pass

    @abstractmethod
    def normalize(self, canonicalized_rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize canonicalized rewards using fitted normalization constant.

        Args:
            canonicalized_rewards: Batch of canonicalized rewards [batch_size, 1].

        Returns:
            Normalized rewards [batch_size, 1].
        """
        pass


class L1Norm(Normalizer):
    """L1 norm normalization that normalizes over the entire function space."""

    def __init__(self):
        """Initialize L1 normalizer."""
        super().__init__()

    def fit(
        self,
        canonicalize_fn: Callable[[Any, int, Any], float],
        mdp: Any,
        state_dim: int,
        device: str = 'cpu'
    ):
        """
        Fit L1 norm constant over all state-action-next_state pairs.

        For stochastic MDPs: beta = Σ_s Σ_a Σ_s' P(s'|s,a) |c(R)(s,a,s')|
        For deterministic MDPs: beta = Σ_s Σ_a |c(R)(s,a,s')|
        """
        l1_sum = 0.0

        # Check if MDP has get_all_transitions (stochastic)
        if hasattr(mdp, 'get_all_transitions'):
            # Stochastic MDP: iterate over all possible transitions
            all_transitions = mdp.get_all_transitions()

            for state, action, next_state, prob in all_transitions:
                # Convert to tensors for canonicalize_fn
                state_row, state_col = state
                state_onehot = np.zeros(state_dim)
                state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                action_tensor = torch.LongTensor([action]).to(device)

                reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                # Include probability weighting for stochastic transitions
                l1_sum += prob * abs(reward.item())
        else:
            # Deterministic MDP: original logic
            if hasattr(mdp, 'grid_size'):
                size = mdp.grid_size
                all_states = [(x, y) for x in range(size) for y in range(size)]
            else:
                raise ValueError("MDP must have a 'grid_size' attribute")

            for state in all_states:
                for action in range(mdp.get_num_actions()):
                    next_state, _, _ = mdp.step(state, action)

                    # Convert to tensors for canonicalize_fn
                    state_row, state_col = state
                    state_onehot = np.zeros(state_dim)
                    state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                    next_state_row, next_state_col = next_state
                    next_state_onehot = np.zeros(state_dim)
                    next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                    state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                    next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                    action_tensor = torch.LongTensor([action]).to(device)

                    reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                    l1_sum += abs(reward.item())

        self.normalization_constant = l1_sum if l1_sum > 0 else 1e-8
        self.is_fitted = True

    def normalize(self, canonicalized_rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize using fitted L1 norm constant.

        norm = (c(R) / beta) where beta = Σ_s Σ_a |c(R)(s,a,s')|
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted first. Call fit()")
        return canonicalized_rewards / self.normalization_constant


class L2Norm(Normalizer):
    """L2 norm normalization that normalizes over the entire function space."""

    def __init__(self):
        """Initialize L2 normalizer."""
        super().__init__()

    def fit(
        self,
        canonicalize_fn: Callable[[Any, int, Any], float],
        mdp: Any,
        state_dim: int,
        device: str = 'cpu'
    ):
        """
        Fit L2 norm constant over all state-action-next_state pairs.

        For stochastic MDPs: beta = sqrt(Σ_s Σ_a Σ_s' P(s'|s,a) (c(R)(s,a,s'))^2)
        For deterministic MDPs: beta = sqrt(Σ_s Σ_a (c(R)(s,a,s'))^2)
        """
        l2_sum = 0.0

        # Check if MDP has get_all_transitions (stochastic)
        if hasattr(mdp, 'get_all_transitions'):
            # Stochastic MDP: iterate over all possible transitions
            all_transitions = mdp.get_all_transitions()

            for state, action, next_state, prob in all_transitions:
                # Convert to tensors for canonicalize_fn
                state_row, state_col = state
                state_onehot = np.zeros(state_dim)
                state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                action_tensor = torch.LongTensor([action]).to(device)

                reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                # Include probability weighting for stochastic transitions
                l2_sum += prob * (reward.item() ** 2)
        else:
            # Deterministic MDP: original logic
            if hasattr(mdp, 'grid_size'):
                size = mdp.grid_size
                all_states = [(x, y) for x in range(size) for y in range(size)]
            else:
                raise ValueError("MDP must have a 'grid_size' attribute")

            for state in all_states:
                for action in range(mdp.get_num_actions()):
                    next_state, _, _ = mdp.step(state, action)

                    # Convert to tensors for canonicalize_fn
                    state_row, state_col = state
                    state_onehot = np.zeros(state_dim)
                    state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                    next_state_row, next_state_col = next_state
                    next_state_onehot = np.zeros(state_dim)
                    next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                    state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                    next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                    action_tensor = torch.LongTensor([action]).to(device)

                    reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                    l2_sum += reward.item() ** 2

        self.normalization_constant = np.sqrt(l2_sum) if l2_sum > 0 else 1e-8
        self.is_fitted = True

    def normalize(self, canonicalized_rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize using fitted L2 norm constant.

        norm = (c(R) / beta) where beta = sqrt(Σ_s Σ_a (c(R)(s,a,s'))^2)
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted first. Call fit()")
        return canonicalized_rewards / self.normalization_constant


class MaxNorm(Normalizer):
    """Max norm normalization that normalizes over the entire function space."""

    def __init__(self):
        """Initialize max normalizer."""
        super().__init__()

    def fit(
        self,
        canonicalize_fn: Callable[[Any, int, Any], float],
        mdp: Any,
        state_dim: int,
        device: str = 'cpu'
    ):
        """
        Fit max norm constant over all state-action-next_state pairs.

        For stochastic MDPs: beta = max_s max_a max_s' |c(R)(s,a,s')|
        For deterministic MDPs: beta = max_s max_a |c(R)(s,a,s')|
        """
        max_val = 0.0

        # Check if MDP has get_all_transitions (stochastic)
        if hasattr(mdp, 'get_all_transitions'):
            # Stochastic MDP: iterate over all possible transitions
            all_transitions = mdp.get_all_transitions()

            for state, action, next_state, prob in all_transitions:
                # Convert to tensors for canonicalize_fn
                state_row, state_col = state
                state_onehot = np.zeros(state_dim)
                state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                action_tensor = torch.LongTensor([action]).to(device)

                reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                max_val = max(max_val, abs(reward.item()))
        else:
            # Deterministic MDP: original logic
            if hasattr(mdp, 'grid_size'):
                size = mdp.grid_size
                all_states = [(x, y) for x in range(size) for y in range(size)]
            else:
                raise ValueError("MDP must have a 'grid_size' attribute")

            for state in all_states:
                for action in range(mdp.get_num_actions()):
                    next_state, _, _ = mdp.step(state, action)

                    # Convert to tensors for canonicalize_fn
                    state_row, state_col = state
                    state_onehot = np.zeros(state_dim)
                    state_onehot[state_row * int(np.sqrt(state_dim)) + state_col] = 1.0

                    next_state_row, next_state_col = next_state
                    next_state_onehot = np.zeros(state_dim)
                    next_state_onehot[next_state_row * int(np.sqrt(state_dim)) + next_state_col] = 1.0

                    state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(device)
                    next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(device)
                    action_tensor = torch.LongTensor([action]).to(device)

                    reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                    max_val = max(max_val, abs(reward.item()))

        self.normalization_constant = max_val if max_val > 0 else 1e-8
        self.is_fitted = True

    def normalize(self, canonicalized_rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize using fitted max norm constant.

        norm = (c(R) / beta) where beta = max_s max_a |c(R)(s,a,s')|
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted first. Call fit()")
        return canonicalized_rewards / self.normalization_constant


class CanonicalizedRewardModel(nn.Module):
    """
    Wrapper around a TransitionRewardModel with canonicalization.

    This model applies canonicalization and normalization to transform
    the original reward predictions into a canonicalized space.
    """

    def __init__(
        self,
        reward_model: TransitionRewardModel,
        canonicalizer: Canonicalizer,
        normalizer: Normalizer,
        mdp: Any,
        state_dim: int,
        device: str = 'cpu',
        rescaling_constant: float = 1.0
    ):
        """
        Initialize canonicalized reward model.

        Args:
            reward_model: The original reward model.
            canonicalizer: The canonicalization function.
            normalizer: The normalization function.
            mdp: The MDP environment.
            state_dim: Dimension of state representation.
            device: Device to run on.
            rescaling_constant: Constant to rescale normalized rewards. Defaults to 1.0 (no rescaling).
        """
        super().__init__()

        self.reward_model = reward_model
        self.canonicalizer = canonicalizer
        self.normalizer = normalizer
        self.mdp = mdp
        self.state_dim = state_dim
        self.device = device
        self.rescaling_constant = rescaling_constant

        # Precompute canonicalization function
        self.canonicalize_fn = canonicalizer.canonicalize(
            reward_model, mdp, state_dim, device
        )

        # Fit normalizer on the entire canonicalized reward function space
        print("  - Fitting normalization constant...")
        self.normalizer.fit(self.canonicalize_fn, mdp, state_dim, device)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute canonicalized and normalized rewards.

        Args:
            states: Batch of states [batch_size, state_dim].
            actions: Batch of actions [batch_size].
            next_states: Batch of next states [batch_size, state_dim].

        Returns:
            Canonicalized and normalized rewards [batch_size, 1].
        """
        # Apply canonicalization
        canonicalized = self.canonicalize_fn(states, actions, next_states)

        # Apply normalization
        normalized = self.normalizer.normalize(canonicalized)

        # Apply rescaling
        rescaled = normalized * self.rescaling_constant

        return rescaled


class CanonicalizedRewardEnsemble:
    """
    Ensemble of canonicalized reward models.

    Takes a trained RewardModelEnsemble and applies STARC canonicalization
    to each model individually, resulting in comparable reward models.
    """

    def __init__(
        self,
        ensemble,
        canonicalizer: Optional[Canonicalizer] = None,
        normalizer: Optional[Normalizer] = None,
        mdp: Optional[Any] = None,
        state_dim: Optional[int] = None,
        device: str = 'cpu'
    ):
        """
        Initialize canonicalized ensemble.

        Args:
            ensemble: The trained RewardModelEnsemble.
            canonicalizer: The canonicalization function. Defaults to ValueAdjustedLevelling.
            normalizer: The normalization function. Defaults to L1Norm.
            mdp: The MDP environment.
            state_dim: Dimension of state representation.
            device: Device to run on.
        """
        self.ensemble = ensemble
        self.mdp = mdp
        self.state_dim = state_dim
        self.device = device

        # Default to ValueAdjustedLevelling and L1Norm
        self.canonicalizer = canonicalizer or ValueAdjustedLevelling(discount_factor=0.99)
        self.normalizer = normalizer or L1Norm()

        self.canonicalized_models = []
        self.is_canonicalized = False
        self.rescaling_constant = 1.0

    def _compute_mean_reward_model(self):
        """
        Create a mean reward model from the ensemble.

        Returns:
            A MeanRewardModel instance that computes mean rewards across the ensemble.
        """
        mean_model = MeanRewardModel(self.ensemble.models, device=self.device)
        return mean_model

    def _compute_rescaling_constant(self) -> float:
        """
        Compute rescaling constant based on the norm of the mean reward model.

        Uses the same norm type as the normalizer to ensure consistency.
        The constant is computed after canonicalization but before the normalization step.

        Returns:
            Rescaling constant C.
        """
        print("  - Computing rescaling constant from mean reward model...")

        # Get mean reward model
        mean_reward_fn = self._compute_mean_reward_model()

        # Canonicalize the mean reward model
        canonicalize_fn = self.canonicalizer.canonicalize(
            mean_reward_fn, self.mdp, self.state_dim, self.device
        )

        # Compute norm of canonicalized mean rewards using same norm type as normalizer
        if hasattr(self.mdp, 'grid_size'):
            size = self.mdp.grid_size
            all_states = [(x, y) for x in range(size) for y in range(size)]
        else:
            raise ValueError("MDP must have a 'grid_size' attribute")

        if isinstance(self.normalizer, L1Norm):
            norm_value = self._compute_l1_norm(canonicalize_fn, all_states)
        elif isinstance(self.normalizer, L2Norm):
            norm_value = self._compute_l2_norm(canonicalize_fn, all_states)
        elif isinstance(self.normalizer, MaxNorm):
            norm_value = self._compute_max_norm(canonicalize_fn, all_states)
        else:
            raise ValueError(f"Unknown normalizer type: {type(self.normalizer)}")

        rescaling_constant = norm_value if norm_value > 0 else 1.0
        print(f"    Rescaling constant: {rescaling_constant:.6f}")

        return rescaling_constant

    def _compute_l1_norm(self, canonicalize_fn, all_states) -> float:
        """Compute L1 norm of canonicalized rewards."""
        l1_sum = 0.0

        # Check if MDP has get_all_transitions (stochastic)
        if hasattr(self.mdp, 'get_all_transitions'):
            # Stochastic MDP: iterate over all transitions
            all_transitions = self.mdp.get_all_transitions()
            for state, action, next_state, prob in all_transitions:
                # Convert to tensors
                state_row, state_col = state
                state_onehot = np.zeros(self.state_dim)
                state_onehot[state_row * int(np.sqrt(self.state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(self.state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(self.state_dim)) + next_state_col] = 1.0

                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)

                reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                l1_sum += prob * abs(reward.item())
        else:
            # Deterministic MDP: original logic
            for state in all_states:
                for action in range(self.mdp.get_num_actions()):
                    next_state, _, _ = self.mdp.step(state, action)

                    # Convert to tensors
                    state_row, state_col = state
                    state_onehot = np.zeros(self.state_dim)
                    state_onehot[state_row * int(np.sqrt(self.state_dim)) + state_col] = 1.0

                    next_state_row, next_state_col = next_state
                    next_state_onehot = np.zeros(self.state_dim)
                    next_state_onehot[next_state_row * int(np.sqrt(self.state_dim)) + next_state_col] = 1.0

                    state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
                    next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(self.device)
                    action_tensor = torch.LongTensor([action]).to(self.device)

                    reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                    l1_sum += abs(reward.item())

        return l1_sum

    def _compute_l2_norm(self, canonicalize_fn, all_states) -> float:
        """Compute L2 norm of canonicalized rewards."""
        l2_sum = 0.0

        # Check if MDP has get_all_transitions (stochastic)
        if hasattr(self.mdp, 'get_all_transitions'):
            # Stochastic MDP: iterate over all transitions
            all_transitions = self.mdp.get_all_transitions()
            for state, action, next_state, prob in all_transitions:
                # Convert to tensors
                state_row, state_col = state
                state_onehot = np.zeros(self.state_dim)
                state_onehot[state_row * int(np.sqrt(self.state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(self.state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(self.state_dim)) + next_state_col] = 1.0

                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)

                reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                l2_sum += prob * (reward.item() ** 2)
        else:
            # Deterministic MDP: original logic
            for state in all_states:
                for action in range(self.mdp.get_num_actions()):
                    next_state, _, _ = self.mdp.step(state, action)

                    # Convert to tensors
                    state_row, state_col = state
                    state_onehot = np.zeros(self.state_dim)
                    state_onehot[state_row * int(np.sqrt(self.state_dim)) + state_col] = 1.0

                    next_state_row, next_state_col = next_state
                    next_state_onehot = np.zeros(self.state_dim)
                    next_state_onehot[next_state_row * int(np.sqrt(self.state_dim)) + next_state_col] = 1.0

                    state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
                    next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(self.device)
                    action_tensor = torch.LongTensor([action]).to(self.device)

                    reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                    l2_sum += reward.item() ** 2

        return np.sqrt(l2_sum)

    def _compute_max_norm(self, canonicalize_fn, all_states) -> float:
        """Compute max norm of canonicalized rewards."""
        max_val = 0.0

        # Check if MDP has get_all_transitions (stochastic)
        if hasattr(self.mdp, 'get_all_transitions'):
            # Stochastic MDP: iterate over all transitions
            all_transitions = self.mdp.get_all_transitions()
            for state, action, next_state, prob in all_transitions:
                # Convert to tensors
                state_row, state_col = state
                state_onehot = np.zeros(self.state_dim)
                state_onehot[state_row * int(np.sqrt(self.state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(self.state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(self.state_dim)) + next_state_col] = 1.0

                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)

                reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                max_val = max(max_val, abs(reward.item()))
        else:
            # Deterministic MDP: original logic
            for state in all_states:
                for action in range(self.mdp.get_num_actions()):
                    next_state, _, _ = self.mdp.step(state, action)

                    # Convert to tensors
                    state_row, state_col = state
                    state_onehot = np.zeros(self.state_dim)
                    state_onehot[state_row * int(np.sqrt(self.state_dim)) + state_col] = 1.0

                    next_state_row, next_state_col = next_state
                    next_state_onehot = np.zeros(self.state_dim)
                    next_state_onehot[next_state_row * int(np.sqrt(self.state_dim)) + next_state_col] = 1.0

                    state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
                    next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(self.device)
                    action_tensor = torch.LongTensor([action]).to(self.device)

                    reward = canonicalize_fn(state_tensor, action_tensor, next_state_tensor)
                    max_val = max(max_val, abs(reward.item()))

        return max_val

    def canonicalize(self, verbose: bool = True):
        """
        Canonicalize all models in the ensemble.

        All models share the same normalizer instance so they use the same
        normalization constant computed over the entire ensemble's average behavior.

        Args:
            verbose: If True, print progress.
        """
        if verbose:
            print(f"Canonicalizing {len(self.ensemble.models)} reward models...")

        # Compute rescaling constant based on mean reward model
        if verbose:
            print("\nComputing rescaling constant...")
        self.rescaling_constant = self._compute_rescaling_constant()

        self.canonicalized_models = []

        # Note: We create fresh normalizer instances for each ensemble canonicalization
        # but the key point is that each CanonicalizedRewardModel will fit its own
        # normalizer based on its canonicalized reward function
        # This ensures each model is normalized consistently within its own function space

        for idx, model in enumerate(self.ensemble.models):
            if verbose:
                print(f"\n  Model {idx + 1}/{len(self.ensemble.models)}:")

            # Create a fresh normalizer for this model
            # It will be fitted during CanonicalizedRewardModel initialization
            from copy import deepcopy
            model_normalizer = deepcopy(self.normalizer)

            canonicalized_model = CanonicalizedRewardModel(
                reward_model=model,
                canonicalizer=self.canonicalizer,
                normalizer=model_normalizer,
                mdp=self.mdp,
                state_dim=self.state_dim,
                device=self.device,
                rescaling_constant=self.rescaling_constant
            )

            self.canonicalized_models.append(canonicalized_model)

        self.is_canonicalized = True

        if verbose:
            print(f"\n  Canonicalization complete!")

    def predict_returns(
        self,
        trajectory,
        return_std: bool = False,
        discount_factor: float = 0.99
    ) -> tuple:
        """
        Predict the return of a trajectory using canonicalized models.

        Args:
            trajectory: Trajectory to evaluate.
            return_std: If True, also return standard deviation.
            discount_factor: Discount factor.

        Returns:
            Mean predicted return (and optionally std dev).
        """
        if not self.is_canonicalized:
            raise RuntimeError("Ensemble must be canonicalized first. Call canonicalize()")

        from .reward_model import trajectory_to_tensors

        states, actions, next_states = trajectory_to_tensors(
            trajectory, self.state_dim, self.device
        )

        predictions = []

        for model in self.canonicalized_models:
            model.eval()
            with torch.no_grad():
                rewards = model(states, actions, next_states).squeeze(-1)

                # Compute discounted return
                discount_factors = discount_factor ** torch.arange(
                    len(rewards), device=self.device, dtype=torch.float32
                )
                pred_return = torch.sum(rewards * discount_factors)
                predictions.append(pred_return.item())

        mean_pred = np.mean(predictions)

        if return_std:
            std_pred = np.std(predictions)
            return mean_pred, std_pred
        else:
            return mean_pred

    def get_canonicalized_models(self):
        """Get the list of canonicalized models."""
        if not self.is_canonicalized:
            raise RuntimeError("Ensemble must be canonicalized first. Call canonicalize()")
        return self.canonicalized_models
