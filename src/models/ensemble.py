"""Ensemble training for reward models."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from .reward_model import TransitionRewardModel, trajectory_to_tensors, preprocess_trajectory_pairs


class RewardModelEnsemble:
    """Ensemble of transition reward models."""

    def __init__(
        self,
        ensemble_size: int,
        state_dim: int,
        num_actions: int,
        hidden_dims: list = [64, 64],
        activation: str = 'relu',
        lr: float = 1e-3,
        max_steps: int = 20,
        device: str = 'cpu'
    ):
        """
        Initialize reward model ensemble.

        Args:
            ensemble_size: Number of models in the ensemble.
            state_dim: Dimension of state representation.
            num_actions: Number of possible actions.
            hidden_dims: Hidden layer dimensions for each model.
            activation: Activation function.
            lr: Learning rate.
            device: Device to use ('cpu' or 'cuda').
        """
        self.ensemble_size = ensemble_size
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.device = device

        # Create ensemble of models
        self.models = []
        self.optimizers = []

        for _ in range(ensemble_size):
            model = TransitionRewardModel(
                state_dim=state_dim,
                num_actions=num_actions,
                hidden_dims=hidden_dims,
                activation=activation
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)

            self.models.append(model)
            self.optimizers.append(optimizer)

    def train_step(
        self,
        trajectory_pairs_tensors: List[Tuple],
        preferences: np.ndarray,
        batch_indices_per_model: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        Perform one training step on the ensemble.

        Args:
            trajectory_pairs: List of (trajectory1, trajectory2) tuples.
            preferences: Array of preference labels (0 or 1).
            batch_indices_per_model: Optional indices to use for this batch.

        Returns:
            List of losses for each model.
        """
        if batch_indices_per_model is None:
            batch_indices_per_model = [np.arange(len(trajectory_pairs_tensors)) for _ in range(self.ensemble_size)]

        losses = []

        discount_factor = 0.99
        self.discount_cache = {
            L: (discount_factor ** torch.arange(L, device=self.device, dtype=torch.float32))
            for L in range(1, self.max_steps + 1)
        }

        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            optimizer.zero_grad()
            model.train()

            batch_returns_1 = []
            batch_returns_2 = []
            batch_targets = []

            # ---- compute discounted returns for all pairs ----
            for idx in batch_indices_per_model[model_idx]:
                traj1, traj2 = trajectory_pairs_tensors[idx]
                preference = preferences[idx]

                states1, actions1, next_states1 = traj1
                states2, actions2, next_states2 = traj2

                # Compute predicted transition rewards
                rewards1 = model(states1, actions1, next_states1).squeeze(-1)
                rewards2 = model(states2, actions2, next_states2).squeeze(-1)

                # Apply precomputed discount vectors
                discounts1 = self.discount_cache[len(rewards1)]
                discounts2 = self.discount_cache[len(rewards2)]

                # Compute discounted returns
                return1 = torch.sum(rewards1 * discounts1)
                return2 = torch.sum(rewards2 * discounts2)

                batch_returns_1.append(return1)
                batch_returns_2.append(return2)
                batch_targets.append(preference)

            # ---- stack into batch tensors ----
            logits = torch.stack([torch.stack(batch_returns_1),
                                torch.stack(batch_returns_2)], dim=1)  # [batch_size, 2]
            targets = torch.tensor(batch_targets, dtype=torch.long, device=self.device)

            # ---- compute batch loss once ----
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, targets)

            # ---- backward and optimize ----
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
        #     # For ensemble diversity, each model can use a bootstrap sample
        #     # For now, use the same data for all models
        #     batch_loss = 0.0
        #     num_pairs = len(batch_indices)

        #     optimizer.zero_grad()

        #     for idx in batch_indices:
        #         traj1, traj2 = trajectory_pairs_tensors[idx]
        #         preference = preferences[idx]

        #         # Convert trajectories to tensors directly on device
        #         states1, actions1, next_states1 = traj1
        #         states2, actions2, next_states2 = traj2

        #         # Compute predicted returns (with gradients enabled)
        #         # Get rewards for all transitions
        #         rewards1 = model.forward(states1, actions1, next_states1).squeeze(-1)
        #         rewards2 = model.forward(states2, actions2, next_states2).squeeze(-1)

        #         discounts1 = self.discount_cache[len(rewards1)]
        #         discounts2 = self.discount_cache[len(rewards2)]

        #         return1 = torch.sum(rewards1 * discounts1)
        #         return2 = torch.sum(rewards2 * discounts2)

        #         # Bradley-Terry preference model loss
        #         # P(traj1 > traj2) = exp(return1) / (exp(return1) + exp(return2))
        #         # If preference = 0, we want return1 > return2
        #         # If preference = 1, we want return2 > return1

        #         logits = torch.stack([return1, return2])
        #         target = torch.tensor([preference], dtype=torch.long, device=self.device)

        #         loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), target)
        #         batch_loss += loss.item()

        #         # Accumulate gradients
        #         loss.backward()

        #     # Update parameters after processing all pairs in batch
        #     optimizer.step()

        #     losses.append(batch_loss / num_pairs)

        return losses

    def train(
        self,
        trajectory_pairs: List[Tuple],
        preferences: np.ndarray,
        num_epochs: int = 100,
        batch_size: Optional[int] = None,
        bootstrap: bool = True,
        verbose: bool = True
    ):
        """
        Train the ensemble on trajectory preference data.

        Args:
            trajectory_pairs: List of trajectory pairs.
            preferences: Preference labels.
            num_epochs: Number of training epochs.
            batch_size: Batch size (if None, use full batch).
            bootstrap: If True, use bootstrap sampling for each model.
            verbose: If True, print training progress.
        """
        num_pairs = len(trajectory_pairs)
        trajectory_pairs_tensors = preprocess_trajectory_pairs(trajectory_pairs, self.state_dim, self.device)

        for epoch in range(num_epochs):
            # Generate indices for this epoch
            if bootstrap:
                # Each model gets a bootstrap sample
                indices_per_model = [
                    np.random.choice(num_pairs, size=num_pairs, replace=True)
                    for _ in range(self.ensemble_size)
                ]
            else:
                # All models use the same shuffled data
                indices = np.random.permutation(num_pairs)
                indices_per_model = [indices] * self.ensemble_size

            # Train each model
            epoch_losses = []

            if batch_size is None:
                # Full batch training
                epoch_losses = self.train_step(trajectory_pairs_tensors, preferences, indices_per_model)
            else:
                # Mini-batch training
                num_batches = (num_pairs + batch_size - 1) // batch_size
                model_losses = [[] for _ in range(self.ensemble_size)]

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_pairs)
                    batch_indices_per_model = [indices_per_model[m][start_idx:end_idx] for m in range(self.ensemble_size)]

                    losses = self.train_step(trajectory_pairs_tensors, preferences, batch_indices_per_model)
                    for model_idx, loss in enumerate(losses):
                        model_losses[model_idx].append(loss)

                epoch_losses = [np.mean(losses) for losses in model_losses]

            if verbose and (epoch % 1 == 0 or epoch == num_epochs - 1):
                mean_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch + 1}/{num_epochs}, Mean Loss: {mean_loss:.4f}")

    def predict_returns(
        self,
        trajectory,
        return_std: bool = False
    ) -> Tuple[float, Optional[float]]:
        """
        Predict the return of a trajectory using the ensemble.

        Args:
            trajectory: Trajectory to evaluate.
            return_std: If True, also return standard deviation.

        Returns:
            Mean predicted return (and optionally std dev).
        """
        states, actions, next_states = trajectory_to_tensors(trajectory, self.state_dim, self.device)

        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred_return = model.predict_trajectory_return(states, actions, next_states)
                predictions.append(pred_return.item())

        mean_pred = np.mean(predictions)

        if return_std:
            std_pred = np.std(predictions)
            return mean_pred, std_pred
        else:
            return mean_pred

    def save(self, path: str):
        """Save ensemble to disk."""
        checkpoint = {
            'ensemble_size': self.ensemble_size,
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'models': [model.state_dict() for model in self.models]
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load ensemble from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        for model, state_dict in zip(self.models, checkpoint['models']):
            model.load_state_dict(state_dict)

    def precompute_rewards(self, mdp, verbose: bool = True) -> dict:
        """
        Precompute mean rewards for all state-action-next_state transitions.

        This creates a lookup table of rewards that can be used for much faster
        evaluation, avoiding the need to run forward passes through the ensemble
        on every reward lookup during value iteration.

        Args:
            mdp: The MDP environment with defined state space.
            verbose: If True, print progress information.

        Returns:
            Dictionary mapping (state, action, next_state) to mean reward.
        """
        if verbose:
            print("Precomputing rewards for all transitions...")

        precomputed_rewards = {}

        # Get all states - works for gridworlds with size or grid_size attribute
        if hasattr(mdp, 'size'):
            size = mdp.size
            all_states = [(x, y) for x in range(size) for y in range(size)]
        elif hasattr(mdp, 'grid_size'):
            size = mdp.grid_size
            all_states = [(x, y) for x in range(size) for y in range(size)]
        else:
            raise ValueError("MDP must have a 'size' or 'grid_size' attribute to precompute rewards")

        num_states = len(all_states)
        num_actions = mdp.get_num_actions()
        total_transitions = num_states * num_actions

        for idx, state in enumerate(all_states):
            for action in range(num_actions):
                # Take action in environment
                next_state, _, _ = mdp.step(state, action)

                # Get mean reward from ensemble
                # Convert state to one-hot encoding
                state_row, state_col = state
                state_onehot = np.zeros(self.state_dim)
                state_onehot[state_row * int(np.sqrt(self.state_dim)) + state_col] = 1.0

                next_state_row, next_state_col = next_state
                next_state_onehot = np.zeros(self.state_dim)
                next_state_onehot[next_state_row * int(np.sqrt(self.state_dim)) + next_state_col] = 1.0

                # Convert to tensors
                state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)

                # Get predictions from each model in ensemble
                predictions = []
                for model in self.models:
                    model.eval()
                    with torch.no_grad():
                        reward_pred = model(state_tensor, action_tensor, next_state_tensor)
                        predictions.append(reward_pred.item())

                # Store mean reward
                mean_reward = np.mean(predictions)
                precomputed_rewards[(state, action, next_state)] = mean_reward

            if verbose and (idx + 1) % max(1, num_states // 10) == 0:
                progress = (idx + 1) * num_actions / total_transitions * 100
                print(f"  Progress: {progress:.1f}% ({(idx + 1) * num_actions}/{total_transitions} transitions)")

        if verbose:
            print(f"  Precomputed {len(precomputed_rewards)} transitions!")

        return precomputed_rewards
