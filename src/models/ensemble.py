"""Ensemble training for reward models."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from .reward_model import TransitionRewardModel, trajectory_to_tensors


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
        trajectory_pairs: List[Tuple],
        preferences: np.ndarray,
        batch_indices: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        Perform one training step on the ensemble.

        Args:
            trajectory_pairs: List of (trajectory1, trajectory2) tuples.
            preferences: Array of preference labels (0 or 1).
            batch_indices: Optional indices to use for this batch.

        Returns:
            List of losses for each model.
        """
        if batch_indices is None:
            batch_indices = np.arange(len(trajectory_pairs))

        losses = []

        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            # For ensemble diversity, each model can use a bootstrap sample
            # For now, use the same data for all models
            batch_loss = 0.0
            num_pairs = len(batch_indices)

            optimizer.zero_grad()

            for idx in batch_indices:
                traj1, traj2 = trajectory_pairs[idx]
                preference = preferences[idx]

                # Convert trajectories to tensors directly on device
                states1, actions1, next_states1 = trajectory_to_tensors(traj1, self.state_dim, self.device)
                states2, actions2, next_states2 = trajectory_to_tensors(traj2, self.state_dim, self.device)

                # Compute predicted returns (with gradients enabled)
                # Get rewards for all transitions
                rewards1 = model.forward(states1, actions1, next_states1).squeeze(-1)
                rewards2 = model.forward(states2, actions2, next_states2).squeeze(-1)

                # Compute discounted returns
                discount_factor = 0.99
                discounts1 = torch.tensor(
                    [discount_factor ** i for i in range(len(rewards1))],
                    dtype=torch.float32,
                    device=self.device
                )
                discounts2 = torch.tensor(
                    [discount_factor ** i for i in range(len(rewards2))],
                    dtype=torch.float32,
                    device=self.device
                )

                return1 = torch.sum(rewards1 * discounts1)
                return2 = torch.sum(rewards2 * discounts2)

                # Bradley-Terry preference model loss
                # P(traj1 > traj2) = exp(return1) / (exp(return1) + exp(return2))
                # If preference = 0, we want return1 > return2
                # If preference = 1, we want return2 > return1

                logits = torch.stack([return1, return2])
                target = torch.tensor([preference], dtype=torch.long, device=self.device)

                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), target)
                batch_loss += loss.item()

                # Accumulate gradients
                loss.backward()

            # Update parameters after processing all pairs in batch
            optimizer.step()

            losses.append(batch_loss / num_pairs)

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
                for model_idx in range(self.ensemble_size):
                    batch_indices = indices_per_model[model_idx]
                    losses = self.train_step(trajectory_pairs, preferences, batch_indices)
                    epoch_losses.append(losses[model_idx] if len(losses) > model_idx else losses[0])
            else:
                # Mini-batch training
                num_batches = (num_pairs + batch_size - 1) // batch_size
                model_losses = [[] for _ in range(self.ensemble_size)]

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, num_pairs)

                    for model_idx in range(self.ensemble_size):
                        batch_indices = indices_per_model[model_idx][start_idx:end_idx]
                        losses = self.train_step(trajectory_pairs, preferences, batch_indices)
                        model_losses[model_idx].append(losses[model_idx] if len(losses) > model_idx else losses[0])

                epoch_losses = [np.mean(losses) for losses in model_losses]

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
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
