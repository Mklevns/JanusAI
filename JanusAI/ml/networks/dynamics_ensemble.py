# File: JanusAI/ml/networks/dynamics_ensemble.py
"""
Dynamics Model Ensemble for Information Gain-based Exploration

This module implements an ensemble of neural networks that predict rewards
from expression embeddings. The variance in predictions serves as a measure
of epistemic uncertainty, which drives exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np
from collections import deque


class DynamicsPredictor(nn.Module):
    """
    A single dynamics model that predicts reward from expression embedding.
    
    Uses a deeper architecture with residual connections for better representation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(ResidualBlock(hidden_dim, dropout))
        
        # Output heads
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Optional: predict uncertainty directly
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Project input
        h = F.relu(self.input_proj(x))
        
        # Apply residual blocks
        for block in self.res_blocks:
            h = block(h)
        
        # Get predictions
        reward_pred = self.reward_head(h)
        uncertainty_pred = self.uncertainty_head(h)
        
        return {
            'reward': reward_pred,
            'uncertainty': uncertainty_pred
        }


class ResidualBlock(nn.Module):
    """Residual block with layer norm and dropout."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class DynamicsEnsemble:
    """
    Ensemble of dynamics models for epistemic uncertainty estimation.
    
    Features:
    - Bootstrap sampling for diversity
    - Adaptive learning rate scheduling
    - Model disagreement metrics
    - Replay buffer for continual learning
    """
    
    def __init__(self, 
                 num_models: int,
                 input_dim: int,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-4,
                 buffer_size: int = 10000,
                 bootstrap_ratio: float = 0.8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.num_models = num_models
        self.input_dim = input_dim
        self.device = device
        self.bootstrap_ratio = bootstrap_ratio
        
        # Create ensemble
        self.models = [
            DynamicsPredictor(input_dim, hidden_dim).to(device) 
            for _ in range(num_models)
        ]
        
        # Optimizers with different learning rates for diversity
        self.optimizers = [
            torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate * (1 + 0.1 * np.random.randn())  # Slight LR variation
            )
            for model in self.models
        ]
        
        # Learning rate schedulers
        self.schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', patience=10, factor=0.5
            )
            for opt in self.optimizers
        ]
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer for continual learning
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.training_stats = {
            'losses': [[] for _ in range(num_models)],
            'disagreements': [],
            'mean_uncertainties': []
        }
        
    def predict(self, expression_embedding: torch.Tensor, 
                return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """
        Get predictions from all models in the ensemble.
        
        Returns:
            Dictionary containing:
            - mean_reward: Average predicted reward
            - reward_variance: Epistemic uncertainty (variance across models)
            - aleatoric_uncertainty: Average predicted uncertainty
            - individual_predictions: (optional) All model predictions
        """
        with torch.no_grad():
            predictions = []
            uncertainties = []
            
            for model in self.models:
                model.eval()
                output = model(expression_embedding)
                predictions.append(output['reward'])
                uncertainties.append(output['uncertainty'])
            
            # Stack predictions
            all_rewards = torch.stack(predictions)  # (num_models, batch_size, 1)
            all_uncertainties = torch.stack(uncertainties)
            
            # Calculate statistics
            mean_reward = all_rewards.mean(dim=0)
            reward_variance = all_rewards.var(dim=0)  # Epistemic uncertainty
            mean_uncertainty = all_uncertainties.mean(dim=0)  # Aleatoric uncertainty
            
            result = {
                'mean_reward': mean_reward,
                'epistemic_uncertainty': reward_variance,
                'aleatoric_uncertainty': mean_uncertainty,
                'total_uncertainty': reward_variance + mean_uncertainty
            }
            
            if return_individual:
                result['individual_predictions'] = all_rewards
                
            return result
    
    def calculate_disagreement(self, expression_embedding: torch.Tensor) -> float:
        """
        Calculate the disagreement between models as a measure of uncertainty.
        
        Uses multiple metrics:
        - Variance of predictions
        - Pairwise distances between predictions
        - Entropy of prediction distribution
        """
        predictions = self.predict(expression_embedding, return_individual=True)
        individual_preds = predictions['individual_predictions'].squeeze(-1)  # (num_models, batch_size)
        
        # Variance-based disagreement
        variance_disagreement = individual_preds.var(dim=0).mean().item()
        
        # Pairwise distance disagreement
        pairwise_distances = []
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                dist = torch.abs(individual_preds[i] - individual_preds[j]).mean()
                pairwise_distances.append(dist)
        
        pairwise_disagreement = torch.tensor(pairwise_distances).mean().item()
        
        # Combined disagreement metric
        total_disagreement = variance_disagreement + 0.5 * pairwise_disagreement
        
        return total_disagreement
    
    def train_step(self, expression_embeddings: torch.Tensor, 
                   true_rewards: torch.Tensor,
                   use_bootstrap: bool = True) -> Dict[str, float]:
        """
        Train all models in the ensemble on a batch of data.
        
        Args:
            expression_embeddings: Batch of expression embeddings
            true_rewards: Corresponding true rewards
            use_bootstrap: Whether to use bootstrap sampling
            
        Returns:
            Dictionary of training metrics
        """
        # Add to replay buffer
        for i in range(expression_embeddings.size(0)):
            self.replay_buffer.append((
                expression_embeddings[i].detach().cpu(),
                true_rewards[i].detach().cpu()
            ))
        
        # Sample from replay buffer for training
        if len(self.replay_buffer) > 32:
            # Mix current batch with replay samples
            replay_batch = self._sample_replay_batch(32)
            replay_embeddings, replay_rewards = zip(*replay_batch)
            replay_embeddings = torch.stack(replay_embeddings).to(self.device)
            replay_rewards = torch.stack(replay_rewards).to(self.device)
            
            # Combine with current batch
            expression_embeddings = torch.cat([expression_embeddings, replay_embeddings])
            true_rewards = torch.cat([true_rewards, replay_rewards])
        
        losses = []
        
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            
            # Bootstrap sampling for diversity
            if use_bootstrap:
                batch_size = expression_embeddings.size(0)
                indices = torch.randint(
                    0, batch_size, 
                    (int(batch_size * self.bootstrap_ratio),),
                    device=self.device
                )
                batch_embeddings = expression_embeddings[indices]
                batch_rewards = true_rewards[indices]
            else:
                batch_embeddings = expression_embeddings
                batch_rewards = true_rewards
            
            # Forward pass
            output = model(batch_embeddings)
            pred_rewards = output['reward']
            pred_uncertainty = output['uncertainty']
            
            # Calculate loss
            # Main loss: reward prediction
            reward_loss = self.loss_fn(pred_rewards, batch_rewards)
            
            # Auxiliary loss: uncertainty should be high for incorrect predictions
            reward_errors = torch.abs(pred_rewards - batch_rewards).detach()
            uncertainty_loss = self.loss_fn(pred_uncertainty, reward_errors)
            
            # Combined loss
            total_loss = reward_loss + 0.1 * uncertainty_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Record loss
            losses.append(total_loss.item())
            self.training_stats['losses'][i].append(total_loss.item())
        
        # Update learning rate schedulers
        avg_loss = np.mean(losses)
        for scheduler in self.schedulers:
            scheduler.step(avg_loss)
        
        # Calculate and record disagreement
        with torch.no_grad():
            disagreement = self.calculate_disagreement(expression_embeddings[:32])
            self.training_stats['disagreements'].append(disagreement)
        
        return {
            'mean_loss': avg_loss,
            'disagreement': disagreement,
            'losses': losses
        }
    
    def _sample_replay_batch(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample a batch from the replay buffer."""
        indices = np.random.choice(len(self.replay_buffer), size=batch_size, replace=True)
        return [self.replay_buffer[i] for i in indices]
    
    def get_information_gain(self, expression_embedding: torch.Tensor) -> float:
        """
        Calculate the expected information gain from evaluating this expression.
        
        This is the key metric for MaxInfoRL - expressions with high information
        gain are those where the ensemble is most uncertain.
        """
        predictions = self.predict(expression_embedding)
        
        # Information gain is proportional to total uncertainty
        info_gain = predictions['total_uncertainty'].mean().item()
        
        # Scale by ensemble disagreement for better exploration
        disagreement = self.calculate_disagreement(expression_embedding)
        
        # Combined metric
        scaled_info_gain = info_gain * (1 + 0.5 * disagreement)
        
        return scaled_info_gain
    
    def save(self, path: str):
        """Save ensemble state."""
        torch.save({
            'models': [model.state_dict() for model in self.models],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'training_stats': self.training_stats,
            'config': {
                'num_models': self.num_models,
                'input_dim': self.input_dim,
                'device': self.device
            }
        }, path)
    
    def load(self, path: str):
        """Load ensemble state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, model in enumerate(self.models):
            model.load_state_dict(checkpoint['models'][i])
        
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint['optimizers'][i])
        
        self.training_stats = checkpoint['training_stats']