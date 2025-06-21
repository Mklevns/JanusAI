"""Observation denoising components."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class NoisyObservationProcessor:
    """Handles noisy observations using denoising autoencoders."""

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()
        self.model = None # Initialize model attribute

    def build_autoencoder(self, input_dim: int):
        """Build denoising autoencoder for preprocessing."""
        class DenoisingAutoencoder(nn.Module):
            def __init__(self, input_dim_ae, latent_dim_ae): # Renamed to avoid conflict
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim_ae, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim_ae)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim_ae, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim_ae)
                )

            def forward(self, x, noise_level=0.1):
                if self.training:
                    noisy_x = x + torch.randn_like(x) * noise_level
                else:
                    noisy_x = x
                latent = self.encoder(noisy_x)
                reconstructed = self.decoder(latent)
                return reconstructed, latent

        self.model = DenoisingAutoencoder(input_dim, self.latent_dim)
        return self.model

    def denoise(self, observations: np.ndarray, epochs: int = 50) -> np.ndarray:
        """Train denoising autoencoder and return cleaned observations."""
        if observations.shape[0] < 100:
            return self._simple_denoise(observations)

        observations_scaled = self.scaler.fit_transform(observations)
        data = torch.FloatTensor(observations_scaled)

        if self.model is None or self.model.encoder[0].in_features != data.shape[1]: # Check if model needs rebuild
            self.build_autoencoder(data.shape[1])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = self.model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            denoised, _ = self.model(data, noise_level=0)

        return self.scaler.inverse_transform(denoised.numpy())

    def _simple_denoise(self, observations: np.ndarray) -> np.ndarray:
        """Simple moving average denoising for small datasets."""
        window = min(5, observations.shape[0] // 10)
        if window < 2:
            return observations

        denoised = np.copy(observations)
        for i in range(observations.shape[1]):
            denoised[:, i] = np.convolve(
                observations[:, i],
                np.ones(window)/window,
                mode='same'
            )
        return denoised
