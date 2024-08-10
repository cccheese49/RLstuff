import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm((210, 160)),
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=10, out_channels=20,
                      kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(1600, latent_size),
            nn.LayerNorm(latent_size)
        )

    def forward(self, x):
        y = self.model(x)
        return y


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2),
            nn.Linear(latent_size * 2, 210 * 160)
        )

    def forward(self, z):
        y = self.model(z)
        return y.reshape(-1, 1, 210, 160)
