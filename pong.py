import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PPO import PPO
from image_dataset import ImageData
from pong_models import Encoder


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Actor(nn.Module):
    def __init__(self, latent_size, action_shape):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.linear = nn.Linear(latent_size, action_shape)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        latent = self.encoder(x).detach()
        y = nn.functional.softmax(self.linear(
            latent), dim=1)
        print(y)
        return y


class Critic(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.linear = nn.Linear(latent_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.linear(nn.functional.relu(self.encoder(x).detach()))


if __name__ == "__main__":
    data = ImageData('./PongData/')
    env = gym.make('ALE/Pong-v5', render_mode="human",
                   obs_type="grayscale", max_episode_steps=100)

    encoder_state_dict = torch.load('encoder.pt')
    actor = Actor(50, env.action_space.n).cuda()
    actor.encoder.load_state_dict(encoder_state_dict)
    actor.encoder.eval()
    critic = Critic(50).cuda()
    critic.encoder.load_state_dict(encoder_state_dict)
    critic.encoder.eval()
    ppo = PPO(env, actor, critic)
    ppo.train()

    fig = plt.figure()
    plt.plot(np.arange(0, len(ppo.scores)), ppo.scores)
    plt.show()
