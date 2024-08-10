import gymnasium as gym
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from PPO import PPO


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state):
        return self.model(state)


def to_tensor(array):
    tensor = torch.from_numpy(array).float()
    return tensor.to('cuda')


if __name__ == "__main__":
    env = gym.make('CartPole-v0', render_mode="human")

    actor = Actor(env.observation_space.shape[0], env.action_space.n).cuda()
    critic = Critic(env.observation_space.shape[0]).cuda()

    ppo = PPO(env, actor, critic)
    ppo.train()

    fig = plt.figure()
    plt.plot(np.arange(0, len(ppo.scores)), ppo.scores)
    plt.show()
    # with torch.no_grad():
    #   obs, _ = env.reset()
    #   obs = to_tensor(obs)
    #   episode_done = True
    #   while True:
    #       if episode_done:
    #           obs, _ = env.reset()
    #           obs = to_tensor(obs)
    #       probs = actor(obs)
    #       dist = torch.distributions.Categorical(probs=probs)
    #       new_obs, reward, episode_done, _, _ = env.step(
    #           dist.sample().item())
    #       obs = to_tensor(new_obs)
