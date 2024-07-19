import gymnasium as gym
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.Tanh(),
            nn.Tanh(),
            nn.Linear(24, action_dim),
            nn.Softmax(dim=0)
        )

    def forward(self, state):
        return self.model(state)


def to_tensor(array):
    tensor = torch.from_numpy(array).float()
    return tensor.to('cuda')


if __name__ == "__main__":
    env = gym.make('CartPole-v0', render_mode="human")

    actor = Actor(env.observation_space.shape[0], env.action_space.n).cuda()
    optim = torch.optim.Adam(actor.parameters())
    gamma = .9
    scores = []
    for i in range(1000):
        obs = to_tensor(env.reset()[0])
        episode_done = False
        policy_logprobs = []
        rewards = []
        while not episode_done:
            probs = actor(obs)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            new_obs, reward, episode_done, _, _ = env.step(
                action.item())

            policy_logprobs.append(dist.log_prob(action))
            rewards.append(reward)
            obs = to_tensor(new_obs)
            env.render()
        scores.append(sum(rewards))
        G = sum([r * (gamma ** i) for i, r in enumerate(rewards)])
        losses = []
        for log_prob in policy_logprobs:
            losses.append(-log_prob * G)
        losses = torch.stack(losses).sum()
        optim.zero_grad()
        losses.backward()
        optim.step()

    fig = plt.figure()
    plt.plot(np.arange(0, len(scores)), scores)
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
