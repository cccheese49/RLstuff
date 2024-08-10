import torch
import torch.nn as nn


def to_tensor(array):
    return torch.from_numpy(array).float().to('cuda')


class PPO:
    def __init__(self, env, actor_model, critic_model):
        self.env = env
        self.actor = actor_model
        self.critic = critic_model
        self.scores = []

    def rollout(self):
        obs, _ = self.env.reset()
        obs = to_tensor(obs)
        prev_obs = torch.zeros_like(obs)
        batch_obs = []
        episode_done = False
        logprobs = []
        rewards = []
        batch_acts = []
        while not episode_done:
            batch_obs.append(obs.detach())
            dist = torch.distributions.Categorical(
                probs=self.actor(obs))
            action = dist.sample()
            batch_acts.append(action.detach())
            new_obs, reward, episode_done, _, _ = self.env.step(
                action.item())

            logprobs.append(dist.log_prob(action).detach())
            rewards.append(reward)
            prev_obs = obs
            obs = to_tensor(new_obs)
            self.env.render()

        self.scores.append(sum(rewards))
        logprobs = torch.stack(logprobs)
        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.stack(batch_acts)
        batch_rtgs = []
        gamma = .9
        for t in range(len(rewards)):
            G = sum([r * (gamma ** i) for i, r in enumerate(rewards[t:])])
            batch_rtgs.append(torch.tensor(G).float().to('cuda'))
        batch_rtgs = torch.stack(batch_rtgs)
        return batch_obs, batch_acts, batch_rtgs, logprobs

    def train(self):
        for i in range(100):
            torch.autograd.set_detect_anomaly(True)
            actor_optim = torch.optim.Adam(self.actor.parameters())
            critic_optim = torch.optim.Adam(self.critic.parameters())
            batch_obs, batch_acts, batch_rtgs, batch_logprobs = self.rollout()
            for _ in range(5):

                V = self.critic(batch_obs).squeeze()
                A = batch_rtgs - V.detach()
                dist = torch.distributions.Categorical(
                    probs=self.actor(batch_obs))
                curr_log_probs = dist.log_prob(batch_acts)
                ratios = torch.exp(curr_log_probs - batch_logprobs)
                eps = .2
                losses = -torch.min(ratios * A,
                                    torch.clamp(ratios, 1 - eps, 1 + eps) * A)

                actor_optim.zero_grad()
                losses.mean().backward(retain_graph=True)
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss = nn.MSELoss()(V, batch_rtgs.detach())
                critic_loss.backward()
                critic_optim.step()
