import torch
import torchvision.io
import gymnasium as gym
import numpy as np
from image_dataset import ImageData
from torch.utils.data import DataLoader
from VAE import VAETrainer
import torch.nn as nn
from pong_models import Encoder, Decoder


def to_tensor(img):
    return torch.from_numpy(img)


if __name__ == "__main__":
    dataset = ImageData("./PongData/")
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    encoder = Encoder(50)
    decoder = Decoder(50)
    vae = VAETrainer(encoder, decoder, 50)
    vae.train(dataloader)
    print("SAVING")
    torch.save(encoder.state_dict(), 'encoder.pt')


def collect():
    env = gym.make('ALE/Pong-v5', obs_type="grayscale")
    episode_done = True
    for i in range(50000):
        if episode_done:
            obs, _ = env.reset()
        obs, _, episode_done, _, _ = env.step(env.action_space.sample())
        torchvision.io.write_png(
            to_tensor(obs).unsqueeze(0), "./PongData/" + str(i) + ".png")
