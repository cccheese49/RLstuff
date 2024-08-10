import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class VAEModel(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.model = nn.Sequential(
            enc,
            dec
        )

    def forward(self, x):
        return self.model(x)


class VAETrainer:
    def __init__(self, enc, dec, hidden_size):
        self.mu = nn.Linear(hidden_size, hidden_size)
        self.log_var = nn.Linear(hidden_size, hidden_size)
        self.model = VAEModel(enc, dec).cuda()

    def train(self, dataloader: DataLoader):
        optim = torch.optim.Adam(self.model.parameters())
        for i in range(1):
            for batch_images in iter(dataloader):
                recon = self.model(batch_images.cuda())
                loss = nn.MSELoss()(recon,
                                    batch_images.cuda())
                optim.zero_grad()
                print(loss)
                loss.backward()
                optim.step()
