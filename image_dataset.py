import os
from torchvision.io import read_image
from torch.utils.data import Dataset


class ImageData(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.length = len([name for name in os.listdir(image_dir)])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, str(idx) + ".png")
        return read_image(path).float()
