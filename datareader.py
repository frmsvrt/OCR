from torch.utils.data import Dataset
import skimage.io as io
import pandas as pd

class DataStream(Dataset):
    def __init__(self, data, transform=None):
        self.imgs = data[0].tolist()
        self.labels = data[1].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # print(self.imgs[idx])
        x = io.imread(self.imgs[idx])
        y = self.labels[idx]
        sample = {'img' : x, 'label' : y}

        if self.transform:
            sample = self.transform(sample)

        return sample
