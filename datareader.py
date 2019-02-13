from torch.utils.data import Dataset
import skimage.io as io
import pandas as pd

class DataStream(Dataset):
    def __init__(self, fname, transform=None):
        self.data = pd.read_csv(fname, sep=';', header=None)
        self.imgs = self.data[0]
        self.labels = self.data[1]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = io.imread(self.imgs[idx])
        y = self.labels[idx]
        sample = {'img' : x, 'label' : y}

        if self.transform:
            sample = self.transform(sample)

        return sample
