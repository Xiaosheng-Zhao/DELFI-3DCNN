import torch
import os
import numpy as np
from torch.utils.data import Dataset

class CNNDataset(Dataset):

    def __init__(self, length, prefix, root_dir, transform=True):
        self.length = length
        self.prefix = prefix
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return int(self.length)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'data_residual_uppermean/',
                                self.prefix+str(idx)+'.npy')
        para_name = os.path.join(self.root_dir, 'data_residual_uppermeano/',
                                self.prefix+str(idx)+'-y'+'.npy')
        image = np.load(img_name)
        para = np.load(para_name)
        para[0] = para[0]/1.431
        sample = [image, para]

        if self.transform:
            sample = self.transform(sample)

        return sample
