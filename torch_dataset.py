# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:16:07 2024

@author: marinamu
"""

from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
