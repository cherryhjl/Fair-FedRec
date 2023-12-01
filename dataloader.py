import torch
import random
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, samples, neg_samples, users, npratio):
        self.samples = []
        self.neg_samples = neg_samples
        self.npratio = npratio

        for user in users:
            self.samples.extend([i for i in samples if i[0] == user])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # uid, [pos,neg],label
        uid, mid = self.samples[idx]
        neg_sample = self.neg_samples[uid]
        neg = random.sample(neg_sample, self.npratio)
        pos_4neg = np.array([mid] + neg)

        label = np.array(0)
        return uid, pos_4neg, label


class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # uid, mid,label
        uid, mid, label = self.samples[idx]
        return uid, mid, label


class ClientDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # uid, mid,label
        uid, mid, label = self.samples[idx]
        return uid, mid, label


class ClientTestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def split_all(self):
        # uid, mid,label
        uid_all, mid_all, label_all = [[] for _ in range(3)]
        for i in self.samples:
            uid, mid, label = i
            uid_all.append(uid)
            mid_all.append(mid)
            label_all.append(label)
        return torch.LongTensor(uid_all), torch.LongTensor(mid_all), torch.LongTensor(label_all)
