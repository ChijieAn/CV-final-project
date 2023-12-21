import os
import math
from abc import abstractmethod

import random
import pandas as pd
from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomTensorDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # suppose the first column is the transformed image and the second folumn is labels
        image = self.dataframe.iloc[idx, 0]  # here is the tensor for a image
        label = self.dataframe.iloc[idx, 1]
        return image, label
def get_subset_loader_1000(full_dataset,datasize,data_loader):

  # randomly pick 1000 non repete indexs
  subset_indices = torch.randperm(len(full_dataset))[:datasize]

  # create a subdataset
  subset = Subset(full_dataset, subset_indices)

  # create a new DataLoader，which only load data from the subset
  subset_dataloader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=True)

  return subset,subset_dataloader

def get_clean_data(sub_data_loader):
    print('Function called.')
    results = []
    print('Initial results length:', len(results))

    for images, labels in sub_data_loader:
        for i in range(len(images)):
           #print(len(images))
            x_start = images[i]

            result = {
                'noisy': x_start,
                'label': labels[i].item()
            }
            results.append(result)
        print('Batch done. Current results length:', len(results))

    print('Final results length:', len(results))
    df = pd.DataFrame(results)
    return df

def get_training_data(begin, end, sub_data_loader,gaussian_diffusion):
    print('Function called.')
    results = []
    print('Initial results length:', len(results))

    for images, labels in sub_data_loader:
        for i in range(len(images)):
           #print(len(images))
            x_start = images[i]
            #print('this is x_start',x_start)
            t = random.randint(begin, end)
            #print('this is timestep',t)
            x_noisy = gaussian_diffusion.q_sample(x_start, t=torch.tensor([t]))

            result = {
                'noisy': x_noisy,
                'label': labels[i].item()
            }
            results.append(result)
        print('Batch done. Current results length:', len(results))

    print('Final results length:', len(results))
    df = pd.DataFrame(results)
    return df