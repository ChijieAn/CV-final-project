import os
import math
from abc import abstractmethod

from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models

from DDPM_image_generation.Gaussian_Diffusion import (GaussianDiffusion,
                                                    linear_beta_schedule,
                                                    cosine_beta_schedule)
from train_function import (train,evaluate_accuracy)
from get_noisy_dataset import (CustomTensorDataset,get_clean_data,get_subset_loader_1000,
                               get_training_data)

# Download and prepare CIFAR-10 dataset
image_size = 128
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

cifar10_data = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(cifar10_data, batch_size=128, shuffle=True)

#define the gaussian diffusion object
gaussian_diffusion = GaussianDiffusion(timesteps=1000)

training_data=get_training_data(1,100,data_loader,gaussian_diffusion)

train_df, test_df = train_test_split(training_data, test_size=0.2, random_state=42)

train_dataset = CustomTensorDataset(train_df)
test_dataset = CustomTensorDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


train_loader_lst=[train_loader]
test_loader_lst=[test_loader]

#define the classifier that we need to train
classifier=models.resnet50(pretrained=True)
classifier.fc=nn.Sequential(
    nn.Linear(in_features=2048,out_features=1000,bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000,out_features=10,bias=True),
    #nn.LogSoftmax(dim=1)
)

classifier_list=[classifier]

for classifier in classifier_list:
  classifier.to('cuda')

criterion = nn.CrossEntropyLoss()

#define the optimizer
optimizer_list=[]
for classifier in classifier_list:
    optimizer=optim.Adam(classifier.parameters(), lr=0.001)
    optimizer_list.append(optimizer)

#train the model
new_model_list=trained_model_list=train(classifier_list, train_loader_lst, test_loader_lst, criterion, optimizer_list, num_epochs=10)

#save the model
model_path = '/content/drive/MyDrive/CV/fine_tuned_resnet50_noisy_100-200.pth'
torch.save(classifier.state_dict(), model_path)





