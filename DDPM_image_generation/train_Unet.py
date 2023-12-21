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

from .Unet_Model import UNetModel
from .Gaussian_Diffusion import (GaussianDiffusion,
                                 linear_beta_schedule,
                                 cosine_beta_schedule)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 128
timesteps = 1000

gaussian_diffusion=GaussianDiffusion(timesteps=timesteps)

def train(epochs, model, train_loader, optimizer):
    for epoch in range(epochs):
      total_loss = 0
      for step, (images, labels) in enumerate(train_loader):
          optimizer.zero_grad()

          batch_size = images.shape[0]
          images = images.to(device)

          # sample t uniformally for every example in the batch
          t = torch.randint(0, timesteps, (batch_size,), device=device).long()

          loss = gaussian_diffusion.train_losses(model, images, t)
          total_loss += loss.item()
          #if step % 200 == 0:
          #    print("Loss:", loss.item())


          loss.backward()
          optimizer.step()
      print(f"Epoch {epoch}, Loss {total_loss/len(train_loader)}")

