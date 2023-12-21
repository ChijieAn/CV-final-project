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

#load the finetuned classifier model, create classifier list
import torchvision.models as models
classifier=models.resnet50(pretrained=True)
classifier.fc=nn.Sequential(
    nn.Linear(in_features=2048,out_features=1000,bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000,out_features=10,bias=True),
    #nn.LogSoftmax(dim=1)
)
model_pth='/content/drive/MyDrive/Classifier_Folder/epoch_8_fine_tuned_resnet50.pth'
classifier.load_state_dict(torch.load(model_pth))
classifier=classifier.to('cuda')
classifier.eval()

model_pth1='/content/drive/MyDrive/Classifier_Folder/epoch_9_fine_tuned_resnet50_1-100.pth'
model_pth2='/content/drive/MyDrive/Classifier_Folder/epoch_9_fine_tuned_resnet50_101-200.pth'
model_pth3='/content/drive/MyDrive/Classifier_Folder/fine_tuned_resnet50_noisy_201-300.pth'
model_pth4='/content/drive/MyDrive/Classifier_Folder/epoch_0_fine_tuned_resnet50_301-400.pth'
model_pth5='/content/drive/MyDrive/Classifier_Folder/epoch_0_fine_tuned_resnet50_401-500.pth'
model_pth6='/content/drive/MyDrive/Classifier_Folder/epoch_0_fine_tuned_resnet50_501-600.pth'
model_pth7='/content/drive/MyDrive/Classifier_Folder/epoch_0_fine_tuned_resnet50_601-700.pth'
model_pth8='/content/drive/MyDrive/Classifier_Folder/epoch_0_fine_tuned_resnet50_701-800.pth'
model_pth9='/content/drive/MyDrive/Classifier_Folder/epoch_0_fine_tuned_resnet50_901-999.pth'
model_pth10='/content/drive/MyDrive/Classifier_Folder/epoch_0_fine_tuned_resnet50_901-999.pth'

model_lst=[model_pth,model_pth2,model_pth3,model_pth4,model_pth5,model_pth6,model_pth7,model_pth8,model_pth9,model_pth10]

classifier_i=models.resnet50(pretrained=True)
classifier_i.fc=nn.Sequential(
    nn.Linear(in_features=2048,out_features=1000,bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000,out_features=10,bias=True),
    #nn.LogSoftmax(dim=1)
    )
classifier_i.load_state_dict(torch.load(model_pth))




classifier_lst=[]
count=0
for i in range(10):
  print(count)
  classifier_i=models.resnet50(pretrained=True)
  classifier_i.fc=nn.Sequential(
    nn.Linear(in_features=2048,out_features=1000,bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000,out_features=10,bias=True),
    #nn.LogSoftmax(dim=1)
    )
  classifier_i.load_state_dict(torch.load(model_lst[i]))
  classifier_i=classifier_i.to('cuda')
  classifier_lst.append(classifier_i)
  count+=1


  #load the data from cifar-10, define the model and optimizer
  batch_size = 128
timesteps = 1000

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

dataset = datasets.CIFAR10('/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetModel(
    in_channels=3,
    model_channels=128,
    out_channels=3,
    channel_mult=(1, 2, 2, 2),
    attention_resolutions=(2,),
    dropout=0.1
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

#display some of the iamges as an example
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

# number of images we need
num_images_to_display = 9

# images already displayed
images_displayed = 0

# traverse the data until we get enough amount of images
for images, labels in train_loader:
    for image, label in zip(images, labels):
        if label == 0:
            ax = axes[images_displayed // 3, images_displayed % 3]
            # reverse standarlization of CIFAR-10
            img = image / 2 + 0.5  
            img = transforms.ToPILImage()(img)  # convert to PIL image
            ax.imshow(img)
            ax.axis('off')
            images_displayed += 1
            if images_displayed == num_images_to_display:
                break
    if images_displayed == num_images_to_display:
        break

# display the images
plt.show()

#some examples of using different strategies to do image generation
my_gaussian_diffusion = GaussianDiffusion(timesteps=1000,beta_schedule='linear',label=0,classifier_lst=classifier_lst,classifier=classifier,guidance_method=0)
generated_images = my_gaussian_diffusion.sample(model, 32, batch_size=64, channels=3)

# display images generated 
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 3, 32, 32)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        img = np.array((imgs[n_row, n_col].transpose([1, 2, 0])+1.0) * 255 / 2, dtype=np.uint8)
        f_ax.imshow(img)
        f_ax.axis("off")


#another example using a different guidance strategy
my_gaussian_diffusion = GaussianDiffusion(timesteps=1000,beta_schedule='linear',label=0,classifier_lst=classifier_lst,classifier=classifier,guidance_method=1)
generated_images = my_gaussian_diffusion.sample(model, 32, batch_size=64, channels=3)

# display images generated 
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 3, 32, 32)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        img = np.array((imgs[n_row, n_col].transpose([1, 2, 0])+1.0) * 255 / 2, dtype=np.uint8)
        f_ax.imshow(img)
        f_ax.axis("off")

#recheck the label of the generated image
labels=classifier(torch.tensor(generated_images[-1]).to('cuda'))
_, predicted = torch.max(labels, 1)
print(predicted)

