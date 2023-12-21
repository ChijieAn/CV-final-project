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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#define the training function
def evaluate_accuracy(model, test_loader):
    # 确保模型在验证模式，这对于某些层如Dropout和BatchNorm是必要的
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # 在评估模式下不计算梯度
        for data in test_loader:
            label_tensor=data[1]
            input_tensor=data[0]
            label_tensor=label_tensor.to('cuda')
            input_tensor=input_tensor.to('cuda')
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            total += label_tensor.size(0)
            correct += (predicted == label_tensor).sum().item()

    # 转换为百分比
    accuracy = 100 * correct / total
    return accuracy

def train(model_list, train_loader_list, test_loader_list, criterion, optimizer_list, num_epochs=10):
    count_model=0

    new_lst=[]

    for i in range(len(model_list)):
      model=model_list[i]
      model.train()

      train_loader=train_loader_list[i]
      test_loader=test_loader_list[i]
      optimizer=optimizer_list[i]

      for epoch in range(num_epochs):
          running_loss = 0.0
          correct = 0
          total = 0

          # 训练过程
          for data in train_loader:
              #labels, inputs = data[0], data[1]
              label_tensor=data[1]
              input_tensor=data[0]
              #print(label_tensor)
              #print(input_tensor)
              label_tensor=label_tensor.to('cuda')
              input_tensor=input_tensor.to('cuda')
              #print(label_tensor)
              optimizer.zero_grad()  # 清空之前的梯度
              #print(input_tensor.size())
              outputs = model(input_tensor)  # 获得模型输出
              #print(outputs.size())
              #print(outputs.shape)
              #print(label_tensor.shape)
              loss = criterion(outputs, label_tensor)  # 计算损失
              loss.backward()  # 反向传播
              optimizer.step()  # 更新权重

              running_loss += loss.item()
              _, predicted = torch.max(outputs.data, 1)
              #print('this is predicted',predicted)
              total += label_tensor.size(0)
              correct += (predicted == label_tensor).sum().item()

          # 训练完一个epoch后的平均损失
          epoch_loss = running_loss / len(train_loader)

          # 计算测试集上的准确率
          test_accuracy = evaluate_accuracy(model, test_loader)

          print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

          model_path = model_path = '/content/drive/MyDrive/Classifier_Folder/epoch_{}_fine_tuned_resnet50_1-100.pth'.format(epoch)
          torch.save(model.state_dict(), model_path)
      count_model+=1

      print('Finished Training',count_model)

      new_lst.append(model)

    return new_lst
