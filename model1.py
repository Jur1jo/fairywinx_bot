import ffmpeg
import torch.nn as nn
import librosa
import numpy as np
import tqdm
import torch

class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout()
        self.norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv1d(128, 256, 5, stride=2)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(256, 512, 5)
        self.conv3 = nn.Conv1d(512, 256, 3)
        self.linl1 = nn.Linear(256 * 5, 80)
        self.linl2 = nn.Linear(80, 8)
        
    def forward(self, x):
        x = x[:,None,:,:]
        x = self.norm(x)
        #x = torch.squeeze(x)
        x = x.reshape(-1, 128, 130)
        #print(x.shape)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        #print(x.shape)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        #print(x.shape)
        x = self.drop(x)
        #print(x.shape);
        x = nn.functional.relu(self.linl1(x.reshape(-1, 256 * 5)));
        x = self.linl2(x)
        return x