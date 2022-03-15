import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.conv = nn.Sequential(nn.Conv1d(128, 256, kernel_size=5, stride = 2),
                                  nn.ReLU(), 
                                  nn.MaxPool1d(3, 3),
                                  nn.Conv1d(256, 384, kernel_size=3, stride = 1), 
                                  nn.ReLU(),
                                  nn.MaxPool1d(3, 3),
                                  nn.Conv1d(384, 512, kernel_size=3),
                                  nn.ReLU())
        self.drop_out = nn.Dropout()
        self.linear = nn.Sequential(nn.Linear(512 * 4, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, 8))
        
    def forward(self, x):
        x = x[:,None,:]
        out = self.norm(x)
        out = out.reshape(-1, 128, 130)
        out = self.conv(out)
        #print(out.shape)
        out = out.reshape(-1, 512 * 4)
        out = self.drop_out(out)
        out = self.linear(out)
        return out