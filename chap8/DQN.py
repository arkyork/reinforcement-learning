import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self,action_size) -> None:
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(4, 120)
        self.layer2 = nn.Linear(120, 120)
        self.layer3 = nn.Linear(120, action_size)

        # 活性化関数
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

