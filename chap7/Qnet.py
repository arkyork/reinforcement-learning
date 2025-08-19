import torch
import torch.nn as nn
import numpy as np
class QNet(nn.Module):
    def __init__(self) -> None:
        super(QNet,self).__init__()
        self.layer1 = nn.Linear(12, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 4)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
    

# 状態を受け取ったときにone-hot vectorに変換
def one_hot(state):
    HEIGHT,WIDTH = 3,4


    vec = np.zeros(HEIGHT * WIDTH,dtype=np.float32)

    y_pos,x_pos = state

    # インデックスの位置の計算
    # idxはそれぞれの高さごとのposに区切られていると考える
    idx = WIDTH * y_pos + x_pos

    vec[idx] = 1

    # ニューラルネットワークに突っ込むため次元を追加
    return torch.tensor(vec[np.newaxis,:],dtype=torch.float32)