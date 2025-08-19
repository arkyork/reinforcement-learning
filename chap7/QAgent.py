
from collections import defaultdict,deque
import numpy as np

from Qnet import QNet

import torch
import torch.nn as nn

# TDターゲットの推定

class QLearnAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.epsilon = 0.1
        self.lr = 0.01    
        self.action_size = 4

        # Qnet

        self.qnet = QNet()
        # 最適化
        self.optimizer = torch.optim.SGD(self.qnet.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()

    def get_actions(self,state):
        # εで条件分岐
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # 状態から推定
            qs = self.qnet(state)
            # Q関数の最大値の引数
            return qs.data.argmax()
    def update(self,state,action,reward,next_state,goal_done):

        if goal_done:
            next_q = torch.zeros(1)
        else:
            # Q(next_state,next_actions)
            next_qs = self.qnet(next_state)
            next_q = torch.max(next_qs).detach()

        td_target = reward + self.gamma * next_q
        # 現在の状態のQ
        qs = self.qnet(state)
        q = qs[:,action]

        loss = self.loss(td_target,q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

