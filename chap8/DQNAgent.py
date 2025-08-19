
from collections import defaultdict,deque
import numpy as np

from DQN import DQN

import torch
import torch.nn as nn
import copy
from ExpBuffer import ExpBuffer

# TDターゲットの推定

class DQNAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.epsilon = 0.1
        self.lr = 0.01    
        self.action_size = 2
        self.buffer_size = 10000
        self.batch_size = 32


        # バッファー
        self.buffer = ExpBuffer(batch_size=self.batch_size,buffer_size=self.buffer_size)
        # Qnet

        self.qnet = DQN(action_size=self.action_size)
        # Target Net
        self.qnet_target = DQN(action_size=self.action_size)

        # 最適化
        self.optimizer = torch.optim.SGD(self.qnet.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()
    @torch.no_grad()
    def get_actions(self,state):
        # print("state",state,type(state))
        # εで条件分岐
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            a = self.qnet(state_t).argmax(dim=1).item()
            return int(a)
    
    def update(self,state,action,reward,next_state,goal_done):
        self.buffer.add(state, action, reward, next_state, goal_done)
        if len(self.buffer) < self.batch_size:
            return
        
        # バッチ取得（np配列が返る想定）
        s, a, r, ns, d = self.buffer.get_batch()

        # Tensor化
        s  = torch.as_tensor(s,  dtype=torch.float32)  # [B, 4]
        ns = torch.as_tensor(ns, dtype=torch.float32)  # [B, 4]
        a  = torch.as_tensor(a,  dtype=torch.long)     # [B]
        r  = torch.as_tensor(r,  dtype=torch.float32)  # [B]
        d  = torch.as_tensor(d,  dtype=torch.float32)  # [B] (0. or 1.)

        # Q(s,a)
        qs = self.qnet(s)                              # [B, A]
        q  = qs.gather(1, a.unsqueeze(1)).squeeze(1)   # [B]

        with torch.no_grad():
            next_qs = self.qnet_target(ns)             # [B, A]
            next_q  = next_qs.max(dim=1).values        # [B]
            td_target = r + (1.0 - d) * self.gamma * next_q  # [B]





        loss = self.loss(td_target,q)

        # 最適化

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)
