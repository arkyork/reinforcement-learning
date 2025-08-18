import numpy as np
from collections import defaultdict

class RandomAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4
        # 方策
        self.pi = defaultdict(lambda : { i : 0.25 for i in range(self.action_size)})
        self.V = defaultdict(lambda : 0)
        self.counts = defaultdict(lambda : 0)
        self.memory = []

    # 行動を一つとる
    def get_action(self,state) -> int:

        # 状態ごとのaction_probを取り出すことに注意
        action_prob = self.pi[state]
        # action_probのkeys
        actions = list(action_prob.keys())
        # action_probのvalues
        probs = list(action_prob.values())

        return np.random.choice(actions,p=probs)
    
    def add(self,state,action,reward) -> None:
        data = (state,action,reward)
        self.memory.append(data)
    def reset(self) -> None:
        self.memory.clear()
    def eval(self):
        # G -> 収益 
        G = 0
        # reversedで後ろから
        for data in reversed(self.memory):
            state,action,reward = data
            # 後ろから繰り返しgammaが掛けられる
            G = self.gamma * G + reward

            self.counts[state] += 1
            # モンテカルロ法
            # 状態 state それぞれの価値関数を更新
            self.V[state] += (G - self.V[state]) /self.counts[state]


