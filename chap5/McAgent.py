import numpy as np
from collections import defaultdict

# ε-greedy
def greedy_probs(Q,state,eps = 0,action_size = 4):

    # Q(state,0),Q(state,1),Q(state,2),Q(state,3)
    qs = [Q[(state,action)] for action in range(action_size)] 
    # Q(s,a)の最大値の引数 aを決定
    max_act = np.argmax(qs)

    prob = eps / action_size

    action_probs = { action : prob for action in range(action_size)}

    action_probs[max_act]  += (1- eps)

    return action_probs

# greedy
def greedy_probs_old(Q,state,eps = 0,action_size = 4):

    # Q(state,0),Q(state,1),Q(state,2),Q(state,3)
    qs = [Q[(state,action)] for action in range(action_size)] 
    # Q(s,a)の最大値の引数 aを決定
    max_act = np.argmax(qs)

    # greedyだと方策の多様性がない　決定論的に決るため
    action_probs = { action : 0.0 for action in range(action_size)}

    action_probs[max_act]  = 1

    return action_probs

# randomAgentと基本的に同じ



class McAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4
        # 方策
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)        
        self.Q = defaultdict(lambda : 0)
        self.counts = defaultdict(lambda : 0)
        self.memory = []
        self.alpha = 0.1
        self.eps = 0.1

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
    def update(self):
        # G -> 収益 
        G = 0
        # reversedで後ろから
        for data in reversed(self.memory):
            state,action,reward = data
            # 後ろから繰り返しgammaが掛けられる
            G = self.gamma * G + reward 
            key = (state,action) # Q(s,a)の引数

            # self.counts[key] += 1
            # モンテカルロ法
            # 状態 state それぞれの価値関数を更新
            # self.Q[key] += (G - self.Q[key]) /self.counts[key]
            # 標本平均から指数移動平均へ
            self.Q[key] += (G - self.Q[key]) * self.alpha

            self.pi[state] = greedy_probs(self.Q,state,self.eps)


