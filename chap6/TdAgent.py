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



class TdAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4
        # 方策
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)        
        self.V = defaultdict(lambda : 0)
        self.counts = defaultdict(lambda : 0)
        
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
    # 行動を一つとる
    def get_action(self,state) -> int:

        # 状態ごとのaction_probを取り出すことに注意
        action_prob = self.pi[state]
        # action_probのkeys
        actions = list(action_prob.keys())
        # action_probのvalues
        probs = list(action_prob.values())

        return np.random.choice(actions,p=probs)
    def eval(self, state, reward, next_state, goal_done):
        next_V = 0 if goal_done else self.V[state]
        td_target = reward + self.gamma * next_V
        self.V[state] += (td_target - self.V[state]) * self.alpha