from collections import defaultdict,deque
import numpy as np
from TdAgent import greedy_probs


class SarasaOnAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.action_size = 4
        # 方策
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)        
        self.Q = defaultdict(lambda : 0)
        self.counts = defaultdict(lambda : 0)
        self.memory = deque(maxlen=2)
        self.alpha = 0.8
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

    def add(self,state,action,reward,goal_done) -> None:
        data = (state,action,reward,goal_done)
        self.memory.append(data)
    def reset(self) -> None:
        self.memory.clear()
    def update(self,state,action,reward,goal_done):
        # t時刻の (s_t, a_t, r_t, done_t) をpush

        self.add(state,action,reward,goal_done)


        if len(self.memory) < 2:
            return 
        
        state,action,reward,goal_done = self.memory[0]
        next_state,next_action,_,_ = self.memory[1]
        
        # Q(next_state,next_action)
        next_q = 0 if goal_done else self.Q[next_state,next_action]

        td_target = reward + self.gamma * next_q

        self.Q[state,action] += (td_target - self.Q[state,action]) * self.alpha
        self.pi[state] = greedy_probs(self.Q, state, self.eps)
