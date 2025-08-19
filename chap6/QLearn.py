
from collections import defaultdict,deque
import numpy as np
from TdAgent import greedy_probs


class QLearnAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.eps = 0.1    
        self.alpha = 0.8
        self.action_size = 4

        # 方策
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        # ターゲット方策 => 評価と改善
        # 挙動方策 => 実際に行動
        self.pi = defaultdict(lambda: random_actions)      
        self.b = defaultdict(lambda: random_actions)        
  
        self.Q = defaultdict(lambda : 0)
        self.memory = deque(maxlen=2)

    # 行動を一つとる
    def get_action(self,state) -> int:

        # 状態ごとのaction_probを取り出すことに注意
        action_prob = self.b[state]
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
    def update(self,state,action,reward,next_state,goal_done):

        if goal_done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state,action] for action in range(self.action_size)]
            next_q_max = max(next_qs)


        # Q(next_state,next_action)

        td_target = reward + self.gamma * next_q_max

        self.Q[state,action] += (td_target - self.Q[state,action]) * self.alpha
        
        
        self.pi[state] = greedy_probs(self.Q, state, 0)
        self.b[state] = greedy_probs(self.Q, state, self.eps)

