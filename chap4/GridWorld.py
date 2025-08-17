import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def argmax(action_values):
    max_value = max(action_values.values())
    max_key = 0
    for key,values in action_values.items():
        if values == max_value :
            max_key = key
            break
    return max_key

class GridWorld:
    def __init__(self):
        # 行動
        self.action_space = [0,1,2,3]
        # それぞれの行動
        self.action_command = {
            0:"up",
            1:"down",
            2:"left",
            3:"right"
        }
        
        # 地図の報酬
        # 1はゴールの報酬
        # -1はゲーム終了　爆弾
        self.reward_mapping = np.array(
            [
                [0,0,0,1.0],
                [0,None,0,-1.0],
                [0,0,0,0]
            ]
        )
        # 座標(状態)
        self.goal_state = (0,3) # ゴールの位置
        self.obstacle_state = (1,1) # 障害物の位置
        self.bomb_state = (1,3) # 爆弾の位置
        self.start_state = (2,0) #初期位置

        # エージェントの状態（位置）
        self.agent_state = self.start_state
    @property
    def height(self):
        return len(self.reward_mapping)
    @property
    def width(self):
        return len(self.reward_mapping[0])
    @property
    def shape(self):
        return self.reward_mapping.shape
    # 行動の一覧
    def actions(self):
        return self.action_space

    # 状態の一覧
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h,w)

    # 次の状態へ遷移
    def next_state(self,state,action):
        # 相対的に移動
        action_move_map = [(-1,0),(1,0),(0,-1),(0,1)]
        # 選択
        move = action_move_map[action]

        next_state = (state[0]+move[0],state[1]+move[1])

        # xとy座標が範囲内に収まっているか？

        next_x ,next_y = next_state

        # 収まっていなければ現在の位置へ
        if (next_x < 0 or next_x >= self.height) or (next_y < 0 or next_y >= self.width):
            next_state = state
        elif next_state == self.obstacle_state:
            next_state = state
        
        return next_state
    
    def reward(self,state,action,next_state):
        return self.reward_mapping[next_state]
    
    # ワンステップだけ更新評価
    def eval_one(self,pi,V,env,gamma=0.9):
        for state in self.states():
            if state == env.goal_state:
                V[state] = 0
                # ゴールは0
                continue
            action_probs = pi[state]
            new_V = 0

            for action,action_prob in action_probs.items():
                next_state = env.next_state(state,action)
                r = env.reward(state,action,next_state)

                new_V += action_prob * (r + gamma * V[next_state])
            V[state] = new_V
        return V
    
    # 価値関数の可視化
    def render_value_function(self, V):
        # Vは {(i,j): value} の辞書を想定
        grid = np.full(self.shape, np.nan, dtype=float)

        # 値を grid に反映
        for (i, j), v in V.items():
            if v is not None:
                grid[i, j] = v
            if self.obstacle_state == (i,j):
                grid[i,j] = None

        plt.figure(figsize=(6,4))
        plt.imshow(grid, cmap="coolwarm", interpolation="nearest")

        # 値をセルに書き込む
        for i in range(self.height):
            for j in range(self.width):
                if not np.isnan(grid[i, j]):
                    plt.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", color="black")

        # 特殊なセルをマーク
        plt.text(self.start_state[1], self.start_state[0], "S", ha="center", va="center", 
                color="blue", fontsize=16, fontweight="bold")
        plt.text(self.goal_state[1], self.goal_state[0], "G", ha="center", va="center", 
                color="green", fontsize=16, fontweight="bold")
        plt.text(self.bomb_state[1], self.bomb_state[0], "B", ha="center", va="center", 
                color="red", fontsize=16, fontweight="bold")

        plt.colorbar(label="Value")
        plt.title("Value Function Heatmap")
        plt.show()
    def policy_eval(self,pi,V,env,gamma ,thershold = 0.001):
        while True:
            old_V = V.copy()
            V = self.eval_one(pi,V,env,gamma)

            delta = 0

            for state in V.keys():
                temp = abs(V[state] -old_V[state])
                if delta < temp:
                    delta = temp

            if delta < thershold:
                break
        return V
    def greedy_policy(self,V,env,gamma):
        # 方策
        pi = {}

        for state in self.states():

            action_values = {}

            for action in self.actions():

                next_state = env.next_state(state,action)
                r = env.reward(state,action,next_state)
                value = r + gamma * V[next_state]
                action_values[action] = value
            
            # 最大の行動
            max_action = argmax(action_values)
            action_probs = { i:0 for i in range(4)}
            # 決定論的に変換
            action_probs[max_action] = 1
            
            pi[state] = action_probs
        return pi
    # greedy_policyを繰り返す
    def greedy_itr(self,env,gamma,thershold = 0.001):
        # 方策
        pi = defaultdict(lambda : { i:0.25 for i in range(4)})
        # 価値関数
        V = defaultdict(lambda : 0)

        while True:
            V = self.policy_eval(pi,V,env,gamma,thershold)
            new_pi = self.greedy_policy(V,env,gamma)

            if new_pi == pi:
                break
            
            pi = new_pi
        return pi
    def value_itr_one(self,V,env,gamma):
        for state in env.states():
            if state == env.goal_state:
                V[state] = 0
                # ゴールは0
                continue
            # 方策反復だと　action_probsやpiが必要
            action_values = []

            for action in env.actions():
                next_state = env.next_state(state,action)
                r = env.reward(state,action,next_state)

                v = (r + gamma * V[next_state])

                action_values.append(v)
            V[state] = max(action_values)
        return V
    # 価値反復
    def value_itr(self,V,env,gamma,thershold = 0.001):
        # 方策は必要ない

        while True:
            
            old_V = V.copy()

            V = self.value_itr_one(V,env,gamma)

            delta = 0
            for state in V.keys():
                temp = abs(V[state]-old_V[state])
                if delta < temp:
                    delta = temp
            if delta < thershold:
                break

        return V