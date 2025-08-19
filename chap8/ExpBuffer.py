from collections import deque
import random
import numpy as np

class ExpBuffer:
    def __init__(self,buffer_size,batch_size) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    # バッファーにデータを追加
    def add(self,state,action,reward,next_state,goal_done):
        data = (state,action,reward,next_state,goal_done)
        self.buffer.append(data)
    def __len__(self):
        # バッファーの長さを返す
        return len(self.buffer)
    def get_batch(self):
        # バッファーから取り出す
        data = random.sample(self.buffer,self.batch_size)

        state = np.vstack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.vstack([x[3] for x in data])
        goal_done = np.array([x[4] for x in data]).astype(np.int32)
        return state,action,reward,next_state,goal_done

