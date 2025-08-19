import gymnasium as gym
import numpy as np
terminated = False
truncated = False

env = gym.make('CartPole-v0', render_mode="human")
state, info = env.reset()

while not (terminated or truncated):
    action = np.random.choice([0,1])
    next_state, reward, terminated, truncated, info = env.step(action)

env.close()