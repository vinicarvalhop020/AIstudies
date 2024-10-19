import gym
import random
from IPython import display
import matplotlib.pyplot as plt
import numpy

env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import gym
import random

env = gym.make('CartPole-v0', render_mode = 'rgb_array')
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10
for episode in range(1,episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        img = env.render()
        plt.imshow(img)
        display(plt.gcf())
        clear_output(wait=True)
        action = random.choice([0,1])
        n_state, reward, done, info,_ = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))   
env.close()
 
