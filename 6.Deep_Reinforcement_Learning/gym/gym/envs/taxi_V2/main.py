#this needs to be run on conda env openAI in mac.
from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
