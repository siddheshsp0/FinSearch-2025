# Imports
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.optimizers import Adam
import os
import time
from td3_implementation import Agent


import gymnasium as gym
env = gym.make("Pendulum-v1", render_mode="human")
# env = gym.make("Pendulum-v1")
observation, _ = env.reset()

# Optional: Load saved model
agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape,
              tau=0.005, env=env, batch_size=100, n_actions=env.action_space.shape[0])
agent.noise = 0.0
agent.load_model()
i = 0
total_reward = 0
while (True):
    i+=1;
    # Use the actor network directly (no noise)
    state = tf.convert_to_tensor([observation], dtype=tf.float32)
    action = agent.actor(state)[0].numpy()

    observation, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    if terminated or truncated:
        observation, _ = env.reset()
    print(total_reward/i)
        
    env.render()
    time.sleep(0.01)

env.close()
print(f"Total reward during test run: {total_reward:.2f}")