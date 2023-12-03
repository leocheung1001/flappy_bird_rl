import gymnasium
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pygame
import random

import tensorflow as tf
# import torch
# import torch.nn as nn
# import torch.optim as optim
from collections import namedtuple, deque
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

import flappy_bird_gymnasium


CUSTOM_MODEL_PATH = ""
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # print(next_state.shape)
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train(epoch=10, audio_on=True, render_mode="human", use_lidar=False):
    batch_size = 64
    # Initialize gym environment and the agent
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=True, render_mode=None, use_lidar=False
    )
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Iterate the game
    episodes = 1000
    for e in range(episodes):
        # reset state in the beginning of each game
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible
        for time_t in range(500):
            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            next_state, reward, done, _, info = env.step(action)
            print(action, reward)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done else -1000

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state
            state = next_state

            # done becomes True when the game ends
            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, time_t))
                break

            # train the agent with the experience of the episode
            if len(agent.memory) > batch_size:
                agent.replay(4)


if __name__ == "__main__":
    train()



