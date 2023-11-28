import gymnasium
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pygame

import tensorflow as tf

import flappy_bird_gymnasium


CUSTOM_MODEL_PATH = ""

class DQN(tf.keras.Model):
    def __init__(self, action_dim, state_dim, epsilon, gamma, mem_size=1000):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.mem_size = mem_size
        self.mem_list = deque(maxlen=self.mem_size)
        self.lr = 0.001
        self.batch_size = batch_size
        self.gamma = gamma

        # build the model itself
        q_model = tf.keras.models.Sequential()
        q_model.add(tf.keras.layers.Dense(512, input_dim=self.state_size, activation='relu'))
        q_model.add(tf.keras.layers.Dense(256, activation='relu'))
        q_model.add(tf.keras.layers.Dense(128, activation='relu'))
        q_model.add(tf.keras.layers.Dense(32, activation='relu'))
        q_model.add(tf.keras.layers.Dense(self.action_dim, activation='linear'))
        q_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.lr))
        self.q_model = q_model


    def store_exp(self, s, a, r, s_prime, is_done):
        self.mem_list.append((s, a, r, s_prime, is_done))


    def experience_replay():
        batch = random.sample(self.mem_list, self.batch_size)
        for exp in batch:
            # unpack the stuff
            state, action, reward, state_prime, is_done = exp
            if is_done:
                y = reward
            else:
                y = reward + self.gamma * torch.max(self.q_model(state_prime))
        

        pred_q = self.q_model.predict(state)
        pred_q[action] = y
        self.q_model.fit(state, pred_q, epochs=1, verbose=0)
            
    def train():
        T = 30
        for i in range(T):
            obs = env.reset()
            pass


        
        



def train(epoch=10, audio_on=True, render_mode="human", use_lidar=False):
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=True, render_mode="human", use_lidar=True
    )
    obs = env.reset()
    print(obs)
    print(obs[0].shape)










if __name__ == "__main__":
    train()



