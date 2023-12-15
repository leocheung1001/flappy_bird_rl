import gymnasium
import numpy as np
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from collections import deque
import flappy_bird_gymnasium
import json
import argparse


class DQNPER(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(DQNPER, self).__init__()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc4 = tf.keras.layers.Dense(32, activation='relu')
        self.fc5 = tf.keras.layers.Dense(action_dim, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

    def get_action(self, state):
        q_values = self(state)
        return np.argmax(q_values[0])


class PrioritizedMemory():
    def __init__(self, capacity, alpha=0.7):
        # roll back to list for better control
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.alpha = alpha
        self.cur_pos = 0

    def get_len(self):
        return len(self.memory)

    def is_full_capacity(self):
        return len(self.memory) == self.capacity

    # TODO: YONG NEEDS REFINEMENT
    def append_memory(self, state, action, reward, next_state, done):
        priority = 1.0
        if len(self.memory) > 0:
            priority = self.priorities.max()

        if self.is_full_capacity():
            id_ = self.priorities.argmin()
            self.memory[id_] = (state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities[self.cur_pos] = priority
            self.cur_pos += 1
        

    def sample(self, batch_size, beta=0.5):
        probs = self.priorities[:self.get_len()] ** self.alpha
        probs = probs / probs.sum()

        ids = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[id_] for id_ in ids]
        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (self.get_len() * probs[ids]) ** (-beta)
        weights /= weights.max()
        # weights = np.array(weights, dtype=np.float32)

        return ids, weights, np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def update_priorities(self, idxs, priorities):
        self.priorities[idxs] = priorities


# class Memory():
#     def __init__(self, capacity):
#         self.memory = deque(maxlen=capacity)

#     def get_len(self):
#         return len(self.memory)

#     def append_memory(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         sample_size = min(batch_size, len(self.memory))
#         samples = random.sample(self.memory, sample_size)
#         states, actions, rewards, next_states, dones = zip(*samples)
#         return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


def save_steps_log(steps_log):
    with open("data/steps_log_per.json", 'w') as file:
        json.dump(steps_log, file, indent=4)


def train():
    steps_log = []
    episodes_ids = []
    env = gymnasium.make("FlappyBird-v0", audio_on=True, render_mode=None, use_lidar=False)
    T = 1000000
    num_episodes = 10000000000
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.03
    epsilon_decay = 0.995
    mem_size = 10000
    epsilon_per = 1e-6

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    sample_input = np.random.random((1, state_dim))
    policy_model = DQNPER(state_dim, action_dim)
    target_model = DQNPER(state_dim, action_dim)
    policy_model(sample_input) 
    target_model(sample_input) 
    target_model.set_weights(policy_model.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    memory = PrioritizedMemory(mem_size)

    for i_episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).reshape((1, state_dim))
        total_reward = 0
        num_steps = 0

        for i in range(T):
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                q_values = policy_model(state)
                action = np.argmax(q_values[0])
            # print(action)
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            memory.append_memory(state, action, reward, next_state, done)

            if memory.get_len() > batch_size and i % 5 == 0:
                ids, weights, states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states = tf.squeeze(states, axis=1)
                # print(f"states.shape {states.shape}")
                # print(f"next_states.shape {next_states.shape}")
                with tf.GradientTape() as tape:
                    q_values = policy_model(states)
                    q_values = tf.reduce_sum(tf.multiply(q_values, tf.one_hot(actions, action_dim)), axis=1)
                    next_max_q_values = tf.reduce_max(target_model(next_states), axis=1)
                    y = rewards + gamma * next_max_q_values * (1 - dones)
                    # loss = tf.keras.losses.mean_squared_error(y, q_values)

                    temp_diff = y - q_values
                    loss = tf.reduce_mean(weights * tf.square(temp_diff))

                gradients = tape.gradient(loss, policy_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))
                memory.update_priorities(ids, np.abs(temp_diff.numpy()) + epsilon_per)

            if done:
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
                break

            state = np.array(next_state, dtype=np.float32).reshape((1, state_dim))
            num_steps = i

        if i_episode % 5 == 0:
            target_model.set_weights(policy_model.get_weights())

        if i_episode % 20 == 0:
            print(f"Episode: {i_episode}, Total Reward: {total_reward}, Survival Steps {num_steps}")
            steps_log.append(num_steps)
            episodes_ids.append(i_episode)

        # if i_episode % 100 == 0:
        #     target_model.save_weights('tf_target_per.h5')
        #     save_steps_log({"episodes_ids": episodes_ids, "num_steps" : steps_log})

    env.close()


def play(epoch=100, audio_on=True, render_mode=None, use_lidar=False):
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    target_net = DQNPER(state_dim, action_dim)
    dummy_input = np.random.random((1, state_dim))
    target_net(dummy_input)
    target_net.load_weights('tf_target_per.h5')
    success = 0
    for i in range(epoch):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        steps = 0
        while True:
            action = target_net.get_action(state)
            # print(action)
            action = np.array(action, copy=False, dtype=env.env.action_space.dtype)

            next_state, _, done, _, info = env.step(action)

            state = np.expand_dims(next_state, axis=0)
            # print(f"Obs: {state}\n" f"Action: {action}\n" f"Score: {info['score']}\n")
            steps += 1
            if done:
                print(f"Episode: {i} \t Steps: {steps}")
                if steps > 1000:
                    success += 1
                break

    print(f"Pass rate: {success / epoch}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="train", action="store_true")
    args = parser.parse_args()

    if args.train:
        print("*" * 50)
        print("Traininig the DQN with PER!")
        print("*" * 50)
        train()
    else:
        print("*" * 50)
        print("Playing the DQN with PER!")
        print("*" * 50)
        play()
