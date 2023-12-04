import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium
import flappy_bird_gymnasium
import json

class DQN(Model):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = Dense(128, activation='relu', input_shape=(inputs,))
        self.fc4 = Dense(32, activation='relu')
        self.fc5 = Dense(outputs, activation=None)

    def call(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)

    def get_action(self, state):
        q_values = self.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

def save_steps_log(steps_log):
    with open("data/steps_log.json", 'w') as file:
    # The indent parameter adds indentation to make the file human-readable.
        json.dump(steps_log, file, indent=4)


def train():
    # Set up the environment
    env = gymnasium.make("FlappyBird-v0", audio_on=True, render_mode=None, use_lidar=False)

    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    MODEL_SAVE = 50
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    policy_net = DQN(n_observations, n_actions)
    target_net = DQN(n_observations, n_actions)
    dummy_input = np.random.random((1, n_observations))
    policy_net(dummy_input)
    target_net(dummy_input)
    target_net.set_weights(policy_net.get_weights())

    optimizer = Adam(learning_rate=1e-3)
    memory = deque(maxlen=10000)
    mse = tf.keras.losses.MeanSquaredError()


    def select_action(state, steps_done):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        # eps_threshold = 0.1
        if np.random.rand() < eps_threshold:
            return np.random.randint(n_actions)
        else:
            q_values = policy_net.predict(state[np.newaxis], verbose=0)
            return np.argmax(q_values[0])

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return

        batch = random.sample(memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        q_values = policy_net.predict(states, verbose=0)
        q_values_next = target_net.predict(next_states, verbose=0)
        max_q_values_next = np.max(q_values_next, axis=1)
        
        targets = rewards + GAMMA * max_q_values_next * (1 - dones)
        targets_full = q_values
        targets_full[np.arange(BATCH_SIZE), actions] = targets

        with tf.GradientTape() as tape:
            q_values = policy_net(states)
            loss = mse(targets_full, q_values)

        grads = tape.gradient(loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

    steps_done = 0
    num_episodes = 100000
    T = 100000

    # load the pretrained models
    target_net.load_weights('target_net.h5')
    policy_net.load_weights('policy_net.h5')
    print("Load pretrained models")


    steps = []
    for i_episode in range(num_episodes):

        # print(f"episode {i_episode}")
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        for t in range(T):
            action = select_action(state, steps_done)
            # print(action)
            steps_done += 1
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            next_state = np.array(next_state, dtype=np.float32)

            memory.append((state, action, reward, next_state, done))
            state = next_state

            optimize_model()

            if done:
                print(f"episode {i_episode} \twith steps {t} \t reward {total_reward}.")
                steps.append(t)
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.set_weights(policy_net.get_weights())
        
        if i_episode % MODEL_SAVE == 6:
            target_net.save_weights('target_net.h5')
            policy_net.save_weights('policy_net.h5')
            save_steps_log({"log" : steps})


    print('Complete')
    env.close()


def play(epoch=10, audio_on=True, render_mode="human", use_lidar=False):
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar
    )

    # init models
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    target_net = DQN(n_observations, n_actions)
    dummy_input = np.random.random((1, n_observations))
    target_net(dummy_input)
    target_net.load_weights('target_net.h5')

    # run
    for _ in range(epoch):
        state, _ = env.reset()
        state = np.expand_dims(state, axis=0)
        while True:
            # Getting action
            action = target_net.get_action(state)
            action = np.array(action, copy=False, dtype=env.env.action_space.dtype)

            # Processing action
            next_state, _, done, _, info = env.step(action)

            state = np.expand_dims(next_state, axis=0)
            print(f"Obs: {state}\n" f"Action: {action}\n" f"Score: {info['score']}\n")

            if done:
                break

    env.close()
    assert state.shape == (1,) + env.observation_space.shape
    assert info["score"] > 0


# def test_play():
#     # play(epoch=1, audio_on=False, render_mode=None, use_lidar=False)
#     # play(epoch=1, audio_on=False, render_mode=None, use_lidar=True)
#     train()


if __name__ == "__main__":
    play()
    # train()
