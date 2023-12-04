import numpy as np
import gymnasium
import flappy_bird_gymnasium
import json




class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_decay = exploration_decay
        self.q_table = {}
        self.factor = 200


    def discretize_state(self, original_state):
        factor = self.factor
        next_x, next_y, velocity, next_next_y = original_state[3], original_state[5], original_state[8], original_state[10]
        state = str(int(next_x * factor)) + "_" + str(int(next_y * factor )) + "_" + str(int(velocity * factor )) + "_" + str(int(next_next_y * factor))
        self.init_state_if_null(state)
        return state


    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])


    def init_state_if_null(self, state):
        if self.q_table.get(state) == None:
            self.q_table[state] = [0, 0]
            num = len(self.q_table.keys())
            # print(state)
            if num > 20000 and num % 1000 == 0:
                print("======== Total state: {} ========".format(num))
            # if num > 30000:
            #     print("======== New state: {0:14s}, Total: {1} ========".format(state, num))


    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (target - predict)

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay


    def save_qtable(self):
        """
        Dump the qvalues to the JSON file
        """
        print("******** Saving Q-table(%d keys) to local file... ********" % len(self.q_table.keys()))
        with open("data/qvalues.json", 'w') as file:
        # The indent parameter adds indentation to make the file human-readable.
            json.dump(self.q_table, file, indent=4)
        print("******** Q-table(%d keys) updated on local file ********" % len(self.q_table.keys()))


def train():
    env = gymnasium.make("FlappyBird-v0", audio_on=True, render_mode=None, use_lidar=False)
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    agent = QLearningAgent(n_observations, n_actions)

    num_episodes = 1000000000
    T = 1000000
    CHECK_POINT = 30000

    # Assuming some environment interaction loop
    for i_episode in range(num_episodes):  # Number of episodes
        current_state, _ = env.reset()  # Reset environment to start state
        current_state = agent.discretize_state(current_state)

        done = False
        total_reward = 0 
        for t in range(T):
            action = agent.choose_action(current_state)
            next_state, reward, done, _, info = env.step(action)  # Take action in the environment
            total_reward += reward
            next_state = agent.discretize_state(next_state)
            agent.learn(current_state, action, reward, next_state)
            current_state = next_state

            if done:
                break
        
        if i_episode % 1000 == 0:
            print(f"episode {i_episode} \twith steps {t} \t reward {total_reward}.")

        if i_episode % CHECK_POINT == 0:
            agent.save_qtable()

    # 3 5 8 10


if __name__ == "__main__":
    train()
    # train()