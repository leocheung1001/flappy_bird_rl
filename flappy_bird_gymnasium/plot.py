import gymnasium
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pygame
import jsonschema
import json
import flappy_bird_gymnasium

def read_json():
    with open('data/steps_log_dueling.json', 'r') as file:
    # Read and parse the JSON data
        data = json.load(file)
        
    return data["episodes_ids"], data["num_steps"]



if __name__ == "__main__":
    episodes, steps = read_json()
    # nums = list(range(len(steps)))
    plt.plot(episodes[:3000], steps[:3000])

    # Adding title
    # plt.title('Training episodes for Double DQN v.s. Survival time', fontsize=18)

    # Adding labels
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Number of episodes",  fontsize=18)
    plt.ylabel('Steps',  fontsize=18)

    # Display the plot
    plt.tight_layout()
    plt.show()


    
