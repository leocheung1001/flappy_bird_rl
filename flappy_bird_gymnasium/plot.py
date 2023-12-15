import gymnasium
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pygame
import jsonschema
import json
import flappy_bird_gymnasium

def read_json():
    with open('data/steps_log.json', 'r') as file:
    # Read and parse the JSON data
        data = json.load(file)
        
    return data["episodes_ids"], data["num_steps"]



if __name__ == "__main__":
    episodes, steps = read_json()
    # nums = list(range(len(steps)))
    plt.plot(episodes, steps)

    # Adding title
    plt.title('Training episodes v.s. Survival time')

    # Adding labels
    plt.xlabel("Number of episodes")
    plt.ylabel('Steps')

    # Display the plot
    plt.show()

    
