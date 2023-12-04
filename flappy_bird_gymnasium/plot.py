import gymnasium
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pygame
import jsonschema

import flappy_bird_gymnasium

def read_json():
    with open('data/steps_log.json', 'r') as file:
    # Read and parse the JSON data
        data = json.load(file)
        
    return data["log"]



if __name__ == "__main__":
    steps = read_json()
    nums = list(range(len(steps)))
    plt.plot(nums, steps)

    # Adding title
    plt.title('Training episodes v.s. Survival time')

    # Adding labels
    plt.xlabel("Number of episodes")
    plt.ylabel('Steps')

    # Display the plot
    plt.show()

    
