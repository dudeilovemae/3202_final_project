import gymnasium as gym
from gymnasium.wrappers import  RecordVideo
from IPython.display import Image
from moviepy.editor import VideoFileClip

from collections import namedtuple
from enum import Enum
import random

# Create a named tuple to hold observation data
Observation = namedtuple("Observation", ["x", "y", "x_vel", "y_vel", "angle", "angular_vel", "touch1", "touch2"])

# Create enum for action space
class LunarAction(Enum):
    IDLE = 0
    FIRE_RIGHT = 1
    FIRE_MAIN = 2
    FIRE_LEFT = 3


def lunar_heuristic(obs):
    obs_tuple = Observation(*obs)

    # Prioity 0: If touch down do nothing
    if obs_tuple.touch1 and obs_tuple.touch2:
        return LunarAction.IDLE.value

    # Priority 1: Control Fall Speed
    if obs_tuple.y_vel < -.3:
        return LunarAction.FIRE_MAIN.value
    
    # Priority 2: Too much tilt
    if obs_tuple.angle > .15:
        return LunarAction.FIRE_LEFT.value
    if obs_tuple.angle < -.15:
        return LunarAction.FIRE_RIGHT.value

    # Priorty 3: Not between flags
    if obs_tuple.x < -.15:
        return LunarAction.FIRE_LEFT.value
    if obs_tuple.x > .15:
        return LunarAction.FIRE_RIGHT.value
    
    
  
    return LunarAction.IDLE.value
    



# Initialise the Lunar Lander Environment 
# No need to graphics 
env = gym.make("LunarLander-v3")
# Let's run more episodes for better stats
num_of_episodes = 500
# statistics to collect
running_score = []
num_of_success = 0

print(f"Running {num_of_episodes} heuristic agent episodes...")

# Repeat for set number of episodes
for episode_count in range(num_of_episodes):
    # We need to reset the environment before first use
    observation, info = env.reset()

    # initialize total score to zero and step count
    total_score = 0
    num_of_steps = 0
    success_criteria = None

    episode_over = False

    while not episode_over:
        # Simple Heuristic agent
        action = lunar_heuristic(observation)

        # Take the random action and recieve information 
        observation, reward, terminated, truncated, info = env.step(action)
        total_score = total_score + reward

        num_of_steps += 1

        episode_over = terminated or truncated

    # The documentation states that a score of 200 or more is a solution
    success_criteria = total_score >= 200
    if success_criteria:
        num_of_success += 1
    running_score.append(total_score)


env.close()
print("\n")

# print statistics 
print("Stats for 500 random agent episodes:")
print(f"Average score: {sum(running_score)/500}")
print(f"Success count: {num_of_success}")
print(f"Success rate: {num_of_success/500}")