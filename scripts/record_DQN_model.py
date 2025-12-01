"""
This python script can be used to run a model and save it's output into a video. 
Created for CSPB 3202 Final Project.

Usage:
record_DQN_model.py file prefix

Author: Ray Franco
"""


# Had to revert to gym for sb3
import gym 
from gym.wrappers import RecordVideo
from IPython.display import Video
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval
import sys
import os


input_file = sys.argv[1]
prefix = sys.argv[2]

if os.path.exists(input_file):
    model_file = input_file
    # Init the lunar lander env
    # had to revert to v2 for sb3
    env = gym.make("LunarLander-v2")
    folder="videos/lunarLander_DQN"
    num_of_episodes = 1

    # install env wrapper to record a video
    env = RecordVideo(
        env,
        video_folder=folder,                  # Folder
        name_prefix=prefix,                # Video filename prefix
        episode_trigger=lambda x: True        # Record every episode
    )

    # load saved model and reset env
    model = DQN.load(model_file, env=env)
    obs = env.reset()

    print(f"Running {num_of_episodes} episodes...")
    print(f"Saving video to: {folder}\n")    

    # Repeat for set number of episodes
    for episode_count in range(num_of_episodes):
        # We need to reset the environment before first use
        obs = env.reset()

        # initialize total score to zero and step count
        total_score = 0
        num_of_steps = 0
        success_criteria = None

        episode_over = False

        while not episode_over:
            # DQN agent
            action, _states = model.predict(obs, deterministic=True)

            # Take model action and recieve information 
            obs, reward, done, info = env.step(action)
            total_score = total_score + reward
            episode_over = done
            num_of_steps += 1
    env.close()

else:
    print(f"can't find file: {input_file}. Exiting program...")
