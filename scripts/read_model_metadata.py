"""
This python script can be used to inspect the metadata of a model. 
Created for CSPB 3202 Final Project.

Usage:
model_metadata_reader.py file

Author: Ray Franco
"""

# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval
import sys
import os
import shutil


input_file = sys.argv[1]

if os.path.exists(input_file):
    print(f"Viewing the metadata for model: {input_file}")

    # Init the lunar lander env
    # had to revert to v2 for sb3
    env = gym.make("LunarLander-v2")
    model_file_name = input_file

    # load saved model
    model = DQN.load(model_file_name, env=env)

    # print model meta data
    print()
    print(f"---- Meta Data for model: {model_file_name} ----")
    print(f"Algorithm: {model.__class__.__name__}")
    print(f"Policy: {model.policy.__class__.__name__}")
    print(f"Obs Space: {model.observation_space}")
    print(f"Action Space: {model.action_space}")
    print(f"Device: {model.device}")
    print(f"-- Hyperparameters:")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"Gamma: {model.gamma}")
    print(f"Batch size: {model.batch_size}")
    print(f"Buffer size: {model.buffer_size}")
    print(f"-- Training Done:")
    print(f"Timesteps: {model.num_timesteps}")
    print(f"Episodes: {model._episode_num}")

else: 
    print(f"can't find file: {input_file}. Exiting program...")

