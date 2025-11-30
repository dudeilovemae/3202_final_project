"""
This python script can be used to run a model with human readable output. 
Created for CSPB 3202 Final Project.

Usage:
run_DQN_model.py file

Author: Ray Franco
"""

# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval
import sys
import os


input_file = sys.argv[1]

if os.path.exists(input_file):
    model_file = input_file
    # Init the lunar lander env
    # had to revert to v2 for sb3
    env = gym.make("LunarLander-v2")

    # load saved model
    model = DQN.load(model_file, env=env)

    obs = env.reset()

    # time steps limit is 1000 so this will at least run 
    # one episode. For well trained models it'll run more.
    for _ in range(1500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
else:
    print(f"can't find file: {input_file}. Exiting program...")
