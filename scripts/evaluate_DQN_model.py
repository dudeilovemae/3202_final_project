"""
This python script can be used to evaluate a model. 
Created for CSPB 3202 Final Project.

Usage:
evaluate_DQN_model.py file

Author: Ray Franco
"""

# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval
import sys
import os

# Init the lunar lander env
# had to revert to v2 for sb3
env = gym.make("LunarLander-v2")
num_of_episodes = 100


input_file = sys.argv[1]

if os.path.exists(input_file):
    model_file = input_file

    # load saved model
    model = DQN.load(model_file, env=env)

    print(f"Now evaluating policy for {model_file} using {num_of_episodes} episodes.")
    avg_reward, std_reward = eval.evaluate_policy(model, env, n_eval_episodes=num_of_episodes, deterministic=True)

    print(f"\nEvaluation has shown: \nAverage Reward: {avg_reward}\nStandard Dev: {std_reward}")
    env.close()
else:
    print(f"can't find file: {input_file}. Exiting program...")
