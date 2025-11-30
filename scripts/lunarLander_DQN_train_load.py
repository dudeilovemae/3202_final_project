# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval
import shutil


time_steps_to_train = 50_000

# Init the lunar lander env
# had to revert to v2 for sb3
env = gym.make("LunarLander-v2")


# load saved model
model = DQN.load("models/dqn_lunarlander_v1", env=env)

time_steps_so_far = model.num_timesteps


model.learn(total_timesteps=time_steps_so_far + time_steps_to_train, log_interval=4)
# save with number of timesteps in name
model.save(f"models/dqn_lunarlander_v1_{time_steps_so_far + time_steps_to_train}")

# copy to original name that way I have a checkpoint
shutil.copy2(f"models/dqn_lunarlander_v1_{time_steps_so_far + time_steps_to_train}.zip", "models/dqn_lunarlander_v1.zip" )


avg_reward, std_reward = eval.evaluate_policy(model, env, n_eval_episodes=25, deterministic=True)

print(f"\nEvaluation has shown: \nAverage Reward: {avg_reward}\nStandard Dev: {std_reward}")

