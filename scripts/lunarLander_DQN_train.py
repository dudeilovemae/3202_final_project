# version controlled in scripts/lunarLander_DQN_train.py

# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval

# Init the lunar lander env
# had to revert to v2 for sb3
env = gym.make("LunarLander-v2")


model = DQN("MlpPolicy", 
            env, 
            verbose=0, # Supressing output 
            tensorboard_log="logs/dqn_tensorboard_log")

print(f"Calling learn for 50000 timesteps...")
model.learn(total_timesteps=50_000, log_interval=4)
model.save("models/dqn_lunarlander_v1")
print(f"Learn complete. ")

print(f"Now evaluating policy")
avg_reward, std_reward = eval.evaluate_policy(model, env, n_eval_episodes=25, deterministic=True)

print(f"\nEvaluation has shown: \nAverage Reward: {avg_reward}\nStandard Dev: {std_reward}")

obs = env.reset()
env.close()
