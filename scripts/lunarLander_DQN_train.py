import gym #gymnasium as gym
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval


# Init the lunar lander env
env = gym.make("LunarLander-v2")

#print(f"env: {env.action_space}")
#exit()

model = DQN("MlpPolicy", 
            env, 
            verbose=1)


model.learn(total_timesteps=200_000, log_interval=4)
model.save("dqn_lunarlander")

avg_reward, std_reward = eval.evaluate_policy(model, env, n_eval_episodes=25, deterministic=True)

print(f"\nEvaluation has shown: \nAverage Reward: {avg_reward}\nStandard Dev: {std_reward}")

obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
