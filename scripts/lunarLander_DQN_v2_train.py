# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval

# Init the lunar lander env
# had to revert to v2 for sb3
env = gym.make("LunarLander-v2")


model = DQN("MlpPolicy", 
            env, 
            verbose=0, # no logs
            exploration_fraction=0.2, # explore longer
            exploration_final_eps=0.1, # Keep some random at the end
            gamma=0.99,
            learning_starts=400, # Dont start learning until the buffer is fuller
            batch_size=32*4, # four times the default batch size
            target_update_interval=500,
            tensorboard_log="logs/dqn_v2_tensorboard_log")

print(f"Calling learn for 800_000 timesteps...")
model.learn(total_timesteps=800_000)
model.save("models/dqn_lunarlander_v2")
print(f"Learn complete. ")

print(f"Now evaluating policy")
avg_reward, std_reward = eval.evaluate_policy(model, env, n_eval_episodes=25, deterministic=True)

print(f"\nEvaluation has shown: \nAverage Reward: {avg_reward}\nStandard Dev: {std_reward}")

obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()