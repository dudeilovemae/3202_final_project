# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval


# Init the lunar lander env
# had to revert to v2 for sb3
env = gym.make("LunarLander-v2")


# load saved model
model = DQN.load("models/dqn_lunarlander_v1", env=env)

obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
