import gymnasium as gym
from gymnasium.wrappers import  RecordVideo
from IPython.display import Image
from moviepy.editor import VideoFileClip

# Initialise the Lunar Lander Environment 
# Don't render for the statistic version
env = gym.make("LunarLander-v3")
# Bump up the number of episodes 
num_of_episodes = 500
# statistics to collect
running_score = []
num_of_success = 0


print(f"Running {num_of_episodes} random agent episodes...")

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
        # For now we will use a random agent
        action = env.action_space.sample()

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
print()
# print statistics 
print("Stats for 500 random agent episodes:")
print(f"Average score: {sum(running_score)/500}")
print(f"Success count: {num_of_success}")
print(f"Success rate: {num_of_success/500}")