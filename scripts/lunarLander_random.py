import gymnasium as gym

# Initialise the Lunar Lander Environment 
env = gym.make("LunarLander-v3", render_mode="human")

# We need to reset the environment before first use
observation, info = env.reset()
# initialize total score to zero 
total_score = 0

# Repeat for 1000 steps
for _ in range(1000):
    # For now we will use a random agent
    action = env.action_space.sample()

    # Take the random action and recieve information 
    observation, reward, terminated, truncated, info = env.step(action)
    total_score = total_score + reward

    # if the episode is over then reset to start new episode
    # and reset total score
    if terminated or truncated:
        print(f"Total Score: {total_score}")
        if total_score >= 200:
            print(f"---- Success!")
        else:
            print(f"---- failure")
        observation, info = env.reset()
        total_score = 0

env.close()