# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval
from gym.wrappers import RecordVideo
from moviepy.editor import VideoFileClip
from IPython.display import Video



# Init the lunar lander env
# had to revert to v2 for sb3
env = gym.make("LunarLander-v2")

num_of_episodes = 1
folder="videos/lunarLander_DQN"

# install env wrapper to record a video
env = RecordVideo(
    env,
    video_folder=folder,                  # Folder
    name_prefix="DQN_v1",              # Video filename prefix
    episode_trigger=lambda x: True        # Record every episode
)

print(f"Running {num_of_episodes} episodes...")
print(f"Saving vides to: {folder}\n")

# load saved model
model = DQN.load("models/dqn_lunarlander_v1_save", env=env)


# Repeat for set number of episodes
for episode_count in range(num_of_episodes):
    # We need to reset the environment before first use
    obs = env.reset()

    # initialize total score to zero and step count
    total_score = 0
    num_of_steps = 0
    success_criteria = None

    episode_over = False

    while not episode_over:
        # DQN agent
        action, _states = model.predict(obs, deterministic=True)

        # Take model action and recieve information 
        obs, reward, done, info = env.step(action)
        total_score = total_score + reward
        episode_over = done
        num_of_steps += 1

        #env.render()


    # The documentation states that a score of 200 or more is a solution
    success_criteria = total_score >= 200
    print(f"Episode {episode_count + 1}: {num_of_steps} steps, reward = {total_score}, success: {success_criteria} ")

env.close()
print("\n")

# convert the first mp4 to a gif and show it
#first_video = VideoFileClip("videos/lunarLander_DQN/DQN_v1-episode-0.mp4")
#first_video = first_video.resize(0.5)
#first_video = first_video.subclip(0, 10)
#first_video.write_gif("videos/lunarLander_hDQN/DQN_v1-episode-0.gif", program="ffmpeg", fps=15)

print("\nDisplaying video from episode 1:")
Video(filename="videos/lunarLander_DQN/DQN_v1-episode-0.mp4", embed=True)