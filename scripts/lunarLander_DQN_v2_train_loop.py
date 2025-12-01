# Had to revert to gym for sb3
import gym 
from stable_baselines3 import DQN
import stable_baselines3.common.evaluation as eval



# Init the lunar lander env
# had to revert to v2 for sb3
env = gym.make("LunarLander-v2")
explor_fract = 0.28

for i in range(18):
    print(f"Attempt {i+1} of 18")
    print(f"exploration fraction: {explor_fract}")
    model = DQN("MlpPolicy", 
                env, 
                verbose=0, # no logs
                exploration_fraction=explor_fract, # explore longer
                exploration_final_eps=0.1, # Keep some random at the end
                gamma=0.99,
                learning_starts=400, # Dont start learning until the buffer is fuller
                buffer_size=300_000,
                learning_rate=0.0004, # increasing learning rate slightly 
                batch_size=32*4, # four times the default batch size
                target_update_interval=500,
                tensorboard_log="logs/dqn_v2_tensorboard_log")

    if i > 6:
        explor_fract = 0.35
    if i > 11:
        explor_fract = 0.70

    print(f"Calling learn for 800_000 timesteps...in 100k chunks")
    
    for chunk in range(800_000//100_000):
        print(f"chunk: {chunk+1}")
        model.learn(total_timesteps=100_000, reset_num_timesteps=False, tb_log_name=f"dqn_v2_run_{i}")
        #model.save("models/dqn_lunarlander_v2")
        #print(f"Learn complete. ")

        print(f"Now evaluating policy")
        avg_reward, std_reward = eval.evaluate_policy(model, env, n_eval_episodes=25, deterministic=True)

        print(f"\nEvaluation has shown: \nAverage Reward: {avg_reward}\nStandard Dev: {std_reward}")

        if avg_reward > 200:
            print(f"Avg reward greater than 200!")
            print(f"Saving model to: models/batch/dqn_lunarlander_v2_{avg_reward}")
            model.save(f"models/batch/dqn_lunarlander_v2_{avg_reward}")

        obs = env.reset()

env.close()