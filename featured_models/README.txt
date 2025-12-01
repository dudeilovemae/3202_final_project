This directory contains a save of notable models for reference by jupyter notebook.

---- Meta Data for model: ./dqn_lunarlander_model_default.zip ----
Algorithm: DQN
Policy: DQNPolicy
Obs Space: Box([-inf -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf], (8,), float32)
Action Space: Discrete(4)
Device: cpu
-- Hyperparameters:
Learning Rate: 0.0001
Gamma: 0.99
Batch size: 32
Buffer size: 1000000
-- Training Done:
Timesteps: 300000
Episodes: 797

--- Now evaluating policy for dqn_lunarlander_model_default.zip using 50 episodes.

Evaluation has shown: 
Average Reward: -17.93202602169171
Standard Dev: 18.843608629971794


---- Meta Data for model: ./dqn_lunarlander_solution.zip ----
Algorithm: DQN
Policy: DQNPolicy
Obs Space: Box([-inf -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf], (8,), float32)
Action Space: Discrete(4)
Device: cpu
-- Hyperparameters:
Learning Rate: 0.0004
Gamma: 0.99
Batch size: 128
Buffer size: 300000
-- Training Done:
Timesteps: 600000
Episodes: 1303


--- Now evaluating policy for dqn_lunarlander_v2_266_solution using 50 episodes.

Evaluation has shown: 
Average Reward: 253.87798247905462
Standard Dev: 52.600668908196724