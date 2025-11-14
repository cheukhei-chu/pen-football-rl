import ray
import torch
import numpy as np
import time
import pygame
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
# Import the specific connector we need
from ray.rllib.connectors.env_to_module import FlattenObservations

# We need to import the environment class so we can register it
from multiagent import FootballMultiAgentEnv
from pen_football import TICK_RATE

# Path to your checkpoint file
# CHECKPOINT_PATH = "/Users/cheukheichu/ray_results/PPO_Football_SelfPlay_Experiment1/PPO_football_v1_963ee_00000_0_2025-09-25_04-21-08/checkpoint_000119"
CHECKPOINT_PATH = "/Users/cheukheichu/ray_results/PPO_Football_SelfPlay_Experiment3/PPO_football_v1_dbfd7_00000_0_2025-11-13_21-37-27/checkpoint_000001"

ray.init()
register_env("football_v1", lambda config: FootballMultiAgentEnv(config))
algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
module = algo.get_module("shared_policy")

env = FootballMultiAgentEnv(config={"render_mode": "human"})
obs, info = env.reset()
clock = pygame.time.Clock()

terminated = truncated = False
while not terminated and not truncated:

    agent_ids = sorted(obs.keys())
    obs_list = [obs[agent_id] for agent_id in agent_ids]
    obs_batch = np.stack(obs_list)
    input_tensor = torch.from_numpy(obs_batch)

    # This call is correct.
    output_dict = module.forward_inference({"obs": input_tensor})

    # 1. Get the raw logit tensor of shape (2, 6) from the output dictionary.
    action_logits_tensor = output_dict["action_dist_inputs"]

    # 2. Slice the tensor to get the logits for each action type for the entire batch.
    #    - Columns 0-1 are for 'left' (logits for 0 and 1)
    #    - Columns 2-3 are for 'right'
    #    - Columns 4-5 are for 'jump'
    left_logits = action_logits_tensor[:, 0:2]   # Shape: (2, 2)
    right_logits = action_logits_tensor[:, 2:4]  # Shape: (2, 2)
    jump_logits = action_logits_tensor[:, 4:6]   # Shape: (2, 2)

    # 3. Create probability distributions from the logits and sample actions for the batch.
    # sampled_left_actions = torch.distributions.Categorical(logits=left_logits).sample()    # Shape: (2,)
    # sampled_right_actions = torch.distributions.Categorical(logits=right_logits).sample()   # Shape: (2,)
    # sampled_jump_actions = torch.distributions.Categorical(logits=jump_logits).sample()    # Shape: (2,)
    sampled_left_actions = torch.argmax(left_logits, dim=-1)    # Shape: (2,)
    sampled_right_actions = torch.argmax(right_logits, dim=-1)   # Shape: (2,)
    sampled_jump_actions = torch.argmax(jump_logits, dim=-1)     # Shape: (2,)

    # 4. Reconstruct the action dictionary for the environment by picking the
    #    i-th action for the i-th agent.
    actions = {}
    for i, agent_id in enumerate(agent_ids):
        actions[agent_id] = {
            "left": sampled_left_actions[i].item(),
            "right": sampled_right_actions[i].item(),
            "jump": sampled_jump_actions[i].item(),
        }
    # actions = algo.compute_actions(obs, explore=False)
    if not terminated:
        obs, reward, terminated_dict, truncated_dict, info = env.step(actions)

        env.render()
        clock.tick(TICK_RATE)

        terminated = terminated_dict["__all__"]
        truncated = truncated_dict["__all__"]

env.close()
ray.shutdown()
