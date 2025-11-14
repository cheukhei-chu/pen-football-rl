import ray
import torch
import numpy as np
import pygame
import subprocess
import threading
import os

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.callback import Callback

from multiagent import *

# =============================================================================
# === 1. TENSORBOARD LAUNCHER =================================================
# =============================================================================
def launch_tensorboard(logdir):
    """Launches TensorBoard in a background thread."""
    def run_tensorboard():
        # Note: You may need to use `tensorboard.main` if the command isn't in your path
        # from tensorboard.main import run_main
        # run_main(['', f'--logdir={logdir}'])
        subprocess.run(["tensorboard", "--logdir", logdir], check=True)

    print(f"\nLaunching TensorBoard...\n")
    # Give the user the link before starting the process
    # The default port is 6006
    # print(f"--> TensorBoard available at: http://localhost:6006/ <--\n")

    thread = threading.Thread(target=run_tensorboard, daemon=True)
    thread.start()

# =============================================================================
# === 2. CUSTOM EVALUATION CALLBACK ===========================================
# =============================================================================
# class EvaluateOnCheckpoint(Callback):
#     """
#     A custom Ray Tune callback to run an evaluation loop every time a checkpoint is saved.
#     """
#     def on_checkpoint(self, iteration, trials, trial, checkpoint, **kwargs):
#         print(f"\n--- Running Evaluation for Checkpoint: {checkpoint} ---")

#         # Build the Algorithm from the checkpoint
#         algo = Algorithm.from_checkpoint(checkpoint)
#         module = algo.get_module("shared_policy")

#         # Create the environment for rendering
#         env = FootballMultiAgentEnv(config={"render_mode": "human"})
#         obs, info = env.reset()
#         clock = pygame.time.Clock()

#         terminated = truncated = False
#         total_reward_red = 0
#         total_reward_blue = 0

#         while not terminated and not truncated:
#             # Handle Pygame events like closing the window
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     terminated = True

#             # Use the same logic from evaluate.py to get actions
#             agent_ids = sorted(obs.keys())
#             obs_list = [obs[agent_id] for agent_id in agent_ids]
#             obs_batch = np.stack(obs_list)
#             input_tensor = torch.from_numpy(obs_batch)
#             output_dict = module.forward_inference({"obs": input_tensor})
#             action_logits_tensor = output_dict["action_dist_inputs"]
#             left_logits, right_logits, jump_logits = action_logits_tensor[:, 0:2], action_logits_tensor[:, 2:4], action_logits_tensor[:, 4:6]
#             actions = {}
#             for i, agent_id in enumerate(agent_ids):
#                 actions[agent_id] = {
#                     "left": torch.distributions.Categorical(logits=left_logits[i]).sample().item(),
#                     "right": torch.distributions.Categorical(logits=right_logits[i]).sample().item(),
#                     "jump": torch.distributions.Categorical(logits=jump_logits[i]).sample().item(),
#                 }

#             if not terminated:
#                 obs, reward, terminated_dict, truncated_dict, info = env.step(actions)
#                 total_reward_red += reward.get("player_red", 0)
#                 total_reward_blue += reward.get("player_blue", 0)
#                 env.render()
#                 clock.tick(TICK_RATE)
#                 terminated = terminated_dict["__all__"]
#                 truncated = truncated_dict["__all__"]

#         print(f"Evaluation finished. Final Score: Red {total_reward_red:.2f} - Blue {total_reward_blue:.2f}")
#         env.close()
#         # Clean up the loaded Algorithm to free up resources
#         algo.stop()
#         print("--- End of Evaluation ---\n")

# =============================================================================
# === 3. MAIN TRAINING SCRIPT =================================================
# =============================================================================
def train_football():
    # 1. Initialize Ray
    ray.init()

    # 2. Register the custom environment
    register_env("football_v1", lambda config: FootballMultiAgentEnv(**config))

    # Define the base directory for Ray results
    ray_results_dir = os.path.expanduser("~/ray_results")

    # Launch TensorBoard pointing to the base results directory
    launch_tensorboard(ray_results_dir)

    # 3. Configure the PPO algorithm for Multi-Agent Self-Play
    config = (
        PPOConfig()
        .environment(env="football_v1") #, env_config={"render_mode": None})
        .framework("torch") # or "tf2"
        .env_runners(
            num_env_runners=4, # Renamed from num_rollout_workers
            rollout_fragment_length='auto'
        )
        .training(
            gamma=0.99,
            lr=5e-5,
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
        )
        .multi_agent(
            # We have two agents, but they will both share the same policy.
            policies={"shared_policy"},
            # The policy mapping function is the key to self-play.
            # It maps both agent IDs to the same policy ID.
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy")
        )
        # For evaluation (watching the agent play)
        # .evaluation(
        #     evaluation_interval=10,
        #     evaluation_num_env_runners=1,
        #     evaluation_config=PPOConfig.overrides(render_env=True)
        # )
    )

    config.kl_coeff = 0.2
    config.clip_param = 0.2
    config.vf_clip_param = 10.0
    config.entropy_coeff = 0.1
    config.sgd_minibatch_size = 128
    config.num_sgd_iter = 10

    # 4. Run the training process
    # You can use `tune.run` for more advanced experiment tracking or just `config.build().train()` for simplicity.
    stop_criteria = {"training_iteration": 3000}

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            stop=stop_criteria,
            name="PPO_Football_SelfPlay_Experiment1",
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=25, # Save a checkpoint every 25 iterations
                checkpoint_at_end=True,
            ),
        ),
    )
    results = tuner.fit()

    print("Training finished.")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best checkpoint saved at: {best_result.checkpoint}")
    ray.shutdown()

if __name__ == "__main__":
    train_football()
