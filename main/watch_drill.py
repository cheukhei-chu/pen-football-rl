import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pygame
import random

from multiagent import FootballMultiAgentEnv
from policy import *
from pen_football import *

def visualize_drill(policy: str, select_drill, episodes=10, render=False):
    """
    Loads policies from a checkpoint file and evaluates them.
    """
    assert os.path.exists(policy), f"Error: Checkpoint file not found at {policy}"

    policy_red, _ = policy_from_checkpoint_path(policy)
    policy_blue = DummyPolicy()

    policy_red.eval()
    policy_blue.eval()

    env = FootballMultiAgentEnv({"render_mode": "human" if render else None})
    clock = pygame.time.Clock()
    scores = []
    for ep in range(episodes):
        drill = select_drill()
        env.set_setting(drill)
        obs, _ = env.reset()
        done = False
        total_reward = np.zeros(4)

        while not done:
            with torch.no_grad():
                a_red = policy_red.sample_action(obs["player_red"])
                a_blue = policy_blue.sample_action(obs["player_blue"])

            obs, rewards, terminated, truncated, _ = env.step(
                {"player_red": a_red, "player_blue": a_blue}
            )
            done = terminated["__all__"] or truncated["__all__"]
            total_reward += np.array(rewards["player_red"])

            if render:
                env.render()
                clock.tick(30)

        print(f"Episode {ep} final reward for red: {total_reward.sum():.1f} (Score: {total_reward[0]:.1f}, Move: {total_reward[1]:.1f}, Kick: {total_reward[2]:.1f}, Jump: {total_reward[3]:.1f})")
        scores.append(total_reward)

    env.close()
    return scores

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pen Football - One Player")
    visualize_drill(
        policy="../checkpoints/shoot_left_ppo/checkpoint_524288.pth",
        # policy="../checkpoints/block_drill_nobounce/football_episode_500000.pth",
        select_drill=lambda: {"drill": "shoot_left", "par": random.uniform(-1, -40/150)},
        # select_drill=lambda: {"drill": "block_nobounce"},
        episodes=10, render=True
        )
