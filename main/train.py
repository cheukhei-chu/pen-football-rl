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

def compute_returns(rewards_list, gamma=0.99):
    G = 0
    returns = []
    for r in reversed(rewards_list):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns = np.array(returns)
    return torch.tensor(returns, dtype=torch.float32)

def sim_episode(env, drill, policy_red, policy_blue, gamma=0.99):
    env.set_setting(drill)
    policy_red.set_setting(drill)
    obs, _ = env.reset()

    logps_red, rewards_red, reports_red = [], [], []
    logps_blue, rewards_blue, reports_blue = [], [], []

    done = False

    while not done:
        a_red  = policy_red.sample_action(obs["player_red"])
        a_blue = policy_blue.sample_action(obs["player_blue"])

        next_obs, rewards, terminated, truncated, info = env.step(
            {"player_red": a_red, "player_blue": a_blue}
        )
        done = terminated["__all__"] or truncated["__all__"]

        obs_tensor_red = torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)
        logits_red = policy_red.forward(obs_tensor_red)
        logp_r = 0
        for key in ["left", "right", "jump"]:
            dist = torch.distributions.Categorical(logits=logits_red[key])
            logp_r += dist.log_prob(torch.tensor(a_red[key]))
        logps_red.append(logp_r)

        rewards_red.append(rewards["player_red"])
        rewards_blue.append(rewards["player_blue"])

        reports_red.append(info["reports"]["player_red"])
        reports_blue.append(info["reports"]["player_blue"])

        obs = next_obs

    returns_red = compute_returns(rewards_red, gamma=gamma)
    loss_red = -(torch.stack(logps_red) * returns_red).mean()

    return loss_red, (np.array(rewards_red), np.array(rewards_blue)), (np.array(reports_red), np.array(reports_blue))

def train_drill(name, policy: tuple[str, dict] | str, select_drill, num_episodes=2000, lr=1e-4, gamma=0.99, pool_size=20, print_episodes=5, save_episodes=50):
    opponent_pool = []

    env = FootballMultiAgentEnv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoints_path = os.path.join(parent_dir, "checkpoints")
    checkpoint_dir = os.path.join(checkpoints_path, name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    if isinstance(policy, tuple):
        policy_name, policy_kwargs = policy
        policy_red = make_policy(policy_name, **policy_kwargs)
    else:
        policy_red, checkpoint = policy_from_checkpoint_path(policy)
        policy_kwargs = checkpoint["policy_kwargs"]
    policy_blue = None

    opt_red = optim.Adam(policy_red.parameters(), lr=lr)

    def select_opponent():
        # if random.random() < 1/(len(opponent_pool)+1):
        #     return policy_red
        # else:
        #     opponent_path = random.choice(opponent_pool)
        #     checkpoint = torch.load(opponent_path)
        #     policy = make_policy(checkpoint['policy_class'], **checkpoint['policy_kwargs'])
        #     if checkpoint['policy_state_dict']:
        #         policy.load_state_dict(checkpoint['policy_state_dict'])
        #     return policy
        return DummyPolicy()

    for episode in range(num_episodes):
        policy_blue = select_opponent()
        drill = select_drill()

        loss_red, (rewards_red, rewards_blue), (reports_red, reports_blue) = sim_episode(env, drill, policy_red, policy_blue, gamma=gamma)

        opt_red.zero_grad()
        loss_red.backward()
        opt_red.step()

        if episode % print_episodes == 0:
            print(f"Episode {episode}: loss red {loss_red:.1f} rewards red {np.sum(rewards_red):.1f}, blue {np.sum(rewards_blue):.1f}, score red {np.sum(reports_red[:, 0]):.1f}, blue {np.sum(reports_blue[:, 0]):.1f}, move red {np.sum(reports_red[:, 1]):.1f}, blue {np.sum(reports_blue[:, 1]):.1f}, kick red {np.sum(reports_red[:, 2]):.1f}, blue {np.sum(reports_blue[:, 2]):.1f}, jump red {np.sum(reports_red[:, 3]):.1f}, blue {np.sum(reports_blue[:, 3]):.1f}, dist red {np.sum(reports_red[:, 4]):.1f}, blue {np.sum(reports_blue[:, 4]):.1f}")

        if (episode + 1) % save_episodes == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"football_episode_{episode+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'episode': episode,
                'policy_class': policy_red.__class__.__name__,
                'policy_kwargs': policy_kwargs,
                'policy_state_dict': policy_red.state_dict(),
                'optimizer_state_dict': opt_red.state_dict(),
            }, checkpoint_path)
            opponent_pool.append(checkpoint_path)
            if len(opponent_pool) == pool_size: opponent_pool.pop(0)

def train_league(name, policy: tuple[str, dict] | str, num_episodes=2000, lr=1e-4, gamma=0.99, pool_size=20, print_episodes=5, save_episodes=50):
    opponent_pool = []

    env = FootballMultiAgentEnv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoints_path = os.path.join(parent_dir, "checkpoints")
    checkpoint_dir = os.path.join(checkpoints_path, name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    if isinstance(policy, tuple):
        policy_name, policy_kwargs = policy
        policy_red = make_policy(policy_name, **policy_kwargs)
    else:
        policy_red, checkpoint = policy_from_checkpoint_path(policy)
        policy_kwargs = checkpoint["policy_kwargs"]
    policy_blue = None

    opt_red = optim.Adam(policy_red.parameters(), lr=lr)

    def select_opponent():
        # if random.random() < 1/(len(opponent_pool)+1):
        #     return policy_red
        # else:
        #     opponent_path = random.choice(opponent_pool)
        #     checkpoint = torch.load(opponent_path)
        #     policy = make_policy(checkpoint['policy_class'], **checkpoint['policy_kwargs'])
        #     if checkpoint['policy_state_dict']:
        #         policy.load_state_dict(checkpoint['policy_state_dict'])
        #     return policy
        return atulPolicy()

    for episode in range(num_episodes):
        policy_blue = select_opponent()

        loss_red, (rewards_red, rewards_blue), (reports_red, reports_blue) = sim_episode(env, policy_red, policy_blue, gamma=gamma)

        opt_red.zero_grad()
        loss_red.backward()
        opt_red.step()

        if episode % print_episodes == 0:
            print(f"Episode {episode}: loss red {loss_red:.1f} rewards red {np.sum(rewards_red):.1f}, blue {np.sum(rewards_blue):.1f}, score red {np.sum(reports_red[:, 0]):.1f}, blue {np.sum(reports_blue[:, 0]):.1f}, move red {np.sum(reports_red[:, 1]):.1f}, blue {np.sum(reports_blue[:, 1]):.1f}, kick red {np.sum(reports_red[:, 2]):.1f}, blue {np.sum(reports_blue[:, 2]):.1f}, jump red {np.sum(reports_red[:, 3]):.1f}, blue {np.sum(reports_blue[:, 3]):.1f}, dist red {np.sum(reports_red[:, 4]):.1f}, blue {np.sum(reports_blue[:, 4]):.1f}")

        if (episode + 1) % save_episodes == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"football_episode_{episode+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'episode': episode,
                'policy_class': policy_red.__class__.__name__,
                'policy_kwargs': policy_kwargs,
                'policy_state_dict': policy_red.state_dict(),
                'optimizer_state_dict': opt_red.state_dict(),
            }, checkpoint_path)
            opponent_pool.append(checkpoint_path)
            if len(opponent_pool) == pool_size: opponent_pool.pop(0)

def evaluate_from_checkpoint(checkpoint_path1, checkpoint_path2, episodes=10, render=False):
    """
    Loads policies from a checkpoint file and evaluates them.
    """
    assert os.path.exists(checkpoint_path1), f"Error: Checkpoint file not found at {checkpoint_path1}"


    # Step 1: Instantiate new policy models
    policy_red = MLPPolicy()
    policy_blue = MLPPolicy()

    if not isinstance(checkpoint_path2, FootballPolicy):
        assert os.path.exists(checkpoint_path2), f"Error: Checkpoint file not found at {checkpoint_path2}"
        checkpoint2 = torch.load(checkpoint_path2)
        policy_blue.load_state_dict(checkpoint2['policy_state_dict'])
    else:
        policy_blue = checkpoint_path2

    # Step 2: Load the checkpoint dictionary
    print(f"Loading checkpoint from {checkpoint_path1} and {checkpoint_path2}...")
    checkpoint = torch.load(checkpoint_path1)

    # Step 3: Load the saved weights into the models
    policy_red.load_state_dict(checkpoint['policy_state_dict'])

    # Step 4: Set the models to evaluation mode
    policy_red.eval()
    policy_blue.eval()

    env = FootballMultiAgentEnv({"render_mode": "human" if render else None})
    clock = pygame.time.Clock()
    scores = []
    for ep in range(episodes):
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
    # train_league(
    #     name="red_league_test3",
    #     policy="../checkpoints/red_league_test2/football_episode_38000.pth",
    #     num_episodes=100000, save_episodes=500, print_episodes=100
    #     )
    select_drill = lambda: {"drill": "shoot_left", "par": random.uniform(-1, -40/150)}
    train_drill(
        name="shoot_left_drill",
        policy=("CurriculumMLPPolicy", {}),
        select_drill=select_drill,
        num_episodes=10000000, save_episodes=500, print_episodes=50
    )
