import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from multiagent import FootballMultiAgentEnv
import os
import time
import pygame

class FootballPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(128, 2)
        self.head_right = nn.Linear(128, 2)
        self.head_jump  = nn.Linear(128, 2)

    def forward(self, obs):
        x = self.net(obs)
        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
        }

    def sample_action(self, obs):
        """Sample action from the policy given a numpy observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs_t)
        return {
            k: torch.distributions.Categorical(logits=v).sample().item()
            for k, v in logits.items()
        }

def train(name, num_episodes=2000, lr=1e-4, gamma=0.99):
    env = FootballMultiAgentEnv()

    policy_red = FootballPolicy()
    policy_blue = FootballPolicy()

    opt_red = optim.Adam(policy_red.parameters(), lr=lr)
    opt_blue = optim.Adam(policy_blue.parameters(), lr=lr)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoints_path = os.path.join(parent_dir, "checkpoints")
    checkpoint_dir = os.path.join(checkpoints_path, name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    for episode in range(num_episodes):
        obs, _ = env.reset()

        # Trajectory buffers
        logps_red, rewards_red = [], []
        logps_blue, rewards_blue = [], []

        done = False

        while not done:
            a_red  = policy_red.sample_action(obs["player_red"])
            a_blue = policy_blue.sample_action(obs["player_blue"])

            next_obs, rewards, terminated, truncated, _ = env.step(
                {"player_red": a_red, "player_blue": a_blue}
            )
            done = terminated["__all__"] or truncated["__all__"]

            # --- RED ---
            obs_tensor_red = torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)
            logits_red = policy_red.forward(obs_tensor_red)
            logp_r = 0
            for key in ["left", "right", "jump"]:
                dist = torch.distributions.Categorical(logits=logits_red[key])
                logp_r += dist.log_prob(torch.tensor(a_red[key]))
            logps_red.append(logp_r)
            rewards_red.append(rewards["player_red"])

            # --- BLUE ---
            obs_tensor_blue = torch.tensor(obs["player_blue"], dtype=torch.float32).unsqueeze(0)
            logits_blue = policy_blue.forward(obs_tensor_blue)
            logp_b = 0
            for key in ["left", "right", "jump"]:
                dist = torch.distributions.Categorical(logits=logits_blue[key])
                logp_b += dist.log_prob(torch.tensor(a_blue[key]))
            logps_blue.append(logp_b)
            rewards_blue.append(rewards["player_blue"])

            obs = next_obs

        # ---- Update policies using REINFORCE ----

        def compute_returns(rewards_list):
            G = np.zeros(4)
            returns = []
            for r in reversed(rewards_list):
                G = np.array(r)[[0, 1, 2, 3]] + gamma * G
                returns.append(G)
            returns.reverse()
            returns = np.stack(returns)
            return torch.tensor(returns, dtype=torch.float32)

        # RED
        returns_red = compute_returns(rewards_red)
        loss_red = -(torch.stack(logps_red) * returns_red).mean()
        opt_red.zero_grad()
        loss_red.backward()
        opt_red.step()

        # BLUE
        returns_blue = compute_returns(rewards_blue)
        loss_blue = -(torch.stack(logps_blue) * returns_blue).mean()
        opt_blue.zero_grad()
        loss_blue.backward()
        opt_blue.step()

        rewards_red = np.array(rewards_red).T
        rewards_blue = np.array(rewards_blue).T
        # print(rewards_red.shape)

        if episode % 5 == 0:
            print(f"Episode {episode}: rewards red {np.sum(rewards_red):.1f}, blue {np.sum(rewards_blue):.1f}, score red {np.sum(rewards_red[0, :]):.1f}, blue {np.sum(rewards_blue[0, :]):.1f}, move red {np.sum(rewards_red[1, :]):.1f}, blue {np.sum(rewards_blue[1, :]):.1f}, kick red {np.sum(rewards_red[2, :]):.1f}, blue {np.sum(rewards_blue[2, :]):.1f}, jump red {np.sum(rewards_red[3, :]):.1f}, blue {np.sum(rewards_blue[3, :]):.1f}")

        if (episode + 1) % 300 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"football_episode_{episode+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'episode': episode,
                'policy_red_state_dict': policy_red.state_dict(),
                'policy_blue_state_dict': policy_blue.state_dict(),
                'optimizer_red_state_dict': opt_red.state_dict(),
                'optimizer_blue_state_dict': opt_blue.state_dict(),
            }, checkpoint_path)

# def evaluate(policy_red, policy_blue, episodes=10, render=False):
#     env = FootballMultiAgentEnv({"render_mode": "human" if render else None})

#     scores = []
#     for ep in range(episodes):
#         obs, _ = env.reset()
#         done = False
#         total_reward = 0

#         while not done:
#             # Deterministic actions (greedy)
#             a_red = {
#                 k: torch.argmax(v).item()
#                 for k, v in policy_red.forward(torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)).items()
#             }
#             a_blue = {
#                 k: torch.argmax(v).item()
#                 for k, v in policy_blue.forward(torch.tensor(obs["player_blue"], dtype=torch.float32).unsqueeze(0)).items()
#             }

#             obs, rewards, terminated, truncated, _ = env.step(
#                 {"player_red": a_red, "player_blue": a_blue}
#             )
#             done = terminated["__all__"] or truncated["__all__"]
#             total_reward += rewards["player_red"]

#             if render:
#                 env.render()

#         print(f"Episode {ep} final reward for red: {total_reward:.1f}")
#         scores.append(total_reward)

#     env.close()
#     return scores

def evaluate_from_checkpoint(checkpoint_path, episodes=10, render=False):
    """
    Loads policies from a checkpoint file and evaluates them.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    # Step 1: Instantiate new policy models
    policy_red = FootballPolicy()
    policy_blue = FootballPolicy()

    # Step 2: Load the checkpoint dictionary
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)

    # Step 3: Load the saved weights into the models
    policy_red.load_state_dict(checkpoint['policy_red_state_dict'])
    policy_blue.load_state_dict(checkpoint['policy_blue_state_dict'])

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
            # Use torch.no_grad() for faster inference and to prevent gradient tracking
            with torch.no_grad():
                a_red = {
                    # k: torch.argmax(v).item()
                    k: torch.distributions.Categorical(logits=v).sample().item()
                    for k, v in policy_red.forward(torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)).items()
                }
                a_blue = {
                    # k: torch.argmax(v).item()
                    k: torch.distributions.Categorical(logits=v).sample().item()
                    for k, v in policy_blue.forward(torch.tensor(obs["player_blue"], dtype=torch.float32).unsqueeze(0)).items()
                }

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

# ---- Run training and evaluation ----
if __name__ == "__main__":
    train(name=None, num_episodes=10000)
# evaluate(policy_red, policy_blue, episodes=5, render=True)
