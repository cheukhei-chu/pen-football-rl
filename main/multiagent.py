import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from pen_football import *

class FootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        render_mode = config.get("render_mode") if config else None
        self.game = FootballGame(render_mode=render_mode)
        self.possible_agents = ["player_red", "player_blue"]
        self.agents = self.possible_agents[:]

        # --- FIX LINGERING DTYPE WARNINGS ---
        # Define bounds as float32 to prevent casting warnings

        low_bound = np.float32(-1.5)
        high_bound = np.float32(1.5)

        single_agent_action_space = spaces.Dict({
            "left": spaces.Discrete(2),  # 0 for not pressed, 1 for pressed
            "right": spaces.Discrete(2),
            "jump": spaces.Discrete(2),
        })

        # Now, create the full multi-agent action space dictionary.
        self.action_space = spaces.Dict({
            agent_id: single_agent_action_space for agent_id in self.possible_agents
        })
        self.observation_space = spaces.Dict(
            {
                agent_id: spaces.Box(low=low_bound, high=high_bound, shape=(12,), dtype=np.float32)
                for agent_id in self.possible_agents
            }
        )
        # x \in (-1, 1), y \in (-151/150, 1), vx \in (-0.5, 0.5), vy \in (-0.5, 0.6)
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

    def _get_obs_dict(self):
        # This function can now be simplified as the game returns the correct dtype
        game_obs = self.game._get_internal_observation()
        red_state = game_obs[0:4]
        blue_state = game_obs[4:8]
        ball_state = game_obs[8:12]

        blue_state_symmetric = blue_state * np.array([-1, 1, -1, 1], dtype=np.float32)
        red_state_symmetric_for_blue = red_state * np.array([-1, 1, -1, 1], dtype=np.float32)
        ball_state_symmetric_for_blue = ball_state * np.array([-1, 1, -1, 1], dtype=np.float32)

        obs_red = np.concatenate([red_state, blue_state, ball_state]).astype(np.float32)
        obs_blue = np.concatenate([
            blue_state_symmetric,
            red_state_symmetric_for_blue,
            ball_state_symmetric_for_blue
        ]).astype(np.float32)

        return {"player_red": obs_red, "player_blue": obs_blue}

    def reset(self, *, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.game.reset()
        return self._get_obs_dict(), {}

    def step(self, action_dict):
        action_red = action_dict["player_red"]
        action_blue = action_dict["player_blue"]
        red_keys = {'left': action_red["left"] == 1, 'right': action_red["right"] == 1, 'jump': action_red["jump"] == 1}
        blue_keys = {'left': action_blue["right"] == 1, 'right': action_blue["left"] == 1, 'jump': action_blue["jump"] == 1}

        _, (red_kicked, blue_kicked), terminated, truncated, _ = self.game.step(red_keys, blue_keys)

        red_x, red_y = self.game.red['x'], self.game.red['y']
        blue_x, blue_y = self.game.blue['x'], self.game.blue['y']
        ball_x, ball_y = self.game.ball['x'], self.game.ball['y']

        reward_red, reward_blue = 0, 0

        if ball_y < -40:
            if ball_x > 210:
                reward_red += 100
                reward_blue -= 100
            elif ball_x < -210:
                reward_red -= 100
                reward_blue += 100

        # # Give a small reward for moving horizontally in the correct direction
        # if action_red["right"] == 1 and ball_x > red_x:
        #     reward_red += 10
        # if action_red["left"] == 1 and ball_x < red_x:
        #     reward_red += 10
        # reward_red -= abs(ball_y - red_y)

        # if action_blue["right"] == 1 and ball_x < blue_x:
        #     reward_blue += 10
        # if action_blue["left"] == 1 and ball_x > blue_x:
        #     reward_blue += 10
        # reward_blue -= abs(ball_y - blue_y)

        # reward_red += red_kicked * 10
        # reward_blue += blue_kicked * 10

        reward_red -= action_red["jump"] * 1000
        reward_blue -= action_blue["jump"] * 1000

        observations = self._get_obs_dict()
        rewards = {"player_red": float(reward_red), "player_blue": float(reward_blue)}
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": truncated}
        return observations, rewards, terminateds, truncateds, {}

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
