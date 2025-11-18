import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pen_football import *

class FootballMultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config=None):
        super().__init__()
        render_mode = config.get("render_mode") if config else None
        self.game = FootballGame(render_mode=render_mode)

        self.possible_agents = ["player_red", "player_blue"]

        # --- Spaces ---
        # Each agent has 3 discrete actions (left/right/jump)
        single_action = spaces.Dict({
            "left": spaces.Discrete(2),
            "right": spaces.Discrete(2),
            "jump": spaces.Discrete(2)
        })

        self.action_space = spaces.Dict({
            agent: single_action for agent in self.possible_agents
        })

        # Observations: 12 floats per agent
        low = np.float32(-1.5)
        high = np.float32(1.5)

        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=low, high=high, shape=(12,), dtype=np.float32)
            for agent in self.possible_agents
        })

    # -----------------------------------------------------

    def _get_obs(self):
        game_obs = self.game._get_internal_observation()
        red_state = game_obs[0:4]
        blue_state = game_obs[4:8]
        ball_state = game_obs[8:12]

        # mirror for blue
        flip = np.array([-1, 1, -1, 1], dtype=np.float32)
        blue_state_sym = blue_state * flip
        red_state_sym = red_state * flip
        ball_state_sym = ball_state * flip

        obs_red = np.concatenate([red_state, blue_state, ball_state]).astype(np.float32)
        obs_blue = np.concatenate([blue_state_sym, red_state_sym, ball_state_sym]).astype(np.float32)

        return {
            "player_red": obs_red,
            "player_blue": obs_blue
        }

    # -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self.game.reset()
        return self._get_obs(), {}

    # -----------------------------------------------------

    def step(self, action_dict):
        a_red = action_dict["player_red"]
        a_blue = action_dict["player_blue"]

        red_keys = {
            "left": a_red["left"] == 1,
            "right": a_red["right"] == 1,
            "jump": a_red["jump"] == 1
        }
        blue_keys = {
            "left": a_blue["right"] == 1,
            "right": a_blue["left"] == 1,
            "jump": a_blue["jump"] == 1
        }

        _, (red_state, blue_state), terminated, truncated, _ = self.game.step(red_keys, blue_keys)

        obs = self._get_obs()

        reports = self.comp_reports(red_state, blue_state)
        rewards = self.comp_rewards(red_state, blue_state)

        terminateds = {"__all__": red_state['scored'] or blue_state['scored']}
        truncateds = {"__all__": truncated}

        return obs, rewards, terminateds, truncateds, {"reports": reports}

    # -----------------------------------------------------

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()

    def comp_reports(self, red_state: dict, blue_state: dict):
        score_red = (red_state['scored'] - blue_state['scored']) * 1
        score_blue = (blue_state['scored'] - red_state['scored']) * 1

        move_red = (red_state['move_towards_ball']) * 1
        move_blue = (blue_state['move_towards_ball']) * 1

        kick_red = red_state['kicked'] * 1
        kick_blue = blue_state['kicked'] * 1

        jump_red = red_state['jump_failed'] * 1
        jump_blue = blue_state['jump_failed'] * 1

        reports = {
            "player_red": [float(score_red), float(move_red), float(kick_red), float(jump_red)],
            "player_blue": [float(score_blue), float(move_blue), float(kick_blue), float(jump_blue)]
        }
        return reports

    def comp_rewards(self, red_state: dict, blue_state: dict):
        score_red = (red_state['scored'] - blue_state['scored']) * 100
        score_blue = (blue_state['scored'] - red_state['scored']) * 100

        move_red = (red_state['move_towards_ball']) * 0.1
        move_blue = (blue_state['move_towards_ball']) * 0.1

        kick_red = red_state['kicked'] * 10
        kick_blue = blue_state['kicked'] * 10

        jump_red = red_state['jump_failed'] * (-1)
        jump_blue = blue_state['jump_failed'] * (-1)

        rewards = {
            "player_red": score_red + move_red + kick_red + jump_red,
            "player_blue": score_blue + move_blue + kick_blue + jump_blue
        }
        return rewards
