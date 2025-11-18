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

        # blue is mirrored
        blue_keys = {
            "left": a_blue["right"] == 1,
            "right": a_blue["left"] == 1,
            "jump": a_blue["jump"] == 1
        }

        _, (red_kicked, blue_kicked, red_scored, blue_scored, red_jump_failed, blue_jump_failed), terminated, truncated, _ = self.game.step(red_keys, blue_keys)

        red_x, red_y = self.game.red['x'], self.game.red['y']
        blue_x, blue_y = self.game.blue['x'], self.game.blue['y']
        ball_x, ball_y = self.game.ball['x'], self.game.ball['y']

        #reward_red = 0
        #reward_blue = 0

        score_red = (red_scored - blue_scored) * 100
        score_blue = (blue_scored - red_scored) * 100

        move_red = 0
        move_blue = 0

        kick_red = 0
        kick_blue = 0

        jump_red = red_jump_failed * (-1)
        jump_blue = blue_jump_failed * (-1)

        # # Scoring logic
        # if ball_y < -40:
        #     if ball_x > 210:
        #         #reward_red += 100
        #         score_red += 100
        #         #reward_blue -= 100
        #         score_blue -= 100
        #     elif ball_x < -210:
        #         #reward_red -= 100
        #         score_red -= 100
        #         #reward_blue += 100
        #         score_blue += 100

        # Movement reward shaping
        if a_red["right"] and ball_x > red_x:
            #reward_red += 10
            move_red += 0.1
        if a_red["left"] and ball_x < red_x:
            #reward_red += 10
            move_red += 0.1
        #reward_red -= abs(ball_y - red_y)

        if a_blue["right"] and ball_x < blue_x:
            #reward_blue += 10
            move_blue += 0.1
        if a_blue["left"] and ball_x > blue_x:
            #reward_blue += 10
            move_blue += 0.1
        #reward_blue -= abs(ball_y - blue_y)

        # Kicking bonuses
        kick_red += red_kicked * 10
        kick_blue += blue_kicked * 10

        # # Jump penalties
        # jump_red -= a_red["jump"]
        # jump_blue -= a_blue["jump"]

        obs = self._get_obs()

        rewards = {
            "player_red": [float(score_red),float(move_red),float(kick_red),float(jump_red)],
            "player_blue": [float(score_blue),float(move_blue),float(kick_blue),float(jump_blue)]
        }

        # terminateds = {"__all__": terminated}
        terminateds = {"__all__": red_scored or blue_scored}
        truncateds = {"__all__": truncated}

        return obs, rewards, terminateds, truncateds, {}

    # -----------------------------------------------------

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
