import numpy as np
import pygame
import os
import onnxruntime
import random

# Import your core game logic
from pen_football import *

# --- ONNX Configuration ---
# Ensure this matches the filename inside your docs folder
ONNX_MODEL_PATH = "model.onnx"

# --- Global ONNX Session Setup ---
try:
    # Load the ONNX Worker Model
    sess = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

    # Get Input/Output names
    # Your model only has ONE input now ("obs") because you used plan_net
    INPUT_NAME_OBS = sess.get_inputs()[0].name
    OUTPUT_NAMES = [output.name for output in sess.get_outputs()]
    print(f"Model loaded successfully. Input: {INPUT_NAME_OBS}")

except Exception as e:
    print(f"Error loading ONNX model at {ONNX_MODEL_PATH}: {e}")
    sess = None


def get_action_from_onnx(obs_array):
    """Runs inference on the ONNX model to get action logits and samples an action."""

    if sess is None:
        # Fallback if model failed to load
        return {"left": 0, "right": 0, "jump": 0}

    # 1. Reshape Observation Input
    obs_input = obs_array.reshape(1, 12).astype(np.float32)

    # 2. Run Inference
    # We ONLY pass the observation now. The model calculates the plan internally.
    result = sess.run(
        OUTPUT_NAMES,
        {
            INPUT_NAME_OBS: obs_input
        }
    )

    # 3. Extract Logits
    # (Output order matches your export script: left, right, jump, value)
    left_logits, right_logits, jump_logits, _ = result

    # 4. Convert Logits to Actions (Sampling)
    def sample_action(logits):
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return np.random.choice([0, 1], p=probs.flatten())

    a_left = sample_action(left_logits)
    a_right = sample_action(right_logits)
    a_jump = sample_action(jump_logits)

    return {
        "left": a_left,
        "right": a_right,
        "jump": a_jump
    }

def _get_obs(game: FootballGame):
    """Calculates the mirrored observation required for the Blue Player (AI)."""
    game_obs = game._get_internal_observation()

    red_state = game_obs[0:4]
    blue_state = game_obs[4:8]
    ball_state = game_obs[8:12]

    # Mirroring transformation
    flip = np.array([-1, 1, -1, 1], dtype=np.float32)
    blue_state_sym = blue_state * flip
    red_state_sym = red_state * flip
    ball_state_sym = ball_state * flip

    obs_blue = np.concatenate([blue_state_sym, red_state_sym, ball_state_sym]).astype(np.float32)
    return obs_blue


def play_one_player_onnx():
    pygame.init()
    screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pen Football - One Player (Autonomous AI)")
    clock = pygame.time.Clock()
    game = FootballGame(screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.VIDEORESIZE:
                new_scale = event.w / BASE_WIDTH
                new_width = BASE_WIDTH * new_scale
                new_height = BASE_HEIGHT * new_scale
                screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                game.scale, game.width, game.height = new_scale, new_width, new_height

        keys = pygame.key.get_pressed()

        # Human (Red)
        red_keys = {
            'jump': keys[pygame.K_w] or keys[pygame.K_UP],
            'left': keys[pygame.K_a] or keys[pygame.K_LEFT],
            'right': keys[pygame.K_d] or keys[pygame.K_RIGHT]
            }

        # AI (Blue)
        obs_blue_array = _get_obs(game)
        a_blue = get_action_from_onnx(obs_blue_array)

        blue_keys = {
            "left": a_blue["right"] == 1,
            "right": a_blue["left"] == 1,
            "jump": a_blue["jump"] == 1
        }

        _, _, terminated, _, _ = game.step(red_keys, blue_keys)

        if terminated:
            print(f"Game Over! Final Score: Red {game.score_red} - Blue {game.score_blue}")
            game.reset()

        game.render()
        clock.tick(TICK_RATE)

    pygame.quit()


if __name__ == "__main__":
    play_one_player_onnx()
