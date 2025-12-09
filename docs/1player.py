import numpy as np
import pygame
import os
import onnxruntime # Pygbag will install this via requirements.txt
import random

# Import your core game logic
from pen_football import * # --- ONNX and Policy Configuration ---
ONNX_MODEL_PATH = "league_ppo (misc rewards) checkpoint_7100000.onnx"
# If the file is bundled correctly, Pygbag should resolve the path.

# Policy Settings (MUST match the setting used during training for this checkpoint)
# You need to determine the correct index/par value for your specific checkpoint.
# We will use a typical placeholder setting for demonstration.
# Reference your policy.py for the full drill_to_index map:
# "block_nobounce": 1, "shoot_left": 2, etc.
DRILL_TO_INDEX = {
    "block_nobounce": 1,
    "shoot_left": 2,
}

FIXED_DRILL_NAME = "shoot_left" # CHANGE THIS if you know the actual trained setting
FIXED_PAR_VALUE = 0.1           # CHANGE THIS if you know the actual trained setting
FIXED_DRILL_INDEX = DRILL_TO_INDEX.get(FIXED_DRILL_NAME, 0)


# --- Global ONNX Session and Inputs/Outputs Setup ---
try:
    # Load the ONNX Worker Model once
    sess = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

    # Get Input/Output names from the ONNX model
    INPUT_NAME_OBS = sess.get_inputs()[0].name # 'obs'
    INPUT_NAME_IDX = sess.get_inputs()[1].name # 'drill_idx'
    INPUT_NAME_PAR = sess.get_inputs()[2].name # 'par_val'
    OUTPUT_NAMES = [output.name for output in sess.get_outputs()]

except Exception as e:
    print(f"Error loading ONNX model at {ONNX_MODEL_PATH}: {e}")
    # Fallback/stub session to allow Pygame to start, but AI will do nothing
    sess = None

# Reshaped fixed inputs for ONNX inference
# The shapes must exactly match the ONNX model inputs: (1, 12), (1,), (1, 1)
IDX_INPUT = np.array([FIXED_DRILL_INDEX], dtype=np.int64)
PAR_INPUT = np.array([[FIXED_PAR_VALUE]], dtype=np.float32)


def get_action_from_onnx(obs_array):
    """Runs inference on the ONNX model to get action logits and samples an action."""

    if sess is None:
        # Fallback: return no-op action if model failed to load
        return {"left": 0, "right": 0, "jump": 0}

    # 1. Reshape Observation Input
    obs_input = obs_array.reshape(1, 12).astype(np.float32)

    # 2. Run Inference
    result = sess.run(
        OUTPUT_NAMES,
        {
            INPUT_NAME_OBS: obs_input,
            INPUT_NAME_IDX: IDX_INPUT,
            INPUT_NAME_PAR: PAR_INPUT,
        }
    )

    # 3. Extract Logits (index 0 for left, 1 for right, 2 for jump)
    left_logits, right_logits, jump_logits, _ = result

    # 4. Convert Logits to Actions (Sampling) using NumPy
    def sample_action(logits):
        # Softmax: exp(logits) / sum(exp(logits))
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        # Sample action index (0 or 1)
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
    # Indices: Red [0:4], Blue [4:8], Ball [8:12]

    red_state = game_obs[0:4]
    blue_state = game_obs[4:8]
    ball_state = game_obs[8:12]

    # Mirroring transformation: Flip X-position and X-velocity
    flip = np.array([-1, 1, -1, 1], dtype=np.float32)
    blue_state_sym = blue_state * flip
    red_state_sym = red_state * flip
    ball_state_sym = ball_state * flip

    # Blue's observation is its own state + mirrored opponent + mirrored ball
    # The ONNX model expects the player being controlled to be first:
    obs_blue = np.concatenate([blue_state_sym, red_state_sym, ball_state_sym]).astype(np.float32)

    return obs_blue


def play_one_player_onnx():
    """
    Main loop for one-player game using ONNX inference.
    """
    pygame.init()
    screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pen Football - One Player (ONNX AI)")
    clock = pygame.time.Clock()
    game = FootballGame(screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.VIDEORESIZE:
                # Handle window resizing for Pygbag
                new_scale = event.w / BASE_WIDTH
                new_width = BASE_WIDTH * new_scale
                new_height = BASE_HEIGHT * new_scale
                screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                game.scale, game.width, game.height = new_scale, new_width, new_height

        keys = pygame.key.get_pressed()

        # Human Player (Red) Keys
        red_keys = {
            'jump': keys[pygame.K_w] or keys[pygame.K_UP],
            'left': keys[pygame.K_a] or keys[pygame.K_LEFT],
            'right': keys[pygame.K_d] or keys[pygame.K_RIGHT]
            }

        # AI Player (Blue) Action
        obs_blue_array = _get_obs(game)
        a_blue = get_action_from_onnx(obs_blue_array)

        # Convert AI action back to game keys
        # Note: Your original code swaps left/right for the mirrored AI:
        # "left": a_blue["right"] == 1,
        # "right": a_blue["left"] == 1,
        blue_keys = {
            "left": a_blue["right"] == 1, # Action 'right' in mirrored obs is 'left' in global frame
            "right": a_blue["left"] == 1, # Action 'left' in mirrored obs is 'right' in global frame
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
    # Call the new ONNX-based function instead of the old policy loader
    play_one_player_onnx()
