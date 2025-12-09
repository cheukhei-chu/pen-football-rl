import pygame
import numpy as np
import json
import os

# Import game logic
from pen_football import FootballGame, BASE_WIDTH, BASE_HEIGHT, TICK_RATE

# --- CONFIGURATION ---
WEIGHTS_FILE = "model_weights.json"
W = {}

def load_weights():
    global W
    if not os.path.exists(WEIGHTS_FILE):
        print(f"ERROR: {WEIGHTS_FILE} not found! AI will be disabled.")
        return
    with open(WEIGHTS_FILE, "r") as f:
        W = json.load(f)
    print("Weights loaded successfully.")

# --- MANUAL NEURAL NETWORK MATH ---
def relu(x):
    return np.maximum(0, x)

def linear(x, name):
    # Implements: y = x * W^T + b
    if f"{name}_w" not in W: return np.zeros(1)

    w = np.array(W[f"{name}_w"])
    b = np.array(W[f"{name}_b"])

    # Dot product
    return np.dot(x, w.T) + b

def get_action_numpy(obs):
    # If weights failed to load, do nothing
    if not W: return {"left": 0, "right": 0, "jump": 0}

    # 1. Prepare Input (Shape: 1, 12)
    obs = obs.reshape(1, -1)

    # 2. Plan Net Forward Pass
    #    Layer 0 (Linear) -> ReLU -> Layer 2 (Linear) -> ReLU
    p = linear(obs, "plan_0")
    p = relu(p)
    p = linear(p, "plan_2")
    p = relu(p) # This is the "Task Embedding"

    # 3. Action Net Forward Pass
    #    Input = Subset of Obs + Task Embedding
    #    Indices: [0, 1, 2, 3, 8, 9, 10, 11] (Red + Blue + Ball, skipping velocities?)
    obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]

    # Concatenate along the last axis
    x = np.concatenate([obs_subset, p], axis=-1)

    #    Layer 0 (Linear) -> ReLU -> Layer 2 (Linear) -> ReLU
    x = linear(x, "action_0")
    x = relu(x)
    x = linear(x, "action_2")
    x = relu(x)

    # 4. Heads (Output Logits)
    left_logits = linear(x, "head_left").flatten()
    right_logits = linear(x, "head_right").flatten()
    jump_logits = linear(x, "head_jump").flatten()

    # 5. Sampling (Softmax -> Random Choice)
    def sample(logits):
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return np.random.choice([0, 1], p=probs)

    return {
        "left": sample(left_logits),
        "right": sample(right_logits),
        "jump": sample(jump_logits)
    }

def _get_obs(game):
    # Get standard observation
    game_obs = game._get_internal_observation()

    red_state = game_obs[0:4]
    blue_state = game_obs[4:8]
    ball_state = game_obs[8:12]

    # Mirror for Blue (AI) perspective
    flip = np.array([-1, 1, -1, 1], dtype=np.float32)

    # Concatenate: Blue(Mirrored), Red(Mirrored), Ball(Mirrored)
    obs_blue = np.concatenate([
        blue_state * flip,
        red_state * flip,
        ball_state * flip
    ]).astype(np.float32)

    return obs_blue

# --- MAIN GAME LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pen Football - NumPy AI")
    clock = pygame.time.Clock()
    game = FootballGame(screen)

    # Load the weights before starting
    load_weights()

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
        red_keys = {
            'jump': keys[pygame.K_w] or keys[pygame.K_UP],
            'left': keys[pygame.K_a] or keys[pygame.K_LEFT],
            'right': keys[pygame.K_d] or keys[pygame.K_RIGHT]
        }

        # AI Inference
        obs_blue = _get_obs(game)
        a_blue = get_action_numpy(obs_blue)

        # Convert AI action to Keys
        # Note: Swap left/right because AI sees mirrored world
        blue_keys = {
            "left": a_blue["right"] == 1,
            "right": a_blue["left"] == 1,
            "jump": a_blue["jump"] == 1
        }

        _, _, terminated, _, _ = game.step(red_keys, blue_keys)

        if terminated:
            print(f"Game Over! Score: {game.score_red} - {game.score_blue}")
            game.reset()

        game.render()
        clock.tick(TICK_RATE)

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Simple error printing to console if something still breaks
        import traceback
        traceback.print_exc()
