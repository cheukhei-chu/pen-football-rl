import sys
import traceback

# ---------------------------------------------------------
# GLOBAL ERROR HANDLER
# This runs if anything crashes, even imports.
# ---------------------------------------------------------
def show_error_screen(error_msg):
    # We have to import pygame here strictly for the error screen
    # If pygame itself is broken, we are out of luck, but that's rare.
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    screen.fill((50, 0, 0)) # Dark Red Background

    font = pygame.font.SysFont("monospace", 14)
    y = 10

    lines = error_msg.splitlines()
    for line in lines:
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (10, y))
        y += 20

    pygame.display.flip()

    # Wait loop so you can read the error
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return

def main():
    try:
        # --- 1. MOVED IMPORTS INSIDE TRY BLOCK ---
        # If any of these fail, we will now catch it!
        import pygame
        import numpy as np
        import json
        import os

        # Check if pen_football exists before importing
        if not os.path.exists("pen_football.py"):
            print("WARNING: pen_football.py not found in current directory")

        from pen_football import FootballGame, BASE_WIDTH, BASE_HEIGHT, TICK_RATE

        # --- 2. CONFIGURATION ---
        # We will attempt to use the JSON weights you generated
        WEIGHTS_FILE = "model_weights.json"
        W = {}

        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, "r") as f:
                W = json.load(f)

        # --- 3. HELPER FUNCTIONS ---
        def linear(x, name):
            if f"{name}_w" not in W: return np.zeros(1)
            w = np.array(W[f"{name}_w"])
            b = np.array(W[f"{name}_b"])
            return np.dot(x, w.T) + b

        def relu(x):
            return np.maximum(0, x)

        def get_action(obs):
            if not W: return {"left": 0, "right": 0, "jump": 0}

            # Simple Forward Pass (No loops, just math)
            obs = obs.reshape(1, -1)

            # Plan Net
            p = relu(linear(relu(linear(obs, "plan_0")), "plan_2"))

            # Action Net
            obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]
            x = np.concatenate([obs_subset, p], axis=-1)
            x = relu(linear(relu(linear(x, "action_0")), "action_2"))

            # Heads
            l = linear(x, "head_left").flatten()
            r = linear(x, "head_right").flatten()
            j = linear(x, "head_jump").flatten()

            def sample(logits):
                e = np.exp(logits - np.max(logits))
                return np.random.choice([0, 1], p=e/np.sum(e))

            return {"left": sample(l), "right": sample(r), "jump": sample(j)}

        def get_obs_blue(game):
            o = game._get_internal_observation()
            flip = np.array([-1, 1, -1, 1], dtype=np.float32)
            return np.concatenate([o[4:8]*flip, o[0:4]*flip, o[8:12]*flip]).astype(np.float32)

        # --- 4. GAME LOOP (Standard Sync Loop) ---
        pygame.init()
        screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        game = FootballGame(screen)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                # Handle resize if needed
                if event.type == pygame.VIDEORESIZE:
                     screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            keys = pygame.key.get_pressed()
            red_keys = {'jump': keys[pygame.K_w] or keys[pygame.K_UP], 'left': keys[pygame.K_a] or keys[pygame.K_LEFT], 'right': keys[pygame.K_d] or keys[pygame.K_RIGHT]}

            # AI Logic
            blue_keys = {"left": False, "right": False, "jump": False}
            if W:
                a = get_action(get_obs_blue(game))
                blue_keys = {"left": a["right"]==1, "right": a["left"]==1, "jump": a["jump"]==1}

            game.step(red_keys, blue_keys)
            game.render()
            clock.tick(TICK_RATE)

    except Exception:
        # Catch ANY error and show it on the red screen
        error_msg = traceback.format_exc()
        print(error_msg) # Print to console too
        show_error_screen(error_msg)

if __name__ == "__main__":
    main()
