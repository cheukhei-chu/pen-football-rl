import sys
import traceback
import pygame

# --- DEBUG WRAPPER START ---
def main():
    try:
        # --- YOUR ORIGINAL GAME CODE STARTS HERE ---
        import numpy as np
        import os
        import random

        # 1. Try to import ONNX (Common crash point)
        print("Attempting to import onnxruntime...")
        import onnxruntime
        print("onnxruntime imported successfully.")

        # Import your core game logic
        from pen_football import FootballGame, BASE_WIDTH, BASE_HEIGHT, TICK_RATE

        # --- ONNX Configuration ---
        ONNX_MODEL_PATH = "model.onnx"

        # 2. Try to load the model (Common crash point)
        print(f"Attempting to load model: {ONNX_MODEL_PATH}")
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(f"File not found: {ONNX_MODEL_PATH}")

        sess = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
        INPUT_NAME_OBS = sess.get_inputs()[0].name
        OUTPUT_NAMES = [output.name for output in sess.get_outputs()]
        print("Model loaded successfully.")

        def get_action_from_onnx(obs_array):
            if sess is None: return {"left": 0, "right": 0, "jump": 0}
            obs_input = obs_array.reshape(1, 12).astype(np.float32)
            result = sess.run(OUTPUT_NAMES, {INPUT_NAME_OBS: obs_input})
            left_logits, right_logits, jump_logits, _ = result

            def sample_action(logits):
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                return np.random.choice([0, 1], p=probs.flatten())

            return {
                "left": sample_action(left_logits),
                "right": sample_action(right_logits),
                "jump": sample_action(jump_logits)
            }

        def _get_obs(game):
            game_obs = game._get_internal_observation()
            red_state = game_obs[0:4]
            blue_state = game_obs[4:8]
            ball_state = game_obs[8:12]
            flip = np.array([-1, 1, -1, 1], dtype=np.float32)
            obs_blue = np.concatenate([blue_state * flip, red_state * flip, ball_state * flip]).astype(np.float32)
            return obs_blue

        # --- Game Loop ---
        pygame.init()
        screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Pen Football - Debug Mode")
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
            red_keys = {'jump': keys[pygame.K_w] or keys[pygame.K_UP], 'left': keys[pygame.K_a] or keys[pygame.K_LEFT], 'right': keys[pygame.K_d] or keys[pygame.K_RIGHT]}

            # AI Step
            obs_blue_array = _get_obs(game)
            a_blue = get_action_from_onnx(obs_blue_array)
            blue_keys = {"left": a_blue["right"] == 1, "right": a_blue["left"] == 1, "jump": a_blue["jump"] == 1}

            _, _, terminated, _, _ = game.step(red_keys, blue_keys)

            if terminated:
                game.reset()

            game.render()
            clock.tick(TICK_RATE)

        pygame.quit()

    except Exception as e:
        # --- ERROR HANDLING SCREEN ---
        # This catches the crash and prints it to the window
        pygame.init()
        screen = pygame.display.set_mode((640, 480))
        screen.fill((0, 0, 100)) # Dark blue background
        font = pygame.font.SysFont("arial", 20)

        # Format the traceback
        error_lines = traceback.format_exc().splitlines()

        y = 10
        for line in error_lines:
            # Render each line of the error
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, y))
            y += 25
            
        pygame.display.flip()

        # Keep the error screen open
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return

if __name__ == "__main__":
    main()
