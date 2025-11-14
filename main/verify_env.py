import pygame
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from PenFootball.pen_football import FootballEnv # Assuming your file is named football_env.py

# --- Method 1: The Official Gymnasium Environment Checker ---
print("--- Running Gymnasium API Compliance Check ---")
check_env_instance = FootballEnv(render_mode=None)
try:
    check_env(check_env_instance.unwrapped)
    print("✅  SUCCESS: Environment passed the official check.")
except Exception as e:
    print(f"❌ FAILURE: Environment failed the official check: {e}")

# --- Method 2: The Random Agent Test ---
print("\n--- Running Random Agent Test (1000 steps) ---")
env = FootballEnv(render_mode='human')
obs, info = env.reset()
terminated, truncated = False, False
for i in range(1000):
    if terminated or truncated:
        obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
print("✅  SUCCESS: Random agent test completed without crashing.")

# --- Method 3: The Human Player Test ---
print("\n--- Running Human Player Test ---")
print("Controls: WAD for Red Player. Close the window to exit.")
obs, info = env.reset()
terminated, truncated = False, False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if terminated or truncated:
        obs, info = env.reset()

    # --- Manual Control Mapping for Red Player (MultiBinary format) ---
    keys = pygame.key.get_pressed()
    red_action = np.array([0, 0, 0]) # [left, right, jump]
    if keys[pygame.K_a]:
        red_action[0] = 1
    if keys[pygame.K_d]:
        red_action[1] = 1
    if keys[pygame.K_w]:
        red_action[2] = 1

    # Blue player does nothing
    blue_action = np.array([0, 0, 0])

    # Package actions into a tuple and step the environment
    action = (red_action, blue_action)
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

print("✅  SUCCESS: Human player test finished.")
env.close()
print("\n--- Verification Complete ---")
