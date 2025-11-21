import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# Game Canvas
BASE_WIDTH, BASE_HEIGHT = 480, 360

# --- Game Constants (in base coordinates) ---
GROUND_Y = -151
CEILING_Y = 150
WALL_X = 230
TICK_RATE = 30

# --- Colors ---
COLOR_SKY = (204, 255, 255)
COLOR_GRASS = (0, 153, 51)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 51, 255)

from pen_football import *

def sample():
    while True:
        empty = {"left":0,"right":0,"jump":0}
        game = FootballGame()
        pos_x = (random.random()-0.5)*2/230
        pos_y = min((random.random()-0.5)*2*151,150)/150
        vel_x = (random.random()-1)*30/20
        vel_y = (random.random()-0.5)*2*10/20
        game.preset([0,0,0,0,0,0,0,0,pos_x,pos_y,vel_x,vel_y])
        for _ in range(90):
            _, (red_state, blue_state, game_state), _, _, _,  = game.step(empty,empty)
            
            if blue_state['scored']:
                #print("YAY")
                return [pos_x,pos_y,vel_x,vel_y]
            if game_state['ball_hit_ground'] or game_state['ball_hit_ceiling']:
                break
        #print("HM")
        



def generate_samples(n, filename="samples_nobounce.npy"):
    # allocate array: each sample is [pos_x, pos_y, vel_x, vel_y]
    data = np.zeros((n, 4), dtype=float)

    for i in range(n):
        if i%50==0:
            print(i,"DONE")
        data[i] = sample()   # your function returns a list of 4 numbers

    np.save(filename, data)
    print(f"Saved {n} samples to {filename}")


if __name__ == "__main__":
    # pygame.init()
    # screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    # pygame.display.set_caption("Pen Football - Two Player")
    # clock = pygame.time.Clock()
    # game = FootballGame(screen)
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT: running = False
    #         if event.type == pygame.VIDEORESIZE:
    #             new_scale = event.w / BASE_WIDTH
    #             new_width = BASE_WIDTH * new_scale
    #             new_height = BASE_HEIGHT * new_scale
    #             screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
    #             game.scale, game.width, game.height = new_scale, new_width, new_height
    #     keys = pygame.key.get_pressed()
    #     red_keys = { 'jump': keys[pygame.K_w], 'left': keys[pygame.K_a], 'right': keys[pygame.K_d] }
    #     blue_keys = { 'jump': keys[pygame.K_UP], 'left': keys[pygame.K_LEFT], 'right': keys[pygame.K_RIGHT] }
    #     _, _, terminated, _, _ = game.step(red_keys, blue_keys)
    #     if terminated:
    #         print(f"Game Over! Final Score: Red {game.score_red} - Blue {game.score_blue}")
    #         game.reset()
    #     game.render()
    #     clock.tick(TICK_RATE)
    # pygame.quit()
    generate_samples(1000)