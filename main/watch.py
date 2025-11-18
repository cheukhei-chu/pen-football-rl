from train import evaluate_from_checkpoint
from policy import *

evaluate_from_checkpoint(
    '../checkpoints/red_league_test2/football_episode_18500.pth',
    atulPolicy(),
    episodes=5, render=True)
