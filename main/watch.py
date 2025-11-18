from train_simple import evaluate_from_checkpoint

evaluate_from_checkpoint(
    '../checkpoints/red_league_new/football_episode_600.pth',
    '../checkpoints/league_new/kick_football_episode_1800_2.pth',
    episodes=5, render=True)
