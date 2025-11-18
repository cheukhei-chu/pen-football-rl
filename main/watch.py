from train_simple import evaluate_from_checkpoint

evaluate_from_checkpoint(
    '../checkpoints/red_league_test/football_episode_50.pth',
    '../checkpoints/red_league_test/football_episode_50.pth',
    episodes=5, render=True)
