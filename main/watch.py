from train_simple import evaluate_from_checkpoint

evaluate_from_checkpoint(
    '../checkpoints/league_new/all_football_episode_9900_1.pth',
    '../checkpoints/league_new/kick_football_episode_1800_2.pth',
    episodes=5, render=True)
