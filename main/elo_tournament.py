import os
import random
import math
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import trange, tqdm

# Import your project code (assumes elo_tournament.py is in same folder as train.py / policy.py)
from multiagent import FootballMultiAgentEnv
from policy import make_policy

# ---------------------------
# Helpers: load policy from checkpoint path
# ---------------------------
def load_policy_from_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy_class = ckpt.get("policy_class") or ckpt.get("policy_name")
    policy_kwargs = ckpt.get("policy_kwargs", {})
    policy = make_policy(policy_class, **policy_kwargs)
    state = ckpt.get("policy_state_dict") or ckpt.get("state_dict")
    if state:
        policy.load_state_dict(state)
    policy.eval()
    # ensure on cpu (or specified device) — sample_action doesn't require device, but forward might
    policy.to(device)
    return policy

# ---------------------------
# Play a single episode (one "round") between two policies.
# Returns reward tuple (red_reward, blue_reward) and winner string 'red'/'blue'/'draw'
# ---------------------------
def play_one_round(env, policy_red, policy_blue, max_steps=500):
    """
    env: FootballMultiAgentEnv instance
    policy_red, policy_blue: policy objects with sample_action(obs) method OR used via forward + categorical
    Returns: winner ('red','blue','draw'), red_reward_sum, blue_reward_sum, steps
    """
    # Ensure env has no persistent setting
    try:
        env.set_setting(None)
    except Exception:
        pass

    obs, _ = env.reset()
    done = False
    steps = 0
    red_total = 0.0
    blue_total = 0.0

    while not done and steps < max_steps:
        # use policy.sample_action to support both checkpoint-loaded and custom policies
        a_red = policy_red.sample_action(obs["player_red"])
        a_blue = policy_blue.sample_action(obs["player_blue"])

        next_obs, rewards, terminated, truncated, _ = env.step({
            "player_red": a_red,
            "player_blue": a_blue
        })

        red_total += float(rewards["player_red"])
        blue_total += float(rewards["player_blue"])

        done = terminated["__all__"] or truncated["__all__"]
        obs = next_obs
        steps += 1

    # decide winner using reward sign heuristic (keeps consistent with comp_rewards in multiagent.py)
    if red_total > blue_total:
        winner = "red"
    elif blue_total > red_total:
        winner = "blue"
    else:
        winner = "draw"

    return winner, red_total, blue_total, steps

# ---------------------------
# Elo update helpers
# ---------------------------
def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

def update_elo(r_a, r_b, score_a, K=32):
    """
    score_a : 1.0 for win, 0.5 draw, 0.0 loss
    returns: new_r_a, new_r_b
    """
    e_a = expected_score(r_a, r_b)
    e_b = 1.0 - e_a
    new_r_a = r_a + K * (score_a - e_a)
    new_r_b = r_b + K * ((1.0 - score_a) - e_b)  # symmetrical
    return new_r_a, new_r_b

# ---------------------------
# Main runner
# ---------------------------
def run_elo_tournament(
    models_dir,
    n_matches_per_pair=100,
    print_every=50,
    K=32,
    device="cpu",
    seed=0,
):
    """
    models_dir: directory containing checkpoint .pth files (only .pth files are considered)
    n_matches_per_pair: how many independent rounds to play for each unordered pair (A,B)
    print_every: print Elo table after this many matches have been played
    K: Elo K-factor
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # discover checkpoints
    ckpt_files = sorted([
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".pth")
    ])
    if len(ckpt_files) == 0:
        raise RuntimeError(f"No .pth files found in {models_dir}")

    # give short names
    names = [os.path.basename(p) for p in ckpt_files]
    n_models = len(ckpt_files)
    print(f"Found {n_models} models: {names}")

    # Initial Elo ratings (default 1500)
    ratings = {p: 1500.0 for p in ckpt_files}
    games_played = {p: 0 for p in ckpt_files}
    wins = {p: 0 for p in ckpt_files}
    losses = {p: 0 for p in ckpt_files}
    draws = {p: 0 for p in ckpt_files}

    # Preload all policies into memory (optional; note memory/time tradeoff)
    print("Loading policies into memory...")
    policies = {}
    for p in ckpt_files:
        policies[p] = load_policy_from_checkpoint(p, device=device)

    env = FootballMultiAgentEnv()

    total_pairs = (n_models * (n_models - 1)) // 2
    total_matches = total_pairs * n_matches_per_pair
    match_counter = 0

    # iterate unordered pairs (i<j)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            p_i = ckpt_files[i]
            p_j = ckpt_files[j]
            pol_i = policies[p_i]
            pol_j = policies[p_j]

            # play n_matches_per_pair independent rounds, update Elo after each match
            for m in range(n_matches_per_pair):
                match_counter += 1
                # flip red/blue sides randomly to reduce side bias
                if random.random() < 0.5:
                    winner, rscore, bscore, steps = play_one_round(env, pol_i, pol_j)
                    # pol_i played red, pol_j blue
                    if winner == "red":
                        s_i = 1.0; s_j = 0.0
                        wins[p_i] += 1; losses[p_j] += 1
                    elif winner == "blue":
                        s_i = 0.0; s_j = 1.0
                        losses[p_i] += 1; wins[p_j] += 1
                    else:
                        s_i = 0.5; s_j = 0.5
                        draws[p_i] += 1; draws[p_j] += 1
                else:
                    # reversed sides
                    winner, rscore, bscore, steps = play_one_round(env, pol_j, pol_i)
                    # pol_j was red, pol_i blue
                    if winner == "red":
                        s_j = 1.0; s_i = 0.0
                        wins[p_j] += 1; losses[p_i] += 1
                    elif winner == "blue":
                        s_j = 0.0; s_i = 1.0
                        losses[p_j] += 1; wins[p_i] += 1
                    else:
                        s_i = 0.5; s_j = 0.5
                        draws[p_i] += 1; draws[p_j] += 1

                # Elo update
                r_i, r_j = ratings[p_i], ratings[p_j]
                new_r_i, new_r_j = update_elo(r_i, r_j, s_i, K=K)
                ratings[p_i], ratings[p_j] = new_r_i, new_r_j

                games_played[p_i] += 1
                games_played[p_j] += 1

                # Periodic print / table
                if match_counter % print_every == 0 or match_counter == total_matches:
                    print("\n=== Elo standings after match {}/{} ===".format(match_counter, total_matches))
                    # sort by rating
                    sorted_items = sorted(ratings.items(), key=lambda x: -x[1])
                    print("{:4s} {:40s} {:>8s} {:>6s} {:>6s} {:>6s}".format("rank","model","rating","games","wins","draws"))
                    for rank, (path, r) in enumerate(sorted_items, start=1):
                        nm = os.path.basename(path)
                        print("{:4d} {:40s} {:8.1f} {:6d} {:6d} {:6d}".format(
                            rank, nm, r, games_played[path], wins[path], draws[path]
                        ))

    print("\nDone. Final Elo ratings:")
    final_sorted = sorted(ratings.items(), key=lambda x: -x[1])
    for rank, (path, r) in enumerate(final_sorted, start=1):
        print(f"{rank:2d}. {os.path.basename(path):40s} {r:8.1f}  games={games_played[path]:4d}  wins={wins[path]:4d} draws={draws[path]:4d}")

    return ratings

if __name__ == "__main__":
    run_elo_tournament("../checkpoints/elo_tournament_test")