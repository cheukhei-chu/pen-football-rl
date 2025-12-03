# elo_incremental.py
import os
import random
import math
import json
import csv
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from multiagent import FootballMultiAgentEnv
from policy import make_policy

import matplotlib.pyplot as plt


# -------------------------
# Helper: load policy from checkpoint path
# -------------------------
def load_policy_from_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    policy_class = ckpt.get("policy_class") or ckpt.get("policy_name")
    policy_kwargs = ckpt.get("policy_kwargs", {})
    policy = make_policy(policy_class, **policy_kwargs)
    state = ckpt.get("policy_state_dict") or ckpt.get("state_dict")
    if state:
        policy.load_state_dict(state)
    policy.eval()
    policy.to(device)
    return policy

# -------------------------
# Play a single episode (one round)
# -------------------------
def play_one_round(env, policy_red, policy_blue, max_steps=500):
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
        a_red = policy_red.sample_action(obs["player_red"])
        a_blue = policy_blue.sample_action(obs["player_blue"])

        next_obs, rewards, terminated, truncated, _ = env.step({
            "player_red": a_red,
            "player_blue": a_blue
        })

        red_total += float(rewards["player_red"])
        blue_total += float(rewards["player_blue"])
        obs = next_obs
        steps += 1
        done = terminated["__all__"] or truncated["__all__"]

    if red_total > blue_total:
        return "red", red_total, blue_total, steps
    elif blue_total > red_total:
        return "blue", red_total, blue_total, steps
    else:
        return "draw", red_total, blue_total, steps

# -------------------------
# ELO helpers
# -------------------------
def expected_score(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def update_elo(r_a, r_b, score_a, K=16):
    e_a = expected_score(r_a, r_b)
    e_b = 1.0 - e_a
    new_r_a = r_a + K * (score_a - e_a)
    new_r_b = r_b + K * ((1.0 - score_a) - e_b)
    return new_r_a, new_r_b

# -------------------------
# Save standings
# -------------------------
def save_standings_csv(path, ratings, games, wins, draws):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "rating", "games", "wins", "draws"])
        for m, r in sorted(ratings.items(), key=lambda x: -x[1]):
            w.writerow([m, f"{r:.2f}", games[m], wins[m], draws[m]])

def save_standings_json(path, ratings, games, wins, draws):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for m, r in sorted(ratings.items(), key=lambda x: -x[1]):
        out.append({"model": m, "rating": r, "games": games[m], "wins": wins[m], "draws": draws[m]})
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

# -------------------------
# Main incremental Elo tournament
# -------------------------
def run_incremental_elo_tournament(
    model_folders,                 # list of folders to scan for .pth files
    batch_size=10,                 # how many models to add per batch (you asked 10)
    matches_per_model=100,         # target matches per model before adding next batch
    n_matches_total=250_000,       # total matches to run overall
    print_every=5000,              # print top-k every this many matches
    save_standings_every=10000,    # save standings to file every this many matches
    save_path=None,                # e.g. "../results/elo_table.csv" or .json
    save_format="csv",             # "csv" or "json"
    top_k=20,                      # print top 20
    K=16,                          # Elo K-factor (I recommend 16)
    device="cpu",
    seed=0,
):
    """
    Incremental Elo tournament:
      - discover models in model_folders (label them folder_filename)
      - random shuffle ordering of models (you requested random order)
      - start with first `batch_size` models
      - every add_every matches add next up to `batch_size` models
        where add_every = batch_size * matches_per_model (1000 for batch_size=10,matches_per_model=100)
      - new models start with Elo 1500
      - print only top_k
      - periodically save standings to save_path
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # discover all .pth files and label as folder_filename
    discovered = []
    for folder in model_folders:
        base = os.path.basename(folder.rstrip("/\\"))
        if not os.path.isdir(folder):
            print(f"[WARN] model folder not found: {folder} -- skipping")
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".pth"):
                path = os.path.join(folder, fname)
                label = f"{base}_{fname}"
                discovered.append((label, path))

    if len(discovered) == 0:
        raise RuntimeError("No .pth files found in the provided folders.")

    # randomize order (you asked "randomize")
    random.shuffle(discovered)

    # partition into pending queue and active initial batch
    pending = discovered.copy()
    active = []

    # helper to add up to `batch_size` models from pending to active
    def add_next_batch():
        nonlocal pending, active
        added = []
        for _ in range(batch_size):
            if not pending:
                break
            added.append(pending.pop(0))
        for label, path in added:
            # load policy lazily and add to active policy dict
            policies[label] = load_policy_from_checkpoint(path, device=device)
            # initialize stats
            ratings[label] = 1500.0
            games[label] = 0
            wins[label] = 0
            draws[label] = 0
            # add to active names list
            active.append(label)
        if added:
            print(f"[INFO] Added {len(added)} models to league; active_size={len(active)}")
        return len(added)

    # statistics containers
    ratings = {}
    games = {}
    wins = {}
    draws = {}

    # dict of loaded policies for currently active models only
    policies = {}

    # initialize: add first batch
    add_next_batch()
    if len(active) == 0:
        raise RuntimeError("No active models after initial batch. Check your folders and files.")

    env = FootballMultiAgentEnv()

    matches_done = 0
    add_every = batch_size * matches_per_model   # e.g. 10 * 100 = 1000 matches between additions

    # Main loop
    while matches_done < n_matches_total:
        # if time to add (except immediate re-add when matches_done==0)
        if matches_done > 0 and matches_done % add_every == 0 and len(pending) > 0:
            added = add_next_batch()
            # small safeguard: if no models were added (pending empty), break if finished
            if added == 0 and len(pending) == 0:
                print("[INFO] No more pending models to add.")
        # choose two distinct active players
        if len(active) < 2:
            print("[WARN] Less than 2 active models; stopping early.")
            break
        p1, p2 = random.sample(active, 2)
        pol1, pol2 = policies[p1], policies[p2]

        # randomize side assignment
        if random.random() < 0.5:
            winner, _, _, _ = play_one_round(env, pol1, pol2)
            if winner == "red":
                s1, s2 = 1.0, 0.0
            elif winner == "blue":
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5
        else:
            winner, _, _, _ = play_one_round(env, pol2, pol1)
            if winner == "red":
                s2, s1 = 1.0, 0.0
            elif winner == "blue":
                s2, s1 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5

        # update counts
        if s1 == 1.0:
            wins[p1] += 1
        if s2 == 1.0:
            wins[p2] += 1
        if s1 == 0.5:
            draws[p1] += 1
        if s2 == 0.5:
            draws[p2] += 1
        games[p1] += 1
        games[p2] += 1

        # update Elo
        r1, r2 = ratings[p1], ratings[p2]
        new_r1, new_r2 = update_elo(r1, r2, s1, K=K)
        ratings[p1], ratings[p2] = new_r1, new_r2

        matches_done += 1

        # periodic prints (top-k only)
        if matches_done % print_every == 0:
            print(f"\n=== Standings after {matches_done}/{n_matches_total} matches (top {top_k}) ===")
            sorted_items = sorted(ratings.items(), key=lambda x: -x[1])
            for rank, (name, rating) in enumerate(sorted_items[:top_k], start=1):
                print(f"{rank:2d}. {name:35s} {rating:8.1f}  games={games[name]:4d}  wins={wins[name]:4d}  draws={draws[name]:4d}")

        # periodic save
        if save_path and matches_done % save_standings_every == 0:
            if save_format == "csv":
                save_standings_csv(save_path, ratings, games, wins, draws)
            else:
                save_standings_json(save_path, ratings, games, wins, draws)
            print(f"[INFO] Saved standings to {save_path} at match {matches_done}")

    # final save + display
    if save_path:
        if save_format == "csv":
            save_standings_csv(save_path, ratings, games, wins, draws)
        else:
            save_standings_json(save_path, ratings, games, wins, draws)

    print("\n=== FINAL TOP {0} ===".format(top_k))
    final_sorted = sorted(ratings.items(), key=lambda x: -x[1])
    for rank, (name, rating) in enumerate(final_sorted[:top_k], start=1):
        print(f"{rank:2d}. {name:35s} {rating:8.1f}  games={games[name]:4d}  wins={wins[name]:4d}  draws={draws[name]:4d}")

    return ratings

# --------------------------------------------------------------------
# Plot final Elo rating vs. epoch for each folder
# --------------------------------------------------------------------
import matplotlib.pyplot as plt

def extract_epoch_from_filename(fname):
    """
    Extracts the number after the last underscore.
    Examples:
        model_45000.pth -> 45000
        league_policy_120000_extra.pth -> 120000
    """
    base = os.path.basename(fname)
    if "_" not in base:
        return None
    parts = base.split("_")
    last = parts[-1]          # e.g. "45000.pth"
    num = last.split(".")[0]  # "45000"
    if num.isdigit():
        return int(num)
    return None


def plot_elo_vs_epoch(
    standings_csv_path,
    model_folders,
    save_dir="../results/elo_plots"
):
    os.makedirs(save_dir, exist_ok=True)

    # --- Load Elo standings CSV ---
    final_ratings = {}
    with open(standings_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            final_ratings[row["model"]] = float(row["rating"])

    # --- Process each folder ---
    for folder in model_folders:
        base = os.path.basename(folder.rstrip("/\\"))
        epoch_list = []
        elo_list = []

        for fname in os.listdir(folder):
            if not fname.endswith(".pth"):
                continue

            full_label = f"{base}_{fname}"   # this matches how your tournament labeled models
            if full_label not in final_ratings:
                continue  # this .pth model never got added or played

            epoch = extract_epoch_from_filename(fname)
            if epoch is None:
                continue

            epoch_list.append(epoch)
            elo_list.append(final_ratings[full_label])

        if len(epoch_list) == 0:
            print(f"[WARN] No usable checkpoints in folder: {folder}")
            continue

        # Sort points by epoch
        pairs = sorted(zip(epoch_list, elo_list))
        xs, ys = zip(*pairs)

        # --- Plot ---
        plt.figure(figsize=(8,5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Epoch number")
        plt.ylabel("Final Elo rating")
        plt.title(f"Elo vs Epoch — {base}")
        plt.grid(True)

        save_path = os.path.join(save_dir, f"{base}_elo_vs_epoch.png")
        plt.savefig(save_path)
        plt.close()

        print(f"[INFO] Saved Elo-vs-epoch plot: {save_path}")

if __name__ == "__main__":
    model_folders = [
        "../checkpoints/league_ppo (misc rewards)",
        "../checkpoints/league_ppo (score reward)",
        "../checkpoints/league_ppo_real (score reward)",
        "../checkpoints/league_ppo_regular (score reward)",
        "../checkpoints/shoot_left_ppo (without embedding)",
    ]
    # Where to save standings (CSV recommended). Make sure parent directory exists or script will create it.
    save_path = "../results/elo_incremental_standings.csv"
    # ratings = run_incremental_elo_tournament(
    #     model_folders=model_folders,
    #     batch_size=10,
    #     matches_per_model=100,
    #     n_matches_total=120000,        # choose your desired total
    #     print_every=5000,
    #     save_standings_every=10000,
    #     save_path=save_path,
    #     save_format="csv",
    #     top_k=20,
    #     K=16,
    #     device="cpu",
    #     seed=42,
    # )
    standings_csv_path = "../results/elo_incremental_standings.csv"
    plot_elo_vs_epoch(
        standings_csv_path=standings_csv_path,
        model_folders=[
            "../checkpoints/league_ppo (misc rewards)",
            "../checkpoints/league_ppo (score reward)",
            "../checkpoints/league_ppo_real (score reward)",
            "../checkpoints/league_ppo_regular (score reward)",
            "../checkpoints/shoot_left_ppo (without embedding)",
        ],
        save_dir="../results/elo_plots",
    )
