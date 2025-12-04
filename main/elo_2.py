# elo_tournament.py
import os
import random
import math
import json
import csv
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

from multiagent import FootballMultiAgentEnv
from policy import *   # brings in atulPolicy, make_policy, etc.

from scipy.signal import savgol_filter

naming_dict = {"league_ppo_real (score reward)":"AlphaStar League",
               "league_ppo (score reward)":"score",
               "league_ppo (misc rewards)":"{score, move, kick, jump}",
               }

def compute_elo_instability(
    folder,
    final_ratings,
    window=11,
    poly=3
):
    """
    Computes Elo instability metrics for all checkpoint files in a folder.

    Returns:
        {
            "base": <folder name>,
            "points": [
                (epoch, filename, elo),
                ...
            ],
            "metrics": {
                "residual_std": ...,
                "delta_variance": ...,
                "mean_abs_delta": ...
            }
        }
    """
    base = os.path.basename(folder.rstrip("/\\"))
    points = []   # (epoch, filename, elo)

    # Collect all usable checkpoints
    for fname in os.listdir(folder):
        if not fname.endswith(".pth"):
            continue

        full_label = f"{base}_{fname}"
        if full_label not in final_ratings:
            continue

        epoch = extract_epoch_from_filename(fname)
        if epoch is None:
            continue

        elo = final_ratings[full_label]
        points.append((epoch, fname, elo))

    if len(points) == 0:
        print(f"[WARN] No usable checkpoints in folder: {folder}")
        return None

    # Sort by epoch
    points.sort(key=lambda x: x[0])

    # Extract ELO values (sorted)
    ys = np.array([p[2] for p in points])

    # ----------------------------
    # 1. Detrended residual noise
    # ----------------------------
    # Ensure smoothing window is valid
    if len(ys) >= window and window % 2 == 1:
        trend = savgol_filter(ys, window, poly)
        residuals = ys - trend
        residual_std = float(np.std(residuals))
    else:
        # Not enough points to smooth meaningfully
        residual_std = None

    # ----------------------------
    # 2. Delta variance
    # ----------------------------
    if len(ys) >= 2:
        deltas = np.diff(ys)
        delta_variance = float(np.var(deltas))
        mean_abs_delta = float(np.mean(np.abs(deltas)))
    else:
        delta_variance = None
        mean_abs_delta = None

    return {
        "base": base,
        "points": points,
        "metrics": {
            "residual_std": residual_std,
            "delta_variance": delta_variance,
            "mean_abs_delta": mean_abs_delta,
        }
    }

def compute_instability_for_folders(standings_csv_path, model_folders):
    # Load ratings
    final_ratings = {}
    with open(standings_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            final_ratings[row["model"]] = float(row["rating"])

    results = {}
    for folder in model_folders:
        info = compute_elo_instability(folder, final_ratings)
        if info is not None:
            results[info["base"]] = info

    return results

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
    """
    Plays a single episode between policy_red and policy_blue.
    Returns winner string: "red", "blue", or "draw".
    Behavior:
      - tries to use info["result"] if provided by env.step()
      - otherwise sums time-step rewards and compares totals
    """
    try:
        env.set_setting(None)
    except Exception:
        pass

    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        a_red = policy_red.sample_action(obs["player_red"])
        a_blue = policy_blue.sample_action(obs["player_blue"])

        next_obs, rewards, terminated, truncated, info = env.step({
            "player_red": a_red,
            "player_blue": a_blue
        })

        obs = next_obs
        steps += 1
        done = terminated["__all__"] or truncated["__all__"]

        if done:
            # Try common formats for result
            return info["result"]


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
            w.writerow([m, f"{r:.2f}", games.get(m, 0), wins.get(m, 0), draws.get(m, 0)])




# -------------------------
# Renormalize helper to pin atulPolicy to 1500
# -------------------------
def renormalize_to_atul(ratings, atol=1e-9):
    """
    Shift all ratings so that ratings['atulPolicy'] == 1500.0, if atulPolicy exists.
    This is an affine shift applied to all entries, which preserves Elo predictions.
    """
    if "atulPolicy" not in ratings:
        return  # nothing to do
    offset = ratings["atulPolicy"] - 1500.0
    if abs(offset) < atol:
        return
    for k in list(ratings.keys()):
        ratings[k] -= offset


# -------------------------
# Baseline (full round-robin) tournament
# -------------------------
def run_baseline_tournament(
    baseline_folder,
    atul_policy=atulPolicy,            # either a policy instance (atulPolicy()) or a .pth path
    games_per_pair=100,
    K=16,
    device="cpu",
):
    """
    Plays a round-robin among all .pth files in baseline_folder plus atulPolicy.
    atul_policy may be:
      - an instance of atulPolicy (preferred)
      - a path to a checkpoint .pth file (will be loaded)
    Ratings are renormalized after every update so atulPolicy == 1500 at all times.
    Returns ratings dict.
    """

    # discover baseline files
    baseline_files = []
    for fname in sorted(os.listdir(baseline_folder)):
        if fname.endswith(".pth"):
            label = f"baseline_{fname}"
            path = os.path.join(baseline_folder, fname)
            baseline_files.append((label, path))

    # prepare atulPolicy
    atul_label = "atulPolicy"
    policies = {}

    if isinstance(atul_policy, str) and os.path.isfile(atul_policy) and atul_policy.endswith(".pth"):
        # load checkpoint version and keep it under the atul label
        policies[atul_label] = load_policy_from_checkpoint(atul_policy, device=device)
    elif isinstance(atul_policy, FootballPolicy):
        policies[atul_label] = atul_policy
    elif callable(atul_policy):
        # maybe user passed the class itself (atulPolicy), instantiate
        try:
            policies[atul_label] = atul_policy()
        except Exception:
            raise RuntimeError("Unable to instantiate provided atul_policy callable/class.")
    else:
        raise ValueError("atul_policy must be either an atulPolicy instance/class or a .pth path")

    # load baseline policies
    for label, path in baseline_files:
        policies[label] = load_policy_from_checkpoint(path, device=device)

    # initialize stats
    ratings = {label: 1500.0 for label in policies.keys()}
    games = defaultdict(int)
    wins = defaultdict(int)
    draws = defaultdict(int)

    env = FootballMultiAgentEnv()

    pairs = list(itertools.combinations(list(policies.keys()), 2))
    total_matches = len(pairs) * games_per_pair
    print(f"[BASELINE] Round-robin: {len(pairs)} pairs, {games_per_pair} games per pair → {total_matches} matches total.")

    for (p1, p2) in tqdm(pairs, desc="baseline pairs"):
        for _ in range(games_per_pair):
            # randomize sides
            winner = play_one_round(env, policies[p1], policies[p2])
            if winner == "red":
                s1, s2 = 1.0, 0.0
            elif winner == "blue":
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5

            # update stats
            games[p1] += 1; games[p2] += 1
            if s1 == 1.0: wins[p1] += 1
            if s2 == 1.0: wins[p2] += 1
            if s1 == 0.5: draws[p1] += 1
            if s2 == 0.5: draws[p2] += 1

            # update Elo
            r1, r2 = ratings[p1], ratings[p2]
            new_r1, new_r2 = update_elo(r1, r2, s1, K=K)
            ratings[p1], ratings[p2] = new_r1, new_r2

            # renormalize so atulPolicy==1500 always
            renormalize_to_atul(ratings)

    print("[BASELINE] Completed baseline round-robin.")
    save_standings_csv("../results/baseline_ratings.txt",ratings,games,wins,draws)
    return [ratings,games,wins,draws]


# -------------------------
# Main incremental Elo tournament
# -------------------------
def run_incremental_elo_tournament(
    model_folders,
    baseline_path,
    matches_per_model=100,
    print_every=10,
    save_path=None,
    top_k=20,
    K=16,
    device="cpu",
    seed=0,
    baseline_ratings=None,   # required!
):
    """
    New version (simplified):
      - load baseline ratings (these models never play each other)
      - discover new models in folders
      - for EACH new model:
            • load policy
            • set rating = 1500
            • play `matches_per_model` games against random baseline opponent
            • update both ratings
            • renormalize so atulPolicy == 1500
      - print final top-k
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Load baseline ratings ---
    if not isinstance(baseline_ratings, dict) or len(baseline_ratings) == 0:
        raise ValueError("baseline_ratings must be provided.")
    def load_baseline_paths(folders, rating_keys):
        """Return {model_name: path_to_pth} for all baseline models."""
        name_to_path = {}
        #print("KEYS ARE \n\n\n",rating_keys)
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            base = os.path.basename(folder.rstrip("/\\"))
            for fname in os.listdir(folder):
                #print(fname)
                if not fname.endswith(".pth"):
                    continue
                if "baseline_"+fname in rating_keys:
                    name_to_path["baseline_"+fname] = os.path.join(folder, fname)
        return name_to_path
    
    ratings = baseline_ratings.copy()
    games  = {k: 0 for k in ratings}
    wins   = {k: 0 for k in ratings}
    draws  = {k: 0 for k in ratings}
    #print(ratings)
    baseline_paths = load_baseline_paths([baseline_path], list(baseline_ratings.keys()))
    #print(baseline_paths)
    # Load baseline policies
    baseline_policies = {
        name: load_policy_from_checkpoint(path, device=device)
        for name, path in baseline_paths.items()
    }
    baseline_names = list(baseline_policies.keys())
    #print(baseline_names)
    env = FootballMultiAgentEnv()
    # --- Discover new models ---
    discovered = []
    model_index = -1
    prev_label = ""
    for folder in model_folders:
        if not os.path.isdir(folder):
            continue
        base = os.path.basename(folder.rstrip("/\\"))
        for fname in os.listdir(folder):
            if fname.endswith(".pth"):
                path = os.path.join(folder, fname)
                label = f"{base}_{fname}"
                if label not in ratings:   # skip baseline entries
                    discovered.append((label, path))
                    model_index += 1
                    policy_new = load_policy_from_checkpoint(path, device=device)

                    # initialize
                    if model_index==0:
                        ratings[label] = 1500.0
                    else:
                        ratings[label] = ratings[prev_label]
                    prev_label = label
                    games[label] = 0
                    wins[label] = 0
                    draws[label] = 0

                    # play matches_per_model games vs random baseline opponent
                    for _ in range(matches_per_model):
                        opp = random.choice(baseline_names)
                        pol_opp = baseline_policies[opp]

                        # play match
                        winner = play_one_round(env, policy_new, pol_opp)
                        if winner == "red":
                            s_new, s_opp = 1.0, 0.0
                        elif winner == "blue":
                            s_new, s_opp = 0.0, 1.0
                        else:
                            s_new, s_opp = 0.5, 0.5
                        # update stats
                        if s_new == 1.0: wins[label] += 1
                        if s_opp == 1.0: wins[opp] += 1
                        if s_new == 0.5: draws[label] += 1
                        if s_opp == 0.5: draws[opp] += 1
                        games[label] += 1
                        games[opp]   += 1

                        # update Elo
                        r_new = ratings[label]
                        r_opp = ratings[opp]
                        new_r_new, new_r_opp = update_elo(r_new, r_opp, s_new, K=K)
                        ratings[label] = new_r_new
                        ratings[opp]   = new_r_opp

                        # renormalize so atulPolicy = 1500
                        renormalize_to_atul(ratings)

                    if model_index % print_every == 0:
                        print(f"[INFO] Finished {model_index}/{len(discovered)} new models")
                        save_standings_csv(save_path,ratings,games,wins,draws)      
                        final_sorted = sorted(ratings.items(), key=lambda x: -x[1])
                        print("\n=== FINAL TOP {0} ===".format(top_k))
                        for rank, (name, rating) in enumerate(final_sorted[:top_k], start=1):
                            print(f"{rank:2d}. {name:35s} {rating:8.1f}  "
                                f"games={games.get(name,0):4d}  wins={wins.get(name,0):4d}  draws={draws.get(name,0):4d}")              

    # --- Final standings ---
    final_sorted = sorted(ratings.items(), key=lambda x: -x[1])

    print("\n=== FINAL TOP {0} ===".format(top_k))
    for rank, (name, rating) in enumerate(final_sorted[:top_k], start=1):
        print(f"{rank:2d}. {name:35s} {rating:8.1f}  "
              f"games={games.get(name,0):4d}  wins={wins.get(name,0):4d}  draws={draws.get(name,0):4d}")
        
    save_standings_csv(save_path,ratings,games,wins,draws)

    return ratings

# --------------------------------------------------------------------
# Plot helpers: final Elo rating vs. epoch for each folder
# --------------------------------------------------------------------
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

def load_model_ratings(csv_path):
    """Return a dict mapping model names → ratings from a CSV file."""
    ratings = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings[row["model"]] = float(row["rating"])
    return ratings

import os
import csv
import matplotlib.pyplot as plt

def plot_multiple_elo(
    standings_csv_path,
    model_folders,
    save_path="../results/elo_plots/latent.png",
    filter=False,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --- Load final Elo ratings CSV ---
    final_ratings = {}
    with open(standings_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            final_ratings[row["model"]] = float(row["rating"])

    plt.figure(figsize=(10, 6))

    # --- Process each model folder ---
    for folder in model_folders:
        base = os.path.basename(folder.rstrip("/\\"))
        epoch_list = []
        elo_list = []

        for fname in os.listdir(folder):
            if not fname.endswith(".pth"):
                continue

            full_label = f"{base}_{fname}"  # how tournament names models
            if full_label not in final_ratings:
                continue

            epoch = extract_epoch_from_filename(fname)
            if epoch is None:
                continue

            epoch_list.append(epoch)
            elo_list.append(final_ratings[full_label])

        if len(epoch_list) == 0:
            print(f"[WARN] No usable checkpoints in folder: {folder}")
            continue

        # Sort by epoch
        pairs = sorted(zip(epoch_list, elo_list))
        xs, ys = zip(*pairs)

        xs_new = []
        ys_new = []
        for i,elem in enumerate(xs):
            if elem <= 5e8:
                xs_new.append(xs[i])
                ys_new.append(ys[i])
        
        if base in naming_dict:
            base = naming_dict[base]
        # Plot on shared figure
        if filter:
            ys_new = smooth_series(xs_new,ys_new,"moving_average")
        plt.plot(xs_new, ys_new, label=base)

    # --- Final formatting ---
    plt.xlabel("Epoch number")
    plt.ylabel("Final Elo rating")
    plt.title("Elo vs Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved combined Elo-vs-epoch plot to {save_path}")

def smooth_series(xs, ys, method="moving_average", window=5, polyorder=2, alpha=0.3):
    """
    Smooth a 1D series of values.

    Args:
        xs (list or np.array): x-values (kept unchanged)
        ys (list or np.array): y-values (to smooth)
        method (str): "moving_average", "ema", or "savgol"
        window (int): window size for moving average or Savitzky-Golay
        polyorder (int): polynomial order for Savitzky-Golay
        alpha (float): smoothing factor for EMA (0 < alpha < 1)

    Returns:
        np.array: smoothed y-values
    """
    ys = np.array(ys)
    
    if method == "moving_average":
        if window < 1:
            window = 1
        cumsum = np.cumsum(np.insert(ys, 0, 0)) 
        smoothed = (cumsum[window:] - cumsum[:-window]) / window
        # pad to match original length
        pad_left = window // 2
        pad_right = ys.size - smoothed.size - pad_left
        smoothed = np.pad(smoothed, (pad_left, pad_right), mode='edge')
        return smoothed
    
    elif method == "ema":
        smoothed = np.zeros_like(ys)
        smoothed[0] = ys[0]
        for i in range(1, len(ys)):
            smoothed[i] = alpha * ys[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    elif method == "savgol":
        # window must be odd and >= polyorder+2
        if window % 2 == 0:
            window += 1
        window = max(window, polyorder + 2)
        smoothed = savgol_filter(ys, window_length=window, polyorder=polyorder)
        return smoothed
    
    else:
        raise ValueError("Unknown method. Choose from 'moving_average', 'ema', 'savgol'.")

# -------------------------
# Example usage (edit paths, save_path)
# -------------------------
if __name__ == "__main__":
    # model_folders = [
    #     "../checkpoints/league_ppo (misc rewards)",
    #     "../checkpoints/league_ppo (score reward)",
    #     "../checkpoints/league_ppo_real (score reward)",
    #     "../checkpoints/league_ppo_regular (score reward)",
    #     "../checkpoints/shoot_left_ppo (without embedding)",
    # ]
    # model_folders = ["../checkpoints/league_ppo_real (score reward) (latent_dims 128 128 128)"]
    model_folders = ["../checkpoints/league_ppo_regular_real (score reward) (latent_dims 128 128)", ]
    baseline_path = "../checkpoints/elo_tournament_baseline"
    # ratings = load_model_ratings(baseline_path)
    

    # Example baseline run:
    # baseline_folder = "../checkpoints/baseline_models"
    # atul_inst = atulPolicy()               # pass the class instance, not a path
    # base_ratings = run_baseline_tournament(baseline_folder, atul_inst, games_per_pair=50, K=16, device="cpu")
    # # base_ratings now has atulPolicy pinned at 1500 throughout

    # Example incremental run (using base_ratings as initialization):
    # save_path = "../results/elo_incremental_standings.csv"
    # ratings = run_incremental_elo_tournament(
    #     model_folders=model_folders,
    #     batch_size=10,
    #     matches_per_model=100,
    #     n_matches_total=120000,
    #     print_every=5000,
    #     save_standings_every=10000,
    #     save_path=save_path,
    #     save_format="csv",
    #     top_k=20,
    #     K=16,
    #     device="cpu",
    #     seed=42,
    #     baseline_ratings=None,  # or base_ratings
    # )

    # Plotting (after running incremental tournament and saving standings csv):
    standings_csv_path = "results/incremental_elo_4.csv"
    # plot_elo_vs_epoch(
    #     standings_csv_path=standings_csv_path,
    #     model_folders=model_folders,
    #     save_dir="../results/elo_plots2",
    # )
    # instability = compute_instability_for_folders("results/incremental_elo_2.csv", model_folders)

    # for base, data in instability.items():
    #     print(f"\n=== {base} ===")
    #     print("metrics:", data["metrics"])
    #     print("first few points:", data["points"][:5])
    plot_multiple_elo(standings_csv_path=standings_csv_path,
                      model_folders=model_folders,
                      filter=False)

