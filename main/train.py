import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, time, pygame, random

from multiagent import FootballMultiAgentEnv
from policy import *
from collections import deque, defaultdict

###############################################################
# =======================  GAE  ============================= #
###############################################################

def compute_gae(rewards, values, dones, last_val, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = last_val if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    return advantages


###############################################################
# ====================  PPO LOSS  =========================== #
###############################################################

def ppo_loss(policy, obs, actions, old_logps, advantages, returns,
             clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01):

    logits = policy.forward(obs)

    logps = []
    entropies = []

    for k in ["left", "right", "jump"]:
        dist = torch.distributions.Categorical(logits=logits[k])
        logps.append(dist.log_prob(actions[k]))
        entropies.append(dist.entropy())

    logp = sum(logps)                # shape (batch,)
    entropy = sum(entropies)         # shape (batch,)

    # Ensure old_logps, advantages, returns are tensors of matching shape
    # ratio shape: (batch,)
    ratio = torch.exp(logp - old_logps)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    value_pred = logits["value"].squeeze(-1)
    value_loss = ((returns - value_pred) ** 2).mean()

    return policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()


###############################################################
# ===================  ROLLOUT CODE  ======================== #
###############################################################

def rollout(env, policy_red, policy_blue, select_drill,
            rollout_len=2048, gamma=0.99, lam=0.95):

    obs_list = []
    act_list = {"left": [], "right": [], "jump": []}
    logp_list = []
    rew_list = []
    done_list = []
    val_list = []

    steps = 0
    obs = None

    while steps < rollout_len:

        # ------ NEW EPISODE ------
        drill = select_drill()
        env.set_setting(drill)
        policy_red.set_setting(drill)
        obs, _ = env.reset()

        done = False

        while not done and steps < rollout_len:

            # observation for red
            obs_tensor = torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)

            logits = policy_red.forward(obs_tensor)
            value = logits["value"].item()

            # sample red action
            a = {
                k: torch.distributions.Categorical(logits=logits[k]).sample().item()
                for k in ["left", "right", "jump"]
            }

            # compute log probability
            logp = 0.0
            for k in ["left", "right", "jump"]:
                dist = torch.distributions.Categorical(logits=logits[k])
                logp += dist.log_prob(torch.tensor(a[k])).item()

            # environment step
            next_obs, rewards, terminated, truncated, info = env.step({
                "player_red": a,
                "player_blue": policy_blue.sample_action(obs["player_blue"]),
            })

            done = terminated["__all__"] or truncated["__all__"]

            # store transition
            obs_list.append(obs_tensor)            # list of (1, obs_dim) tensors
            for k in a:
                act_list[k].append(a[k])           # list of ints
            logp_list.append(logp)                 # list of floats
            rew_list.append(rewards["player_red"]) # list of floats
            done_list.append(done)                 # list of bools
            val_list.append(value)                 # list of floats

            steps += 1
            obs = next_obs

            # if we've reached rollout_len exactly mid-episode, break cleanly
            if steps >= rollout_len:
                break

        # episode ends here — loop automatically restarts new drill

    # ------ BOOTSTRAP VALUE ------
    with torch.no_grad():
        last_val = policy_red.forward(
            torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)
        )["value"].item()

    return {
        "obs": obs_list,
        "acts": act_list,
        "logp": logp_list,
        "rew": rew_list,
        "done": done_list,
        "val": val_list,
        "last_val": last_val,
    }


###############################################################
# ===================  PPO UPDATE  ========================== #
###############################################################

def ppo_update(policy, optimizer, obs, actions, old_logps, advantages, returns,
               epochs=10, batch_size=64, clip_ratio=0.2):
    """
    obs: list of (1,obs_dim) tensors OR a stacked tensor (N, obs_dim)
    actions: dict of lists -> will be converted to tensors
    old_logps: list or tensor
    advantages, returns: tensors or lists
    """

    # ---------- Convert/stack observations ----------
    if isinstance(obs, list):
        obs = torch.cat(obs, dim=0)            # (N, obs_dim)
    # else assume obs is already a tensor of shape (N, obs_dim)

    # ---------- Convert actions and other lists to tensors ----------
    actions = {
        k: torch.tensor(v, dtype=torch.long)
        for k, v in actions.items()
    }

    old_logps = torch.tensor(old_logps, dtype=torch.float32)
    advantages = advantages.clone().detach().requires_grad_(True)
    returns = returns.clone().detach().requires_grad_(True)

    N = len(returns)
    idxs = np.arange(N)

    # ---------- PPO training ----------
    for _ in range(epochs):
        np.random.shuffle(idxs)

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch = idxs[start:end]
            if len(batch) == 0:
                continue

            # convert batch indices to torch LongTensor for safe indexing
            batch_idx = torch.tensor(batch, dtype=torch.long)

            obs_b = obs[batch_idx]
            act_b = {k: v[batch_idx] for k, v in actions.items()}
            old_log_b = old_logps[batch_idx]
            adv_b = advantages[batch_idx]
            ret_b = returns[batch_idx]

            loss = ppo_loss(
                policy, obs_b, act_b, old_log_b, adv_b, ret_b, clip_ratio
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


###############################################################
# ===============  PPO DRILL TRAINING LOOP  ================= #
###############################################################
def train_drill_ppo(name, policy, select_drill,
                    total_steps=3_000_000, rollout_len=4096,
                    lr=3e-4, gamma=0.99, lam=0.95,
                    epochs=10, batch_size=256,
                    print_every=10_000, save_every=100_000):

    env = FootballMultiAgentEnv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    # Load or construct policy
    if isinstance(policy, tuple):
        pname, kwargs = policy
        policy_red = make_policy(pname, **kwargs)
        policy_kwargs = kwargs
    else:
        policy_red, checkpoint = policy_from_checkpoint_path(policy)
        policy_kwargs = checkpoint["policy_kwargs"]

    policy_blue = DummyPolicy()
    optimizer = optim.Adam(policy_red.parameters(), lr=lr)

    steps = 0
    rewards_save = []

    while steps < total_steps:

        # -------- GET ROLLOUT DATA --------
        roll = rollout(
            env, policy_red, policy_blue, select_drill,
            rollout_len=rollout_len, gamma=gamma, lam=lam
        )

        obs      = roll["obs"]       # list of tensors (1, obs_dim)
        actions  = roll["acts"]      # dict of lists
        old_logps= roll["logp"]      # list of floats
        rewards  = roll["rew"]       # list of floats
        dones    = roll["done"]      # list of bools
        values   = roll["val"]       # list of floats
        last_val = roll["last_val"]  # float

        steps += rollout_len

        # -------- COMPUTE ADV + RETURNS --------
        adv = compute_gae(rewards, values, dones, last_val, gamma, lam)   # tensor (N,)
        ret = adv + torch.tensor(values, dtype=torch.float32)             # tensor (N,)

        # NOTE: DO NOT cat obs here; we pass the list into ppo_update which handles stacking
        ppo_update(
            policy_red, optimizer,
            obs, actions, old_logps, adv, ret,
            epochs=epochs, batch_size=batch_size
        )

        rewards_save.append(sum(rewards)/len(rewards))
        if steps % print_every < rollout_len:
            print(f"[{steps - steps % print_every}] PPO update completed | mean reward = {sum(rewards_save)/len(rewards_save):.3f}")
            rewards_save = []


        if steps % save_every < rollout_len:
            save_path = os.path.join(checkpoint_dir, f"checkpoint_{steps - steps % save_every}.pth")
            torch.save({
                "policy_state_dict": policy_red.state_dict(),
                "policy_class": policy_red.__class__.__name__,
                "policy_kwargs": policy_kwargs
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

def train_league_ppo(
    name, policy,
    total_steps=3_000_000, rollout_len=4096,
    lr=3e-4, gamma=0.99, lam=0.95,
    epochs=10, batch_size=256,
    pool_size=20,
    self_play_prob=None,
    print_every=10_000, save_every=50_000
):

    env = FootballMultiAgentEnv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    if isinstance(policy, tuple):
        pname, kwargs = policy
        policy_red = make_policy(pname, **kwargs)
        policy_kwargs = kwargs
    else:
        policy_red, checkpoint = policy_from_checkpoint_path(policy)
        policy_kwargs = checkpoint["policy_kwargs"]

    optimizer = optim.Adam(policy_red.parameters(), lr=lr)

    opponent_pool = []

    if self_play_prob is None: self_play_prob = 1/(pool_size+1)
    def select_opponent():
        if len(opponent_pool) == 0:
            return DummyPolicy()

        if random.random() < self_play_prob:
            return policy_red

        opp_path = random.choice(opponent_pool)
        ckpt = torch.load(opp_path, map_location="cpu")

        opponent = make_policy(ckpt["policy_class"], **ckpt["policy_kwargs"])
        opponent.load_state_dict(ckpt["policy_state_dict"])
        opponent.eval()
        return opponent

    steps = 0
    rewards_save = []

    while steps < total_steps:

        policy_blue = select_opponent()

        roll = rollout(
            env,
            policy_red,
            policy_blue,
            select_drill=lambda: None,
            rollout_len=rollout_len,
            gamma=gamma,
            lam=lam
        )

        obs      = roll["obs"]
        actions  = roll["acts"]
        old_logps= roll["logp"]
        rewards  = roll["rew"]
        dones    = roll["done"]
        values   = roll["val"]
        last_val = roll["last_val"]

        steps += rollout_len

        adv = compute_gae(rewards, values, dones, last_val, gamma, lam)
        ret = adv + torch.tensor(values, dtype=torch.float32)

        ppo_update(
            policy_red, optimizer,
            obs, actions, old_logps, adv, ret,
            epochs=epochs, batch_size=batch_size
        )

        rewards_save.append(sum(rewards)/len(rewards))
        if steps % print_every < rollout_len:
            print(f"[{steps - (steps % print_every)}] PPO update | mean reward = {sum(rewards_save)/len(rewards_save):.3f}")
            rewards_save = []

        if steps % save_every < rollout_len:
            save_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_{steps - (steps % save_every)}.pth"
            )
            torch.save({
                "policy_state_dict": policy_red.state_dict(),
                "policy_class": policy_red.__class__.__name__,
                "policy_kwargs": policy_kwargs
            }, save_path)

            print(f"Saved checkpoint to {save_path}")

            opponent_pool.append(save_path)
            if len(opponent_pool) > pool_size:
                opponent_pool.pop(0)

def train_league_ppo_real(
    name, policy,
    total_steps=3_000_000, rollout_len=4096,
    lr=3e-4, gamma=0.99, lam=0.95,
    epochs=10, batch_size=256,
    pool_size=20,
    self_play_prob=None,
    print_every=10_000, save_every=50_000,
    eval_win_window=20,         # number of recent matches to track per opponent
    difficulty_alpha=2.0,       # skew exponent for weighting (>=1 -> more skew)
    min_opponent_weight=1e-3,   # don't let weight go to zero
):
    env = FootballMultiAgentEnv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    # Load or construct policy
    if isinstance(policy, tuple):
        pname, kwargs = policy
        policy_red = make_policy(pname, **kwargs)
        policy_kwargs = kwargs
    else:
        policy_red, checkpoint = policy_from_checkpoint_path(policy)
        policy_kwargs = checkpoint["policy_kwargs"]

    optimizer = optim.Adam(policy_red.parameters(), lr=lr)

    # ---------- Opponent pool + bookkeeping ----------
    opponent_pool = []  # list of checkpoint file paths
    # For each opponent checkpoint path we keep a deque of last eval_win_window results (1 = learner win, 0 = loss)
    win_history = defaultdict(lambda: deque(maxlen=eval_win_window))

    if self_play_prob is None:
        self_play_prob = 1.0 / (pool_size + 1)

    def compute_win_rate(opponent_path):
        hist = win_history.get(opponent_path, None)
        if hist is None or len(hist) == 0:
            return 0.5  # unknown opponents treated as 50/50 initially
        return float(sum(hist)) / len(hist)

    def opponent_weight(opponent_path):
        # weight should be higher when learner has LOWER win rate vs that opponent
        win_rate = compute_win_rate(opponent_path)
        difficulty = 1.0 - win_rate
        # exponentiate to control skew; ensure nonzero
        return max(min_opponent_weight, (difficulty ** difficulty_alpha))

    def select_opponent_weighted():
        # If no opponents yet, return DummyPolicy
        if len(opponent_pool) == 0:
            return None  # signal to use DummyPolicy

        # With small probability pick self (self-play)
        if random.random() < self_play_prob:
            return "self"

        # weighted sample from opponent_pool by difficulty
        weights = [opponent_weight(p) for p in opponent_pool]
        total = sum(weights)
        if total <= 0:
            # fallback to uniform
            choice = random.choice(opponent_pool)
            return choice
        probs = [w / total for w in weights]
        choice = random.choices(opponent_pool, weights=probs, k=1)[0]
        return choice

    def load_opponent_from_path(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        opponent = make_policy(ckpt["policy_class"], **ckpt["policy_kwargs"])
        opponent.load_state_dict(ckpt["policy_state_dict"])
        opponent.eval()
        return opponent

    def record_result_vs_opponent(opponent_path, learner_score):
        """
        learner_score: numeric performance metric from rollout vs opponent (higher = better).
        For simplicity we treat learner_score > 0 as win; customize as needed.
        """
        if opponent_path is None or opponent_path == "self":
            return
        win = 1 if learner_score > 0 else 0
        win_history[opponent_path].append(win)

    # ---------- training loop ----------
    steps = 0
    rewards_save = []

    while steps < total_steps:
        # pick and load opponent
        opp_choice = select_opponent_weighted()
        if opp_choice is None:
            policy_blue = DummyPolicy()
            opp_path_for_record = None
        elif opp_choice == "self":
            policy_blue = policy_red
            opp_path_for_record = "self"
        else:
            policy_blue = load_opponent_from_path(opp_choice)
            opp_path_for_record = opp_choice

        # -------- GET ROLLOUT DATA --------
        roll = rollout(
            env,
            policy_red,
            policy_blue,
            select_drill=lambda: None,
            rollout_len=rollout_len,
            gamma=gamma, lam=lam
        )

        obs      = roll["obs"]
        actions  = roll["acts"]
        old_logps= roll["logp"]
        rewards  = roll["rew"]
        dones    = roll["done"]
        values   = roll["val"]
        last_val = roll["last_val"]

        # Compute a simple scalar metric of learner performance vs opponent over this rollout:
        # you can replace this with per-episode result processing if you want.
        learner_score = float(np.mean(rewards))  # average time-step reward across rollout
        record_result_vs_opponent(opp_path_for_record, learner_score)

        steps += rollout_len

        adv = compute_gae(rewards, values, dones, last_val, gamma, lam)
        ret = adv + torch.tensor(values, dtype=torch.float32)

        ppo_update(
            policy_red, optimizer,
            obs, actions, old_logps, adv, ret,
            epochs=epochs, batch_size=batch_size
        )

        rewards_save.append(sum(rewards)/len(rewards))
        if steps % print_every < rollout_len:
            print(f"[{steps - (steps % print_every)}] PPO update | mean reward = {sum(rewards_save)/len(rewards_save):.3f}")
            rewards_save = []

        # ---------- periodic save & add to pool ----------
        if steps % save_every < rollout_len:
            save_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_{steps - (steps % save_every)}.pth"
            )
            torch.save({
                "policy_state_dict": policy_red.state_dict(),
                "policy_class": policy_red.__class__.__name__,
                "policy_kwargs": policy_kwargs
            }, save_path)

            print(f"Saved checkpoint to {save_path}")

            # add to opponent pool
            opponent_pool.append(save_path)
            if len(opponent_pool) > pool_size:
                # drop oldest from both pool and win_history
                removed = opponent_pool.pop(0)
                if removed in win_history:
                    del win_history[removed]


###############################################################
# ========================= MAIN ============================ #
###############################################################

if __name__ == "__main__":
    # train_drill_ppo(
    #     name="misc_drill",
    #     policy=("CurriculumMLPPolicy", {}),
    #     select_drill=lambda: random.choices([
    #         {"drill": "block"},
    #         {"drill": "block_nobounce"},
    #         {"drill": "shoot_left", "par": random.uniform(-1, -40/150)},
    #         {"drill": "shoot_right", "par": random.uniform(-1, -40/150)},
    #         {"drill": "hit_left_wall", "par": random.uniform(-40/150, 1)},
    #         {"drill": "hit_right_wall", "par": random.uniform(-40/150, 1)},
    #         {"drill": "volley", "par": random.uniform(-0.9, 0.3)},
    #         {"drill": "prepare", "par": random.uniform(-1, 1)},
    #     ],
    #     [0.1, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1],
    #     )[0],
    #     total_steps=30_000_000,
    #     rollout_len=2048,
    #     print_every=10_000,
    #     save_every=100_000,
    # )
    # train_league_ppo(
    #     name="league_ppo_regular (score reward)",
    #     policy=("RegularMLPPolicy", {}),
    #     total_steps=30_000_000,
    #     rollout_len=2048,
    #     print_every=10_000,
    #     save_every=100_000,
    # )
    train_league_ppo_real(
        name="league_ppo_regular_real (score reward) (latent_dims 128 128)",
        policy=("ActorCriticMLPPolicy", {}),
        total_steps=15_000_000,
        rollout_len=2048,
        print_every=10_000,
        save_every=100_000,
    )
