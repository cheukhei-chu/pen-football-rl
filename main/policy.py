import os
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

# -------------------------
# Abstract base policy
# -------------------------
class FootballPolicy(nn.Module, ABC):
    """
    Abstract base class for football policies.
    Subclasses must implement forward() and sample_action().
    forward() must return a dict containing:
        - "left", "right", "jump": logits (un-normalized) for each discrete action
        - "value": a (batch, 1) tensor of state values
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs: torch.Tensor):
        """Given an observation tensor (batch, obs_dim), return dict of logits + value."""
        raise NotImplementedError

    @abstractmethod
    def sample_action(self, obs):
        """Given a single observation (numpy array or torch tensor), return sampled action dict."""
        raise NotImplementedError


# -------------------------
# Small utility for sampling
# -------------------------
def _to_tensor(obs):
    """Convert obs (np array or torch tensor) to a batched torch.float32 tensor on CPU."""
    if isinstance(obs, torch.Tensor):
        t = obs
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.float()
    else:
        return torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)


# -------------------------
# MLPPolicy (with value head)
# -------------------------
class MLPPolicy(FootballPolicy):
    def __init__(self, obs_dim=12, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(hidden_dim, 2)
        self.head_right = nn.Linear(hidden_dim, 2)
        self.head_jump  = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        returns dict with logits for each action and "value": (batch,1)
        """
        x = self.net(obs)
        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D torch tensor)."""
        obs_t = _to_tensor(obs)
        with torch.no_grad():
            logits = self.forward(obs_t)
            # logits values are batched; take index 0
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out


# -------------------------
# DummyPolicy (deterministic zeros, returns logits & value)
# -------------------------
class DummyPolicy(FootballPolicy):
    def __init__(self, obs_dim=12):
        super().__init__()
        # We'll create simple constant logits and zero value
        # so forward() works in rollouts
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor):
        batch = obs.shape[0]
        device = obs.device
        # constant logits: prefer action 0 (logits [1.0, 0.0]) but it's arbitrary
        left = torch.zeros(batch, 2, device=device)
        right = torch.zeros(batch, 2, device=device)
        jump = torch.zeros(batch, 2, device=device)
        # small bias to first action
        left[:, 0] = 1.0
        right[:, 0] = 1.0
        jump[:, 0] = 1.0
        value = torch.zeros(batch, 1, device=device)
        return {"left": left, "right": right, "jump": jump, "value": value}

    def sample_action(self, obs):
        # deterministic no-op
        return {"left": 0, "right": 0, "jump": 0}


# -------------------------
# atulPolicy (simple hand-coded policy)
# -------------------------
class atulPolicy(FootballPolicy):
    def __init__(self, obs_dim=12):
        super().__init__()
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor):
        # Return logits that correspond to the actions produced by sample_action.
        # We'll produce batched logits with mild preference for the sampled action.
        batch = obs.shape[0]
        device = obs.device
        left = torch.zeros(batch, 2, device=device)
        right = torch.zeros(batch, 2, device=device)
        jump = torch.zeros(batch, 2, device=device)
        value = torch.zeros(batch, 1, device=device)

        # evaluate heuristic in batch
        # obs layout (per your doc): [rx, ry, rvx, rvy, bx, by, bvx, bvy, ballx, bally, ballvx, ballvy]
        # but original doc used indices 8-11 for ball; keep same indexing
        rx = obs[:, 0]
        ballx = obs[:, 8]
        ballvy = obs[:, 11].abs()

        # if ball is near and ball is to the right of red, prefer move right
        cond = (ballvy < 0.2) & (ballx > rx)
        # set preference
        right[cond, 1] = 1.0  # prefer right action (index 1)
        # otherwise do nothing (left/right/jump pref index 0)
        return {"left": left, "right": right, "jump": jump, "value": value}

    def sample_action(self, obs):
        # Accept numpy or tensor single observation
        arr = np.asarray(obs)
        # indexes: as in your doc
        ballvy = abs(arr[11])
        ballx = arr[8]
        rx = arr[0]
        if (ballvy < 0.2) and (ballx > rx):
            return {"left": 0, "right": 1, "jump": 0}
        return {"left": 0, "right": 0, "jump": 0}


# -------------------------
# CurriculumMLPPolicy (fixed)
# -------------------------
class CurriculumMLPPolicy(FootballPolicy):
    def __init__(self, embed_dim=3, obs_dim=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim

        # plan_net used when no explicit setting is provided
        self.plan_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # index embedding for tasks
        self.index_embed = nn.Embedding(10, embed_dim)  # support up to 10 tasks

        # combine (index_emb + par_scalar) -> task embedding
        self.embed_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 20),
            nn.ReLU(),
            nn.Linear(20, embed_dim),
            nn.ReLU(),
        )

        # action network: uses a subset of obs + task embedding
        # subset length = 8 (indices [0,1,2,3,8,9,10,11])
        self.action_net = nn.Sequential(
            nn.Linear(8 + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(128, 2)
        self.head_right = nn.Linear(128, 2)
        self.head_jump  = nn.Linear(128, 2)

        # value head
        self.value_head = nn.Linear(128, 1)

        # setting holds current drill dict (or None)
        self.setting = None

        # mapping from drill name to small integer index (keep in sync with index_embed size)
        self.drill_to_index = {
            "block_nobounce": 1,
            "shoot_left": 2,
        }

    def set_setting(self, drill):
        """drill is expected to be a dict, e.g. {'drill': 'shoot_left', 'par': 0.1}"""
        self.setting = drill

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        Returns dict with "left","right","jump" logits and "value" (batch,1)
        """
        batch = obs.shape[0]
        device = obs.device

        if self.setting is None:
            # use plan_net applied to the observation to compute a task embedding
            task_emb = self.plan_net(obs)  # (batch, embed_dim)
        else:
            # build index embedding + par scalar and run through embed_net
            drill_name = self.setting.get("drill")
            idx = self.drill_to_index.get(drill_name, 0)
            idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)  # shape (1,)
            idx_emb = self.index_embed(idx_tensor).float()  # (1, embed_dim)
            # expand to batch
            idx_emb = idx_emb.repeat(batch, 1)  # (batch, embed_dim)

            par_val = float(self.setting.get("par", 0.0))
            par_tensor = torch.tensor([[par_val]], dtype=torch.float32, device=device).repeat(batch, 1)  # (batch,1)

            embed_input = torch.cat([idx_emb, par_tensor], dim=-1)  # (batch, embed_dim+1)
            task_emb = self.embed_net(embed_input)  # (batch, embed_dim)

        # pick the relevant obs features: [0,1,2,3,8,9,10,11]
        obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]  # (batch, 8)

        x = self.action_net(torch.cat([obs_subset, task_emb], dim=-1))  # (batch, 128)

        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D tensor)."""
        obs_t = _to_tensor(obs)  # (1, obs_dim)
        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out

# -------------------------
# RegularMLPPolicy (simple MLP using same obs subset as Curriculum, with value head)
# -------------------------
class RegularMLPPolicy(FootballPolicy):
    def __init__(self, obs_dim=12):
        """
        A plain MLP policy that uses the same 8-element observation subset as
        CurriculumMLPPolicy (indices [0,1,2,3,8,9,10,11]) but has no planning/embedding.
        Produces logits for left/right/jump and a scalar value.
        """
        super().__init__()
        self.obs_dim = obs_dim

        # action network: takes the 8-element subset and produces a 128-d representation
        self.action_net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(128, 2)
        self.head_right = nn.Linear(128, 2)
        self.head_jump  = nn.Linear(128, 2)

        # value head on top of same representation
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        returns dict with logits for each action and "value": (batch,1)
        """
        # pick the relevant obs features: [0,1,2,3,8,9,10,11]
        obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]  # (batch, 8)
        x = self.action_net(obs_subset)  # (batch, 128)

        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D tensor)."""
        obs_t = _to_tensor(obs)  # (1, obs_dim)
        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out
    
    def set_setting(self,setting):
        pass

class CurriculumMLPPolicyScaled(FootballPolicy):
    def __init__(self, latent_dims=[128, 128], embed_dim=3, obs_dim=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim

        # plan_net used when no explicit setting is provided
        self.plan_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # index embedding for tasks
        self.index_embed = nn.Embedding(10, embed_dim)  # support up to 10 tasks

        # combine (index_emb + par_scalar) -> task embedding
        self.embed_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 20),
            nn.ReLU(),
            nn.Linear(20, embed_dim),
            nn.ReLU(),
        )

        # action network: uses a subset of obs + task embedding
        # subset length = 8 (indices [0,1,2,3,8,9,10,11])
        dims = [8 + embed_dim] + latent_dims
        self.nets = []
        for i in range(len(latent_dims)):
            self.nets.append(nn.Linear(dims[i], dims[i+1]))
            self.nets.append(nn.ReLU())
        self.action_net = nn.Sequential(*self.nets)
        self.head_left  = nn.Linear(latent_dims[-1], 2)
        self.head_right = nn.Linear(latent_dims[-1], 2)
        self.head_jump  = nn.Linear(latent_dims[-1], 2)

        # value head
        self.value_head = nn.Linear(latent_dims[-1], 1)

        # setting holds current drill dict (or None)
        self.setting = None

        # mapping from drill name to small integer index (keep in sync with index_embed size)
        self.drill_to_index = {
            "block_nobounce": 1,
            "shoot_left": 2,
        }

    def set_setting(self, drill):
        """drill is expected to be a dict, e.g. {'drill': 'shoot_left', 'par': 0.1}"""
        self.setting = drill

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        Returns dict with "left","right","jump" logits and "value" (batch,1)
        """
        batch = obs.shape[0]
        device = obs.device

        if self.setting is None:
            # use plan_net applied to the observation to compute a task embedding
            task_emb = self.plan_net(obs)  # (batch, embed_dim)
        else:
            # build index embedding + par scalar and run through embed_net
            drill_name = self.setting.get("drill")
            idx = self.drill_to_index.get(drill_name, 0)
            idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)  # shape (1,)
            idx_emb = self.index_embed(idx_tensor).float()  # (1, embed_dim)
            # expand to batch
            idx_emb = idx_emb.repeat(batch, 1)  # (batch, embed_dim)

            par_val = float(self.setting.get("par", 0.0))
            par_tensor = torch.tensor([[par_val]], dtype=torch.float32, device=device).repeat(batch, 1)  # (batch,1)

            embed_input = torch.cat([idx_emb, par_tensor], dim=-1)  # (batch, embed_dim+1)
            task_emb = self.embed_net(embed_input)  # (batch, embed_dim)

        # pick the relevant obs features: [0,1,2,3,8,9,10,11]
        obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]  # (batch, 8)

        x = self.action_net(torch.cat([obs_subset, task_emb], dim=-1))  # (batch, 128)

        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D tensor)."""
        obs_t = _to_tensor(obs)  # (1, obs_dim)
        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out

# -------------------------
# Factory + checkpoint loader
# -------------------------
def make_policy(class_name, **kwargs):
    """Initialize policy class given class name and kwargs."""
    name_to_class = {
        "MLPPolicy": MLPPolicy,
        "CurriculumMLPPolicy": CurriculumMLPPolicy,
        "RegularMLPPolicy":RegularMLPPolicy,
        "DummyPolicy": DummyPolicy,
        "atulPolicy": atulPolicy,
        "CurriculumMLPPolicyScaled": CurriculumMLPPolicyScaled,
    }
    if class_name in name_to_class:
        return name_to_class[class_name](**kwargs)
    raise ValueError(f"Unknown policy: {class_name}")


def policy_from_checkpoint_path(checkpoint_path):
    assert os.path.exists(checkpoint_path), f"Error: Checkpoint file not found at {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # older checkpoints might not include policy_kwargs; handle gracefully
    policy_kwargs = checkpoint.get("policy_kwargs", {})
    policy_class_name = checkpoint.get("policy_class") or checkpoint.get("policy_name") or "MLPPolicy"

    policy = make_policy(policy_class_name, **policy_kwargs)

    state = checkpoint.get("policy_state_dict") or checkpoint.get("state_dict")
    if state:
        policy.load_state_dict(state)

    return policy, checkpoint
