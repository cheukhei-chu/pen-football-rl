import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pygame
import random
from abc import ABC, abstractmethod

class FootballPolicy(nn.Module, ABC):
    """
    Abstract base class for football policies.
    Subclasses must implement forward() and sample_action().
    """
    def __init__(self):
        super().__init__()

    # @abstractmethod
    # def forward(self, obs):
    #     """Given an observation (tensor), return the raw action logits/scores."""
    #     pass

    @abstractmethod
    def sample_action(self, obs):
        """Given an observation (numpy array or tensor), return a sampled action."""
        pass

class MLPPolicy(FootballPolicy):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(128, 2)
        self.head_right = nn.Linear(128, 2)
        self.head_jump  = nn.Linear(128, 2)

    def forward(self, obs):
        x = self.net(obs)
        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
        }

    def sample_action(self, obs):
        """Sample action from the policy given a numpy observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs_t)
        return {
            k: torch.distributions.Categorical(logits=v).sample().item()
            for k, v in logits.items()
        }

class DummyPolicy(FootballPolicy):
    def __init__(self):
        super().__init__()

    def sample_action(self, obs):
        return {
            "left":  0,
            "right": 0,
            "jump":  0,
        }

class atulPolicy(FootballPolicy):
    def __init__(self):
        super().__init__()

    def sample_action(self, obs):
        '''
        Format of observation (the variable obs) is as follows:

        indices 0 through 3 contain x_pos, y_pos, x_vel, y_vel for red
        indices 4 through 7 contain this for blue
        indices 8 through 11 contain this for ball
        '''
        if abs(obs[11])<0.2 and obs[8] > obs[0]:
            return {"left":0, "right":1, "jump":0}
        return {"left":0,"right":0,"jump":0}

class CurriculumMLPPolicy(FootballPolicy):
    def __init__(self, embed_dim=3):
        super().__init__()
        self.plan_net = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )
        self.index_embed = nn.Embedding(10, embed_dim)
        self.embed_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 20),
            nn.ReLU(),
            nn.Linear(20, embed_dim),
            nn.ReLU(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(12 + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(128, 2)
        self.head_right = nn.Linear(128, 2)
        self.head_jump  = nn.Linear(128, 2)

    def forward(self, obs, index=None, par=None):
        if index:
            assert par is not None, "par is None"
            index_emb = torch.cat([self.index_emb(index), par], dim=-1)
            task_emb = self.embed_net(index_emb)
        else:
            task_emb = self.plan_net(obs)
        x = self.action_net(torch.cat([obs, task_emb], dim=-1))
        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
        }

    def sample_action(self, obs):
        """Sample action from the policy given a numpy observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(obs_t)
        return {
            k: torch.distributions.Categorical(logits=v).sample().item()
            for k, v in logits.items()
        }

def make_policy(class_name, **kwargs):
    """Initialize policy class given class name and kwargs."""
    name_to_class = {
        "MLPPolicy": MLPPolicy,
        "CurriculumMLPPolicy": CurriculumMLPPolicy
    }
    if class_name in name_to_class:
        return name_to_class[class_name](**kwargs)
    raise ValueError("Unknown policy:", class_name)
