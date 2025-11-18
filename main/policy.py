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

def make_policy(class_name, **kwargs):
    if class_name == "MLPPolicy":
        return MLPPolicy(**kwargs)
    elif class_name == "DummyPolicy":
        return DummyPolicy(**kwargs)
    raise ValueError("Unknown policy:", class_name)
