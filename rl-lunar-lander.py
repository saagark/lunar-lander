import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import time
import random

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple(
    "Transition",
    ("state",
     "action",
     "reward",
     "next_state",
     "done"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


def main():
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5)


if __name__ == '__main__':

    t = time.time()
    main()
    elapsed = time.time() - t
    mins, secs = divmod(elapsed, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")
