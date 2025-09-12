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
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


def select_action(state, policy_net, eps, action_size):
    if random.random() > eps:
        # Exploitation
        with torch.no_grad():
            print(state)
            state = torch.Tensor(state)
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()
    else:
        # Exploration
        action = random.choice(np.arange(action_size))
    return action


def main():
    seed = 42  # for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Device:", device)

    # Hyperparameters
    max_t = 1000
    batch_size = 64
    gamma = 0.99
    lr = 3e-4
    memory_size = 100000
    target_update = 10
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    min_memory_for_training = 1000
    render = False
    render_every = 50
    solved_score = 200
    scores_window = deque(maxlen=100)

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size, seed).to(device)
    target_net = DQN(state_size, action_size, seed).to(device)

    state, _ = env.reset(seed=seed)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayMemory(memory_size)
    eps = eps_start
    state, _ = env.reset(seed=seed)
    total_reward = 0
    for t in range(max_t):
        action = select_action([state], policy_net, eps, action_size)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) > min_memory_for_training:
            transitions = buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            states = torch.tensor(
                batch.state, dtype=torch.float32).to(device)
            actions = torch.tensor(
                batch.action, dtype=torch.int64).unsqueeze(1).to(device)
            rewards = torch.tensor(
                batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
            next_states = torch.tensor(
                batch.next_state, dtype=torch.float32).to(device)
            dones = torch.tensor(
                batch.done, dtype=torch.float32).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[
                    0].unsqueeze(1)
                target = rewards + (gamma * next_q_values * (1 - dones))

            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    env.close()


if __name__ == '__main__':

    t = time.time()
    main()
    elapsed = time.time() - t
    mins, secs = divmod(elapsed, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")
