import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import time
import random
from gymnasium.wrappers import RecordVideo
import os

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

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
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        # Kaiming (He) initialization for all Linear layers
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def select_action(state, policy_net, eps, action_size):
    if random.random() > eps:
        # Exploitation
        with torch.no_grad():
            # print(state)
            state = torch.Tensor(state).to(device)
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
    num_episodes = 700            # More episodes for stable learning
    max_t = 10000                 # Max steps per episode
    batch_size = 64               # Standard batch size
    gamma = 0.99                  # Discount factor
    lr = 1e-3                     # Learning rate (higher for faster learning)
    memory_size = 100000          # Large replay buffer
    eps_start = 1.0               # Initial epsilon for exploration
    eps_end = 0.01                # Final epsilon
    eps_decay = 0.995             # Epsilon decay rate
    min_memory_for_training = 1000  # Minimum buffer size before training
    render_every = 100
    scores_window = deque(maxlen=100)
    tau = 0.005                   # Slightly higher tau for faster soft update

    chkpt_save_dir = "artifacts"

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size, seed).to(device)
    target_net = DQN(state_size, action_size, seed).to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayMemory(memory_size)
    eps = eps_start
    writer = SummaryWriter()

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        total_reward = 0
        losses = []
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
                    # Bellman equation
                    target = rewards + (gamma * next_q_values * (1 - dones))

                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                # Soft update of target network
                with torch.no_grad():
                    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                        target_param.data.copy_(
                            tau * policy_param.data + (1.0 - tau) * target_param.data)

            if done:
                break

        # Record video
        if (episode + 1) % render_every == 0:
            video_env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                                 enable_wind=False, wind_power=15.0, turbulence_power=1.5,
                                 render_mode="rgb_array")
            video_env = RecordVideo(video_env, video_folder="videos",
                                    episode_trigger=lambda ep: True, name_prefix=f"episode_{episode+1}")
            state, _ = video_env.reset(seed=seed)
            total_reward_video = 0
            for _ in range(max_t):
                action = select_action(
                    [state], policy_net, 0.0, action_size)  # Greedy
                next_state, reward, terminated, truncated, _ = video_env.step(
                    action)
                state = next_state
                total_reward_video += reward
                if terminated or truncated:
                    break
            print(
                f"Saved video for episode {episode+1}, reward: {total_reward_video:.2f}")
            video_env.close()

        scores_window.append(total_reward)
        eps = max(eps_end, eps_decay * eps)
        avg_reward = np.mean(scores_window)
        avg_loss = np.mean(losses) if losses else 0
        writer.add_scalar('Reward/episode', total_reward, episode)
        writer.add_scalar('Reward/avg100', avg_reward, episode)
        writer.add_scalar('Loss/avg', avg_loss, episode)
        writer.add_scalar('Epsilon', eps, episode)
        print(
            f"Episode {episode}\tReward: {total_reward:.2f}\tAverage100: {avg_reward:.2f}\tEpsilon: {eps:.3f}")

    writer.close()

    final_ckpt = {
        "episode": episode,
        "avg_reward": float(avg_reward),
        "eps": float(eps),
        "seed": seed,
        "state_dict": policy_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    os.makedirs(chkpt_save_dir, exist_ok=True)
    torch.save(final_ckpt, os.path.join(chkpt_save_dir, "dqn_final.pth"))

    # Save a video of the trained agent
    print("\nSaving video of the trained agent...")
    video_env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                         enable_wind=False, wind_power=15.0, turbulence_power=1.5,
                         render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_folder="videos",
                            episode_trigger=lambda ep: True, name_prefix="final_agent")
    state, _ = video_env.reset(seed=seed)
    total_reward_video = 0
    for t in range(max_t):
        action = select_action([state], policy_net, 0.0, action_size)  # Greedy
        next_state, reward, terminated, truncated, _ = video_env.step(action)
        state = next_state
        total_reward_video += reward
        if terminated or truncated:
            break
    print(f"Saved final agent video, reward: {total_reward_video:.2f}")
    video_env.close()


if __name__ == '__main__':

    t = time.time()
    main()
    elapsed = time.time() - t
    mins, secs = divmod(elapsed, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")
