import gymnasium as gym
import ale_py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random

# set seeds
seed = 42
torch.manual_seed(seed)
random.seed(seed)

env = gym.make("ALE/Pong-v5")
env.action_space.seed(seed)

# print some information
print('Action space:', env.action_space)
print('Observation space:', env.observation_space)
print('Max episode steps:', env.spec.max_episode_steps)


# ==Classes=========================================================================================================


class DQN(nn.Module):
    def __init__(self, output_dim, epsilon_start, epsilon_end, epsilon_decay, env, device):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        h = conv2d_size(conv2d_size(conv2d_size(84, 8, 4), 4, 2), 3, 1)
        w = conv2d_size(conv2d_size(conv2d_size(84, 8, 4), 4, 2), 3, 1)

        self.fc1 = nn.Linear(64 * h * w, 512)
        self.fc2 = nn.Linear(512, output_dim)

        self.n_steps = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.device = device

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state):
        self.n_steps += 1
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.n_steps / self.epsilon_decay)
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.env.action_space.n)]],
                                dtype=torch.long, device=self.device)
        return action


class FrameProcessor:
    def __init__(self, device, height=84, width=84):
        self.device = device
        self.height = height
        self.width = width
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((height, width)),
            T.ToTensor(),
        ])
        self.prev_frame = None

    def reset(self):
        self.prev_frame = None

    def process(self, frame):
        current_frame = self.transform(frame)
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return torch.zeros_like(current_frame).unsqueeze(0)

        frame_diff = current_frame - self.prev_frame
        self.prev_frame = current_frame

        return frame_diff.unsqueeze(0)


from dqn import ReplayBuffer, train_dqn


# ==Hyperparams and Train=========================================================================================================


# Hyperparams
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
target_update_freq = 1_000
tau = 0  # No polyak averaging seemed to do okay
gamma = 0.99

lr = 1e-4
batch_size = 64
buffer_size = 100_000  # Big buffer size
init_buffer_size = 10_000
eps_init, eps_final, eps_decay = 1.0, 0.02, 100_000  # exponential decay. Super slow decay was key
n_episodes = 1_000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policy_net = DQN(
    output_dim=env.action_space.n,
    epsilon_start=eps_init,
    epsilon_end=eps_final,
    epsilon_decay=eps_decay,
    env=env,
    device=device
).to(device)

target_net = DQN(
    output_dim=env.action_space.n,
    epsilon_start=eps_init,
    epsilon_end=eps_final,
    epsilon_decay=eps_decay,
    env=env,
    device=device
).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=lr)
criterion = nn.SmoothL1Loss()  # Huber loss
replay_buffer = ReplayBuffer(buffer_size)

frame_processor = FrameProcessor(device)


policy_net, all_rewards, all_losses = train_dqn(
                                        policy_net,
                                        target_net,
                                        optimizer,
                                        criterion,
                                        env,
                                        n_episodes=n_episodes,
                                        replay_buffer=replay_buffer,
                                        tau=tau,
                                        batch_size=batch_size,
                                        gamma=gamma,
                                        target_update_freq=1,
                                        log_dir='./tensorboard',
                                        device=device,
                                        modifier=frame_processor,
                                        is_pong=True,
                                        init_buffer_size=init_buffer_size,
                                        save_every=10,
                                        save_path='pong_dqn.pth'
                                    )
