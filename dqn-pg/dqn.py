import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time

# set seeds
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ReplayBuffer():

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def len(self):
        return len(self.memory)


# ==Train Function=========================================================================================================

def train_dqn(
    policy_net,
    target_net,
    optimizer,
    criterion,
    env,
    n_episodes,
    replay_buffer,
    tau,
    batch_size,
    gamma,
    target_update_freq,
    log_dir='./tensorboard',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    modifier=None,
    solved=None,
    seed=42,
    is_pong=False,
    init_buffer_size=None,
    save_every=None,
    save_path=None
):
    writer = SummaryWriter(log_dir)
    policy_net.to(device)
    target_net.to(device)
    policy_net.train()
    running_loss_size = 10
    all_rewards, all_losses = [], []
    best_reward = -np.inf
    pbar = tqdm(range(n_episodes), desc='Training')

    for episode in pbar:
        if solved is not None:
            avg, last_n = solved
            if len(all_rewards) >= last_n and sum(all_rewards[-last_n:]) / last_n >= avg:
                print(f'Solved in {episode} episodes!')
                break

        if modifier is not None:
            modifier.reset()
        state, _ = env.reset(seed=(seed + episode))

        if is_pong:
            
            if modifier is None:
                processed_state = torch.FloatTensor(state).unsqueeze(0)
            else:
                processed_state = modifier.process(state)

            frame_stack = [processed_state] * 4  # Initialize frame stack with initial state repeated 4 times
            state = torch.cat(frame_stack, dim=1)
        else:
            if modifier is None:
                state = torch.FloatTensor(state).unsqueeze(0)
            else:
                state = modifier.process(state)

        episode_reward, episode_loss, n_steps = 0, 0, 0

        # --Collect Experiences----------------------------------------------------------------------------------------------
        done = False
        while not done:
            state = state.to(device)
            action = policy_net.act(state)
            state = state.cpu()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            if is_pong:
                if modifier is None:
                    processed_next_state = torch.FloatTensor(next_state).unsqueeze(0)
                else:
                    processed_next_state = modifier.process(next_state)

                frame_stack.pop(0)
                frame_stack.append(processed_next_state)
                next_state = torch.cat(frame_stack, dim=1)
            else:
                if modifier is None:
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)
                else:
                    next_state = modifier.process(next_state)

            reward = torch.FloatTensor([reward])
            done = torch.BoolTensor([done])

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward.item()
            n_steps += 1

            # --Replay Experiences----------------------------------------------------------------------------------------------
            if init_buffer_size is None:
                init_buffer_size = batch_size
            if replay_buffer.len() > init_buffer_size:
                transitions = replay_buffer.sample(batch_size)

                batch = tuple(zip(*transitions))
                states, actions, rewards, next_states, dones = torch.cat(batch[0]).to(device), torch.cat(batch[1]).to(device), torch.cat(batch[2]).to(device), torch.cat(batch[3]).to(device), torch.cat(batch[4]).to(device)

                current_q_values = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards.unsqueeze(1) + gamma * next_q_values * ~dones.unsqueeze(1)

                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                episode_loss += loss.item()

                with torch.no_grad():
                    if n_steps % target_update_freq == 0:
                        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                            target_param.data.copy_(tau * target_param.data + (1.0 - tau) * policy_param.data)

        all_rewards.append(episode_reward)
        all_losses.append(episode_loss)

        pbar.set_postfix({
            'Reward': sum(all_rewards[-running_loss_size:]) / len(all_rewards[-running_loss_size:]),
            'Loss': sum(all_losses[-running_loss_size:]) / len(all_losses[-running_loss_size:])
        })

        if save_every is not None and episode % save_every == 0:
            torch.save(policy_net.state_dict(), save_path)
            print(f'Model saved at episode {episode}')

        cur_reward = sum(all_rewards[-running_loss_size:]) / len(all_rewards[-running_loss_size:])
        if cur_reward > best_reward:
            best_reward = cur_reward
            if save_path is not None:
                best_save_path_name = save_path.split('.')[0] + '_best.pth'
                torch.save(policy_net.state_dict(), best_save_path_name)
                print(f'Best model saved at episode {episode}')

        writer.add_scalar('Reward', episode_reward, episode)
        writer.add_scalar('Loss', episode_loss, episode)

    writer.close()
    return policy_net, all_rewards, all_losses
