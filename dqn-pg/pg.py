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


# ==Train Functions=========================================================================================================


def train_pg(
    policy_net,
    optimizer,
    env,
    n_episodes,
    gamma,
    batch_size=1,
    value_net=None,
    value_optimizer=None,
    log_dir='./tensorboard_pg',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    solved=None,
    seed=42,
    save_every=None,
    save_path=None,
    temporal=False,
    advantage=False,
):
    writer = SummaryWriter(log_dir)
    policy_net.to(device)
    if value_net:
        value_net.to(device)
    all_rewards, all_losses = [], []
    pbar = tqdm(range(n_episodes), desc='Training')

    for episode in range(0, n_episodes):
        batch_rewards, batch_states, batch_actions = [], [], []
        full_rewards = []
        for b in range(batch_size):
            rewards, states, actions = [], [], []
            with torch.no_grad():
                state, _ = env.reset(seed=(seed + episode*batch_size + b))
                done = False
                while not done:
                    state = torch.FloatTensor(state).to(device)
                    states.append(state)
                    probs = policy_net(state)
                    action_distribution = torch.distributions.Categorical(probs=probs)
                    action = action_distribution.sample()

                    next_state, reward, terminated, truncated, _ = env.step(action.item())

                    rewards.append(reward)
                    actions.append(action)
                    state = next_state
                    done = terminated or truncated
            full_rewards.append(sum(rewards))

            # --Form Returns----------------------------------------------------------------------------------------------
            discounted_rewards = []
            running_add = 0
            for r in reversed(rewards):
                running_add = r + gamma * running_add
                discounted_rewards.insert(0, running_add)
            discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)

            batch_rewards.append(discounted_rewards)
            batch_states.append(states)
            batch_actions.append(actions)

            # Calculate advantage if required
            if advantage:
                states_copy = torch.stack([state.clone().detach() for state in states])
                values = value_net(states_copy).squeeze()
                advantages = discounted_rewards - values.clone().detach()  # A(s) = G - V(s)

                value_loss = F.mse_loss(values, discounted_rewards.clone().detach())
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
            else:
                advantages = discounted_rewards

            batch_rewards[-1] = advantages  # Store advantages instead of rewards if advantage is True

        # The total loss is the sum of the losses over each trajectory
        total_loss = 0
        for states, actions, advantages in zip(batch_states, batch_actions, batch_rewards):
            states_tensor = torch.stack(states).detach()
            actions_tensor = torch.tensor([a.item() for a in actions]).to(device)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

            probs = policy_net(states_tensor)
            selected_probs = probs[range(len(probs)), actions_tensor]

            if temporal:
                total_loss += -torch.sum(torch.log(selected_probs) * advantages_tensor)
            else:
                total_loss += -torch.sum(torch.log(selected_probs) * advantages_tensor[0])

        total_loss /= batch_size
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        episode_reward = sum(full_rewards) / batch_size
        all_rewards.append(episode_reward)
        all_losses.append(total_loss.item())

        writer.add_scalar('Reward', episode_reward, episode)
        writer.add_scalar('Loss', total_loss.item(), episode)

        pbar.set_description(f"Episode {episode + 1}/{n_episodes} | Reward: {np.mean(all_rewards[-100:])}, Loss: {np.mean(all_losses[-100:])}")

        if solved is not None and np.mean(all_rewards[-solved[1]:]) >= solved[0]:
            print(f"Solved in {episode + 1} episodes!")
            break

        if save_every is not None and (episode + 1) % save_every == 0 and save_path:
            torch.save(policy_net.state_dict(), save_path)

    writer.close()

    return policy_net, all_rewards, all_losses


# --A similar function just for advantage normalization for the lunar lander ;-/ ----------------------------------------------------------------------------------------------


def train_pg_lunar_lander(
    policy_net,
    optimizer,
    env,
    n_episodes,
    gamma,
    batch_size=1,
    log_dir='./tensorboard_pg',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    solved=None,
    seed=42,
    save_every=None,
    save_path=None,
    temporal=False,
    advantage=False,
):
    writer = SummaryWriter(log_dir)
    policy_net.to(device)
    all_rewards, all_losses = [], []
    pbar = tqdm(range(n_episodes), desc='Training')

    for episode in range(0, n_episodes):
        batch_rewards, batch_states, batch_actions = [], [], []
        full_rewards = []

        for b in range(batch_size):
            rewards, states, actions = [], [], []
            with torch.no_grad():
                state, _ = env.reset(seed=(seed + episode*batch_size + b))
                done = False
                while not done:
                    state = torch.FloatTensor(state).to(device)
                    states.append(state)
                    probs = policy_net(state)
                    action_distribution = torch.distributions.Categorical(probs=probs)
                    action = action_distribution.sample()

                    next_state, reward, terminated, truncated, _ = env.step(action.item())

                    rewards.append(reward)
                    actions.append(action)
                    state = next_state
                    done = terminated or truncated
            full_rewards.append(sum(rewards))

            discounted_rewards = []
            running_add = 0
            for r in reversed(rewards):
                running_add = r + gamma * running_add
                discounted_rewards.insert(0, running_add)
            discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)

            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)  # Normalize rewards
            batch_rewards.append(discounted_rewards)
            batch_states.append(states)
            batch_actions.append(actions)

        # Calculate the average return (avg(rewards[i])) for the batch
        max_len = max(len(rewards) for rewards in batch_rewards)
        padded_rewards = [torch.nn.functional.pad(rewards, (0, max_len - len(rewards))) for rewards in batch_rewards]
        avg_batch_return = [sum(rewards[i] for rewards in padded_rewards if i < len(rewards)) / batch_size for i in range(max_len)]
        batch_advantages = [[rewards[i] - avg_batch_return[i] for i in range(len(rewards))] for rewards in batch_rewards]

        total_loss = 0
        for states, actions, advantages in zip(batch_states, batch_actions, batch_advantages):
            states_tensor = torch.stack(states).detach()
            actions_tensor = torch.tensor([a.item() for a in actions]).to(device)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)  # Normalize advantages

            probs = policy_net(states_tensor)
            selected_probs = probs[range(len(probs)), actions_tensor]

            if temporal:
                total_loss += -torch.sum(torch.log(selected_probs) * advantages_tensor)
            else:
                total_loss += -torch.sum(torch.log(selected_probs) * advantages_tensor[0])

        total_loss /= batch_size
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        episode_reward = sum(full_rewards) / batch_size
        all_rewards.append(episode_reward)
        all_losses.append(total_loss.item())

        writer.add_scalar('Reward', episode_reward, episode)
        writer.add_scalar('Loss', total_loss.item(), episode)

        pbar.set_description(f"Episode {episode + 1}/{n_episodes} | Reward: {np.mean(all_rewards[-100:])}, Loss: {np.mean(all_losses[-100:])}")

        if solved is not None and np.mean(all_rewards[-solved[1]:]) >= solved[0]:
            print(f"Solved in {episode + 1} episodes!")
            break

        if save_every is not None and (episode + 1) % save_every == 0 and save_path:
            torch.save(policy_net.state_dict(), save_path)

    writer.close()

    return policy_net, all_rewards, all_losses
