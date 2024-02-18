import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from global_vatiance import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        inner_dim = 512
        self.layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, NUM_ACTIONS),
        )
        self.conv.apply(self.init_weights)
        self.layers.apply(self.init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.layers(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # m.weight.data.normal_(0, 0.1)
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class ExperienceReplay:
    def __init__(self, capacity):
        self.MEMORY_CAPACITY = capacity
        self.memory_counter = 0
        self.memory = np.zeros((self.MEMORY_CAPACITY, NUM_STATES * 2 + 2), dtype=np.uint8)

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index] = transition
        self.memory_counter += 1

    def get_batch(self, batch_size):
        sample_index = np.random.choice(len(self), batch_size, replace=False)
        batch_memory = self.memory[sample_index]
        batch_state = torch.tensor(batch_memory[:, :NUM_STATES], dtype=torch.float32, device=DEVICE)
        batch_action = torch.tensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int), dtype=torch.long,
                                    device=DEVICE)
        batch_reward = torch.tensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2], dtype=torch.float32, device=DEVICE)
        batch_next_state = torch.tensor(batch_memory[:, -NUM_STATES:], dtype=torch.float32, device=DEVICE)
        batch_next_state_not_done_mask = ~torch.any(torch.isnan(batch_next_state), dim=1)
        return batch_state, batch_action, batch_reward, batch_next_state, batch_next_state_not_done_mask

    def get_cur_buffer(self):
        return self.memory[:len(self)]

    def __len__(self):
        return self.memory_counter if self.memory_counter < self.MEMORY_CAPACITY else self.MEMORY_CAPACITY


class DQN:
    def __init__(self):
        super(DQN, self).__init__()
        self.possible_actions = list(range(0, NUM_ACTIONS))
        self.eval_net = Net().to(DEVICE)
        self.target_net = Net().to(DEVICE)

        self.buffer = ExperienceReplay(MEMORY_CAPACITY)

        self.learning_rate = LR
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

    def get_action(self, state, epsilon):
        r = np.random.random()
        if r < epsilon:
            return np.random.choice(self.possible_actions, 1)[0], r
        with torch.no_grad():
            state = state.reshape(SHAPE_STATES)
            state = torch.tensor(state / 255.0, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            out = self.eval_net(state)
            max = out.argmax().item()
            prob = F.softmax(out[0])[max].item()
            return max, prob

    def update(self, num_steps=20):
        for _ in range(num_steps):
            batch_state, batch_action, batch_reward, batch_next_state, next_not_done_mask = self.buffer.get_batch(
                BATCH_SIZE)

            next_not_done_states = batch_next_state[next_not_done_mask]
            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
            q_out = self.eval_net((batch_state / 255.0).view(INP_SHAPE))
            q_a = q_out.gather(1, batch_action)
            with torch.no_grad():
                next_state_values[next_not_done_mask] = self.target_net(
                    (next_not_done_states / 255.0).view((next_not_done_states.shape[0],) + SHAPE_STATES)).max(1)[0]
            target = batch_reward + GAMMA * next_state_values.unsqueeze(1)

            loss = F.huber_loss(q_a, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
