import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # 使用 torch.stack 来堆叠每个元素
        state, action, reward, next_state, done = zip(*batch)
        state = torch.stack(state)
        action = torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        next_state = torch.stack(next_state)
        done = torch.tensor(done).to(device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)