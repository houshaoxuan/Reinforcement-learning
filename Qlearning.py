import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 5

def get_states(states):
    states = states[0]
    return torch.stack([torch.FloatTensor(state).permute(2, 0, 1) for state in states]).to(device)

def get_next_states(next_states):
    return torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)

class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2),  # 输出: [32, 103, 78]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),  # 输出: [64, 51, 38]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),  # 输出: [64, 25, 18]
            nn.ReLU(),
            nn.Flatten(),  # 展平后的大小为 64*25*18 = 28800
            nn.Linear(28800, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)  # 输出动作维度
        )

    def forward(self, x):
        return self.model(x)

class QlearningNN:
    def __init__(self, env):
        self.env = env
        shape = env.single_observation_space.shape
        self.state_size = [shape[2], shape[0], shape[1]]
        self.action_size = env.single_action_space.n
        self.alpha = 0.001  # 学习率
        self.gamma = 0.6  # 折扣因子
        self.epsilon = 0.1  # 探索率
        self.batch_size = batch_size
        self.model = QNetwork(self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def choose_action(self, state_batch):
        if np.random.uniform() < self.epsilon:
            return [self.env.single_action_space.sample() for _ in range(self.batch_size)]
        with torch.no_grad():
            q_values = self.model(state_batch)
        return torch.argmax(q_values, dim=1).tolist()

    def learn(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        action_batch = torch.LongTensor(action_batch).to(device)
        target_batch = reward_batch + self.gamma * torch.max(self.model(next_state_batch), dim=1)[0]
        current_q_batch = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        loss = self.criterion(current_q_batch, target_batch)
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def run(self):
        state_batch = self.env.reset()
        state_batch = get_states(state_batch)
        total_reward = 0
        step = 0 # 每 5 step 输出一次
        episode = 0
        while episode < 200:
            actions = self.choose_action(state_batch)
            next_state_batch, rewards, dones, _, infos = self.env.step(actions)
            next_state_batch = get_next_states(next_state_batch)
            dones = np.array(dones)
            rewards = np.array(rewards)
            actions = np.array(actions)
            loss = self.learn(state_batch, actions, rewards, next_state_batch, dones)
            total_reward += sum(rewards)  # 累加非终止状态的奖励
            state_batch = next_state_batch
            if dones.any():
                step += np.sum(dones)
                if step >= batch_size:
                    episode += 1
                    total_reward = total_reward * batch_size / step
                    print(f'Epoch [{episode + 1}/200], Loss: {loss.item()}, Total Reward: {total_reward}')
                    total_reward = 0
                    step = 0
        self.env.close()