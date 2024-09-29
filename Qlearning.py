import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ReplayBuffer import ReplayBuffer
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 20
latest_episode_num = 5
td_steps = 4  # TD(4)

def get_state(state):
    return torch.FloatTensor(state[0]).to(device)

def get_next_state(next_state):
    return torch.FloatTensor(next_state).to(device)

class DoubleQNetwork(nn.Module):
    def __init__(self, num_actions, in_channels):
        super(DoubleQNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class QlearningNN:
    def __init__(self, env):
        self.env = env
        self.in_channels = 1
        self.num_actions = env.action_space.n
        self.alpha = 0.06  # 学习率
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.992  # 探索率衰减
        self.batch_size = batch_size
        self.model = DoubleQNetwork(self.num_actions, self.in_channels).to(device)
        self.target_model = DoubleQNetwork(self.num_actions, self.in_channels).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.update_target_model()
        self.replay_buffer_size = 10000 # 经验回放缓冲区大小
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)  # 经验回放缓冲区
        self.training_frames = int(1e6) # 训练步数
        self.training_episodes = 2000 # 训练的最大回合数
        self.update_frequency = 4 # 每隔 4 步更新一次
        self.target_network_update_freq = 1000
        self.latest_episode_num = latest_episode_num
        self.print_interval = 100
        self.td_steps = td_steps

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_value = self.model(state.unsqueeze(0))
            return torch.argmax(q_value, dim=1).item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        # 计算当前状态的 Q 值
        current_q_batch = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 使用目标网络计算下一状态的最大 Q 值
        with torch.no_grad():
            next_q_values_target = self.target_model(next_state_batch)
            max_next_q_values = next_q_values_target.max(1)[0]
            target_batch = reward_batch + self.gamma * max_next_q_values * (~done_batch)

        # 计算损失
        loss = self.criterion(current_q_batch, target_batch)
        loss = loss.mean()

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def run(self):
        state = self.env.reset()
        state = get_state(state)
        episode_reward = 0
        total_step = 0
        rewards = []
        latest_rewards = deque(maxlen = self.latest_episode_num)
        episode = 0
        while episode < self.training_episodes:
            action = self.choose_action(state)
            next_state, reward, done, _, info = self.env.step(action)
            next_state = get_next_state(next_state)
            current_td_step = 1
            while current_td_step < self.td_steps and not done:
                action = self.choose_action(next_state)
                next_state, reward_step, done, _, info = self.env.step(action)
                reward += reward_step
                next_state = get_next_state(next_state)
                current_td_step += 1

            self.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward  # 累加非终止状态的奖励
            state = next_state

            if total_step > self.replay_buffer_size:
                self.learn()

            if total_step % self.target_network_update_freq == 0 and total_step > self.replay_buffer_size:
                self.update_target_model()

            total_step += 1
            if done:
                episode += 1
                latest_rewards.append(episode_reward)
                if episode % self.print_interval == 0:
                    mean = np.mean(latest_rewards)
                    print(f'group [{round(episode / self.print_interval)}], episode {episode} '
                          f'Average Reward: {mean} Total Steps: {total_step}')

                if episode % self.latest_episode_num == 0:
                    mean = np.mean(latest_rewards)
                    rewards.append(mean)

                episode_reward = 0
                state = self.env.reset()
                state = get_state(state)

                # 衰减 epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

        self.env.close()
        return rewards

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        print(f'Model loaded from {file_path}')