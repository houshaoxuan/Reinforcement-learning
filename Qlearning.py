import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1

def get_states(states):
    states = states[0]
    return torch.stack([torch.FloatTensor(state).permute(2, 0, 1) for state in states]).to(device)

def get_next_states(next_states):
    return torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)

class DoubleQNetwork(nn.Module):
    def __init__(self, num_actions, in_channels):
        super(DoubleQNetwork, self).__init__()
        self.normalize = lambda x: x / 255.0
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(4480, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
class QlearningNN:
    def __init__(self, env):
        self.env = env
        self.in_channels = env.single_observation_space.shape[2]
        self.num_actions = env.single_action_space.n
        shape = env.single_observation_space.shape
        self.state_size = [shape[2], shape[0], shape[1]]
        self.alpha = 0.06  # 学习率
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 0.1  # 探索率
        self.batch_size = batch_size
        self.model = DoubleQNetwork(self.num_actions, self.in_channels).to(device)
        self.target_model = DoubleQNetwork(self.num_actions, self.in_channels).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state_batch):
        if np.random.uniform() < self.epsilon:
            return [self.env.single_action_space.sample() for _ in range(self.batch_size)], None
        else :
            q_values = self.model(state_batch)
            return torch.argmax(q_values, dim=1).tolist(), q_values

    def learn(self, state_batch, action_batch, reward_batch, next_state_batch, q_values):
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        action_batch = torch.LongTensor(action_batch).to(device)

        # 计算当前状态的 Q 值
        if q_values is not None:
            current_q_batch = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        else:
            current_q_batch = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 使用目标网络计算下一状态的最大 Q 值
        with torch.no_grad():
            next_q_values_target = self.target_model(next_state_batch)
            max_next_q_values = next_q_values_target.max(1)[0]

        # 计算目标 Q 值
        target_batch = reward_batch + self.gamma * max_next_q_values

        # 计算损失
        loss = self.criterion(current_q_batch, target_batch)
        loss = loss.sum()

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def run(self):
        rewards = []
        state_batch = self.env.reset()
        state_batch = get_states(state_batch)
        average_reward = 0
        step = 0 # 每 5 step 输出一次
        episode = 0
        while episode < 200:
            actions, q_values = self.choose_action(state_batch)
            next_state_batch, reward_batch, dones, _, infos = self.env.step(actions)
            next_state_batch = get_next_states(next_state_batch)
            dones = np.array(dones)
            reward_batch = np.array(reward_batch)
            actions = np.array(actions)
            self.learn(state_batch, actions, reward_batch, next_state_batch, q_values)
            average_reward += sum(reward_batch)  # 累加非终止状态的奖励
            state_batch = next_state_batch
            if dones.any():
                step += np.sum(dones)
                if step >= batch_size:
                    episode += 1
                    average_reward = round(average_reward / step)
                    print(f'Epoch [{episode}/200], Average Reward: {average_reward}')
                    rewards.append(average_reward)
                    average_reward = 0
                    step = 0
                    self.update_target_model()  # 更新目标网络
        self.env.close()
        return rewards

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        print(f'Model loaded from {file_path}')