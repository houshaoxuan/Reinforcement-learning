import matplotlib.pyplot as plt
import numpy as np
import csv

from Qlearning import latest_episode_num

def plot(x, y):
    x = [x*latest_episode_num for x in x]
    plt.scatter(x, y, label='Data Points', s=10)
    # 计算拟合曲线并绘制
    kandb = np.polyfit(x, y, 1)
    f = np.poly1d(kandb)
    y_fit = f(x)
    plt.plot(x, y_fit, color='#E74C3C',label='Reward', linewidth=4)
    # 计算每50个group的平均并绘制
    num = 50
    x_50 = []
    y_50 = []
    for i in range(0, len(y), num):
        end = i + num
        if end > len(y):
            end = len(y)
        mean = sum(y[i:end]) / (end - i)
        x_50.append(end * latest_episode_num)
        y_50.append(mean)
    plt.plot(x_50, y_50, color='#8A2BE2', label='Reward',linewidth=4)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards over Episodes')
    plt.legend()
    plt.show()  # 显示图像

def save_rewards_to_csv(rewards, filename='rewards.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward'])
        for i, reward in enumerate(rewards):
            writer.writerow([i + 1, reward])