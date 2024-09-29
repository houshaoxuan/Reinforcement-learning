import matplotlib.pyplot as plt
import csv

from Qlearning import latest_episode_num

color = [
    '#FF6347',
    '#7FFF00',
    '#00FFFF',
    '#9400D3'
]

def plot(data, model):
    x = data[2]['Episode']
    y = data[2]['Reward']
    x = [x*latest_episode_num for x in x]
    plt.scatter(x, y, label='Data Points', s=10)
    # 计算每50个group的平均并绘制

    for [idx, item] in enumerate(data):
        y = item['Reward']
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
        plt.plot(x_50, y_50, color=color[idx], label=f"Reward_{model[idx]}",linewidth=4)
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