import matplotlib.pyplot as plt
import csv

def plot(rewards):
    plt.plot(rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards over Episodes')
    plt.legend()
    plt.show()  # 显示图像

def save_rewards_to_csv(rewards, filename='rewards.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward'])
        for i, reward in enumerate(rewards):
            writer.writerow([i + 1, reward])