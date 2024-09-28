import gym
import sys
import pandas as pd
from Qlearning import QlearningNN, get_state
from utils.plt import plot
from game import demon_attack_dqn

# pip install "gym[accept-rom-license, atari]"

def main():
    demon_attack_dqn()
    # 创建环境
    # env = gym.make("CarRacing-v2", continuous=False, render_mode="human")  # 禁用渲染
    # qlearning_agent = QlearningNN(env)

    # 训练
    # rewards = qlearning_agent.run()
    # qlearning_agent.save_model("double_dqn_model.pth")
    # save_rewards_to_csv(rewards, "experience_back")  # 保存数据到 CSV 文件

    # plot(rewards)

def draw_rewards(file_path = 'experience_back'):
    data = pd.read_csv(file_path, header=1, names=['Episode', 'Reward'])
    x = data['Episode'].astype(float).tolist()
    y = data['Reward'].astype(float).tolist()

    plot(x, y)

def show_agent_play_trained(file_path = 'double_dqn_model.pth'):
    # 训练完成后启用渲染
    env_name = 'ALE/DemonAttack-v5'
    env = gym.make(env_name, obs_type="rgb", render_mode="human")
    qlearning_agent = QlearningNN(env)
    qlearning_agent.load_model(file_path)
    qlearning_agent.env = env

    # 展示结果
    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = qlearning_agent.choose_action(get_state(state))
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f'Total Reward: {total_reward}')
    env.close()

def show_agent_play_initial():
    # 训练完成后启用渲染
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    qlearning_agent = QlearningNN(env)

    # 展示结果
    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = qlearning_agent.choose_action(get_state(state))
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f'Total Reward: {total_reward}')
    env.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'draw':
            draw_rewards()
        elif sys.argv[1] == 'play':
            show_agent_play_trained()
        elif sys.argv[1] == 'raw_play':
            show_agent_play_initial()
    else :main()
