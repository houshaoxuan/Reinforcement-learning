import gym
from gym.vector import make as vector_make
from Qlearning import QlearningNN, get_states, batch_size
from plt import plot, save_rewards_to_csv


def main():
    # 创建环境
    env = vector_make("ALE/Breakout-v5", num_envs=batch_size, asynchronous=True)  # 禁用渲染
    qlearning_agent = QlearningNN(env)

    # 训练
    rewards = qlearning_agent.run()

    plot(rewards)
    save_rewards_to_csv(rewards)  # 保存数据到 CSV 文件


    # 训练完成后启用渲染
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    qlearning_agent.env = env

    # 展示结果
    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = qlearning_agent.choose_action(get_states(state))
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f'Total Reward: {total_reward}')
    env.close()


if __name__ == '__main__':
    main()
