import gym

from Qlearning import QlearningNN
from utils.plt import save_rewards_to_csv
from envWrapper import BaseSkipFrame, GrayScaleObservation, ResizeObservation

def demon_attack_dqn():
    env_name = 'ALE/DemonAttack-v5'
    env = gym.make(env_name, obs_type="rgb")
    env = ResizeObservation(
        GrayScaleObservation(BaseSkipFrame(
            env,
            skip=5,
            cut_slices=[[15, 188], [0, 160]],
            start_skip=14,
            )),
        shape=84
    )
    qlearning_agent = QlearningNN(env)

    # 训练
    rewards = qlearning_agent.run()
    qlearning_agent.save_model("double_dqn_model.pth")
    save_rewards_to_csv(rewards, "experience_back")  # 保存数据到 CSV 文件