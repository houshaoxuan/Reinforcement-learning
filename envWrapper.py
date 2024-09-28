import gymnasium as gym
import torch
import numpy as np
from torchvision import transforms
from gymnasium.spaces import Box
from typing import List, Dict


class BaseSkipFrame(gym.Wrapper):
    def __init__(
            self,
            env,
            skip: int,
            cut_slices: List[List[int]] = None,
            start_skip: int = None,
            action_map: Dict = None
    ):
        """
        Args:
            env (_type_): _description_
            skip (int): skip frames
            cut_slices (List[List[int]], optional): pic observation cut. Defaults to None.
            start_skip (int, optional): skip several frames to start. Defaults to None.
        """
        super().__init__(env)
        self._skip = skip
        self.pic_cut_slices = cut_slices
        self.start_skip = start_skip
        self.action_map = action_map

    def _get_need_action(self, action):
        if self.action_map is None:
            return action
        return self.action_map[action]

    def _cut_slice(self, obs):
        if self.pic_cut_slices is None:
            return obs
        slice_list = []
        for idx, dim_i_slice in enumerate(self.pic_cut_slices):
            slice_list.append(eval('np.s_[{st}:{ed}]'.format(st=dim_i_slice[0], ed=dim_i_slice[1])))

        obs = obs[tuple(i for i in slice_list)]
        return obs

    def step(self, action):
        action = self._get_need_action(action)
        tt_reward_list = []
        done = False
        total_reward = 0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done_f = terminated or truncated
            total_reward += reward
            tt_reward_list.append(reward)
            if done:
                break

        obs = self._cut_slice(obs) if self.pic_cut_slices is not None else obs
        return obs, total_reward, done_f, truncated, info

    def _start_skip(self):
        for i in range(self.start_skip):
            obs, reward, terminated, truncated, info = self.env.step(0)
        return obs, info

    def reset(self, seed=42, options=None):
        s, info = self.env.reset(seed=seed, options=options)
        if self.start_skip is not None:
            obs, info = self._start_skip()
        obs = self._cut_slice(obs) if self.pic_cut_slices is not None else obs

        return obs, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        """RGP -> Gray
        (high, width, channel) -> (1, high, width)
        """
        super().__init__(env)
        # change observation type for [ sync_vector_env ]
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.float32
        )

    def observation(self, observation):
        tf = transforms.Grayscale()
        observation = tf(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))
        # channel first
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape: int):
        """reshape observe
        Args:
            env (_type_): _description_
            shape (int): reshape size
        """
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        # change observation type for [ sync_vector_env ]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        #  Normalize -> input[channel] - mean[channel]) / std[channel]
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        observation = transformations(observation)
        return observation