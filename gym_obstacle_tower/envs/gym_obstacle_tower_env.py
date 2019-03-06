import gym
from gym import error, spaces, utils
from gym.utils import seeding
from obstacle_tower_env import ObstacleTowerEnv
import numpy as np

class GymObstacleTowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    worker_id = 0

    def __init__(self):
        self.env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=False, realtime_mode=False, worker_id=GymObstacleTowerEnv.worker_id)
        GymObstacleTowerEnv.worker_id += 1

        self.original_action_vec = self.env.action_space.nvec
        self.original_action_count = self.original_action_vec.prod()
        self.action_space = spaces.Discrete(self.original_action_count)
        self.action_table = []
        for action in range(self.original_action_count):
            action_count = int(self.original_action_count)
            action_vec = []
            for current_space in self.original_action_vec:
                action_count /= current_space
                action_vec.append(int(action // action_count))
                action = action % action_count
            self.action_table.append(action_vec)

    def step(self, action):
        action_vec = self._convert_action(action)
        obs, reward, done, info = self.env.step(action_vec)
        return np.uint8(obs[0] * 255), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return np.uint8(obs[0] * 255)
        # obs is consist of image, keys, time. Let's use only image for this time

    def render(self, mode='human', close=False):
        pass

    def _convert_action(self, action):
        return self.action_table[action]

'''
Action Space Sample
    Original Action Space
        Movement Forward/Back, Camera, Jump, Movement Left/Right [3 3 2 3]
        
    # action_table = [
        #     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2],
        #     [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 2], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 2],
        #     [0, 2, 0, 0], [0, 2, 0, 1], [0, 2, 0, 2], [0, 2, 1, 0], [0, 2, 1, 1], [0, 2, 1, 2],
        #     [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 2], [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 2],
        #     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 2], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 2],
        #     [1, 2, 0, 0], [1, 2, 0, 1], [1, 2, 0, 2], [1, 2, 1, 0], [1, 2, 1, 1], [1, 2, 1, 2],
        #     [2, 0, 0, 0], [2, 0, 0, 1], [2, 0, 0, 2], [2, 0, 1, 0], [2, 0, 1, 1], [2, 0, 1, 2],
        #     [2, 1, 0, 0], [2, 1, 0, 1], [2, 1, 0, 2], [2, 1, 1, 0], [2, 1, 1, 1], [2, 1, 1, 2],
        #     [2, 2, 0, 0], [2, 2, 0, 1], [2, 2, 0, 2], [2, 2, 1, 0], [2, 2, 1, 1], [2, 2, 1, 2],
        # ]

    0 [0, 0, 0, 0]
    1 [0, 0, 0, 1]
    2 [0, 0, 0, 2]
    3 [0, 0, 1, 0]
    4 [0, 0, 1, 1]
    5 [0, 0, 1, 2]
    6 [0, 1, 0, 0]
    7 [0, 1, 0, 1]
    8 [0, 1, 0, 2]
    9 [0, 1, 1, 0]
    10[0, 1, 1, 1]
    11[0, 1, 1, 2]
    12[0, 2, 0, 0]
    13[0, 2, 0, 1]
    14[0, 2, 0, 2]
    15[0, 2, 1, 0]
    16[0, 2, 1, 1]
    17[0, 2, 1, 2]
    18[1, 0, 0, 0]
    19[1, 0, 0, 1]
    20[1, 0, 0, 2]
    21[1, 0, 1, 0]
    22[1, 0, 1, 1]
    23[1, 0, 1, 2]
    24[1, 1, 0, 0]
    25[1, 1, 0, 1]
    26[1, 1, 0, 2]
    27[1, 1, 1, 0]
    28[1, 1, 1, 1]
    29[1, 1, 1, 2]
    30[1, 2, 0, 0]
    31[1, 2, 0, 1]
    32[1, 2, 0, 2]
    33[1, 2, 1, 0]
    34[1, 2, 1, 1]
    35[1, 2, 1, 2]
    36[2, 0, 0, 0]
    37[2, 0, 0, 1]
    38[2, 0, 0, 2]
    39[2, 0, 1, 0]
    40[2, 0, 1, 1]
    41[2, 0, 1, 2]
    42[2, 1, 0, 0]
    43[2, 1, 0, 1]
    44[2, 1, 0, 2]
    45[2, 1, 1, 0]
    46[2, 1, 1, 1]
    47[2, 1, 1, 2]
    48[2, 2, 0, 0]
    49[2, 2, 0, 1]
    50[2, 2, 0, 2]
    51[2, 2, 1, 0]
    52[2, 2, 1, 1]
    53[2, 2, 1, 2]

'''
