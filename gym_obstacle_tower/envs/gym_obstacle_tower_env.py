import gym
from gym import error, spaces, utils
from gym.utils import seeding
from obstacle_tower_env import ObstacleTowerEnv
import numpy as np
import pygame


class GymObstacleTowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    worker_id = 0

    def __init__(self):
        self.local_worker_id = 0
        while True:
            try:
                self.env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=False, realtime_mode=False, worker_id=self.local_worker_id, timeout_wait=30)
                break
            except:
                self.local_worker_id += 1

        self.initialized = False
        self.discrete = False

        self.render_enabled = False
        self.recent_obs = None
        self.display = None
        self.clock = None
        self.num_key = 0
        self.remain_time = 0
        self.last_action = []
        self.last_action_raw = -1
        self.init()

    def init(self, discrete=False):
        self.discrete = discrete
        if discrete:
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
        else:
            self.action_space = self.env.action_space

        self.observation_space = self.env.observation_space.spaces[0]
        self.initialized = True

    def set_render(self, render):
        if not self.render_enabled and render:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((168*8, 168*5), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()

        self.render_enabled = render

    def step(self, action):
        if not self.initialized:
            self.init()

        if self.discrete:
            action_vec = self._convert_action(action)
        else:
            action_vec = action

        obs, reward, done, info = self.env.step(action_vec)
        rgb = np.uint8(obs[0] * 255)

        self.num_key = obs[1]
        self.remain_time = obs[2]
        self.last_action = action_vec
        self.last_action_raw = action

        if self.render_enabled:
            self.recent_obs = rgb

        return rgb, reward, done, info

    def reset(self):
        if not self.initialized:
            self.init()

        obs = self.env.reset()
        rgb = np.uint8(obs[0] * 255)
        if self.render_enabled:
            self.recent_obs = rgb

        return rgb
        # obs is consist of image, keys, time. Let's use only image for this time

    def render(self, mode='human', close=False):
        if not self.initialized:
            self.init()

        if not self.render_enabled:
            self.set_render(True)

        if self.display is not None and self.recent_obs is not None:
            obs_surface = pygame.surfarray.make_surface(self.recent_obs.swapaxes(0, 1))
            obs_surface = pygame.transform.scale(obs_surface, (840, 840))
            self.display.blit(obs_surface, (0, 0))
            pygame.display.update()
            self.clock.tick_busy_loop(20)
            if int(self.remain_time) % 1 == 0:
                print('keys: {}, time: {:.2f}, action: {}, action_raw:{}'.format(
                    self.num_key, self.remain_time / 100, self.last_action, self.last_action_raw))

    def _convert_action(self, action):
        return self.action_table[action]

    @classmethod
    def set_workerid(cls, worker_id):
        cls.worker_id = worker_id

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
