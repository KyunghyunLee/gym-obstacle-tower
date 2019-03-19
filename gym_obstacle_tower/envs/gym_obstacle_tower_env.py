import gym
from gym import error, spaces, utils
from gym.utils import seeding
from obstacle_tower_env import ObstacleTowerEnv
import numpy as np
import pygame
import cv2
import sys


class GymObstacleTowerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    worker_id = 0

    def __init__(self):
        self.local_worker_id = 0
        self._retro = True
        if self._retro:
            print('--------------------------------')
            print('------- Retro Activated --------')
            print('--------------------------------')

        for retry_count in range(20):
            try:
                self.env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=self._retro, realtime_mode=False, worker_id=self.local_worker_id, timeout_wait=30)
                break
            except:
                self.local_worker_id += 1

        self.initialized = False
        self.discrete = False
        self._seed = None

        self.render_enabled = False
        self.recent_obs = None
        self.display = None
        self.clock = None
        self.num_key = 0
        self.remain_time = 0
        self.last_action = []
        self.last_action_raw = -1
        self.render_timesleep = 20
        self.action_meanings = None
        self.action_mask = None
        self.use_action_mask = True
        self.init()
        self.use_preprocessing = False
        self.preprocessing_size = None
        self.game_over = False

    def set_preprocessing(self, size=(84, 84)):
        self.use_preprocessing = True
        self.preprocessing_size = size
        self.observation_space = spaces.Box(shape=size, low=0, high=1, dtype=np.float32)

    def get_action_meanings(self):
        action_meanings = [
            ['NOOP', 'FORWARD', 'BACKWARD'],
            ['NOOP', 'CLOCKWISE', 'C_CLOCKWISE'],
            ['NOOP', 'JUMP'],
            ['NOOP', 'RIGHT', 'LEFT']

        ]
        if not self.discrete:
            self.init(discrete=True)

        if self.action_meanings is None:
            self.action_meanings = []
            for i in range(self.original_action_count):
                if i == 0:
                    action = 'NOOP'
                else:
                    action_vec = self.action_table[i]
                    action = ''
                    for j in range(len(self.action_table[0])):
                        if action_vec[j] != 0:
                            action += action_meanings[j][action_vec[j]]

                self.action_meanings.append(action)

        return self.action_meanings

    def set_timesleep(self, timesleep):
        self.render_timesleep = timesleep

    def seed(self, seed=None):
        self._seed = seed
        # self.env.seed(seed)

    def init(self, discrete=False):
        if self._retro:
            discrete = True
        self.discrete = discrete
        if discrete:
            if self._retro:
                self.original_action_vec = [3, 3, 2, 3]
                self.original_action_count = 54
            else:
                self.original_action_vec = self.env.action_space.nvec
                self.original_action_count = self.original_action_vec.prod()
            self.action_space = spaces.Discrete(self.original_action_count)
            self.action_table = []
            self.action_mask = []

            # for action in range(self.original_action_count):
            #     action_count = int(self.original_action_count)
            #     action_vec = []
            #     for current_space in self.original_action_vec:
            #         action_count /= current_space
            #         action_vec.append(int(action // action_count))
            #         action = action % action_count
            #     self.action_table.append(action_vec)
            #     if self.use_action_mask:
            #         if not (action_vec[1] != 0 and action_vec[3] != 0):  # Camera with left or right
            #             self.action_mask.append(action_vec)
            self.action_mask = [
                [0, 0, 0, 0],  # Noop
                [0, 0, 1, 0],  # Jump
                [0, 1, 0, 0],  # CW
                [0, 2, 0, 0],  # CCW
                [1, 0, 0, 0],  # Forward
                [1, 0, 1, 0],  # Forward Jump
                [1, 1, 0, 0],  # Forward CW
                [1, 2, 0, 0],  # Forward CCW
                [2, 0, 0, 0],  # Backward
            ]
            self.action_mast_index = [
                0, 3, 6, 12, 18, 21, 24, 30, 36
            ]
            if self.use_action_mask:
                self.action_space = spaces.Discrete(len(self.action_mask))

        else:
            self.action_space = self.env.action_space
        if self._retro:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = self.env.observation_space.spaces[0]
        self.initialized = True

    def set_render(self, render):
        if not self.render_enabled and render:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((420, 420), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()

        self.render_enabled = render

    def preprocessor(self, obs):
        if self._retro:
            rgb = obs
        else:
            rgb = obs[0]

        rgb = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        if rgb.shape[0] != self.preprocessing_size[0] or rgb.shape[1] != self.preprocessing_size[1]:
            rgb = cv2.resize(rgb, self.preprocessing_size, interpolation=cv2.INTER_AREA)

        if self._retro:
            rgb = np.float32(rgb) / 255.0
            new_obs = rgb
        else:
            new_obs = (rgb, obs[1], obs[2])
        return new_obs

    def step(self, action):
        if not self.initialized:
            self.init()
        if self._retro:
            if self.use_action_mask:
                action_vec = self.action_mast_index[action]
            else:
                action_vec = action
        else:
            if self.discrete:
                action_vec = self._convert_action(action)
            else:
                action_vec = action

        obs, reward, done, info = self.env.step(action_vec)

        rgb = self._get_obs(obs)

        if self._retro:
            self.num_key = 0
            self.remain_time = 1
        else:
            self.num_key = obs[1]
            self.remain_time = obs[2]
        self.last_action = action_vec
        self.last_action_raw = action


        self.game_over = done
        return rgb, reward, done, info

    def _get_obs(self, obs):
        if self._retro:
            self.recent_obs = obs

            if self.use_preprocessing:
                obs = self.preprocessor(obs)

            rgb = obs
        else:
            self.recent_obs = obs[0]
            if self.use_preprocessing:
                obs = self.preprocessor(obs)

            rgb = obs[0]
        return rgb

    def reset(self):
        if not self.initialized:
            self.init()
        if self._seed is not None:
            # print('seed is fixed to {}'.format(self._seed))
            self.seed(self._seed)

        obs = self.env.reset()


        # rgb = np.uint8(obs[0] * 255)
        rgb = self._get_obs(obs)

        self.game_over = False
        return rgb
        # obs is consist of image, keys, time. Let's use only image for this time

    def render(self, mode='human', close=False):
        if not self.initialized:
            self.init()

        if not self.render_enabled:
            self.set_render(True)

        if self.display is not None and self.recent_obs is not None:
            if self._retro:
                rgb = self.recent_obs
            else:
                rgb = np.uint8(self.recent_obs * 255)
            obs_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            obs_surface = pygame.transform.scale(obs_surface, (420, 420))
            self.display.blit(obs_surface, (0, 0))
            pygame.display.update()
            self.clock.tick_busy_loop(self.render_timesleep)
            if int(self.remain_time) % 1 == 0:
                str1 = '\rkeys: {}, time: {:.2f}, action: {}, action_raw:{}, converted_action:{}'.format(
                    self.num_key, self.remain_time / 100, self.last_action, self.last_action_raw, self._convert_action(self.last_action_raw) )
                sys.stdout.write(str1)
                sys.stdout.flush()
                if self.remain_time == 0:
                    sys.stdout.write('\n')
                    sys.stdout.flush()

    def _convert_action(self, action):
        if self.use_action_mask:
            return self.action_mask[action]
        else:
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
    7 [0, 1, 0, 1] x
    8 [0, 1, 0, 2] x
    9 [0, 1, 1, 0]
    10[0, 1, 1, 1] x
    11[0, 1, 1, 2] x
    12[0, 2, 0, 0]
    13[0, 2, 0, 1] x 
    14[0, 2, 0, 2] x
    15[0, 2, 1, 0]
    16[0, 2, 1, 1] x
    17[0, 2, 1, 2] x
    18[1, 0, 0, 0]
    19[1, 0, 0, 1]
    20[1, 0, 0, 2]
    21[1, 0, 1, 0]
    22[1, 0, 1, 1]
    23[1, 0, 1, 2]
    24[1, 1, 0, 0]
    25[1, 1, 0, 1] x
    26[1, 1, 0, 2] x
    27[1, 1, 1, 0]
    28[1, 1, 1, 1] x 
    29[1, 1, 1, 2] x
    30[1, 2, 0, 0]
    31[1, 2, 0, 1] x
    32[1, 2, 0, 2] x
    33[1, 2, 1, 0]
    34[1, 2, 1, 1] x
    35[1, 2, 1, 2] x
    36[2, 0, 0, 0]
    37[2, 0, 0, 1]
    38[2, 0, 0, 2]
    39[2, 0, 1, 0]
    40[2, 0, 1, 1]
    41[2, 0, 1, 2]
    42[2, 1, 0, 0]
    43[2, 1, 0, 1] x
    44[2, 1, 0, 2] x
    45[2, 1, 1, 0]
    46[2, 1, 1, 1] x
    47[2, 1, 1, 2] x
    48[2, 2, 0, 0]
    49[2, 2, 0, 1] x
    50[2, 2, 0, 2] x
    51[2, 2, 1, 0]
    52[2, 2, 1, 1] x
    53[2, 2, 1, 2] x
    
    Reduced Action space
    
    0 [0, 0, 0, 0]  Noop
    1 [0, 0, 0, 1]  Right
    2 [0, 0, 0, 2]  Left
    3 [0, 0, 1, 0]  Jump
    4 [0, 0, 1, 1]  Right Jump
    5 [0, 0, 1, 2]  Left Jump
    6 [0, 1, 0, 0]  CW
    7 [0, 1, 1, 0]  CW Jump
    8 [0, 2, 0, 0]  CCW
    9 [0, 2, 1, 0]  CCW Jump
    10[1, 0, 0, 0]  Forward
    11[1, 0, 0, 1]  Forward Right
    12[1, 0, 0, 2]  Forward Left
    13[1, 0, 1, 0]  Forward Jump
    14[1, 0, 1, 1]  Forward Jump Right
    15[1, 0, 1, 2]  Forward Jump Left
    16[1, 1, 0, 0]  Forward CW
    17[1, 1, 1, 0]  Forward CW Jump
    18[1, 2, 0, 0]  Forward CCW
    19[1, 2, 1, 0]  Forward CCW Jump
    20[2, 0, 0, 0]  Backward
    21[2, 0, 0, 1]  Backward Right
    22[2, 0, 0, 2]  Backward Left
    23[2, 0, 1, 0]  Backward Jump
    24[2, 0, 1, 1]  Backward Jump Right
    25[2, 0, 1, 2]  Backward Jump Left
    26[2, 1, 0, 0]  Backward CW
    27[2, 1, 1, 0]  Backward CW Jump
    28[2, 2, 0, 0]  Backward CCW
    29[2, 2, 1, 0]  Backward CCW Jump
    
    More Reduced Action space
    0 [0, 0, 1, 0]  Jump
    1 [0, 1, 0, 0]  CW
    2 [0, 2, 0, 0]  CCW
    3 [1, 0, 0, 0]  Forward
    4 [1, 0, 1, 0]  Forward Jump
    5 [1, 1, 0, 0]  Forward CW
    6 [1, 2, 0, 0]  Forward CCW
    7 [2, 0, 0, 0]  Backward

'''
