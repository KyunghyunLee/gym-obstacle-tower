from gym.envs.registration import register

register(
    id='obstacle-tower-v0',
    entry_point='gym_obstacle-tower.envs:GymObstacleTowerEnv',
)
