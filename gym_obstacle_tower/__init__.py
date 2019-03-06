from gym.envs.registration import register

register(
    id='obstacle-tower-v0',
    entry_point='gym_obstacle_tower.envs:GymObstacleTowerEnv',
)
