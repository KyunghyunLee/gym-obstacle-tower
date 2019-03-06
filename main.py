if __name__ == '__main__':
    import gym
    from gym_obstacle_tower.envs.gym_obstacle_tower_env import GymObstacleTowerEnv
    env = gym.make('obstacle-tower-v0')
    # for i in range(100):
    #     sample_action = env.action_space.sample()
    #     converted_action = env.step(sample_action)
    #     print(sample_action, converted_action)
