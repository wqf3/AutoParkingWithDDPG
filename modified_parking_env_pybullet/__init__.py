from gym.envs.registration import register

register(
    id='modified_parking_env_pybullet-v0',
    entry_point='modified_parking_env_pybullet.env:CustomEnv',
)