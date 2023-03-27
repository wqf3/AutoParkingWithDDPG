from gym.envs.registration import register

register(
    id='newmap_modified_parking_env-v0',
    entry_point='newmap_modified_parking_env.env:CustomEnv',
)