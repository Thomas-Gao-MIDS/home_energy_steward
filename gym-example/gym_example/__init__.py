from gym.envs.registration import register
register(
    id="henv",
    entry_point="gym_example.envs:HEnv",
)