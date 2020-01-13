import gym
from gym.spaces import Discrete


class Environment(gym.Wrapper):
    def __init__(self, environment: gym.Wrapper):
        super().__init__(environment)
        if isinstance(environment.observation_space, Discrete):
            self.observation_space_size = environment.observation_space.n
        else:
            self.observation_space_size = environment.observation_space.shape[0]
        if isinstance(environment.action_space, Discrete):
            self.action_space_size = environment.action_space.n
        else:
            self.action_space_size = environment.action_space.shape[0]

    @property
    def state(self):
        return self._state

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        return next_state, reward, done
