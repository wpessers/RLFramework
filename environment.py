import gym

from percept import Percept


class Environment(gym.Wrapper):
    def __init__(self, environment: gym.Wrapper):
        super().__init__(environment)

    @property
    def state(self):
        return self._state

    def step(self, action: int):
        next_state, reward, done, info = super().step(action)
        return next_state, reward, done
