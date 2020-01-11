import numpy as np

from abc import ABC, abstractmethod
from numpy.random import choice
from environment import Environment
from percept import Percept


class LearningStrategy(ABC):
    def __init__(self, learning_rate: float, discount_factor: float, expl_prob_max: float, expl_prob_min: float,
                 expl_decay_rate: float, environment: Environment):
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.expl_prob = expl_prob_max  # initieel is epsilon = epsilon_max
        self.expl_prob_max = expl_prob_max  # epsilon max
        self.expl_prob_min = expl_prob_min  # epsilon min
        self.expl_decay_rate = expl_decay_rate  # lambda
        self.environment = environment
        self.policy = np.full((environment.observation_space_size, environment.action_space_size),
                              1 / environment.action_space_size)

    def learn(self, percept: Percept, episode_count):
        self.evaluate(percept)
        self.improve(episode_count)

    @abstractmethod
    def evaluate(self, percept: Percept):
        pass

    @abstractmethod
    def improve(self, episode_count):
        pass

    def next_action(self, current_state: int):
        return choice(np.arange(0, self._environment.action_space.n, 1), 1, p=self._policy[current_state])[0]
