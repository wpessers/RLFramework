import math
import numpy as np

from numpy.random import choice
from environment import Environment
from misc_functions import Functions
from percept import Percept
from strategies.learningstrategy import LearningStrategy


class QLearnStrategy(LearningStrategy):
    def __init__(self, learning_rate: float, discount_factor: float, expl_prob_max: float, expl_prob_min: float,
                 expl_decay_rate: float, environment: Environment):
        super().__init__(learning_rate, discount_factor, expl_prob_max, expl_prob_min, expl_decay_rate, environment)
        #TODO: FECKING OBSERVATION SPACE
        self.qsa = np.zeros((environment.observation_space.shape[0], environment.action_space.n))

    def evaluate(self, percept: Percept):
        self.qsa[percept.current_state, percept.action] = \
            self.qsa[percept.current_state, percept.action] + self.learning_rate * (
                    percept.reward + self.discount_factor * max(self.qsa[percept.next_state]) -
                    self.qsa[percept.current_state, percept.action]
            )

    def improve(self, episode_count: int):
        state = 0
        while state < self.environment.observation_space.n:
            a_max = Functions.argmax_float(self.qsa[state])
            action = 0
            while action < self.environment.action_space.n:
                if action == a_max:
                    self.policy[state, action] = 1 - self.expl_prob + (self.expl_prob / self.environment.action_space.n)
                else:
                    self.policy[state, action] = self.expl_prob / self.environment.action_space.n
                action += 1
            state += 1

        self.expl_prob = self.expl_prob_min + \
                         (self.expl_prob_max - self.expl_prob_min) * math.e ** (- self.expl_decay_rate * episode_count)

    def next_action(self, current_state: int):
        return choice(np.arange(0, self.environment.action_space.n, 1), 1, p=self.policy[current_state])[0]
