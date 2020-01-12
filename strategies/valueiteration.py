import math
import time

import numpy as np

from numpy.random import choice

from environment import Environment
from mdp import MDP
from misc_functions import Functions
from percept import Percept
from strategies.learningstrategy import LearningStrategy


class ValueIteration(LearningStrategy):
    def __init__(self, mdp: MDP, precision: float, discount_factor: float, expl_prob_max: float, expl_prob_min: float,
                 expl_decay_rate: float, environment: Environment, learning_rate=0):
        super().__init__(learning_rate, discount_factor, expl_prob_max, expl_prob_min, expl_decay_rate, environment)
        self.mdp = mdp
        self.precision = precision
        self.v = np.zeros(mdp.states_amount)
        self.policy = np.full((mdp.states_amount, mdp.actions_amount), 1 / mdp.actions_amount)

    def learn(self, percept: Percept, episode_count: int):
        self.evaluate(percept)
        self.improve(episode_count)

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        max_reward = np.amax(self.mdp.Rtsa)

        delta = math.inf
        while delta > self.precision * max_reward * ((1-self.discount_factor)/self.discount_factor):
            delta = 0
            for state in range(self.mdp.states_amount):
                u = self.v[state]
                self.v[state] = np.amax(self.valuefunction(state))
                delta = max(delta, math.fabs(u - self.v[state]))

    def improve(self, episode_count: int):
        state = 0
        while state < self.mdp.states_amount:
            a_max = Functions.argmax_float(self.v[state])
            action = 0
            while action < self.mdp.actions_amount:
                if action == a_max:
                    self.policy[state, action] = 1 - self.expl_prob + (self.expl_prob / self.mdp.actions_amount)
                else:
                    self.policy[state, action] = self.expl_prob / self.mdp.actions_amount
                action += 1
            state += 1

        self.expl_prob = self.expl_prob_min + \
                         (self.expl_prob_max - self.expl_prob_min) * math.e ** (- self.expl_decay_rate * episode_count)

    def valuefunction(self, state):
        eu = np.zeros(self.mdp.actions_amount)
        action = 0
        while action < self.mdp.actions_amount:
            sum_value = 0
            next_state = 0
            while next_state < self.mdp.states_amount:
                sum_value += self.mdp.Ptsa[next_state, state, action] * (
                        self.mdp.Rtsa[next_state, state, action] + self.discount_factor * self.v[next_state]
                )
                next_state += 1

            eu[action] = self.policy[state, action] * sum_value
            action += 1
        return eu

    def next_action(self, current_state: int):
        return choice(np.arange(0, self.mdp.actions_amount, 1), 1, p=self.policy[current_state])[0]
