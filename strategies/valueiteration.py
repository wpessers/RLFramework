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

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        max_reward = np.amax(self.mdp.Rtsa)

        delta = math.inf
        while delta > self.precision * max_reward * ((1 - self.discount_factor) / self.discount_factor):
            delta = 0
            for state in range(self.mdp.states_amount):
                u = self.v[state]
                self.v[state] = np.amax(self.valuefunction(state))
                delta = max(delta, math.fabs(u - self.v[state]))

    def improve(self, episode_count: int):
        for state in range(self.mdp.states_amount):
            actions = np.empty(self.mdp.actions_amount)
            for action in range(self.mdp.actions_amount):
                sum_value = 0
                for next_state in range(self.mdp.states_amount):
                    sum_value += self.mdp.Ptsa[next_state, state, action] * (
                            self.mdp.Rtsa[next_state, state, action] + self.discount_factor * self.v[next_state]
                    )
                actions[action] = sum_value
            a_max = Functions.argmax_float(actions)
            for action in range(self.mdp.actions_amount):
                if action == a_max:
                    self.policy[state, action] = 1 - self.expl_prob + (self.expl_prob / self.mdp.actions_amount)
                else:
                    self.policy[state, action] = self.expl_prob / self.mdp.actions_amount


        self.expl_prob = self.expl_prob_min + \
                         (self.expl_prob_max - self.expl_prob_min) * math.e ** (- self.expl_decay_rate * episode_count)

    def valuefunction(self, state):
        eu = np.zeros(self.mdp.actions_amount)
        for action in range(self.mdp.actions_amount):
            sum_value = 0
            for next_state in range(self.mdp.states_amount):
                sum_value += self.mdp.Ptsa[next_state, state, action] * (
                        self.mdp.Rtsa[next_state, state, action] + self.discount_factor * self.v[next_state]
                )
            eu[action] = self.policy[state, action] * sum_value
        return eu

    def next_action(self, current_state: int):
        return choice(np.arange(0, self.mdp.actions_amount, 1), 1, p=self.policy[current_state])[0]
