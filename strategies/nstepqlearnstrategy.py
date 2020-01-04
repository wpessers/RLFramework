import numpy as np

from environment import Environment
from percept import Percept
from strategies.qlearnstrategy import QLearnStrategy


class NStepQLearnStrategy(QLearnStrategy):
    def __init__(self, learning_rate: float, discount_factor: float, expl_prob_max: float, expl_prob_min: float,
                 expl_decay_rate: float, environment: Environment, steps_amount: int):
        super().__init__(learning_rate, discount_factor, expl_prob_max, expl_prob_min, expl_decay_rate, environment)
        self.steps_amount = steps_amount
        self.buffered_percepts_list = []

    def evaluate(self, percept: Percept):
        self.buffered_percepts_list.append(
            Percept(percept.current_state, percept.action, percept.next_state, percept.reward, percept.done)
        )
        if len(self.buffered_percepts_list) >= self.steps_amount:
            for p in self.buffered_percepts_list:
                self.qsa[p.current_state, p.action] = self.qsa[p.current_state, p.action] + self.learning_rate * (
                        p.reward + self.discount_factor * max(self.qsa[p.next_state]) - self.qsa[
                    p.current_state, p.action]
                )
            self.buffered_percepts_list = []
