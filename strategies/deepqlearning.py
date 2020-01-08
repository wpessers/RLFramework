import math
from random import sample, choice, random
from typing import List

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

from environment import Environment
from percept import Percept
from strategies.qlearnstrategy import QLearnStrategy


class DeepQLearning(QLearnStrategy):
    def __init__(self, batch_size: int, update_interval: int,
                 learning_rate: float, discount_factor: float, expl_prob_max: float, expl_prob_min: float,
                 expl_decay_rate: float, environment: Environment):
        super().__init__(learning_rate, discount_factor, expl_prob_max, expl_prob_min, expl_decay_rate, environment)
        self.model_1 = self.init_model(self.environment.observation_space.n, self.environment.action_space.n)
        self.model_2 = self.init_model(self.environment.observation_space.n, self.environment.action_space.n)
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.buffered_percepts_list = []
        self.update_count = 0

    def improve(self, episode_count: int):
        self.expl_prob = self.expl_prob_min + \
                         (self.expl_prob_max - self.expl_prob_min) * math.e ** (- self.expl_decay_rate * episode_count)

    def init_model(self, states_amount, actions_amount):
        dqn_model = Sequential()
        dqn_model.add(Dense(24, input_shape=(states_amount,), activation="relu"))
        dqn_model.add(Dense(24, activation="relu"))
        dqn_model.add(Dense(actions_amount, activation="linear"))
        dqn_model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return dqn_model

    def learn(self, percept: Percept, episode_count):
        self.buffered_percepts_list.append(
            Percept(percept.current_state, percept.action, percept.next_state, percept.reward, percept.done)
        )
        if len(self.buffered_percepts_list) >= self.batch_size:
            percept_sample = sample(self.buffered_percepts_list, self.batch_size)
            self.learn_from_batch(percept_sample)
            self.buffered_percepts_list.clear()
        self.improve(episode_count)

    def learn_from_batch(self, percept_sample):
        training_set = self.build_training_set(percept_sample)
        self.train_network(training_set)
        self.update_count += 1
        if self.update_count % self.update_interval == 0:
            self.model_2.set_weights(self.model_1.get_weights())

    def build_training_set(self, percept_sample: List[Percept]):
        training_set = []
        for percept in percept_sample:
            state = to_categorical(percept.current_state, num_classes=self.environment.observation_space.n)
            next_state = to_categorical(percept.next_state, num_classes=self.environment.observation_space.n)
            q = self.model_1.predict(state.reshape((1, -1)))
            q_ster = np.amax(self.model_2.predict(next_state.reshape((1, -1))))
            if not percept.done:
                self.qsa[percept.current_state, percept.action] = (percept.reward + self.discount_factor * q_ster)
            else:
                self.qsa[percept.current_state, percept.action] = percept.reward
            training_set.append([state, self.qsa[percept.current_state]])
        return training_set

    def train_network(self, training_set):
        for training_pair in training_set:
            self.model_1.fit(training_pair[0].reshape((1, -1)), training_pair[1].reshape((1, -1)), verbose=0)

    def next_action(self, current_state: int):
        state = to_categorical(current_state, num_classes=self.environment.observation_space.n)
        if random() < self.expl_prob:
            return choice(np.arange(0, self.environment.action_space.n, 1))
        else:
            return np.argmax(self.model_1.predict(state.reshape((1, -1))))


            # user_details = [self.model_1.predict() for i in range(self.environment.action_space.n)]
            # np.zeros(self.environment.action_space.n).map(index => )

