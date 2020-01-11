from typing import List

import numpy as np
from keras.utils import to_categorical

from percept import Percept
from strategies.deepqlearning import DeepQLearning


class DoubleDeepQLearning(DeepQLearning):
    def build_training_set(self, percept_sample: List[Percept]):
        training_set = []
        for percept in percept_sample:
            if isinstance(percept.current_state, np.ndarray):
                state = percept.current_state
                next_state = percept.next_state
            else:
                state = to_categorical(percept.current_state, num_classes=self.environment.observation_space_size)
                next_state = to_categorical(percept.next_state, num_classes=self.environment.observation_space_size)
            q_s = self.model_1.predict(state.reshape((1, self.environment.observation_space_size))).squeeze()
            a_ster = np.argmax(self.model_1.predict(next_state.reshape((1, self.environment.observation_space_size))))
            q_ster = self.model_2.predict(next_state.reshape((1, self.environment.observation_space_size))).squeeze()[a_ster]
            if not percept.done:
                q_s[percept.action] = (percept.reward + self.discount_factor * q_ster)
            else:
                q_s[percept.action] = percept.reward
            training_set.append([state, q_s])
        return training_set