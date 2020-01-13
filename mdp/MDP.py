import numpy as np

from percept import Percept


class MDP:
    def __init__(self, states_amount, actions_amount):
        self.states_amount = states_amount
        self.actions_amount = actions_amount
        self.Rtsa = np.zeros((states_amount, states_amount, actions_amount), dtype=float)
        self.Nsa = np.zeros((states_amount, actions_amount), dtype=int)
        self.Ntsa = np.zeros((states_amount, states_amount, actions_amount), dtype=int)
        self.Ptsa = np.zeros((states_amount, states_amount, actions_amount), dtype=float)

    def update(self, percept: Percept):
        self.Rtsa[percept.next_state, percept.current_state, percept.action] = percept.reward
        self.Nsa[percept.current_state, percept.action] += 1
        self.Ntsa[percept.next_state, percept.current_state, percept.action] += 1

        for action in range(self.actions_amount):
            if self.Nsa[percept.current_state, int(action)] != 0:
                self.Ptsa[percept.next_state, percept.current_state, int(action)] = \
                    self.Ntsa[percept.next_state, percept.current_state, int(action)] / self.Nsa[percept.current_state, int(action)]
