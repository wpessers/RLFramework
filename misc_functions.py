import numpy as np


class Functions:
    @staticmethod
    def argmax_int(a):
        return np.random.choice(np.flatnonzero(a == a.max()))

    @staticmethod
    def argmax_float(a):
        return np.random.choice(np.flatnonzero(np.isclose(a, a.max())))
